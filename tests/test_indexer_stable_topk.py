"""Mac CPU reference/contract tests. CUDA correctness is a separate kit."""
import ast
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import struct
import sys
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'overlay/bringup'))
import glm_indexer_stable_topk as S


class StableReference(unittest.TestCase):
    def check(self, row, start=0, end=None, k=2048):
        end = len(row) if end is None else end
        self.assertEqual(S.radix_reference(row, start, end, k), S.reference(row, start, end, k))

    def test_all_equal_boundary(self):
        self.assertEqual(S.radix_reference([3.] * 4097, 0, 4097), list(range(2048)))

    def test_tie_after_strict_winners(self):
        row = [1.] * 35000
        for i in range(30000, 30500): row[i] = 2.
        self.check(row)
        self.assertEqual(S.reference(row, 0, len(row)), list(range(30000,30500)) + list(range(1548)))

    def test_long_equal_contexts(self):
        for n in (8192, 35000, 129000):
            with self.subTest(n=n): self.check([0.] * n)

    def test_random_quantized(self):
        rng = random.Random(39)
        for n in (0, 1, 2047, 2048, 2049, 8192, 35000):
            self.check([rng.randrange(-32, 33) / 8 for _ in range(n)])

    def test_random_fp32_bitpatterns(self):
        rng = random.Random(53)
        row = [struct.unpack('<f', struct.pack('<I', rng.getrandbits(32)))[0] for _ in range(9000)]
        self.check(row)

    def test_unique_boundary(self):
        row = list(map(float, range(35000)))
        random.Random(1).shuffle(row)
        self.check(row)

    def test_nonzero_start_relative_index(self):
        row = [float('nan')] * 17 + [1.] * 2050 + [float('inf')] * 3
        self.assertEqual(S.radix_reference(row, 17, 2067), list(range(2048)))

    def test_signed_zero_ties(self):
        row = [-0., 0.] * 1100
        self.assertEqual(S.radix_reference(row, 0, len(row)), list(range(2048)))
        self.assertEqual(S.score_key(-0.), S.score_key(0.))

    def test_infinities_nan(self):
        self.check([float('-inf'), 1., float('inf'), float('nan'), -0., 0.] * 500)

    def test_subnormals(self):
        row = [struct.unpack('<f', struct.pack('<I', b))[0] for b in (0,1,2,0x80000001,0x80000002)] * 500
        self.check(row)

    def test_empty_padding(self):
        self.assertEqual(S.reference([9.] * 50, 3, 3), [-1] * 2048)

    def test_short_padding(self):
        self.assertEqual(S.reference([3., 8., 8.], 0, 3, 5), [1,2,0,-1,-1])
        self.check([3., 8., 8.], k=5)

    def test_bad_bounds(self):
        for start,end,k in ((-1,1,1),(1,0,1),(0,3,1),(0,1,0)):
            for fn in (S.reference, S.radix_reference):
                with self.assertRaises(ValueError): fn([1.,2.],start,end,k)

    def test_row_partitioning(self):
        rng = random.Random(7)
        rows = [[rng.randrange(5) for _ in range(2300)] for _ in range(17)]
        baseline = [S.reference(row,5,2250) for row in rows]
        for boundaries in ((0,17),(0,1,4,11,17),(0,0,3,16,17)):
            result = [S.radix_reference(row,5,2250) for a,b in zip(boundaries,boundaries[1:]) for row in rows[a:b]]
            self.assertEqual(result,baseline)

    def test_score_key_numeric_order(self):
        row = [-math.inf,-3.,-1e-30,-0.,0.,1e-30,3.,math.inf,math.nan]
        keys = [S.score_key(v) for v in row]
        self.assertEqual(keys,sorted(keys))


class Integration(unittest.TestCase):
    def test_default_off_no_imports_or_hooks(self):
        with patch.dict(sys.modules, {'glm_glue_lite': None}):
            self.assertFalse(S.register({}))
            self.assertFalse(S.register({'GLM_INDEXER_STABLE_TOPK':'0'}))

    def test_invalid_flag(self):
        for flag in ('', 'yes', '2', ' 1'):
            with self.assertRaises(ValueError): S.enabled({'GLM_INDEXER_STABLE_TOPK':flag})

    def test_registration_and_order(self):
        calls = []
        glue = types.SimpleNamespace(after_import=lambda name,fn: calls.append((name,fn)))
        with patch.object(S,'REGISTERED',False), patch.dict(sys.modules,{'glm_glue_lite':glue}):
            self.assertTrue(S.register({'GLM_INDEXER_STABLE_TOPK':'1'}))
            self.assertEqual([c[0] for c in calls],[S.OPS,S.SPARSE,S.WORKER])
            with self.assertRaises(RuntimeError): S.register({'GLM_INDEXER_STABLE_TOPK':'1'})

    def test_late_registration_refused(self):
        with patch.dict(sys.modules,{S.OPS:types.SimpleNamespace()}), patch.object(S,'REGISTERED',False):
            with self.assertRaises(RuntimeError): S.register({'GLM_INDEXER_STABLE_TOPK':'1'})

    def test_ops_prefill_only(self):
        stock_decode = object()
        mod = types.SimpleNamespace(top_k_per_row_prefill=object(),top_k_per_row_decode=stock_decode)
        with patch.object(S,'pin'),patch.object(S,'load'),patch.object(S,'INSTALLED',False):
            S.install_ops(mod)
            self.assertIs(mod.top_k_per_row_prefill,S.top_k_per_row_prefill)
            self.assertIs(mod.top_k_per_row_decode,stock_decode)

    def test_source_pins(self):
        pins = json.loads((ROOT/'overlay/bringup/stable_topk/source_pins.json').read_text())
        self.assertEqual(pins,S.PINS)

    def test_stock_call_sites_resolve_ops_dynamically(self):
        image = Path(os.environ.get('GLM_IMAGE_SRC','/tmp/glm-adaptive-image'))
        source = (image/'vllm/model_executor/layers/sparse_attn_indexer.py').read_text()
        fn = next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='sparse_attn_indexer')
        calls = [n for n in ast.walk(fn) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='top_k_per_row_prefill']
        self.assertEqual(len(calls),1)
        self.assertEqual(calls[0].func.value.id,'ops')
        fp4 = (ROOT/'overlay/bringup/glm_fp4_prefill.py').read_text()
        self.assertIn('fp8_fp4_mqa_logits, ops.top_k_per_row_prefill)',fp4)

    def test_launcher_default_and_optin(self):
        spec=importlib.util.spec_from_file_location('stable_cluster', ROOT/'scripts/cluster.py')
        mod=importlib.util.module_from_spec(spec)
        fixture=dict(RECIPE_ROOT=str(ROOT),RECIPE_HOSTS='s1 s2 s3 s4',
                     RECIPE_IPS='192.0.2.1 192.0.2.2 192.0.2.3 192.0.2.4',IMAGE='image:local')
        with patch.dict(os.environ,fixture): spec.loader.exec_module(mod)
        with patch.dict(mod.ENV, {'GLM_INDEXER_STABLE_TOPK':'0'}):
            self.assertNotIn('GLM_INDEXER_STABLE_TOPK',mod.optional_env())
        with patch.dict(mod.ENV, {'GLM_INDEXER_STABLE_TOPK':'1'}):
            self.assertEqual(mod.optional_env()['GLM_INDEXER_STABLE_TOPK'],'1')
        with patch.dict(mod.ENV, {'GLM_INDEXER_STABLE_TOPK':'yes'}):
            with self.assertRaises(ValueError):mod.optional_env()

    def test_no_profiling_in_gpu_kit(self):
        source = (ROOT/'bench/indexer_stable_topk_micro.py').read_text()
        tree = ast.parse(source)
        imported = [n.name for x in ast.walk(tree) if isinstance(x,ast.Import) for n in x.names]
        self.assertFalse(any('profiler' in name for name in imported))
        self.assertIn('torch.cuda.Event(enable_timing=True)',source)


if __name__ == '__main__': unittest.main()
