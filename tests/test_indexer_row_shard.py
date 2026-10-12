"""Stdlib-only CPU tests; never import CUDA, connect to fleet, or sort indices."""
import ast
import hashlib
import os
from pathlib import Path
import random
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'overlay/bringup'))
import glm_indexer_row_shard as R
import glm_dsa_short as D
import glm_fp4_prefill as P
SRC = Path(os.environ.get('GLM_IMAGE_SRC', '/tmp/glm-adaptive-image/vllm'))


class Partition(unittest.TestCase):
    def test_even_chunks(self):
        self.assertEqual(R.partition(2048), ((0,512),(512,1024),(1024,1536),(1536,2048)))
        self.assertEqual(R.partition(4096), ((0,1024),(1024,2048),(2048,3072),(3072,4096)))

    def test_uneven_and_empty(self):
        self.assertEqual(R.partition(7), ((0,2),(2,4),(4,6),(6,7)))
        self.assertEqual(R.partition(1), ((0,1),(1,1),(1,1),(1,1)))
        self.assertEqual(R.partition(0), ((0,0),)*4)

    def test_original_tile_ownership(self):
        self.assertEqual(R.partition(2048, quantum=1024), ((0,1024),(1024,2048),(2048,2048),(2048,2048)))
        for n in (2048,2049,3072,4096):
            for quantum in (256,512,1024,2048):
                slices = R.partition(n, quantum=quantum)
                original = list(range(0,n,quantum))
                owned = [o for begin,end in slices for o in range(begin,end,quantum)]
                self.assertEqual(original, owned)

    def test_fuzz_cover(self):
        rng = random.Random(39)
        for _ in range(1500):
            n,w,q = rng.randrange(5000),rng.randrange(1,9),rng.choice((1,128,256,512,1024))
            slices = R.partition(n,w,q)
            self.assertEqual(slices[0][0],0)
            self.assertEqual(slices[-1][1],n)
            self.assertTrue(all(a <= b for a,b in slices))
            self.assertTrue(all(slices[i][1] == slices[i+1][0] for i in range(w-1)))

    def test_bad_geometry(self):
        for args in ((-1,), (1,0), (1,4,0), (1.2,)):
            with self.assertRaises(ValueError): R.partition(*args)

    def test_metadata_splits_gate_on_scheduled_rows(self):
        spans = [1040,1040,1040,976]
        tasks,owners,slices = R.tile_plan(spans,[512]*4)
        self.assertEqual(len(tasks),11)
        self.assertEqual(sum(b-a for a,b in slices),4096)
        self.assertEqual([b-a for a,b in owners],[3,3,3,2])
        # No inner chunk reaches 2048; aggregate gate must still activate.
        with patch.object(R,'MODE','check'):
            self.assertTrue(R.eligible(sum(spans),129000))
        compiled = R.transform((SRC/'model_executor/layers/sparse_attn_indexer.py').read_text())
        self.assertIn('scheduled_rows=attn_metadata_narrowed.num_prefill_tokens',compiled)

    def test_multiple_requests_tile_plan(self):
        tasks,owners,slices = R.tile_plan([2048,2048],[512,1024])
        self.assertEqual(tasks,[(0,0,512,512),(0,512,512,512),(0,1024,512,512),
                               (0,1536,512,512),(1,0,1024,1024),(1,1024,1024,1024)])
        rows=list(range(4096))
        self.assertEqual(R.reassemble([rows[a:b] for a,b in slices],slices),rows)
        with self.assertRaises(ValueError): R.tile_plan([0],[512])


class Gather(unittest.TestCase):
    def test_ties_and_sentinels_keep_order(self):
        # Equal-score indices intentionally unsorted; bytes must stay as emitted.
        rows = [[r,9,2,8,0,-1,-1] for r in range(13)]
        slices = R.partition(len(rows))
        parts = [rows[a:b] + [[999]*7]*(4-(b-a)) for a,b in slices]
        self.assertEqual(R.reassemble(parts,slices),rows)
        self.assertNotEqual(R.reassemble(parts,slices)[0],sorted(rows[0]))

    def test_zero_rank_and_partial_tile(self):
        rows = [(r, -1, 3) for r in range(2051)]
        slices = R.partition(len(rows),quantum=1024)
        parts = [rows[a:b]+[('pad',)]*1024 for a,b in slices]
        self.assertEqual(R.reassemble(parts,slices),rows)

    def test_fuzz_exact(self):
        for n in range(0,200):
            rows = [(r,3,0,-1) for r in range(n)]
            slices = R.partition(n,quantum=17)
            parts = [rows[a:b]+[None]*50 for a,b in slices]
            self.assertEqual(R.reassemble(parts,slices),rows)

    def test_bad_gather(self):
        for parts,slices in (([],[]), ([[1]],[(1,2)]), ([[1],[]],[(0,1),(2,2)]), ([[]],[(0,1)])):
            with self.assertRaises(ValueError): R.reassemble(parts,slices)

    def test_payload_is_per_rank_limit(self):
        slices = R.partition(4096)
        size = max(b-a for a,b in slices)*2048*4
        self.assertEqual(size,8<<20)
        self.assertEqual(size*4,32<<20)
        code = (ROOT/'roce/b12x/b12x/comm/roce/roce_oneshot.py').read_text()
        self.assertIn('nbytes = inp.numel() * inp.element_size()',code)
        self.assertIn('return 0 < nbytes <= self.max_gather_bytes',code)


class Policy(unittest.TestCase):
    def test_default_off(self):
        before = list(sys.meta_path)
        self.assertFalse(R.register({}))
        self.assertFalse(R.register({'GLM_INDEXER_ROW_SHARD':'0'}))
        self.assertEqual(before,sys.meta_path)

    def test_no_unchecked_mode(self):
        for mode in ('on','fast','1','unchecked'):
            with self.assertRaises(ValueError):
                R.options(dict(GLM_INDEXER_ROW_SHARD='1',GLM_INDEXER_ROW_SHARD_INIT=mode,VLLM_SERVER_DEV_MODE='1'))

    def test_options(self):
        self.assertEqual(R.options(dict(GLM_INDEXER_ROW_SHARD='1',VLLM_SERVER_DEV_MODE='1')),('off','nccl'))
        with self.assertRaises(ValueError): R.options({'GLM_INDEXER_ROW_SHARD':'yes'})
        with self.assertRaises(ValueError): R.options({'GLM_INDEXER_ROW_SHARD':'1'})
        with self.assertRaises(ValueError): R.options({'GLM_INDEXER_ROW_SHARD_TRANSPORT':'auto'})

    def test_gate_2048_4096_and_depth(self):
        with patch.object(R,'MODE','check'):
            for rows in (2048,4096): self.assertTrue(R.eligible(rows,35000))
            for rows in (1,16,2047,4097): self.assertFalse(R.eligible(rows,129000))
            self.assertFalse(R.eligible(4096,16384))
            self.assertTrue(R.eligible(4096,16385))

    def test_gate_unsupported_and_graph(self):
        with patch.object(R,'MODE','check'):
            for kw in (dict(world=1),dict(capture=True),dict(pcp=True),dict(dcp=2),
                       dict(fp4_cache=True),dict(topk=1024),dict(head_dim=64)):
                self.assertFalse(R.eligible(4096,129000,**kw))
        with patch.object(R,'MODE','off'): self.assertFalse(R.eligible(4096,129000))

    def test_runtime_selection(self):
        with patch.object(R,'REGISTERED',True),patch.object(R,'MODE','off'):
            for mode in ('self','check','off'): self.assertEqual(R.set_mode(mode)['mode'],mode)
            with self.assertRaises(ValueError): R.set_mode('fast')


class Source(unittest.TestCase):
    def sparse(self): return (SRC/'model_executor/layers/sparse_attn_indexer.py').read_text()

    def test_pins(self):
        for module,digest in R.PINS.items():
            path = SRC/(module.removeprefix('vllm.').replace('.','/')+'.py')
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),digest)

    def test_transform_compile_and_decode_preserved(self):
        raw = self.sparse()
        compiled = R.transform(raw)
        compile(compiled,'<row-shard>','exec')
        original_fn = next(n for n in ast.parse(raw).body if isinstance(n,ast.FunctionDef) and n.name=='sparse_attn_indexer')
        original = ''.join(raw.splitlines(keepends=True)[original_fn.lineno-1:original_fn.end_lineno])
        self.assertEqual(compiled[compiled.index('    if has_decode:'):],original[original.index('    if has_decode:'):])
        self.assertIn('ops.indexer_k_quant_and_cache(',compiled)
        self.assertIn('cp_gather_indexer_k_quant_cache(',compiled)

    def test_short_composition(self):
        compiled = R.transform(D.transform_sparse(self.sparse()))
        compile(compiled,'<short+row-shard>','exec')
        self.assertIn('elif _dsa_short:',compiled)
        self.assertIn('_row_score(',compiled)

    def test_fp4_compile_short_composition(self):
        compiled = P.transform_indexer(D.transform_sparse(self.sparse()))
        compile(compiled,'<fp4+short>','exec')
        self.assertIn('tiled_prefill(',compiled)
        self.assertIn('elif _dsa_short:',compiled)

    def test_register_last(self):
        source = (ROOT/'overlay/bringup/sitecustomize.py').read_text()
        row = source.index('glm_indexer_row_shard.register()')
        for hook in ('glm_fp4_kv.register()','glm_dsa_short.register()','glm_mtp_kstop.register()','glm_glue_lite.register()'):
            self.assertGreater(row,source.index(hook))

    def test_strict_guard_no_sort_or_dump(self):
        source = (ROOT/'overlay/bringup/glm_indexer_row_shard.py').read_text()
        self.assertIn('(candidate != reference).any()',source)
        self.assertIn('dist.all_reduce(bad, op=dist.ReduceOp.MAX',source)
        self.assertNotIn('.sort(',source)
        self.assertNotIn('torch.save(',source)


if __name__ == '__main__': unittest.main()
