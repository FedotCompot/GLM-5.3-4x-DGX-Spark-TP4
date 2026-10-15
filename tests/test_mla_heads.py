"""TP8 sparse MLA (8 local heads) against the released H16 kernels in the Triton interpreter.

Needs torch and triton; runs with TRITON_INTERPRET=1 and no GPU. Heads are independent rows of
the released 16-row tile, so H8 must equal the first eight heads of an H16 call bit for bit.
This proves masking, grids and scratch indexing only: the CPU interpreter's BF16 tl.dot is not
numerically faithful (triton 3.5 and 3.8), so GB10 numerics need the GPU gate.
"""
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

os.environ.setdefault('TRITON_INTERPRET', '1')
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'overlay/bringup'))
import glm_mla_heads as H8
import glm_full_mla_kernel
import glm_full_mla_split_kernel
import glm_fp4_mla_kernel
import glm_fp4_mla_split_kernel
from glm_fp4_kv_kernel import _pack_store

SCALE = 0.0625


def queries(t, heads, gen):
    q = torch.randn(t, heads, 512, generator=gen).bfloat16()
    qr = torch.randn(t, heads, 64, generator=gen).bfloat16()
    return q, qr


def slots(t, width, n, gen):
    s = torch.randint(0, n, (t, width), generator=gen, dtype=torch.int32)
    s[:, -5:] = -1
    s[0, 3] = n + 7
    return s


def first8(q):
    return q[:, :8].contiguous()


def packed_cache(n, gen):
    x = torch.randn(n, 512, generator=gen).bfloat16()
    rope = torch.randn(n, 64, generator=gen).bfloat16()
    cache = torch.zeros(n, 368, dtype=torch.uint8)
    index = torch.arange(n, dtype=torch.int64)
    _pack_store[(n,)](x, rope, cache, index, torch.ones(1), x.stride(0), rope.stride(0), 368, n, n,
                      num_warps=4, num_stages=1, enable_fp_fusion=False)
    return cache


class Plain(unittest.TestCase):
    def setUp(self):
        self.gen = torch.Generator().manual_seed(8080)
        self.cache = torch.randn(96, 576, generator=self.gen).bfloat16()
        self.q, self.qr = queries(3, 16, self.gen)
        self.s = slots(3, 80, 96, self.gen)

    def test_unsplit_equals_released_first_heads(self):
        stock = glm_full_mla_kernel.sparse_mla(self.q, self.qr, self.cache, self.s, SCALE)
        mine = H8.mla(first8(self.q), first8(self.qr), self.cache, self.s, SCALE)
        self.assertTrue(torch.equal(mine, stock[:, :8]))
        self.assertFalse(torch.equal(mine[:, 0], mine[:, 1]))

    def test_split_equals_released_first_heads(self):
        for splits in ('8', '32'):
            with self.subTest(splits=splits), patch.dict(os.environ, GLM_MLA_SPLIT_K=splits, GLM_DIRTY_L2='0'):
                stock = glm_full_mla_split_kernel.sparse_mla(self.q, self.qr, self.cache, self.s, SCALE)
                mine = H8.split_mla(first8(self.q), first8(self.qr), self.cache, self.s, SCALE)
                self.assertTrue(torch.equal(mine, stock[:, :8]))

    def test_fp8_cache_and_scale(self):
        cache = self.cache.to(torch.float8_e4m3fn)
        stock = glm_full_mla_split_kernel.sparse_mla(self.q, self.qr, cache, self.s, SCALE, 0.5)
        mine = H8.split_mla(first8(self.q), first8(self.qr), cache, self.s, SCALE, 0.5)
        self.assertTrue(torch.equal(mine, stock[:, :8]))

    def test_refuses_other_shapes(self):
        with self.assertRaisesRegex(ValueError, 'H8'):
            H8.mla(self.q, self.qr, self.cache, self.s, SCALE)
        with patch.dict(os.environ, GLM_MLA_SPLIT_K='4'), self.assertRaisesRegex(ValueError, 'split'):
            H8.split_mla(first8(self.q), first8(self.qr), self.cache, self.s, SCALE)
        with patch.dict(os.environ, GLM_MLA_SPLIT_K='16', GLM_DIRTY_L2='discard'), \
                self.assertRaisesRegex(ValueError, 'split32'):
            H8.split_mla(first8(self.q), first8(self.qr), self.cache, self.s, SCALE)


class Packed(unittest.TestCase):
    def setUp(self):
        self.gen = torch.Generator().manual_seed(8081)
        self.cache = packed_cache(80, self.gen)
        self.q, self.qr = queries(2, 16, self.gen)
        self.s = slots(2, 64, 80, self.gen)
        glm_fp4_mla_split_kernel._SCRATCH.clear()
        self.enterContext(patch.object(torch.cuda, 'is_current_stream_capturing', lambda: False))

    def test_unsplit_equals_released_first_heads(self):
        stock = glm_fp4_mla_kernel.sparse_mla(self.q, self.qr, self.cache, self.s, SCALE)
        mine = H8.packed_mla(first8(self.q), first8(self.qr), self.cache, self.s, SCALE)
        self.assertTrue(torch.equal(mine, stock[:, :8]))

    def test_split_equals_released_first_heads(self):
        with patch.dict(os.environ, GLM_MLA_SPLIT_K='32', GLM_DIRTY_L2='0', GLM_MLA_SPLIT_MAX_ROWS='36'):
            stock = glm_fp4_mla_split_kernel.sparse_mla(self.q, self.qr, self.cache, self.s, SCALE)
            mine = H8.packed_split_mla(first8(self.q), first8(self.qr), self.cache, self.s, SCALE)
        self.assertTrue(torch.equal(mine, stock[:, :8]))

    def test_refuses_plain_cache(self):
        with self.assertRaisesRegex(ValueError, 'uint8 D368'):
            H8.packed_mla(first8(self.q), first8(self.qr), self.cache.view(torch.bfloat16)[:, :184], self.s, SCALE)


class Routing(unittest.TestCase):
    def test_forward_routes_like_release(self):
        calls = []
        names = ('mla', 'split_mla', 'packed_mla', 'packed_split_mla')
        stubs = {n: (lambda n: lambda *a: calls.append(n) or a[0])(n) for n in names}
        stubs['packed_prefill'] = lambda q, *a: calls.append('packed_prefill') or q
        impl = type('Impl', (), {'scale': SCALE})()
        md = type('Md', (), {'num_prefills': 0, 'num_decode_tokens': 0})()
        q = (torch.zeros(4, 8, 512), torch.zeros(4, 8, 64))
        with patch.multiple(H8, **stubs):
            for split, packed, expect in (('32', False, 'split_mla'), ('0', False, 'mla'),
                                          ('32', True, 'packed_split_mla'), ('0', True, 'packed_mla')):
                with patch.dict(os.environ, GLM_MLA_SPLIT_K=split, GLM_MLA_SPLIT_MAX_ROWS='36'):
                    calls.clear()
                    H8.forward(impl, q, None, torch.zeros(4, 1), md, 1.0, packed)
                    self.assertEqual(calls, [expect])
            md.num_prefills, md.num_decode_tokens = 1, 1
            with patch.dict(os.environ, GLM_MLA_SPLIT_K='32', GLM_MLA_SPLIT_MAX_ROWS='36'):
                calls.clear()
                out = H8.forward(impl, q, None, torch.zeros(4, 1), md, 1.0, True)
                self.assertEqual(calls, ['packed_split_mla', 'packed_prefill'])
                self.assertEqual(out.shape, q[0].shape)
                md.num_decode_tokens = 0
                calls.clear()
                H8.forward(impl, q, None, torch.zeros(4, 1), md, 1.0, True)
                self.assertEqual(calls, ['packed_prefill'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
