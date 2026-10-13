# SPDX-License-Identifier: Apache-2.0
"""Exact stock FP32 RoPE with BF16 boundary, plus unchanged FP4x writer body."""
import torch
import triton
import triton.language as tl
from glm_fp4_kv_kernel import _parts, _e4m3_rne

@triton.jit
def _rotate(A, C, pos, offsets, AS: tl.constexpr, NEOX: tl.constexpr):
    pair = offsets // 2
    first = (offsets % 2) == 0
    if NEOX:
        pair = offsets % 32
        first = offsets < 32
        xi = pair
        yi = pair + 32
    else:
        xi = 2 * pair
        yi = xi + 1
    x = tl.load(A + xi).to(tl.float32)
    y = tl.load(A + yi).to(tl.float32)
    c = tl.load(C + pos * 64 + pair).to(tl.float32)
    s = tl.load(C + pos * 64 + 32 + pair).to(tl.float32)
    # nvcc default contracts x*c-y*s and y*c+x*s this way. Verify retained PTX.
    xr = tl.fma(x, c, -(y * s))
    yr = tl.fma(x, s, y * c)
    return tl.where(first, xr, yr).to(tl.bfloat16)

@triton.jit
def _rope_pack(Q, R, X, C, P, KV, SLOTS, SCALE,
               QS: tl.constexpr, QH: tl.constexpr, RS: tl.constexpr,
               XS: tl.constexpr, KS: tl.constexpr, N,
               NS, H: tl.constexpr, NEOX: tl.constexpr,
               PACK: tl.constexpr):
    t = tl.program_id(0).to(tl.int64)
    pos = tl.load(P + t)
    rd = tl.arange(0, 64)
    rotated_k = _rotate(R + t * RS, C, pos, rd, RS, NEOX)
    tl.debug_barrier()
    tl.store(R + t * RS + rd, rotated_k)
    qd = tl.arange(0, 1024)
    qhead = qd // 64
    qoff = qd % 64
    # H16 only. All pair values are loaded before the in-place store.
    rotated_q = _rotate(Q + t * QS + qhead * QH, C, pos, qoff, QS, NEOX)
    tl.debug_barrier()
    tl.store(Q + t * QS + qhead * QH + qoff, rotated_q)
    if PACK:
        slot = tl.load(SLOTS + t, mask=t < NS, other=-1).to(tl.int64)
        if (slot >= 0) & (slot < N):
            d = tl.arange(0, 512)
            z = tl.load(X + t * XS + d).to(tl.float32).reshape(32, 16)
            # Bad activations must not silently become a valid cache row.
            tl.device_assert(tl.sum((tl.abs(z) == float('inf')).to(tl.int32), 0) == 0, 'nonfinite FP4 latent')
            tl.device_assert(tl.sum((z != z).to(tl.int32), 0) == 0, 'NaN FP4 latent')
            code, block, row = _parts(z)
            pairs = code.reshape(256, 2)
            packed = (tl.sum(pairs * tl.full((1, 2), 1, tl.int32) * (1 << (tl.arange(0, 2)[None, :] * 4)), 1)).to(tl.uint8)
            base = KV.to(tl.pointer_type(tl.uint8)) + slot * KS
            tl.store(base + tl.arange(0, 256), packed)
            fp8_base = base.to(tl.pointer_type(tl.float8e4nv))
            tl.store(fp8_base + 256 + tl.arange(0, 32), block)
            tl.store((base + 288).to(tl.pointer_type(tl.float32)), row)
            pad = tl.arange(0, 16)
            tl.store(base + 292 + pad, 0, mask=pad < 12)
            rd = tl.arange(0, 64)
            rope = rotated_k.to(tl.bfloat16).to(tl.float32)
            ks = tl.load(SCALE)
            rope = _e4m3_rne(tl.minimum(tl.maximum(tl.div_rn(rope, ks), -448.), 448.))
            tl.store(fp8_base + 304 + rd, rope)


def rope_pack(q, k, x, cache, positions, cos_sin, slots, scale, neox):
    if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16 or x.dtype != torch.bfloat16:
        raise ValueError('attention front requires BF16')
    m = x.shape[0]
    if not 1 <= m <= 16 or q.shape != (m,16,64) or k.shape != (m,1,64) or x.shape != (m,512):
        raise ValueError('attention front requires M1..16/H16/D512/R64')
    if (q.stride(0)!=4096 or q.stride(1)!=256 or k.stride(0) not in (576,2112,2624)
            or x.stride(0)!=512 or any(a.stride(-1)!=1 for a in (q,k,x))):
        raise ValueError('attention front requires measured q/k/latent strides')
    if positions.dtype != torch.int64 or positions.shape != (m,) or not positions.is_contiguous():
        raise ValueError('attention front positions must be contiguous int64 M')
    if cos_sin.dtype not in (torch.bfloat16,torch.float32) or cos_sin.ndim != 2 or cos_sin.shape[1] != 64 or not cos_sin.is_contiguous():
        raise ValueError('attention front cos/sin cache unsupported')
    if cache.dtype != torch.uint8 or cache.shape[-1] != 368 or not cache.is_contiguous():
        raise ValueError('attention front requires FP4x v1 cache')
    if scale.dtype != torch.float32 or scale.numel()!=1:
        raise ValueError('attention front requires FP32 scalar scale')
    ns=0 if slots is None else slots.numel()
    if ns > m or (slots is not None and (not slots.is_contiguous() or slots.dtype != torch.int64)):
        raise ValueError('attention front slot mapping unsupported')
    if any(a.device != q.device for a in (k,x,cache,positions,cos_sin,scale)) or (slots is not None and slots.device!=q.device):
        raise ValueError('attention front mixed devices')
    _rope_pack[(m,)](q,k,x,cos_sin,positions,cache,positions if slots is None else slots,scale,
        q.stride(0),q.stride(1),k.stride(0),x.stride(0),368,cache.numel()//368,
        ns,16,neox,ns>0 and cache.numel()>0,
        num_warps=4,num_stages=1,enable_fp_fusion=False)
