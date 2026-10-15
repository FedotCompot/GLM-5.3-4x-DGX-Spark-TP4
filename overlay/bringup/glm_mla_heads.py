# SPDX-License-Identifier: Apache-2.0
"""TP8 sparse MLA: the released Triton kernels at 8 local heads.

The released kernels tile 16 heads and mask ``h < H``; their wrappers stay
pinned to H16. This module repeats only the wrapper checks and sizes each
reduce grid by the real head count. Kernel bodies are imported, not copied.
"""
import os

import torch
import triton

HEADS = 8
LOG2E = 1.4426950408889634


def _queries(q, qr, rope):
    t, h, d = q.shape
    if h != HEADS or d != 512 or qr.shape[-1] not in rope or qr.shape[:2] != q.shape[:2]:
        raise ValueError('glm-mla-heads requires H8/D512/R' + '|'.join(map(str, rope)))
    if q.dtype != torch.bfloat16 or qr.dtype != torch.bfloat16:
        raise ValueError('glm-mla-heads requires BF16 queries')
    if q.stride(-1) != 1 or qr.stride(-1) != 1:
        raise ValueError('glm-mla-heads requires unit inner query strides')
    return t, h, qr.shape[-1]


def _plain(q, qr, cache, slots):
    t, h, p = _queries(q, qr, (0, 64))
    if cache.dtype not in (torch.bfloat16, torch.float8_e4m3fn) or cache.shape[-1] != 512 + p:
        raise ValueError('glm-mla-heads requires a plain BF16/FP8 latent cache')
    if cache.stride(-1) != 1 or slots.dtype != torch.int32:
        raise ValueError('glm-mla-heads requires unit-stride cache and int32 slots')
    return t, h, p


def _packed(q, qr, cache, slots):
    t, h, p = _queries(q, qr, (64,))
    if cache.dtype != torch.uint8 or cache.shape[-1] != 368 or not cache.is_contiguous():
        raise ValueError('glm-mla-heads requires contiguous uint8 D368 storage')
    if slots.dtype != torch.int32 or slots.shape[0] != t or slots.stride(-1) != 1:
        raise ValueError('glm-mla-heads requires int32 unit-stride slots')
    return t, h, p


def _splits():
    splits = int(os.environ.get('GLM_MLA_SPLIT_K', '32'))
    if splits not in (8, 16, 32):
        raise ValueError('qualified split choices:8/16/32')
    discard = os.environ.get('GLM_DIRTY_L2', '0') == 'discard'
    if discard and splits != 32:
        raise ValueError('dirty-L2 requires split32')
    return splits, discard


def _reduce(acc, md, out, t, h, splits, discard):
    # One program per (head, 64-wide latent chunk) of the live heads only.
    if discard:
        from glm_dirty_l2_kernel import _mla_reduce_discard, LINE_BYTES
        if acc.data_ptr() % LINE_BYTES:
            raise RuntimeError('glm-mla-heads: partial buffer is not 128-byte aligned')
        _mla_reduce_discard[(t, h * 8)](acc, md, out, out.stride(0), splits, NT=128, num_warps=4, num_stages=1)
    else:
        from glm_full_mla_split_kernel import _mla_reduce
        _mla_reduce[(t, h * 8)](acc, md, out, out.stride(0), splits, num_warps=4, num_stages=1)


def mla(q, qr, cache, slots, scale, ckv_scale=1.0):
    """Unsplit plain-cache MLA (glm_full_mla_kernel._mla)."""
    from glm_full_mla_kernel import _mla
    t, h, p = _plain(q, qr, cache, slots)
    flat = cache.reshape(-1, cache.shape[-1])
    out = torch.empty_like(q, memory_format=torch.contiguous_format)
    if t:
        _mla[(t, 1)](q, qr, flat, slots, out, q.stride(0), q.stride(1), qr.stride(0), qr.stride(1),
                     flat.stride(0), slots.stride(0), out.stride(0), h, slots.shape[1], p, flat.shape[0],
                     float(scale) * LOG2E, float(ckv_scale), cache.dtype == torch.float8_e4m3fn,
                     num_warps=4, num_stages=1)
    return out


def split_mla(q, qr, cache, slots, scale, ckv_scale=1.0):
    """Split plain-cache MLA (glm_full_mla_split_kernel._mla_partial)."""
    from glm_full_mla_split_kernel import _mla_partial
    t, h, p = _plain(q, qr, cache, slots)
    splits, discard = _splits()
    flat = cache.reshape(-1, cache.shape[-1])
    w = slots.shape[1]
    seg = triton.cdiv(w, splits * 16) * 16
    out = torch.empty_like(q, memory_format=torch.contiguous_format)
    acc = torch.empty((t, splits, 16, 512), dtype=torch.float32, device=q.device)
    md = torch.empty((t, splits, 2, 16), dtype=torch.float32, device=q.device)
    if t:
        _mla_partial[(t, splits)](q, qr, flat, slots, acc, md, q.stride(0), q.stride(1), qr.stride(0),
                                  qr.stride(1), flat.stride(0), slots.stride(0), out.stride(0), h, w, p,
                                  flat.shape[0], float(scale) * LOG2E, float(ckv_scale),
                                  cache.dtype == torch.float8_e4m3fn, splits, seg, num_warps=4, num_stages=1)
        _reduce(acc, md, out, t, h, splits, discard)
    return out


def packed_mla(q, qr, cache, slots, scale, ckv_scale=1.0):
    """Unsplit FP4x MLA (glm_fp4_mla_kernel._mla)."""
    from glm_fp4_mla_kernel import _mla
    from glm_recent_kv import reader_args
    t, h, p = _packed(q, qr, cache, slots)
    flat = cache.reshape(-1, 368)
    out = torch.empty_like(q, memory_format=torch.contiguous_format)
    if t:
        shadow, mapping, tags, positions, lower, flag, cap = reader_args(cache)
        _mla[(t, 1)](q, qr, flat, slots, out, q.stride(0), q.stride(1), qr.stride(0), qr.stride(1), 368,
                     slots.stride(0), out.stride(0), h, slots.shape[1], p, flat.shape[0], float(scale) * LOG2E,
                     1.0, False, SHADOW=shadow, MAP=mapping, TAG=tags, POS=positions, LOW=lower, FLAG=flag,
                     CAP=cap, num_warps=4, num_stages=1, enable_fp_fusion=False)
    return out


def packed_split_mla(q, qr, cache, slots, scale, ckv_scale=1.0):
    """Split FP4x MLA on the released fixed-address scratch."""
    from glm_fp4_mla_split_kernel import _mla_partial, scratch
    from glm_recent_kv import reader_args
    t, h, p = _packed(q, qr, cache, slots)
    splits, discard = _splits()
    flat = cache.reshape(-1, 368)
    w = slots.shape[1]
    seg = triton.cdiv(w, splits * 16) * 16
    out = torch.empty_like(q, memory_format=torch.contiguous_format)
    if t:
        acc, md = scratch(q, splits)
        shadow, mapping, tags, positions, lower, flag, cap = reader_args(cache)
        _mla_partial[(t, splits)](q, qr, flat, slots, acc, md, q.stride(0), q.stride(1), qr.stride(0),
                                  qr.stride(1), 368, slots.stride(0), out.stride(0), h, w, p, flat.shape[0],
                                  float(scale) * LOG2E, 1.0, False, splits, seg, SHADOW=shadow, MAP=mapping,
                                  TAG=tags, POS=positions, LOW=lower, FLAG=flag, CAP=cap,
                                  num_warps=4, num_stages=1, enable_fp_fusion=False)
        _reduce(acc, md, out, t, h, splits, discard)
    return out


def packed_prefill(q, qr, cache, slots, scale):
    """FP4x union prefill: the released expansion, then the H8 plain reader."""
    import glm_fp4_mla_prefill as released
    from glm_fp4_prefill import bank_for, MLA_ROWS
    from glm_recent_kv import reader_args
    t, h, p = _packed(q, qr, cache, slots)
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError('FP4x union prefill must be eager')
    n = cache.numel() // 368
    if n > MLA_ROWS:
        raise RuntimeError('FP4x prefill exceeds release physical cache capacity')
    bank = bank_for(q.device)
    selected = bank['selected'][:n]
    selected.zero_()
    plain = bank['workspace'][:n * 576 * 2].view(torch.bfloat16).view(n, 576)
    if t:
        shadow, mapping, tags, positions, lower, flag, cap = reader_args(cache)
        released._mark_selected[(t, triton.cdiv(slots.shape[1], 256))](
            slots, selected, slots.stride(0), slots.shape[1], n, num_warps=4)
        released._expand_selected[(triton.cdiv(n, 16),)](
            cache, selected, plain, n, SHADOW=shadow, MAP=mapping, TAG=tags, POS=positions, LOW=lower,
            FLAG=flag, CAP=cap, num_warps=4, num_stages=1, enable_fp_fusion=False)
    return mla(q, qr, plain, slots, scale, 1.0)


def forward(impl, q, cache, slots, md, ks, packed):
    """Same prefill/decode/split routing as glm_full_mla's H16 forward."""
    qn, qr = q
    t = qn.shape[0]
    split = (os.environ.get('GLM_MLA_SPLIT_K', '0') != '0'
             and t <= int(os.environ.get('GLM_MLA_SPLIT_MAX_ROWS', '36')))
    if packed and getattr(md, 'num_prefills', 0) and t > md.num_decode_tokens:
        nd = md.num_decode_tokens
        if not 0 <= nd <= t:
            raise RuntimeError('FP4x mixed prefill token bounds drift')
        if not nd:
            return packed_prefill(qn, qr, cache, slots, impl.scale)
        out = torch.empty_like(qn, memory_format=torch.contiguous_format)
        decode = packed_split_mla if split else packed_mla
        out[:nd].copy_(decode(qn[:nd], qr[:nd], cache, slots[:nd], impl.scale, ks))
        out[nd:].copy_(packed_prefill(qn[nd:], qr[nd:], cache, slots[nd:], impl.scale))
        return out
    if packed:
        return (packed_split_mla if split else packed_mla)(qn, qr, cache, slots, impl.scale, ks)
    return (split_mla if split else mla)(qn, qr, cache, slots, impl.scale, ks)
