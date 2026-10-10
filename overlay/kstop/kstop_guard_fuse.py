# SPDX-License-Identifier: Apache-2.0
"""Fused CUDA packet and verdict kernels for the K-stop equality guard."""
import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = tl = None


if triton is not None:
    @triton.jit
    def _packet_kernel(words_bad, confidence, packet, n: tl.constexpr):
        slot = tl.arange(0, 4)
        bits = tl.load(confidence + slot, mask=slot < n, other=0.0)
        bits = bits.to(tl.int32, bitcast=True).to(tl.int64)
        w0 = tl.load(words_bad + 0)
        w1 = tl.load(words_bad + 1)
        bad = tl.load(words_bad + 2)
        tl.store(packet + 0, w0)
        tl.store(packet + 1, w1)
        tl.store(packet + 2 + slot, bits)
        tl.store(packet + 6, bad)
        tl.store(packet + 7, -w0)
        tl.store(packet + 8, -w1)
        tl.store(packet + 9 + slot, -bits)


    @triton.jit
    def _verdict_kernel(reduced, verdict):
        slot = tl.arange(0, 8)
        equal = ((slot < 6) &
                 (tl.load(reduced + slot, mask=slot < 6, other=0) ==
                  -tl.load(reduced + 7 + slot, mask=slot < 6, other=0)))
        count = tl.sum(equal.to(tl.int32), axis=0)
        tl.store(verdict, (count == 6) & (tl.load(reduced + 6) == 0))


def require_available():
    if triton is None:
        raise RuntimeError('GLM_KSTOP_GUARD_FUSE requires Triton')


def launch_packet(words_bad, confidence, n, packet):
    require_available()
    if n not in (0, 1, 2, 3, 4):
        raise ValueError('guard confidence count')
    _packet_kernel[(1,)](words_bad, confidence, packet, n=n, num_warps=1)
    return packet


def launch_verdict(reduced, verdict):
    require_available()
    _verdict_kernel[(1,)](reduced, verdict, num_warps=1)
    return verdict


def reference_packet(words_bad, confidence, n):
    conf = torch.zeros(4, dtype=torch.int64, device=confidence.device)
    if n:
        conf[:n] = confidence[:n].view(torch.int32).to(torch.int64)
    words_bad = words_bad.to(dtype=torch.int64)
    return torch.cat((words_bad[:2], conf, words_bad[2:3], -words_bad[:2], -conf))


def reference_verdict(reduced):
    return ((reduced[:6] == -reduced[7:]).all() & (reduced[6] == 0)).reshape(1)
