// SPDX-License-Identifier: Apache-2.0
// Original prefill radix selection; NVIDIA CUB supplies the final block sort.
#include <cuda_runtime.h>
#include <cub/block/block_radix_sort.cuh>
#include <cassert>
#include <cstdint>

namespace glm_stable {
constexpr int THREADS = 256, K = 2048, ITEMS = K / THREADS;
using Sort = cub::BlockRadixSort<uint64_t, THREADS, ITEMS>;

// Ascending numeric order, NaNs above +inf, signed zeros tied. No arithmetic
// on the scores: distinct FP32 values remain distinct (including subnormals).
__device__ __forceinline__ uint32_t score_key(float x) {
  uint32_t b = __float_as_uint(x);
  if ((b & 0x7fffffffU) == 0) b = 0;
  if ((b & 0x7fffffffU) > 0x7f800000U) return 0xffffffffU;
  return b ^ ((b & 0x80000000U) ? 0xffffffffU : 0x80000000U);
}
__device__ __forceinline__ uint64_t key(float x, int index) {
  return (uint64_t(score_key(x)) << 32) | uint32_t(~uint32_t(index));
}

__global__ void select(const float* logits, const int* starts, const int* ends,
                       int* out, int rows, int cols, int64_t s0, int64_t s1,
                       int64_t o0, int64_t o1) {
  const int row = blockIdx.x, t = threadIdx.x;
  const int start = starts[row], end = ends[row], n = end - start;
  assert(start >= 0 && end >= start && end <= cols);
  logits += int64_t(row) * s0 + int64_t(start) * s1;
  out += int64_t(row) * o0;
  __shared__ uint64_t selected[K];
  __shared__ union { int hist[256]; Sort::TempStorage sort; } scratch;
  __shared__ uint64_t prefix;
  __shared__ int remaining, done, count;
  for (int i = t; i < K; i += THREADS) selected[i] = 0;
  if (t == 0) { prefix = 0; remaining = K; done = 0; count = 0; }
  __syncthreads();

  if (n > K) {
    // Integer histogram counts are independent of atomic arrival order.
    // Refine only the threshold bucket, from score bits through index bits.
    // A bucket fitting remaining exactly permits a safe early stop.
    for (int shift = 56; shift >= 0; shift -= 8) {
      scratch.hist[t] = 0;
      __syncthreads();
      const uint64_t mask = shift == 56 ? 0 : (~uint64_t(0) << (shift + 8));
      for (int base = 0; base < n; base += THREADS) {
        const int i = base + t;
        const uint64_t v = i < n ? key(logits[int64_t(i) * s1], i) : 0;
        const bool active = i < n && (v & mask) == prefix;
        // All lanes execute the vote/match. Aggregate equal bins within a
        // warp: an all-equal score row needs eight histogram atomics/round.
        const int bin = active ? int((v >> shift) & 255) : -1;
        const unsigned peers = __match_any_sync(0xffffffffU, bin);
        if (active && (t & 31) == __ffs(peers) - 1)
          atomicAdd(&scratch.hist[bin], __popc(peers));
      }
      __syncthreads();
      if (t == 0) {
        for (int b = 255; b >= 0; --b) {
          const int size = scratch.hist[b];
          if (remaining > size) remaining -= size;
          else {
            prefix |= uint64_t(b) << shift;
            done = size == remaining;
            break;
          }
        }
      }
      __syncthreads();
      if (done) break;
    }
  }
  // Unique 64-bit keys make this set exact. Atomic compaction may permute
  // its staging slots, but the final unique-key sort removes that order.
  for (int i = t; i < n; i += THREADS) {
    const uint64_t v = key(logits[int64_t(i) * s1], i);
    if (n <= K || v >= prefix) {
      const int dst = atomicAdd(&count, 1);
      assert(dst < K);
      selected[dst] = v;
    }
  }
  __syncthreads();
  assert(count == (n < K ? n : K));
  // Emit the selected set in ascending index order: identical to the stock selector's <=K shortcut (so short
  // prompts keep stock attention order) and canonical for longer rows. Valid keys are >= 2^32; 0 = empty slot.
  uint64_t keys[ITEMS];
  #pragma unroll
  for (int j = 0; j < ITEMS; ++j) {
    const uint64_t v = selected[t * ITEMS + j];
    keys[j] = v ? uint64_t(~uint32_t(v)) : ~uint64_t(0);
  }
  __syncthreads();
  Sort(scratch.sort).SortBlockedToStriped(keys);
  #pragma unroll
  for (int j = 0; j < ITEMS; ++j) {
    const int pos = j * THREADS + t;
    out[int64_t(pos) * o1] = pos < count ? int(keys[j]) : -1;
  }
}
}  // namespace glm_stable

extern "C" int glm_stable_topk(const float* logits, const int* starts,
                               const int* ends, int* out, int rows, int cols,
                               int64_t s0, int64_t s1, int64_t o0, int64_t o1,
                               int device, void* stream) {
  cudaError_t err = cudaSetDevice(device);
  if (err != cudaSuccess || !rows) return int(err);
  glm_stable::select<<<rows, glm_stable::THREADS, 0,
      reinterpret_cast<cudaStream_t>(stream)>>>(
          logits, starts, ends, out, rows, cols, s0, s1, o0, o1);
  return int(cudaGetLastError());
}
