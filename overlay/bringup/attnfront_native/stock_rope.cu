// SPDX-License-Identifier: Apache-2.0
// Unmodified vLLM device code, cache/type instantiations only.
#include <torch/headeronly/util/BFloat16.h>
#include <cuda_runtime.h>
#define VLLM_LDG __ldg
namespace vllm {

template <typename scalar_t, typename cache_t, bool IS_NEOX>
inline __device__ void apply_token_rotary_embedding(
    scalar_t* __restrict__ arr, const cache_t* __restrict__ cos_ptr,
    const cache_t* __restrict__ sin_ptr, int rot_offset, int embed_dim,
    const bool inverse) {
  int x_index, y_index;
  float cos_f, sin_f;
  if (IS_NEOX) {
    x_index = rot_offset;
    y_index = embed_dim + rot_offset;
    cos_f = static_cast<float>(VLLM_LDG(cos_ptr + x_index));
    sin_f = static_cast<float>(VLLM_LDG(sin_ptr + x_index));
  } else {
    x_index = 2 * rot_offset;
    y_index = 2 * rot_offset + 1;
    cos_f = static_cast<float>(VLLM_LDG(cos_ptr + x_index / 2));
    sin_f = static_cast<float>(VLLM_LDG(sin_ptr + x_index / 2));
  }
  if (inverse) {
    sin_f = -sin_f;
  }
  const float x_f = static_cast<float>(arr[x_index]);
  const float y_f = static_cast<float>(arr[y_index]);
  arr[x_index] = static_cast<scalar_t>(x_f * cos_f - y_f * sin_f);
  arr[y_index] = static_cast<scalar_t>(y_f * cos_f + x_f * sin_f);
}

template <typename scalar_t, typename cache_t, bool IS_NEOX>
inline __device__ void apply_rotary_embedding(
    scalar_t* __restrict__ query,  // [batch_size, seq_len, num_heads,
                                   // head_size] or [num_tokens, num_heads,
                                   // head_size]
    scalar_t* __restrict__ key,    // nullptr or
                                   // [batch_size, seq_len, num_kv_heads,
                                   // head_size] or [num_tokens, num_kv_heads,
                                   // head_size]
    const cache_t* cache_ptr, const int head_size, const int num_heads,
    const int num_kv_heads, const int rot_dim, const int token_idx,
    const int64_t query_stride, const int64_t key_stride,
    const int64_t head_stride, const int64_t rope_dim_offset,
    const bool inverse) {
  const int embed_dim = rot_dim / 2;
  const cache_t* cos_ptr = cache_ptr;
  const cache_t* sin_ptr = cache_ptr + embed_dim;

  const int nq = num_heads * embed_dim;
  for (int i = threadIdx.x; i < nq; i += blockDim.x) {
    const int head_idx = i / embed_dim;
    const int64_t token_head =
        token_idx * query_stride + head_idx * head_stride + rope_dim_offset;
    const int rot_offset = i % embed_dim;
    apply_token_rotary_embedding<scalar_t, cache_t, IS_NEOX>(
        query + token_head, cos_ptr, sin_ptr, rot_offset, embed_dim, inverse);
  }

  if (key != nullptr) {
    const int nk = num_kv_heads * embed_dim;
    for (int i = threadIdx.x; i < nk; i += blockDim.x) {
      const int head_idx = i / embed_dim;
      const int64_t token_head =
          token_idx * key_stride + head_idx * head_stride + rope_dim_offset;
      const int rot_offset = i % embed_dim;
      apply_token_rotary_embedding<scalar_t, cache_t, IS_NEOX>(
          key + token_head, cos_ptr, sin_ptr, rot_offset, embed_dim, inverse);
    }
  }
}

template <typename scalar_t, typename cache_t, bool IS_NEOX>
__global__ void rotary_embedding_kernel(
    const int64_t* __restrict__ positions,  // [batch_size, seq_len] or
                                            // [num_tokens]
    scalar_t* __restrict__ query,           // [batch_size, seq_len, num_heads,
                                   // head_size] or [num_tokens, num_heads,
                                   // head_size]
    scalar_t* __restrict__ key,  // nullptr or
                                 // [batch_size, seq_len, num_kv_heads,
                                 // head_size] or [num_tokens, num_kv_heads,
                                 // head_size]
    const cache_t* __restrict__ cos_sin_cache,  // [max_position, rot_dim]
    const int rot_dim, const int64_t query_stride, const int64_t key_stride,
    const int64_t head_stride, const int num_heads, const int num_kv_heads,
    const int head_size, const int64_t rope_dim_offset, const bool inverse) {
  const int token_idx = blockIdx.x;
  int64_t pos = positions[token_idx];
  const cache_t* cache_ptr = cos_sin_cache + pos * rot_dim;

  apply_rotary_embedding<scalar_t, cache_t, IS_NEOX>(
      query, key, cache_ptr, head_size, num_heads, num_kv_heads, rot_dim,
      token_idx, query_stride, key_stride, head_stride, rope_dim_offset,
      inverse);
}

}  // namespace vllm


using bf16=torch::headeronly::BFloat16;
template __global__ void vllm::rotary_embedding_kernel<bf16,bf16,true>(const int64_t*,bf16*,bf16*,const bf16*,int,int64_t,int64_t,int64_t,int,int,int,int64_t,bool);
template __global__ void vllm::rotary_embedding_kernel<bf16,bf16,false>(const int64_t*,bf16*,bf16*,const bf16*,int,int64_t,int64_t,int64_t,int,int,int,int64_t,bool);
template __global__ void vllm::rotary_embedding_kernel<bf16,float,true>(const int64_t*,bf16*,bf16*,const float*,int,int64_t,int64_t,int64_t,int,int,int,int64_t,bool);
template __global__ void vllm::rotary_embedding_kernel<bf16,float,false>(const int64_t*,bf16*,bf16*,const float*,int,int64_t,int64_t,int64_t,int,int,int,int64_t,bool);
