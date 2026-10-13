// SPDX-License-Identifier: Apache-2.0
// Adapted from vLLM g487ecf187; see vendor/BASE.json and CREDITS.md.
#include "vendor/torch_utils.h"
#include <torch/csrc/stable/library.h>
#include <torch/headeronly/util/BFloat16.h>
#include "vendor/cub_helpers.h"
#include "vendor/vectorization_utils.cuh"
using vllm::vec_n_t;
using scalar_t = torch::headeronly::BFloat16;
constexpr int VEC_SIZE = 8;
constexpr bool HasWeight = true;

__global__ void dual_rms_kernel(scalar_t* qo, scalar_t* ko,
    const scalar_t* qi, const scalar_t* ki, const scalar_t* qw, const scalar_t* kw,
    int64_t qs, int64_t ks, float qe, float ke, int m, int q_width, int k_width) {
  const bool is_q = blockIdx.x < m;
  const int row_idx = is_q ? blockIdx.x : blockIdx.x - m;
  const int hidden_size = is_q ? q_width : k_width;
  const int logical_threads = hidden_size / VEC_SIZE;
  const float epsilon = is_q ? qe : ke;
  const scalar_t* input_row = (is_q ? qi : ki) + row_idx * (is_q ? qs : ks);
  const scalar_t* weight_row = is_q ? qw : kw;
  scalar_t* out = is_q ? qo : ko;
  __shared__ float s_variance;
  float variance = 0.0f;
  auto vec_op = [&variance](const vec_n_t<scalar_t, VEC_SIZE>& vec) {
#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
      float x = static_cast<float>(vec.val[i]);
      variance += x * x;
    }
  };
  auto scalar_op = [&variance](const scalar_t& val) {
    float x = static_cast<float>(val);
    variance += x * x;
  };
  if (threadIdx.x < logical_threads) {
    vllm::vectorize_read_with_alignment<VEC_SIZE>(
        input_row, hidden_size, threadIdx.x, logical_threads, vec_op, scalar_op);
  }

  using BlockReduce = cub::BlockReduce<float, 1024>;
  __shared__ typename BlockReduce::TempStorage reduceStore;
  variance = BlockReduce(reduceStore).Reduce(variance, CubAddOp{}, logical_threads);

  if (threadIdx.x == 0) {
    s_variance = rsqrtf(variance / hidden_size + epsilon);
  }
  __syncthreads();

  scalar_t* out_row = out + row_idx * hidden_size;
  auto* v_in = reinterpret_cast<const vec_n_t<scalar_t, VEC_SIZE>*>(input_row);
  auto* v_w = reinterpret_cast<const vec_n_t<scalar_t, VEC_SIZE>*>(weight_row);
  auto* v_out = reinterpret_cast<vec_n_t<scalar_t, VEC_SIZE>*>(out_row);
  for (int i = threadIdx.x; i < hidden_size / VEC_SIZE; i += logical_threads) {
    vec_n_t<scalar_t, VEC_SIZE> dst;
    vec_n_t<scalar_t, VEC_SIZE> src1 = v_in[i];
    vec_n_t<scalar_t, VEC_SIZE> src2;
    if constexpr (HasWeight) {
      src2 = v_w[i];
    }
#pragma unroll
    for (int j = 0; j < VEC_SIZE; j++) {
      float x = static_cast<float>(src1.val[j]);
      if constexpr (HasWeight) {
        float w = static_cast<float>(src2.val[j]);
        dst.val[j] = static_cast<scalar_t>(x * s_variance * w);
      } else {
        dst.val[j] = static_cast<scalar_t>(x * s_variance);
      }
    }
    v_out[i] = dst;
  }
}

void dual_rms(const torch::stable::Tensor& q, const torch::stable::Tensor& k,
    const torch::stable::Tensor& qw, const torch::stable::Tensor& kw,
    double qe, double ke, torch::stable::Tensor qo, torch::stable::Tensor ko) {
  STD_TORCH_CHECK(q.get_device_index()>=0 && k.get_device_index()>=0 && q.dim()==2 && k.dim()==2);
  const auto dev=q.device();
  for (const auto& t : {q,k,qw,kw,qo,ko}) {
    STD_TORCH_CHECK(t.device()==dev && t.scalar_type()==torch::headeronly::ScalarType::BFloat16);
    STD_TORCH_CHECK(reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0);
    STD_TORCH_CHECK(t.stride(-1)==1);
  }
  const int m=q.size(0);
  STD_TORCH_CHECK(m>=1 && m<=16 && k.size(0)==m && (q.size(1)==2048 || q.size(1)==1536) && k.size(1)==512);
  STD_TORCH_CHECK(q.stride(0)%8==0 && k.stride(0)%8==0);
  STD_TORCH_CHECK(qw.dim()==1 && kw.dim()==1 && qw.numel()==q.size(1) && kw.numel()==k.size(1));
  STD_TORCH_CHECK(qo.sizes()==q.sizes() && ko.sizes()==k.sizes() && qo.is_contiguous() && ko.is_contiguous());
  const torch::stable::accelerator::DeviceGuard guard(q.get_device_index());
  // Stock BF16 vec=gcd(8,width)=8; M<256 -> cap1024.
  // Physical q_width/8; CUB valid-items q_width/8 or kv_width/8.
  dual_rms_kernel<<<2*m,q.size(1)/VEC_SIZE,0,get_current_cuda_stream()>>>(
    reinterpret_cast<scalar_t*>(qo.data_ptr()),reinterpret_cast<scalar_t*>(ko.data_ptr()),
    reinterpret_cast<const scalar_t*>(q.data_ptr()),reinterpret_cast<const scalar_t*>(k.data_ptr()),
    reinterpret_cast<const scalar_t*>(qw.data_ptr()),reinterpret_cast<const scalar_t*>(kw.data_ptr()),
    q.stride(0), k.stride(0), float(qe),float(ke),m,q.size(1),k.size(1));
  STD_TORCH_CHECK(cudaGetLastError()==cudaSuccess,"attention front CUDA launch failed");
}
STABLE_TORCH_LIBRARY_FRAGMENT(glm_attnfront, m) {
  m.def("dual_rms(Tensor q, Tensor k, Tensor qw, Tensor kw, float qe, float ke, Tensor(a!) qo, Tensor(b!) ko) -> ()");
}
STABLE_TORCH_LIBRARY_IMPL(glm_attnfront, CUDA, m) { m.impl("dual_rms", TORCH_BOX(&dual_rms)); }
