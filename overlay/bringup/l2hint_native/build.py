# SPDX-License-Identifier: Apache-2.0
"""G3 production selection; runtime flag selects stock or B-only hint kernels."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
HERE=Path(__file__).resolve().parent
TILES=((256,8,8),(128,8,4),(128,4,8))
SCHEMA='''
STABLE_TORCH_LIBRARY_FRAGMENT(_glm_l2hint_C, m) {
 m.def("moe_wna16_marlin_gemm(Tensor! a, Tensor? c_or_none, Tensor! b_q_weight, Tensor? b_bias_or_none, Tensor! b_scales, Tensor? a_scales, Tensor? global_scale, Tensor? b_zeros_or_none, Tensor? g_idx_or_none, Tensor? perm_or_none, Tensor! workspace, Tensor sorted_token_ids, Tensor! expert_ids, Tensor! num_tokens_past_padded, Tensor! topk_weights, int moe_block_size, int top_k, bool mul_topk_weights, int b_type_id, int size_m, int size_n, int size_k, bool is_full_k, bool use_atomic_add, bool use_fp32_reduce, bool is_zp_float, int thread_k, int thread_n, int blocks_per_sm, bool l2_hint=False) -> Tensor");
}
'''
def replace(s,a,b):
    if s.count(a)!=1:raise RuntimeError('source anchor drift: '+a[:60])
    return s.replace(a,b)

def generate(out,reference=False):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((HERE/'vendor/BASE.json').read_text())
    for name,digest in manifest['sha256'].items():
        if hashlib.sha256((HERE/'vendor'/name).read_bytes()).hexdigest()!=digest:raise RuntimeError('stock source drift: '+name)
    shutil.copytree(HERE/'vendor/csrc',out/'csrc',dirs_exist_ok=True)
    source=out/'csrc/libtorch_stable/moe/marlin_moe_wna16'
    if not reference:
        for name in ('kernel.h','marlin_template.h'):
            p=source/name;s=p.read_text()
            s=replace(s,'const bool is_zp_float   // is zero point of float16 type?','const bool is_zp_float,  // is zero point of float16 type?\n          const bool L2Hint')
            if name=='marlin_template.h':
                s=replace(s,'  // Each threadblock processes', '''  unsigned long long policy = 0;
  if constexpr (L2Hint)
    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(policy));
  // Each threadblock processes''')
                s=replace(s,'        cp_async4(&sh_b_stage[threads * i + threadIdx.x], &B[b_gl_idx]);', '''        if constexpr (L2Hint) {
          int dst = static_cast<int>(__cvta_generic_to_shared(&sh_b_stage[threads * i + threadIdx.x]));
          asm volatile("cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, %2;"
                       :: "r"(dst), "l"(&B[b_gl_idx]), "l"(policy));
        } else {
          cp_async4(&sh_b_stage[threads * i + threadIdx.x], &B[b_gl_idx]);
        }''')
            p.write_text(s)
        p=source/'ops.cu';s=p.read_text().replace('_moe_C,','_glm_l2hint_C,')
        s=replace(s,'int threads, bool is_zp_float, int stages) {','int threads, bool is_zp_float, int stages, bool l2_hint = false) {')
        s=replace(s,'bool use_fp32_reduce, bool is_zp_float) {','bool use_fp32_reduce, bool is_zp_float, bool l2_hint) {')
        s=replace(s,'      num_threads, is_zp_float, stages);','      num_threads, is_zp_float, stages, l2_hint);')
        s=replace(s,'int64_t blocks_per_sm) {','int64_t blocks_per_sm, bool l2_hint) {')
        s=replace(s,'      is_zp_float);','      is_zp_float, l2_hint);')
        p.write_text(s+SCHEMA)
    declarations=[];selectors=[]
    for th,nb,kb in TILES:
        for m8 in (False,True):
            params=f'vllm::kBFloat16.id(),vllm::kU4B8.id(),vllm::kBFloat16.id(),vllm::kBFloat16.id(),{th},1,{nb},{kb},{str(m8).lower()},4,8,false'
            for hint in ((False,) if reference else (False,True)):
                inst=params if reference else params+','+str(hint).lower()
                declarations.append('template __global__ void Marlin<'+inst+'>(MARLIN_KERNEL_PARAMS);')
            if not reference:
                cond=f'a_type == vllm::kBFloat16 && b_type == vllm::kU4B8 && c_type == vllm::kBFloat16 && s_type == vllm::kBFloat16 && thread_m_blocks == 1 && thread_n_blocks == {nb} && thread_k_blocks == {kb} && m_block_size_8 == {str(m8).lower()} && !has_act_order && !has_zp && group_blocks == 8 && threads == {th} && !is_zp_float && stages == 4'
                selectors.append(f'if ({cond}) kernel = l2_hint ? Marlin<{params},true> : Marlin<{params},false>;')
    (out/'kernel_selector.h').write_text('\n'.join(selectors)+'\n')
    ns='stock_marlin' if reference else 'l2hint_marlin'
    tu=out/'bench.cu';tu.write_text(f'#define MARLIN_NAMESPACE_NAME {ns}\n#include "libtorch_stable/moe/marlin_moe_wna16/kernel.h"\n#include "libtorch_stable/moe/marlin_moe_wna16/marlin_template.h"\nnamespace MARLIN_NAMESPACE_NAME {{\n'+'\n'.join(declarations)+'\n}\n'+('' if reference else '#include "libtorch_stable/moe/marlin_moe_wna16/ops.cu"\n'))
    return tu

def build(out,reference=False):
    import torch
    import platform
    if platform.machine() not in ("aarch64", "arm64"):
        raise RuntimeError("requires aarch64 pinned serving image")
    if "g487ecf187" not in __import__("vllm.version", fromlist=["__version__"]).__version__:
        raise RuntimeError("requires pinned vLLM g487ecf187")
    from torch.utils.cpp_extension import include_paths,library_paths
    out=Path(out).resolve();tu=generate(out,reference)
    includes=include_paths(device_type='cuda')+[str(out/'csrc'),str(out/'csrc/libtorch_stable/moe/marlin_moe_wna16'),str(out),str(HERE)]
    includes += [str(p) for p in Path('/usr/local/lib/python3.12/dist-packages/nvidia').glob('*/include')]
    libs=library_paths(device_type='cuda');binary=out/'marlin.so'
    cmd=['nvcc','-DUSE_CUDA','-O3','-std=c++17','--expt-relaxed-constexpr','--threads=2','-gencode=arch=compute_121a,code=[sm_121a,compute_121a]','-Xcompiler=-fPIC','-Xptxas=-v,-warn-spills','-D_GLIBCXX_USE_CXX11_ABI='+str(int(torch._C._GLIBCXX_USE_CXX11_ABI)),*['-I'+p for p in includes],str(tu),'-shared','-Xlinker=--no-undefined',*['-L'+p for p in libs],*['-Xlinker=-rpath='+p for p in libs],'-lcudart','-ltorch','-ltorch_cpu','-ltorch_cuda','-lc10','-lc10_cuda','-o',str(binary)]
    (out/'command.json').write_text(json.dumps(cmd,indent=2)+'\n')
    with (out/'ptxas-link.log').open('w') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=150)
    # Do not dlopen a CUDA extension during a no-GPU build (libcuda absent).
    import importlib.util
    root=Path(next(iter(importlib.util.find_spec('vllm').submodule_search_locations)))
    providers=[]
    for path in root.glob('_moe_C*.so'):
        with path.open('rb') as stream:
            tail=b''
            for chunk in iter(lambda: stream.read(1024*1024),b''):
                data=tail+chunk
                if b'moe_wna16_marlin_gemm' in data:
                    providers.append(path);break
                tail=data[-64:]
    if len(providers)!=1:raise RuntimeError('ambiguous/absent stock provider')
    stock_path=providers[0]
    source_hashes={str(p.relative_to(HERE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(HERE.rglob('*')) if p.is_file() and '__pycache__' not in p.parts}
    (out/'build.json').write_text(json.dumps(dict(binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),source_sha256=source_hashes,stock_binary_name=stock_path.name,stock_binary_sha256=hashlib.sha256(stock_path.read_bytes()).hexdigest(),reference=reference,tiles=6 if reference else 12,threads_max=2),indent=2)+'\n')
    from l2hint_native.asm_check import dump,check_hints
    dumps=dump(binary,out);check_hints(dumps['ptx'],reference)
    return binary

def static(out):
    out=Path(out);a=build(out/'candidate');b=build(out/'stock',True)
    from l2hint_native.asm_check import compare
    return compare(a,b,out)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--static',action='store_true');a=p.parse_args()
    print(static(a.out) if a.static else build(a.out))
