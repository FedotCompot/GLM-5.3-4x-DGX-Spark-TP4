# SPDX-License-Identifier: Apache-2.0
"""No-GPU ARM64 sm_121a build, source/binary receipts and integrity check."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess

HERE=Path(__file__).resolve().parent

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def sources():
    return {str(p.relative_to(HERE)):sha(p) for p in sorted(HERE.rglob('*'))
            if p.is_file() and 'build' not in p.relative_to(HERE).parts and '__pycache__' not in p.parts}

def verify(out=None):
    out=HERE/'build' if out is None else Path(out)
    receipt=json.loads((out/'build.json').read_text())
    if receipt['sources']!=sources() or receipt['binary_sha256']!=sha(out/'attnfront.so'):
        raise RuntimeError('attention front native build/source drift')
    import torch
    if torch.__version__!=receipt['torch']:
        raise RuntimeError('attention front native PyTorch ABI drift')
    provider=Path(__import__('importlib.util').util.find_spec('vllm').origin).parent/'_C_stable_libtorch.abi3.so'
    if sha(provider)!=receipt['stock_provider_sha256']:
        raise RuntimeError('attention front stock CUDA provider drift')
    return receipt

def build(out=None):
    import torch
    from torch.utils.cpp_extension import include_paths,library_paths
    out=HERE/'build' if out is None else Path(out)
    out.mkdir(parents=True,exist_ok=True)
    if platform.machine() not in ('aarch64','arm64'):
        raise RuntimeError('attention front build requires ARM64 pinned image')
    inc=[v for p in include_paths(device_type='cuda') for v in ('-I',p)]
    import sys
    for base in sys.path:
        for path in sorted(Path(base).glob('nvidia/*/include')):
            inc += ['-I',str(path)]
    lib=[v for p in library_paths(device_type='cuda') for v in ('-L',p)]
    argv=['nvcc','-DUSE_CUDA','-D_GLIBCXX_USE_CXX11_ABI='+str(int(torch._C._GLIBCXX_USE_CXX11_ABI)),'-std=c++17','-O3','--threads=2','-arch=sm_121a','--expt-relaxed-constexpr',
          '-Xcompiler=-fPIC','-Xptxas=-v,-warn-spills',*inc,'-shared',str(HERE/'dual_rms.cu'),
          *lib,'-ltorch','-ltorch_cpu','-lc10','-lc10_cuda','-ltorch_cuda','-lcudart',
          '-Xlinker=--no-undefined','-o',str(out/'attnfront.so')]
    subprocess.run(argv,check=True)
    # Retain compiler evidence for the stock RoPE expression, every cache dtype/style.
    subprocess.run(['nvcc','-std=c++17','-O3','-arch=compute_121a','--expt-relaxed-constexpr',
                    *inc,'-ptx',str(HERE/'stock_rope.cu'),'-o',str(out/'stock_rope.ptx')],check=True)
    subprocess.run(['nvcc','-std=c++17','-O3','-arch=sm_121a','--expt-relaxed-constexpr',
                    *inc,'-cubin',str(HERE/'stock_rope.cu'),'-o',str(out/'stock_rope.cubin')],check=True)
    with (out/'stock_rope.sass').open('w') as log:
        subprocess.run(['cuobjdump','--dump-sass',str(out/'stock_rope.cubin')],stdout=log,check=True,
                       env=dict(__import__('os').environ,NVDISASM_PATH='/usr/local/lib/python3.12/dist-packages/triton/backends/nvidia/bin'))
    provider=Path(__import__('importlib.util').util.find_spec('vllm').origin).parent/'_C_stable_libtorch.abi3.so'
    import re
    functions=re.findall(r'\.visible \.entry\s+([^\s(]+)',(out/'stock_rope.ptx').read_text())
    functions += ['_ZN4vllm15rms_norm_kernelIN3c108BFloat16ELi8ELi2ELb1EEEvPT_PKS3_lllllS6_lfii']
    env=dict(__import__('os').environ,NVDISASM_PATH='/usr/local/lib/python3.12/dist-packages/triton/backends/nvidia/bin')
    with (out/'stock_provider_rope_rms.sass').open('w') as log:
        subprocess.run(['cuobjdump','--gpu-architecture','sm_120','--function',','.join(functions),
                        '--dump-sass',str(provider)],stdout=log,check=True,env=env,timeout=120)
    text=(out/'stock_provider_rope_rms.sass').read_text()
    if 'rms_norm_kernel' not in text or text.count('Function :')!=5:
        raise RuntimeError('stock sm_120 RMSNorm provider SASS absent')
    cub=Path('/usr/local/cuda/include/cccl/cub/block/block_reduce.cuh')
    if not cub.exists():
        hits=list(Path('/usr/local').glob('**/cub/block/block_reduce.cuh'))
        cub=hits[0]
    (out/'build.json').write_text(json.dumps(dict(sources=sources(),binary_sha256=sha(out/'attnfront.so'),
        torch=torch.__version__,cuda=torch.version.cuda,stock_provider_sha256=sha(provider),
        cub_block_reduce_sha256=sha(cub),cub_warp_reduce_sha256=sha(cub.parent/'specializations/block_reduce_warp_reductions.cuh'),nvcc=subprocess.check_output(['nvcc','--version']).decode(),
        arch='sm_121a',argv=argv),indent=2)+'\n')
    return out

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out');a=p.parse_args();build(a.out)
