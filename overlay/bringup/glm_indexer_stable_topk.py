# SPDX-License-Identifier: Apache-2.0
"""Default-off full-GLM prefill selector; exact (score desc, index asc)."""
import ctypes
import functools
import hashlib
import math
import os
from pathlib import Path
import struct
import subprocess
import sys

OPS = 'vllm._custom_ops'
SPARSE = 'vllm.model_executor.layers.sparse_attn_indexer'
WORKER = 'vllm.v1.worker.gpu_worker'
PINS = {
    OPS: '5ca3b8955ff2dca446813057e6678065a44ec283d73db4a9d8fd24328cb4ed07',
    SPARSE: '22d1d98bd475b1dc22de85d0e414d0490860de50c8bbd26371175e5ea6411116',
    WORKER: 'b2e580d74e7259ff2cbc82dabf38a43409ea5584d143880436583d9fc1ceedf1',
}
REGISTERED = False
INSTALLED = False
CALLS = 0
_LIB = None


def enabled(env):
    value = env.get('GLM_INDEXER_STABLE_TOPK', '0')
    if value not in ('0', '1'):
        raise ValueError('GLM_INDEXER_STABLE_TOPK must be 0 or 1')
    return value == '1'


def score_key(value):
    """CPU specification for the CUDA float-flip, with zero/NaN normalization."""
    bits = struct.unpack('<I', struct.pack('<f', value))[0]
    if bits & 0x7fffffff == 0:
        bits = 0
    if bits & 0x7fffffff > 0x7f800000:
        return 0xffffffff
    return bits ^ (0xffffffff if bits & 0x80000000 else 0x80000000)


def stable_key(value, index):
    return (score_key(value) << 32) | (0xffffffff - index)


def reference(row, start, end, k=2048):
    """Independent stable sort oracle, relative indices and -1 padding."""
    if not 0 <= start <= end <= len(row) or k < 1:
        raise ValueError('invalid row bounds/topk')
    values = [struct.unpack('<f', struct.pack('<f', row[i]))[0]
              for i in range(start, end)]
    # Python sort is stable: input enumeration already has ascending indices.
    order = sorted(range(len(values)),
                   key=lambda i: (not math.isnan(values[i]),
                                  0 if math.isnan(values[i]) else -values[i]))
    return sorted(order[:k]) + [-1] * max(0, k - len(order))   # emitted in ascending index order


def radix_reference(row, start, end, k=2048):
    """CPU model of the threshold-prefix algorithm (not the sort oracle)."""
    if not 0 <= start <= end <= len(row) or k < 1:
        raise ValueError('invalid row bounds/topk')
    keys = [stable_key(x, i) for i, x in enumerate(row[start:end])]
    prefix, remaining = 0, k
    if len(keys) > k:
        for shift in range(56, -1, -8):
            bins = [0] * 256
            mask = 0 if shift == 56 else ((1 << 64) - 1) ^ ((1 << (shift + 8)) - 1)
            for key in keys:
                if key & mask == prefix:
                    bins[key >> shift & 255] += 1
            for b in range(255, -1, -1):
                if remaining > bins[b]:
                    remaining -= bins[b]
                else:
                    prefix |= b << shift
                    done = bins[b] == remaining
                    break
            if done:
                break
    selected = sorted((key for key in keys if len(keys) <= k or key >= prefix), reverse=True)
    assert len(selected) == min(k, len(keys))
    return sorted(0xffffffff - (key & 0xffffffff) for key in selected) + [-1] * max(0, k - len(keys))


def build():
    """One niced nvcc job, no Torch/CUDA initialization needed for compilation."""
    src = Path(__file__).with_name('stable_topk') / 'kernel.cu'
    digest = hashlib.sha256(src.read_bytes()).hexdigest()
    cache = Path(os.environ.get('TORCH_EXTENSIONS_DIR', '/tmp/glm-stable-topk')) / ('stable_topk_' + digest)
    cache.mkdir(parents=True, exist_ok=True)
    output = cache / 'kernel.so'
    import fcntl
    with (cache / 'build.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not output.exists():
            nvcc = Path(os.environ.get('CUDA_HOME', '/usr/local/cuda')) / 'bin/nvcc'
            temporary = cache / 'kernel.tmp.so'
            cmd = ['nice', '-n', '10', str(nvcc), '-O3', '-std=c++17',
                   '-arch=sm_121a', '--shared', '-Xcompiler=-fPIC',
                   '-Xptxas=-v,-warn-spills', str(src), '-o', str(temporary)]
            with (cache / 'build.log').open('w') as log:
                subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT,
                               check=True, timeout=600)
            temporary.replace(output)
    return output


def load():
    global _LIB
    if _LIB is None:
        _LIB = ctypes.CDLL(str(build()))
        _LIB.glm_stable_topk.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int] * 2 + [ctypes.c_int64] * 4 + [ctypes.c_int, ctypes.c_void_p]
        _LIB.glm_stable_topk.restype = ctypes.c_int
    return _LIB


def top_k_per_row_prefill(logits, starts, ends, out, rows, stride0, stride1, topk):
    global CALLS
    import torch
    if (topk != 2048 or logits.dtype != torch.float32 or out.dtype != torch.int32
            or logits.ndim != 2 or out.ndim != 2 or out.shape != (rows, topk)
            or rows > logits.shape[0] or rows < 0
            or (stride0, stride1) != logits.stride()
            or stride0 < 0 or stride1 < 1 or out.stride(0) < topk or out.stride(1) != 1
            or any(x.dtype != torch.int32 or not x.is_contiguous() or x.numel() < rows for x in (starts, ends))
            or any(not x.is_cuda or x.device != logits.device for x in (logits, starts, ends, out))):
        raise RuntimeError('stable prefill top-k ABI drift')
    if torch.cuda.get_device_capability(logits.device) != (12, 1):
        raise RuntimeError('stable prefill top-k requires sm_121 GB10')
    if logits.shape[1] >= 2**31 or rows >= 2**31:
        raise RuntimeError('stable prefill top-k exceeds int32 geometry')
    if _LIB is None and torch.cuda.is_current_stream_capturing():
        raise RuntimeError('stable top-k must be loaded before graph capture')
    lib = load()
    with torch.cuda.device(logits.device):
        err = lib.glm_stable_topk(logits.data_ptr(), starts.data_ptr(), ends.data_ptr(), out.data_ptr(),
                                rows, logits.shape[1], stride0, stride1, out.stride(0), out.stride(1),
                                logits.device.index, torch.cuda.current_stream().cuda_stream)
    if err:
        raise RuntimeError('stable prefill top-k CUDA launch error: ' + str(err))
    CALLS += 1


def pin(mod):
    if hashlib.sha256(Path(mod.__file__).read_bytes()).hexdigest() != PINS[mod.__name__]:
        raise RuntimeError('stable top-k source drift: ' + mod.__name__)


def install_ops(mod):
    global INSTALLED
    pin(mod)
    if INSTALLED:
        raise RuntimeError('stable top-k already installed')
    load()
    mod.top_k_per_row_prefill = top_k_per_row_prefill
    INSTALLED = True
    sys.stderr.write('glm-indexer-stable-topk: installed score-desc/index-asc prefill only\n')


def install_sparse(mod):
    pin(mod)
    original = mod.SparseAttnIndexer.__init__
    @functools.wraps(original)
    def init(self, *args, **kwargs):
        original(self, *args, **kwargs)
        import glm_dsa_short
        config = mod.get_current_vllm_config()
        if not glm_dsa_short.supported(config.model_config.hf_config, config.parallel_config):
            raise RuntimeError('stable top-k requires full GLM TP4/PCP1/DCP1')
    mod.SparseAttnIndexer.__init__ = init


def status():
    return dict(armed=REGISTERED, installed=INSTALLED, calls=CALLS,
                order='set_score_desc_index_asc_emit_index_asc', prefill_only=True)


def install_worker(mod):
    pin(mod)
    def stable_status(self):
        return dict(rank=int(os.environ.get('RANK', '-1')), **status())
    mod.Worker.indexer_stable_topk_status = stable_status


def register(env=None):
    global REGISTERED
    if not enabled(os.environ if env is None else env):
        return False
    if REGISTERED or OPS in sys.modules or SPARSE in sys.modules:
        raise RuntimeError('stable top-k must register once before vLLM imports')
    from glm_glue_lite import after_import
    after_import(OPS, install_ops)
    after_import(SPARSE, install_sparse)
    after_import(WORKER, install_worker)
    REGISTERED = True
    return True
