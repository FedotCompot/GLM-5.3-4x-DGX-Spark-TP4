# SPDX-License-Identifier: Apache-2.0
"""Fixed single-bank serving or private two-bank cache-policy experiment.
No hooks when OFF.

Only BF16/U4B8/group128 target gate_up/down, block8/16. Native MTP is
NVFP4 and remains stock. Alignment, scratch locks, reduction and launch
heuristics are unchanged. Requires mode0 FULL graphs, no torch.compile/PW.
"""
from contextlib import contextmanager
import functools
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys

OPS = 'vllm._custom_ops'
CG = 'vllm.v1.worker.gpu.cudagraph_utils'
WORKER = 'vllm.v1.worker.gpu_worker'
CORE = 'vllm.v1.engine.core'
PINS = {
    OPS: '5ca3b8955ff2dca446813057e6678065a44ec283d73db4a9d8fd24328cb4ed07',
    CG: 'c183937e6eb5b9c28c79d98fb4c64f562e7649d5f6d65743e6640b2f378ecf9f',
    WORKER: 'b2e580d74e7259ff2cbc82dabf38a43409ea5584d143880436583d9fc1ceedf1',
    CORE: '86b8f3b3826504549ef8bea2e7fbf7553728339803abcb3e7d051d146963ff9c',
}
PARAMETERS = ('input', 'output', 'b_qweight', 'b_bias', 'b_scales',
              'a_scales', 'global_scale', 'b_qzeros', 'g_idx', 'perm',
              'workspace', 'sorted_token_ids', 'expert_ids',
              'num_tokens_past_padded', 'topk_weights', 'moe_block_size',
              'top_k', 'mul_topk_weights', 'b_q_type', 'size_m', 'size_n',
              'size_k', 'is_k_full', 'use_atomic_add', 'use_fp32_reduce',
              'is_zp_float', 'thread_k', 'thread_n', 'blocks_per_sm')
_REGISTERED = False
_mode = 'off'
BINARY = Path('/overlay/l2hint-serving/candidate/marlin.so')
_runtime = 0
_capture = None
_profiling = False
_managers = []
_hint_op = None
_u4b8 = None
_binary_hash = None
_config_hash = None
_counts = {}
_replays = [0, 0]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def mode(env):
    serving = env.get('GLM_MARLIN_L2HINT', 'off')
    ab = env.get('GLM_MARLIN_L2HINT_AB', '0')
    if serving not in ('off', 'on'):
        raise ValueError('GLM_MARLIN_L2HINT must be off or on')
    if ab not in ('0', '1'):
        raise ValueError('GLM_MARLIN_L2HINT_AB must be 0 or 1')
    if serving == 'on' and ab == '1':
        raise ValueError('single-bank L2 hint and AB are mutually exclusive')
    selected = 'ab' if ab == '1' else serving
    if selected != 'off' and (env.get('VLLM_USE_V2_MODEL_RUNNER') != '1'
                              or env.get('GLM_MOE_DET_ALIGN') != '1'):
        raise ValueError('L2 hint requires V2 and det-align')
    if selected == 'ab' and env.get('VLLM_SERVER_DEV_MODE') != '1':
        raise ValueError('L2 hint AB requires private dev RPC')
    return selected


def options(env):
    return mode(env) != 'off'


def current():
    return _runtime if _capture is None else _capture


@contextmanager
def capturing(value):
    global _capture
    prev, _capture = _capture, value
    try:
        yield
    finally:
        _capture = prev


def qualified(p, u4b8):
    # Tensor metadata only: no GPU scalar read, allocation or host-device sync.
    a, b, s = p['input'], p['b_qweight'], p['b_scales']
    empty = lambda x: x is None or x.numel() == 0
    return (str(a.dtype) == 'torch.bfloat16' and str(s.dtype) == 'torch.bfloat16'
            and str(b.dtype) == 'torch.int32'
            and (p['output'] is None or str(p['output'].dtype) == 'torch.bfloat16')
            and p['b_q_type'].id == u4b8 and len(b.shape) == 3
            and b.shape[1:] == (p['size_k'] // 16, p['size_n'] * 2)
            and len(s.shape) == 3 and s.shape[0] == b.shape[0]
            and s.shape[1] * 128 == p['size_k'] and s.shape[2] == p['size_n']
            and (p['size_k'], p['size_n']) in ((6144, 1024), (512, 6144))
            and len(a.shape) == 2 and a.shape[1] == p['size_k']
            and p['moe_block_size'] in (8, 16) and p['is_k_full']
            and not p['is_zp_float'] and not p['use_atomic_add']
            and p['use_fp32_reduce']
            and all(empty(p[n]) for n in ('b_qzeros', 'g_idx', 'perm'))
            and p['a_scales'] is None and p['global_scale'] is None
            and ((p['thread_k'], p['thread_n']) == (-1, -1)
                 or (p['thread_k'], p['thread_n']) in ((128, 128), (64, 128), (128, 64))))


def dispatch(stock, hint, flag, u4b8):
    sig = inspect.signature(stock)
    if tuple(sig.parameters) != PARAMETERS:
        raise RuntimeError('L2 hint Marlin wrapper signature drift')

    # Let CPython bind ON arguments; inspect.bind per GEMM would cost much of
    # the G-screen's 3.6-6.2us device saving. No per-call reflection.
    def on(input, output, b_qweight, b_bias, b_scales, a_scales,
           global_scale, b_qzeros, g_idx, perm, workspace, sorted_token_ids,
           expert_ids, num_tokens_past_padded, topk_weights, moe_block_size,
           top_k, mul_topk_weights, b_q_type, size_m, size_n, size_k,
           is_k_full, use_atomic_add, use_fp32_reduce, is_zp_float,
           thread_k=-1, thread_n=-1, blocks_per_sm=-1):
        p = locals()
        if not qualified(p, u4b8):
            _counts['fallback'] = _counts.get('fallback', 0) + 1
            return stock(*(p[n] for n in PARAMETERS))
        key = f'{"single" if flag is None else current()}|{p["size_k"]}x{p["size_n"]}|block{p["moe_block_size"]}'
        _counts[key] = _counts.get(key, 0) + 1
        return hint(input, output, b_qweight, b_bias, b_scales, a_scales,
                    global_scale, b_qzeros, g_idx, perm, workspace, sorted_token_ids,
                    expert_ids, num_tokens_past_padded, topk_weights, moe_block_size,
                    top_k, mul_topk_weights, b_q_type.id, size_m, size_n, size_k,
                    is_k_full, use_atomic_add, use_fp32_reduce, is_zp_float,
                    thread_k, thread_n, blocks_per_sm, True)

    if flag is None:
        # Fixed at boot: never consult runtime/capture state on serving calls.
        on._glm_l2hint = True
        return on

    @functools.wraps(stock)
    def call(*args, **kwargs):
        # OFF returns through the original wrapper without binding/conversion.
        if not flag():
            return stock(*args, **kwargs)
        return on(*args, **kwargs)

    call._glm_l2hint = True
    return call


def install_ops(module):
    if not getattr(module.moe_wna16_marlin_gemm, '_glm_l2hint', False):
        module.moe_wna16_marlin_gemm = dispatch(
            module.moe_wna16_marlin_gemm, _hint_op, None if _mode == 'on' else current, _u4b8)


def install_graphs(module):
    cls = module.CudaGraphManager
    if getattr(cls, '_glm_l2hint', False):
        return
    original, replay, profile = cls.capture, cls.run_fullgraph, cls.profile_memory

    @functools.wraps(original)
    def capture(self, create_forward_fn, progress_bar_desc='Capturing CUDA graphs'):
        if type(self).__name__ != 'ModelCudaGraphManager':
            # MTP uses NVFP4; its one stock graph set is shared.
            return original(self, create_forward_fn, progress_bar_desc)
        cfg = self.compilation_config
        if int(cfg.mode) != 0 or self.use_breakable_cg or any(
                str(mode.name) != 'FULL' for mode in self._capture_descs):
            raise RuntimeError('L2 hint AB requires mode0 FULL-only graph capture')
        sets = [{}, {}]
        receipts = [[], []]
        try:
            for value in (0, 1):
                self.graphs = sets[value]
                def factory(desc, warmup=True):
                    forward = create_forward_fn(desc, warmup=warmup)
                    def checked(*a, **kw):
                        before = dict(_counts)
                        out = forward(*a, **kw)
                        delta = {k: v - before.get(k, 0) for k, v in _counts.items()
                                 if v != before.get(k, 0)}
                        receipts[value].append(dict(descriptor=str(desc), warmup=warmup, calls=delta))
                        return out
                    return checked
                with capturing(value):
                    original(self, factory, f'{progress_bar_desc} [l2hint {value}]')
            if sets[0].keys() != sets[1].keys():
                raise RuntimeError('L2 hint graph descriptor mismatch')
            self._glm_l2hint_sets = sets
            self._glm_l2hint_receipts = receipts
            if not _profiling and self not in _managers:
                _managers.append(self)
        finally:
            self.graphs = sets[_runtime]

    @functools.wraps(replay)
    def run(self, desc):
        sets = self.__dict__.get('_glm_l2hint_sets')
        if sets is not None:
            if self.graphs is not sets[_runtime]:
                raise RuntimeError('L2 hint replay selected wrong bank')
            _replays[_runtime] += 1
        return replay(self, desc)

    @functools.wraps(profile)
    def profile_memory(self, *args, **kwargs):
        global _profiling
        prev, _profiling = _profiling, True
        try:
            # Stock profile calls patched capture: accounts for BOTH banks.
            return profile(self, *args, **kwargs)
        finally:
            sets = self.__dict__.pop('_glm_l2hint_sets', None)
            self.__dict__.pop('_glm_l2hint_receipts', None)
            if sets is not None:
                for table in sets:
                    table.clear()
            _profiling = prev

    cls.capture, cls.run_fullgraph, cls.profile_memory = capture, run, profile_memory
    cls._glm_l2hint = True


def validate_switch(value):
    if _mode == 'on':
        raise RuntimeError('single-bank L2 hint has no runtime switch')
    if type(value) is not str or value not in ('0', '1'):
        raise ValueError('L2 hint RPC takes string 0 or 1')
    if _capture is not None or _profiling:
        raise RuntimeError('L2 hint switch during capture/profile')
    if not _managers or any(not all(m._glm_l2hint_sets) for m in _managers):
        raise RuntimeError('L2 hint both graph sets must be ready before switching')
    return int(value)


def status():
    return dict(enabled=_REGISTERED, value=_runtime, binary_sha256=_binary_hash, config_sha256=_config_hash,
                graphs=[[len(t) for t in m._glm_l2hint_sets] for m in _managers],
                captures=[m._glm_l2hint_receipts for m in _managers],
                calls=dict(_counts), replays=list(_replays), mtp='stock NVFP4')


def worker_switch(value):
    global _runtime
    import torch
    from vllm.distributed import get_tp_group
    group = get_tp_group()
    # Every rank votes before any mutation, including malformed/not-ready calls.
    try:
        selected = validate_switch(value)
        ready = 1
    except (ValueError, RuntimeError):
        selected, ready = -1, 0
    ticket = [ready, selected, _runtime, [len(t) for m in _managers for t in m._glm_l2hint_sets], _config_hash]
    votes = [None] * group.world_size
    torch.distributed.all_gather_object(votes, ticket, group=group.cpu_group)
    if not ready or any(v != ticket for v in votes):
        raise RuntimeError('L2 hint switch not ready or TP votes disagree')
    torch.cuda.synchronize()
    _runtime = selected
    for mgr in _managers:
        mgr.graphs = mgr._glm_l2hint_sets[selected]
    return status()


def install_worker(module):
    module.Worker.glm_l2hint_status = lambda self: status()
    module.Worker.glm_l2hint_switch = lambda self, value: worker_switch(value)


def install_core(module):
    cls = module.EngineCore
    original = cls.collective_rpc
    if getattr(original, '_glm_l2hint', False):
        return

    @functools.wraps(original)
    def rpc(self, method, timeout=None, args=(), kwargs=None):
        if method == 'glm_l2hint_switch':
            busy = max(int(self.scheduler.get_num_unfinished_requests()),
                       len(getattr(self.scheduler, 'running', ()) or ()))
            if busy:
                return [{'busy': busy}]
        return original(self, method, timeout, args, kwargs)

    rpc._glm_l2hint = True
    cls.collective_rpc = rpc


def register(env=None):
    global _REGISTERED, _hint_op, _u4b8, _binary_hash, _config_hash, _mode
    env = os.environ if env is None else env
    selected = mode(env)
    if selected == 'off':
        return False
    if _REGISTERED:
        if selected != _mode:
            raise RuntimeError('L2 hint mode is immutable after registration')
        return True
    if env.get('GLM_AB_VARIANTS', '0') not in ('0', '1'):
        raise ValueError('use dedicated L2 hint banks; GLM_AB harness conflicts')
    root = Path(next(iter(importlib.util.find_spec('vllm').submodule_search_locations)))
    for name, pin in PINS.items():
        if sha256(root / (name.removeprefix('vllm.').replace('.', '/') + '.py')) != pin:
            raise RuntimeError('L2 hint source drift ' + name)
    binary = BINARY
    receipt = json.loads(binary.with_name('build.json').read_text())
    assembly = json.loads(binary.parent.parent.joinpath('assembly.json').read_text())
    selfcheck = json.loads(binary.parent.parent.joinpath('selfcheck.json').read_text())
    _binary_hash = sha256(binary)
    _config_hash = hashlib.sha256(json.dumps(dict(binary=_binary_hash, mode=selected, runtime=sha256(__file__),
        sources=receipt['source_sha256'], det=env.get('GLM_MOE_DET_ALIGN'),
        glue=env.get('GLM_GLUE_MOE_WS')), sort_keys=True).encode()).hexdigest()
    if receipt['reference'] or receipt['binary_sha256'] != _binary_hash or (
            assembly['status'] != 'STATIC_PASS' or assembly['candidate_sha256'] != _binary_hash
            or selfcheck['status'] != 'SELFCHECK_PASS' or selfcheck['cuda_initialized'] is not False):
        raise RuntimeError('L2 hint absent/drifted assembly receipt')
    native = Path(__file__).parent / 'l2hint_native'
    sources = {str(p.relative_to(native)) for p in native.rglob('*')
               if p.is_file() and '__pycache__' not in p.parts}
    if set(receipt['source_sha256']) != sources:
        raise RuntimeError('L2 hint incomplete build source pins')
    for name, pin in receipt['source_sha256'].items():
        if sha256(Path(__file__).parent / 'l2hint_native' / name) != pin:
            raise RuntimeError('L2 hint build source drift ' + name)
    if sha256(root / receipt['stock_binary_name']) != receipt['stock_binary_sha256']:
        raise RuntimeError('L2 hint stock provider differs from build image')
    import torch
    from vllm.scalar_type import scalar_types
    torch.ops.load_library(str(root / receipt['stock_binary_name']))
    torch.ops.load_library(str(binary))
    _hint_op = torch.ops._glm_l2hint_C.moe_wna16_marlin_gemm
    # Actual registrations, not just a saved build receipt, must match at boot.
    from l2hint_native.selfcheck import check_schemas
    stock_schema, hint_schema = check_schemas(torch.ops._moe_C.moe_wna16_marlin_gemm, _hint_op)
    if (selfcheck['stock_schema'], selfcheck['hint_schema']) != (stock_schema, hint_schema):
        raise RuntimeError('L2 hint live ABI differs from selfcheck receipt')
    _mode = selected
    _u4b8 = scalar_types.uint4b8.id
    from glm_skip_mla_plan import Hooks
    hooks = Hooks()
    sys.meta_path.insert(0, hooks)
    installers = [(OPS, install_ops)]
    if selected == 'ab':
        installers += [(CG, install_graphs), (WORKER, install_worker), (CORE, install_core)]
    # Single mode leaves capture/profile/replay and worker/core RPC untouched.
    for name, installer in installers:
        hooks.after_import(name, installer)
    _REGISTERED = True
    print(f'L2HINT_READY mode={selected} binary={binary} sha256={_binary_hash} '
          f'config={_config_hash} hooks={len(installers)} mtp=stock-NVFP4', file=sys.stderr, flush=True)
    return True
