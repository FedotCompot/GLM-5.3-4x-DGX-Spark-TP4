# SPDX-License-Identifier: Apache-2.0
"""Default-off TP row ownership probe for pinned full GLM DSA prefill.

Inspired by vLLM #54951/#54394 (Apache-2.0 contributors). No upstream
implementation is copied. FAST requires the installed stable selector and
original FP4x scorer tiles; sampled raw equality fails closed.
"""
import ast
import functools
import hashlib
import os
from pathlib import Path
import sys

SPARSE = 'vllm.model_executor.layers.sparse_attn_indexer'
WORKER = 'vllm.v1.worker.gpu_worker'
PINS = {
    SPARSE: '22d1d98bd475b1dc22de85d0e414d0490860de50c8bbd26371175e5ea6411116',
    WORKER: 'b2e580d74e7259ff2cbc82dabf38a43409ea5584d143880436583d9fc1ceedf1',
}
MODE = 'off'
TRANSPORT = 'nccl'
REGISTERED = False
HAS_FP4X = False
FAILED = False
SAMPLE_EVERY = 32
FAST_CHUNKS = 0
PENDING_TIMINGS = []
COUNTS = dict(chunks=0, rows=0, checked=0, mismatches=0, payload_bytes=0, fast_chunks=0, sampled=0)
PHASE_MS = dict(owner=0.0, gather=0.0, reference=0.0, equality=0.0)


class Timing:
    """CUDA events only. Read at drained status RPC, never sync FAST chunks."""
    def __init__(self, tensor):
        self.events = []
        self.enabled = tensor.is_cuda
        if self.enabled:
            self.previous = self.event()

    @staticmethod
    def event():
        import torch
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        return event

    def mark(self, phase):
        if self.enabled:
            event = self.event()
            self.events.append((phase, self.previous, event))
            self.previous = event

    def finish(self):
        if self.enabled:
            PENDING_TIMINGS.extend(self.events)
            if len(PENDING_TIMINGS) >= 256:
                flush_timings(wait=False)


def flush_timings(wait=True):
    # Coordinator drains requests before calling status or switching policy.
    if PENDING_TIMINGS:
        if wait:
            PENDING_TIMINGS[-1][2].synchronize()
        completed = 0
        for name, a, b in PENDING_TIMINGS:
            if not wait and not b.query():
                break
            PHASE_MS[name] += a.elapsed_time(b)
            completed += 1
        del PENDING_TIMINGS[:completed]


def sample_interval(value):
    if isinstance(value, bool) or not str(value).isdigit():
        raise ValueError('row-shard SAMPLE_EVERY must be a nonnegative integer')
    return int(value)


def configured_mode(env):
    # Release default: FAST when the stable selector it requires is on, else OFF (probe modes are explicit).
    default = 'fast' if env.get('GLM_INDEXER_STABLE_TOPK') == '1' else 'off'
    mode = env.get('GLM_INDEXER_ROW_SHARD_MODE', env.get('GLM_INDEXER_ROW_SHARD_INIT', default))
    mode = 'check' if mode == 'checked' else mode
    initial = env.get('GLM_INDEXER_ROW_SHARD_INIT', 'off')
    initial = 'check' if initial == 'checked' else initial
    if ('GLM_INDEXER_ROW_SHARD_MODE' in env and 'GLM_INDEXER_ROW_SHARD_INIT' in env
            and initial not in ('', 'off', mode)):
        raise ValueError('conflicting row-shard MODE and INIT')
    return mode


def options(env):
    flag = env.get('GLM_INDEXER_ROW_SHARD', '0')
    if flag not in ('0', '1'):
        raise ValueError('GLM_INDEXER_ROW_SHARD must be 0 or 1')
    if flag == '0':
        if any(env.get(k, '0') not in ('', '0') for k in (
                'GLM_INDEXER_ROW_SHARD_INIT', 'GLM_INDEXER_ROW_SHARD_TRANSPORT',
                'GLM_INDEXER_ROW_SHARD_MODE', 'GLM_INDEXER_ROW_SHARD_SAMPLE_EVERY')):
            raise ValueError('row-shard options require GLM_INDEXER_ROW_SHARD=1')
        return None
    mode = configured_mode(env)
    transport = env.get('GLM_INDEXER_ROW_SHARD_TRANSPORT', 'nccl')
    if mode not in ('off', 'self', 'check', 'fast') or transport not in ('nccl', 'auto'):
        raise ValueError('row-shard MODE=off/self/check/fast, TRANSPORT=nccl/auto')
    # The launcher enables the private worker RPCs (status/switch) itself, as for the draft head.
    sample_interval(env.get('GLM_INDEXER_ROW_SHARD_SAMPLE_EVERY', '32'))
    if mode == 'fast' and env.get('GLM_INDEXER_STABLE_TOPK') != '1':
        raise ValueError('FAST requires GLM_INDEXER_STABLE_TOPK=1')
    return mode, transport


def partition(rows, world=4, quantum=1):
    """Contiguous half-open slices, optionally preserving whole stock Q tiles.

    Zero-work ranks still contribute a padded shard and enter every collective.
    Equal padding permits ordinary NCCL or b12x all_gather (no all_gatherv).
    """
    if type(rows) is not int or rows < 0 or world < 1 or quantum < 1:
        raise ValueError('invalid row geometry')
    units = (rows + quantum - 1) // quantum
    base, extra = divmod(units, world)
    offsets = [min(rows, (r * base + min(r, extra)) * quantum)
               for r in range(world + 1)]
    return tuple(zip(offsets[:-1], offsets[1:]))


def reassemble(parts, slices):
    """CPU specification: remove rank-tail padding, preserve every index byte."""
    if len(parts) != len(slices) or not slices or slices[0][0] != 0:
        raise ValueError('invalid gather geometry')
    out = []
    for rank, (begin, end) in enumerate(slices):
        if end < begin or begin != len(out) or len(parts[rank]) < end - begin:
            raise ValueError('noncontiguous or truncated gather')
        out.extend(parts[rank][:end - begin])
    return out


def tile_plan(spans, quanta, world=4):
    """Own original tiles across metadata splits; one gather for the layer.

    A 512 MiB metadata budget splits long-context scheduled 2048/4096-row
    chunks below 2048 rows. Gating each metadata slice would never activate.
    """
    if len(spans) != len(quanta) or not spans:
        raise ValueError('invalid tile plan')
    tasks, offsets = [], [0]
    for chunk, (rows, quantum) in enumerate(zip(spans, quanta)):
        if rows < 1 or quantum < 1:
            raise ValueError('empty tile span')
        for offset in range(0, rows, quantum):
            count = min(quantum, rows - offset)
            tasks.append((chunk, offset, count, quantum))
            offsets.append(offsets[-1] + count)
    owners = partition(len(tasks), world)
    return tasks, owners, tuple((offsets[a], offsets[b]) for a,b in owners)


def eligible(rows, depth, *, world=4, capture=False, pcp=False, dcp=1,
             fp4_cache=False, topk=2048, head_dim=128):
    # Depth is CPU metadata's max_seq_len, not sum of packed request lengths.
    # Scheduled prefill rows matter: residual APC suffixes below 2048 stay replicated.
    return (MODE != 'off' and 2048 <= rows <= 4096 and depth > 16384
            and world == 4 and not capture and not pcp and dcp == 1
            and not fp4_cache and topk == 2048 and head_dim == 128)


def status():
    flush_timings()
    return dict(armed=REGISTERED, mode=MODE, transport=TRANSPORT,
                checked_only=False, fast_supported=HAS_FP4X, failed=FAILED,
                sample_every=SAMPLE_EVERY, fast_sequence=FAST_CHUNKS,
                phase_ms=dict(PHASE_MS), **COUNTS)


def require_fast(topk_fn=None):
    import glm_indexer_stable_topk as stable
    if (FAILED or not HAS_FP4X or not stable.REGISTERED or not stable.INSTALLED
            or (topk_fn is not None and topk_fn is not stable.top_k_per_row_prefill)):
        raise RuntimeError('FAST requires installed GLM_INDEXER_STABLE_TOPK=1, FP4x, and no prior failure')


def set_mode(mode, sample_every=None):
    global MODE, SAMPLE_EVERY, FAST_CHUNKS
    mode = 'check' if mode == 'checked' else mode
    if not REGISTERED or mode not in ('off', 'self', 'check', 'fast') or FAILED:
        raise ValueError('armed healthy row-shard accepts off/self/check/fast only')
    interval = SAMPLE_EVERY if sample_every is None else sample_interval(sample_every)
    if mode == 'fast':
        require_fast()
    flush_timings()
    if mode != MODE or interval != SAMPLE_EVERY:
        FAST_CHUNKS = 0
    MODE, SAMPLE_EVERY = mode, interval
    return status()


def checked_call(topk_fn):
    global FAST_CHUNKS
    if MODE != 'fast':
        return True
    require_fast(topk_fn)
    # First eligible call after each policy transition, then every N calls.
    check = SAMPLE_EVERY > 0 and FAST_CHUNKS % SAMPLE_EVERY == 0
    FAST_CHUNKS += 1
    COUNTS['fast_chunks'] += 1
    if check:
        COUNTS['sampled'] += 1
    return check


def gather(local, slices, group):
    """Padded dim-0 gather; nccl explicitly bypasses the RoCE adapter."""
    import torch
    size = max(end - begin for begin, end in slices)
    staged = torch.full((size, local.shape[1]), -1, dtype=local.dtype, device=local.device)
    staged[:local.shape[0]].copy_(local)
    if TRANSPORT == 'auto':
        collected = group.all_gather(staged, dim=0)
    else:
        comm = group.device_communicator.pynccl_comm
        if comm is None or comm.disabled:
            raise RuntimeError('row-shard requires enabled TP PyNCCL communicator')
        collected = torch.empty((size * group.world_size, local.shape[1]),
                                dtype=local.dtype, device=local.device)
        comm.all_gather(collected, staged)
    COUNTS['payload_bytes'] += staged.numel() * staged.element_size() * group.world_size
    result = torch.empty((slices[-1][1], local.shape[1]), dtype=local.dtype, device=local.device)
    for rank, (begin, end) in enumerate(slices):
        result[begin:end].copy_(collected[rank * size:rank * size + end - begin])
    return result


def require_equal(candidate, reference, group):
    """One failure vote across TP; every rank stops before consuming indices.

    Compare raw order and -1 tails, never sorted sets or score multisets.
    No prompts, activations or indices are written to disk.
    """
    global FAILED
    import torch
    import torch.distributed as dist
    bad = (candidate != reference).any().to(dtype=torch.int32).reshape(1)
    dist.all_reduce(bad, op=dist.ReduceOp.MAX, group=group.device_group)
    COUNTS['checked'] += 1
    if int(bad.item()):
        FAILED = True
        COUNTS['mismatches'] += 1
        sys.stderr.write('glm-indexer-row-shard: INDEX_ORDER_MISMATCH ' + str(status()) + '\n')
        raise RuntimeError('row-shard raw top-k index equality failed; kill this candidate')


def score(q, scale, keys, key_scale, weights, starts, ends, out, logits_fn, topk_fn):
    logits = logits_fn((q, scale), (keys, key_scale), weights, starts, ends, clean_logits=False)
    topk_fn(logits, starts, ends, out, out.shape[0], logits.stride(0), logits.stride(1), out.shape[1])


def score_rows(q, scale, keys, key_scale, weights, starts, ends, out, logits_fn, topk_fn, depth, scheduled_rows=None):
    import torch
    from vllm.distributed import get_tp_group
    group = get_tp_group()
    if not eligible(out.shape[0] if scheduled_rows is None else scheduled_rows, depth, world=group.world_size,
                    capture=torch.cuda.is_current_stream_capturing()):
        return score(q, scale, keys, key_scale, weights, starts, ends, out, logits_fn, topk_fn)
    if MODE == 'fast':
        raise RuntimeError('FAST requires original FP4x tile geometry')
    slices = partition(out.shape[0], group.world_size)
    timer = Timing(q)
    if MODE == 'self':
        candidate = torch.full_like(out, -1)
        score(q, scale, keys, key_scale, weights, starts, ends, candidate, logits_fn, topk_fn)
        timer.mark('owner')
    else:
        begin, end = slices[group.rank_in_group]
        local = torch.full((end - begin, out.shape[1]), -1, dtype=out.dtype, device=out.device)
        if end > begin:
            score(q[begin:end], None if scale is None else scale[begin:end], keys, key_scale,
                  weights[begin:end], starts[begin:end], ends[begin:end], local, logits_fn, topk_fn)
        timer.mark('owner')
        candidate = gather(local, slices, group)
    timer.mark('gather')
    # The reference is exactly the original full-row scorer/selector; query
    # projection, key insertion/gather, mixed decode and DCP remain outside.
    score(q, scale, keys, key_scale, weights, starts, ends, out, logits_fn, topk_fn)
    timer.mark('reference')
    require_equal(candidate, out, group)
    timer.mark('equality')
    timer.finish()
    COUNTS['chunks'] += 1
    COUNTS['rows'] += out.shape[0]


def transform(source):
    """Compile pinned stock function with one prefill scoring-site replacement."""
    fn = next(n for n in ast.parse(source).body
              if isinstance(n, ast.FunctionDef) and n.name == 'sparse_attn_indexer')
    raw = ''.join(source.splitlines(keepends=True)[fn.lineno - 1:fn.end_lineno])
    begin = raw.index('                if current_platform.is_xpu():\n')
    end = raw.index('\n            _merge_dcp_topk_global(', begin)
    old = raw[begin:end]
    # Keep unsupported paths byte-for-byte, including the full original block.
    replacement = '''                from glm_indexer_row_shard import eligible as _row_ok, score_rows as _row_score
                if current_platform.is_cuda() and _row_ok(
                    attn_metadata_narrowed.num_prefill_tokens, attn_metadata_narrowed.max_seq_len,
                    capture=torch.cuda.is_current_stream_capturing(), pcp=use_pcp,
                    dcp=dcp_world_size, fp4_cache=use_fp4_cache,
                    topk=topk_tokens, head_dim=head_dim):
                    _row_score(q_slice_cast, q_scale_slice, k_quant_cast, k_scale_cast,
                        weights[chunk.token_start:chunk.token_end], cu_seqlen_ks,
                        cu_seqlen_ke, topk_indices, fp8_fp4_mqa_logits,
                        ops.top_k_per_row_prefill, attn_metadata_narrowed.max_seq_len,
                        scheduled_rows=attn_metadata_narrowed.num_prefill_tokens)
                    # DCP1 merge is an immediate no-op but still takes logits.
                    logits = None
                else:
'''
    return raw[:begin] + replacement + ''.join('    ' + line if line.strip() else line
                    for line in old.splitlines(keepends=True)) + raw[end:]


def tiled_prefill(cache, q, q_scale, weights, chunks, out, logits_fn, topk_fn=None, *, depth):
    """FP4x: distribute original whole Q tiles, retaining scorer launch shape.

    Gather keys once per chunk on every rank. Never rerun cache insertion or
    compact MTP's shared buffer. The final tile keeps stock query padding.
    """
    import torch
    import glm_fp4_prefill as p
    from glm_fp4_prefill_kernel import gather_index_tile, prepare_index_queries
    from vllm.distributed import get_tp_group
    if topk_fn is None:
        from vllm import _custom_ops as ops
        topk_fn = ops.top_k_per_row_prefill
    group = get_tp_group()
    b = p.bank_for(q.device)
    if (q_scale is not None or cache.shape[-1] != 132 or cache.dtype != torch.uint8
            or q.shape[1:] != b['q'].shape[1:] or q.dtype != b['q'].dtype
            or not q.is_contiguous() or not weights.is_contiguous()
            or weights.dtype != torch.float32 or weights.shape != q.shape[:2]):
        raise RuntimeError('row-shard FP4x query ABI drift')
    rows = sum(c.token_end - c.token_start for c in chunks)
    if (not chunks or not eligible(rows, depth, world=group.world_size)
            or any(c.local_total_seq_lens == 0 for c in chunks)):
        return p._row_shard_stock(cache, q, q_scale, weights, chunks, out, logits_fn, topk_fn)
    if any(a.token_end != z.token_start for a,z in zip(chunks,chunks[1:])):
        raise RuntimeError('row-shard prefill chunks must cover contiguous rows')
    geometry = [p.geometry(c.local_total_seq_lens) for c in chunks]
    tasks, owners, slices = tile_plan(
        [c.token_end-c.token_start for c in chunks], [qt for qt,kt in geometry], group.world_size)
    if len(tasks) < 2:
        return p._row_shard_stock(cache, q, q_scale, weights, chunks, out, logits_fn, topk_fn)

    def run(first, last, dest):
        previous, written = None, 0
        for chunk_id, qoff, count, qt in tasks[first:last]:
            chunk = chunks[chunk_id]
            kt = geometry[chunk_id][1]
            keys, scales = b['keys'][:kt], b['scales'][:kt]
            if previous != chunk_id:
                gather_index_tile(cache, chunk.block_table, chunk.local_cu_seq_lens, keys, scales, 0)
                previous = chunk_id
            qs, ws = b['q'][:qt], b['weights'][:qt]
            ks, ke = b['starts'][:qt], b['ends'][:qt]
            prepare_index_queries(q, weights, chunk.cu_seqlen_ks, chunk.cu_seqlen_ke,
                                  qs, ws, ks, ke, chunk.token_start + qoff, qoff, count)
            score(qs, None, keys, scales, ws, ks, ke,
                  dest[written:written+count], logits_fn, topk_fn)
            written += count

    check = checked_call(topk_fn)
    timer = Timing(q)
    if MODE == 'self':
        candidate = torch.full((rows, p.TOPK), -1, dtype=out.dtype, device=out.device)
        run(0, len(tasks), candidate)
        timer.mark('owner')
    else:
        first, last = owners[group.rank_in_group]
        begin, end = slices[group.rank_in_group]
        local = torch.full((end-begin, p.TOPK), -1, dtype=out.dtype, device=out.device)
        run(first, last, local)
        timer.mark('owner')
        candidate = gather(local, slices, group)
    timer.mark('gather')
    reference = out[chunks[0].token_start:chunks[-1].token_end, :p.TOPK]
    if check:
        p._row_shard_stock(cache, q, q_scale, weights, chunks, out, logits_fn, topk_fn)
        timer.mark('reference')
        require_equal(candidate, reference, group)
        timer.mark('equality')
    if MODE == 'fast':
        # Publish only after a sampled all-rank equality vote has passed.
        reference.copy_(candidate)
    timer.finish()
    COUNTS['chunks'] += 1
    COUNTS['rows'] += rows


def pin(mod):
    if hashlib.sha256(Path(mod.__file__).read_bytes()).hexdigest() != PINS[mod.__name__]:
        raise RuntimeError('row-shard source drift: ' + mod.__name__)


def install_sparse(mod):
    global HAS_FP4X
    pin(mod)
    cls = mod.SparseAttnIndexer
    original, original_init = cls.forward_cuda, cls.__init__
    fp4x = getattr(mod, '_glm_fp4x_installed', False)
    HAS_FP4X = fp4x
    if MODE == 'fast' and not fp4x:
        raise RuntimeError('FAST requires original FP4x tile geometry')
    if fp4x:
        import glm_fp4_prefill as p
        p._row_shard_stock = p.tiled_prefill
        # Only the wrapper's call is intercepted; inactive/decode goes straight
        # to the original method. ContextVar avoids cross-call metadata leakage.
        from contextvars import ContextVar
        depth_var = ContextVar('glm_row_shard_depth', default=0)
        def tiled(*args, **kwargs):
            if depth_var.get() == 0 or MODE == 'off':
                return p._row_shard_stock(*args, **kwargs)
            return tiled_prefill(*args, **kwargs, depth=depth_var.get())
        p.tiled_prefill = tiled
    else:
        raw = Path(mod.__file__).read_text()
        # Retain the source-pinned short DSA transform inside the fallback op.
        if os.environ.get('GLM_INDEXER_SHORTCUT') == '1':
            import glm_dsa_short
            raw = glm_dsa_short.transform_sparse(raw)
        ns = {}
        exec(compile(transform(raw), mod.__file__, 'exec'), mod.__dict__, ns)
        operation = ns['sparse_attn_indexer']

    @functools.wraps(original_init)
    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        import glm_dsa_short
        config = mod.get_current_vllm_config()
        if not glm_dsa_short.supported(config.model_config.hf_config, config.parallel_config):
            raise RuntimeError('row-shard requires full GLM TP4/PCP1/DCP1')

    @functools.wraps(original)
    def forward(self, hidden_states, q_quant, k, weights):
        import torch
        if FAILED:
            raise RuntimeError('row-shard latched failure; kill this candidate')
        context = mod.get_forward_context()
        md = context.attn_metadata
        layer = md.get(self.k_cache.prefix) if isinstance(md, dict) else None
        if (MODE == 'off' or layer is None or not layer.num_prefills
                or torch.cuda.is_current_stream_capturing()):
            return original(self, hidden_states, q_quant, k, weights)
        if fp4x:
            token = depth_var.set(layer.max_seq_len)
            try:
                return original(self, hidden_states, q_quant, k, weights)
            finally:
                depth_var.reset(token)
        q, scale = q_quant if isinstance(q_quant, tuple) else (q_quant, None)
        return operation(hidden_states, self.k_cache.prefix, self.k_cache.kv_cache,
                         q, scale, k, weights, self.quant_block_size, self.scale_fmt,
                         self.topk_tokens, self.head_dim, self.max_model_len,
                         self.max_total_seq_len, self.topk_indices_buffer,
                         self.skip_k_cache_insert, self.use_pcp, self.dense_mha_metadata_layer_name,
                         self.use_fp4_cache, self.dcp_rank, self.dcp_world_size,
                         self.cp_kv_cache_interleave_size)
    cls.__init__, cls.forward_cuda = init, forward
    sys.stderr.write('glm-indexer-row-shard: armed fp4x=' + str(fp4x) + '\n')


def install_worker(mod):
    pin(mod)
    def row_shard_set(self, mode, sample_every=None):
        return dict(rank=int(os.environ.get('RANK', '-1')), **set_mode(mode, sample_every))
    def row_shard_status(self):
        return dict(rank=int(os.environ.get('RANK', '-1')), **status())
    mod.Worker.indexer_row_shard_set = row_shard_set
    mod.Worker.indexer_row_shard_status = row_shard_status


def register(env=None):
    global MODE, TRANSPORT, REGISTERED, SAMPLE_EVERY
    config = options(os.environ if env is None else env)
    if config is None:
        return False
    if REGISTERED:
        raise RuntimeError('row-shard already armed')
    MODE, TRANSPORT = config
    SAMPLE_EVERY = sample_interval((os.environ if env is None else env).get('GLM_INDEXER_ROW_SHARD_SAMPLE_EVERY', '32'))
    # Register last, outside the short-DSA / Kstop / FP4x source loaders.
    from glm_glue_lite import after_import
    after_import(SPARSE, install_sparse)
    after_import(WORKER, install_worker)
    REGISTERED = True
    return True
