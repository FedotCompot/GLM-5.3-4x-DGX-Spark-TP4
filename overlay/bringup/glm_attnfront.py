# SPDX-License-Identifier: Apache-2.0
"""Boot-fixed E0 attention front. No imports/hooks/native load when OFF."""
import ast
import functools
import hashlib
import importlib.abc
import importlib.util
import json
import os
from pathlib import Path
import sys
import textwrap

FRONT = 'vllm.model_executor.layers.mla'
COMMON = 'vllm.model_executor.layers.attention.mla_attention'
PINS = json.loads(Path(__file__).with_name('attnfront_source_pins.json').read_text())
FLAGS = ('GLM_ATTN_FRONT_PARTS', 'GLM_ATTN_FRONT_NORMS', 'GLM_ATTN_FRONT_ROPE_PACK')
CONFIG = None
COUNTS = dict(parts=0, norms=0, rope_pack=0, parts_fallback=0, norms_fallback=0, rope_fallback=0)
_HITS = set()
DEBUG = False
_RMS_PROVIDERS = {}
# (hidden, q rank, kv rank, rope dim, local heads, nope dim, value dim).
# Admission is explicit; widths/strides passed to operators come from the owner.
ADMITTED_SHAPES = frozenset({
    (6144, 2048, 512, 64, 16, 192, 256),
    (6144, 1536, 512, 64, 16, 192, 256),
})


def owner_shape(owner):
    return tuple(getattr(owner, name, None) for name in (
        'hidden_size', 'q_lora_rank', 'kv_lora_rank', 'qk_rope_head_dim',
        'num_heads', 'qk_nope_head_dim', 'v_head_dim'))


def quantized_owner(owner):
    # Layer zero is unquantized BF16. Quantized MTP may still use stock INT8;
    # its norms/RoPE are independent of the projection's quantization scheme.
    return (owner_shape(owner) in ADMITTED_SHAPES
            and getattr(owner.fused_qkv_a_proj, 'scheme', None) is not None)


def nvfp4_owner(owner):
    layer = owner.fused_qkv_a_proj
    widths = [owner.q_lora_rank, owner.kv_lora_rank + owner.qk_rope_head_dim]
    return (owner_shape(owner) in ADMITTED_SHAPES
            and not getattr(layer, '_use_min_latency_gemm', False)
            and getattr(layer, 'tp_size', 0) == 1 and getattr(layer, 'bias', None) is None
            and type(getattr(layer, 'scheme', None)).__name__ == 'AttentionNVFP4'
            and [p.output_size_per_partition for p in getattr(layer, '_glm_nvfp4_parts', ())] == widths)


def parts_eligible(owner, x):
    return (CONFIG[0] and nvfp4_owner(owner) and x.is_cuda and x.ndim == 2
            and str(x.dtype) == 'torch.bfloat16' and 1 <= x.shape[0] <= 16
            and x.shape[-1] == owner.hidden_size)


def hit(piece, rows, owner):
    COUNTS[piece] += 1
    key = (piece, rows, id(owner))
    if key not in _HITS:
        _HITS.add(key)
        sys.stderr.write('ATTNFRONT_HIT '+piece+' M'+str(rows)+' '+owner.mla_attn.layer_name+'\n')

def options(env=None):
    env = os.environ if env is None else env
    value = env.get('GLM_ATTN_FRONT_FUSE', '0')
    if value not in ('0','1'):
        raise ValueError('GLM_ATTN_FRONT_FUSE must be 0 or 1')
    # Disabled subflags must have no effect, including on DRY vectors.
    if value == '0':
        return None
    flags = tuple(env.get(k,'1') for k in FLAGS)
    if any(v not in ('0','1') for v in flags):
        raise ValueError('attention front piece switches must be 0 or 1')
    if not any(v=='1' for v in flags):
        raise ValueError('attention front requires at least one piece')
    # On by default in the release profile: any other layout keeps the stock path (no hooks), never an error.
    if env.get('GLM_ATTN_WEIGHTS')!='nvfp4' or env.get('GLM_KV_FORMAT')!='fp4x' or env.get('GLM_FULL_MLA')!='triton':
        return None
    for key in ('GLM_NVFP4_MERGED','GLM_FP4_RECENT_WINDOW','GLM_FP4_RECENT_AB','GLM_NVFP4_WSIM'):
        if env.get(key,'0') not in ('','0'):
            return None
    if flags[1]=='1' and env.get('VLLM_BATCH_INVARIANT','0') not in ('','0'):
        return None
    if env.get('VLLM_MARLIN_USE_ATOMIC_ADD','0') not in ('','0'):
        return None
    return tuple(v=='1' for v in flags)


def once(source, old, new):
    if source.count(old)!=1:
        raise RuntimeError('attention front source anchor drift: '+old[:80])
    return source.replace(old,new)


def method(raw, owner, name):
    cls=next(n for n in ast.parse(raw).body if isinstance(n,ast.ClassDef) and n.name==owner)
    node=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name==name)
    return textwrap.dedent(''.join(raw.splitlines(keepends=True)[node.lineno-1:node.end_lineno]))


def transform_front(raw, cached=True):
    s=method(raw,'MultiHeadLatentAttentionWrapper','forward')
    s=once(s,'    q_c = None\n','    _front_norm = _af_norm_eligible(self, hidden_states)\n    _front_dep = None\n    q_c = None\n')
    s=once(s,'''        qkv_lora = self.fused_qkv_a_proj(hidden_states)[0]
        q_c, kv_lora = qkv_lora.split(
            [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
            dim=-1,
        )''','''        q_c, kv_lora = _af_partitions(self, hidden_states)''')
    s=once(s,'        if not self.fuse_qkv_rmsnorm:\n','        if not self.fuse_qkv_rmsnorm and not _front_norm:\n')
    s=once(s,'    if self.fuse_qkv_rmsnorm and q_c is not None:\n', '''    if _front_norm:
        q_proj_input, kv_c_normed = _af_norms(self, q_c, kv_c)
        q_c = q_proj_input
    elif self.fuse_qkv_rmsnorm and q_c is not None:
''')
    s=once(s,'''        q[..., self.qk_nope_head_dim :], k_pe = self.rotary_emb(
            positions, q[..., self.qk_nope_head_dim :], k_pe
        )''','''        q[..., self.qk_nope_head_dim :], k_pe, _front_dep = _af_rope(
            self, positions, q[..., self.qk_nope_head_dim :], k_pe, kv_c_normed
        )''')
    s=once(s,'        q_dcp_replicated=q_dcp_replicated,\n','        q_dcp_replicated=q_dcp_replicated,\n        attnfront_dep=_front_dep,\n')
    if cached:
        s=once(s, '    _front_norm = _af_norm_eligible(self, hidden_states)\n', '''    # Large/mixed prefill takes the unmodified stock method before any work.
    if not 1 <= hidden_states.shape[0] <= 16:
        return self._glm_attnfront_stock(positions, hidden_states, llama_4_scaling)
    # Rank/hidden width follow the loaded model's projection contract.
    if hidden_states.dtype != _af_bf16 or not hidden_states.is_cuda:
        return self._glm_attnfront_stock(positions, hidden_states, llama_4_scaling)
    _front_parts, _front_norm, _front_rope = self._glm_attnfront_pieces
    if _af_debug or _af_capturing():
        _af_record(self, hidden_states.shape[0])
''')
        s=once(s, '        q_c, kv_lora = _af_partitions(self, hidden_states)', '''        if _front_parts:
            q_c, kv_lora = _af_apply_parts(self.fused_qkv_a_proj, hidden_states)
        else:
            qkv_lora = self.fused_qkv_a_proj(hidden_states)[0]
            q_c, kv_lora = qkv_lora.split(
                [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim], dim=-1)
'''.rstrip())
        s=once(s, '''        q[..., self.qk_nope_head_dim :], k_pe, _front_dep = _af_rope(
            self, positions, q[..., self.qk_nope_head_dim :], k_pe, kv_c_normed
        )''', '''        if _front_rope:
            q[..., self.qk_nope_head_dim :], k_pe, _front_dep = _af_rope_cached(
                self, positions, q[..., self.qk_nope_head_dim :], k_pe, kv_c_normed)
        else:
            q[..., self.qk_nope_head_dim :], k_pe = self.rotary_emb(
                positions, q[..., self.qk_nope_head_dim :], k_pe)
'''.rstrip())
    return s


def rms_provider_supported(owner, x, parts):
    """Boot-fixed dispatch; validate every admitted M/layout once per signature.

    Probe metadata only (empty_strided launches no kernel). Pinned vllm_c
    supports_args tests dtype/variance; include shape/stride/epsilon/device so
    another prioritized provider cannot be mistaken for that stock provider.
    """
    import torch
    from vllm import ir
    stride=owner.q_lora_rank+owner.kv_lora_rank+owner.qk_rope_head_dim
    for n,width in ((owner.q_a_layernorm,owner.q_lora_rank),
                    (owner.kv_a_layernorm,owner.kv_lora_rank)):
        rs=(width if width==owner.q_lora_rank else width+owner.qk_rope_head_dim) if parts else stride
        for m in range(1,17):
            key=(m,width,rs,x.dtype,x.device,n.weight.dtype,n.weight.device,
                 n.weight.stride(0),n.variance_epsilon)
            if key not in _RMS_PROVIDERS:
                probe=torch.empty_strided((m,width),(rs,1),dtype=x.dtype,device=x.device)
                _RMS_PROVIDERS[key]=(ir.ops.rms_norm.dispatch(
                    probe,n.weight,n.variance_epsilon,None).provider=='vllm_c')
            if not _RMS_PROVIDERS[key]:
                return False
    return True


def static_pieces(owner, x):
    """Loaded owner contract. Called once, never from the installed fast path."""
    if not quantized_owner(owner):
        return (False,False,False)
    parts=CONFIG[0] and nvfp4_owner(owner)
    norm_contract=(not owner.fuse_qkv_rmsnorm
        and type(owner.q_a_layernorm).__name__=='RMSNorm'
        and type(owner.kv_a_layernorm).__name__=='RMSNorm'
        and all(n.pass_weight and n.variance_size_override is None
                and str(n.weight.dtype)=='torch.bfloat16' and tuple(n.weight.shape)==(width,)
                for n,width in ((owner.q_a_layernorm,owner.q_lora_rank),
                                (owner.kv_a_layernorm,owner.kv_lora_rank))))
    # A CPU/profile first call must not permanently disqualify a CUDA owner.
    norms_ok=CONFIG[1] and norm_contract
    if norms_ok:
        import torch
        probe=x if x.is_cuda and x.dtype==torch.bfloat16 else torch.empty(
            0,device=owner.q_a_layernorm.weight.device,dtype=torch.bfloat16)
        norms_ok=rms_provider_supported(owner,probe,parts)
    rotary=owner.rotary_emb
    native=(getattr(getattr(rotary,'_forward_method',None),'__name__','')=='forward_cuda'
            and (type(rotary).__name__=='RotaryEmbedding' and not rotary.use_flashinfer
                 or type(rotary).__name__=='DeepseekV4ScalingRotaryEmbedding'))
    # The pinned non-fused RMSNorm produces contiguous kv_c_normed; q_b_proj
    # produces [M,H*qk_dim], and admitted projections produce the two k strides.
    rope_ok=(CONFIG[2] and native and norm_contract
        and owner.qk_head_dim==owner.qk_nope_head_dim+owner.qk_rope_head_dim
        and rotary.rotary_dim==owner.qk_rope_head_dim and not owner.dcp_q_replicate
        and not owner.mla_attn.use_pcp
        and getattr(owner.mla_attn.impl,'dcp_world_size',1)==1
        and not getattr(owner.mla_attn.impl,'_glm_recent_bank',None))
    return (bool(parts),bool(norms_ok),bool(rope_ok))


def record(owner, rows):
    for enabled,piece,fallback in zip(owner._glm_attnfront_pieces,
            ('parts','norms','rope_pack'),('parts_fallback','norms_fallback','rope_fallback')):
        if enabled:
            hit(piece,rows,owner)
        else:
            COUNTS[fallback]+=1


def rope_cached(owner, positions, q, k, x):
    import torch
    rotary=owner.rotary_emb
    cache=rotary._match_cos_sin_cache_dtype(q) if owner._glm_attnfront_match_cache else rotary.cos_sin_cache
    dep=torch.ops.glm_attnfront.rope_update(q,k,x,positions,cache,
        owner.mla_attn.layer_name,rotary.is_neox_style)
    return q,k,dep


def bind_forward(owner, original, patched):
    """First-call initialization happens after quantization/weight loading."""
    @functools.wraps(original)
    def initialize(self, positions, hidden_states, llama_4_scaling=None):
        self._glm_attnfront_stock=original.__get__(self,type(self))
        self._glm_attnfront_pieces=static_pieces(self,hidden_states)
        self._glm_attnfront_match_cache=type(self.rotary_emb).__name__=='RotaryEmbedding'
        if hasattr(self,'register_load_state_dict_post_hook') and not getattr(self,'_glm_attnfront_reload_hook',False):
            self.register_load_state_dict_post_hook(lambda module,keys: invalidate(module))
            self._glm_attnfront_reload_hook=True
        self.forward=(patched if any(self._glm_attnfront_pieces) else original).__get__(self,type(self))
        return self.forward(positions,hidden_states,llama_4_scaling)
    owner._glm_attnfront_original=original
    owner._glm_attnfront_initialize=initialize
    owner.forward=initialize


def invalidate(owner):
    """Revalidate after custom in-place reloads; normal state_dict uses a hook."""
    for name in ('forward','_glm_attnfront_pieces','_glm_attnfront_stock',
                 '_glm_attnfront_match_cache'):
        owner.__dict__.pop(name,None)


def transform_common(raw):
    s=method(raw,'MLAAttention','forward')
    s=once(s,'    q_dcp_replicated: torch.Tensor | None = None,\n','    q_dcp_replicated: torch.Tensor | None = None,\n    attnfront_dep: torch.Tensor | None = None,\n')
    a=s.index('        self.impl.do_kv_cache_update(')
    b=s.index('        output = torch.empty(',a)
    block=s[a:b]
    s=s[:a]+'        if attnfront_dep is None:\n'+textwrap.indent(block,'    ')+s[b:]
    a=s.index('        kv_cache_dummy_dep = torch.ops.vllm.unified_mla_kv_cache_update(')
    b=s.index('        output = torch.empty(',a)
    block=s[a:b]
    s=s[:a]+'        kv_cache_dummy_dep = attnfront_dep\n        if kv_cache_dummy_dep is None:\n'+textwrap.indent(block,'    ')+s[b:]
    return s


def partitions(owner, x):
    layer=owner.fused_qkv_a_proj
    if parts_eligible(owner, x):
        from glm_nvfp4_attn import apply_partitions
        hit('parts',x.shape[0],owner)
        return apply_partitions(layer,x)
    COUNTS['parts_fallback']+=1
    value=layer(x)[0]
    return value.split([owner.q_lora_rank,owner.kv_lora_rank+owner.qk_rope_head_dim],dim=-1)


def norm_eligible(owner,x):
    ok=(CONFIG[1] and quantized_owner(owner) and x.is_cuda and x.ndim==2
        and x.shape[-1]==owner.hidden_size and str(x.dtype)=='torch.bfloat16'
        and 1<=x.shape[0]<=16
        and not owner.fuse_qkv_rmsnorm
        and type(owner.q_a_layernorm).__name__=='RMSNorm'
        and type(owner.kv_a_layernorm).__name__=='RMSNorm'
        and all(n.pass_weight and n.variance_size_override is None
                and n.weight.dtype==x.dtype and tuple(n.weight.shape)==(width,)
                for n,width in ((owner.q_a_layernorm,owner.q_lora_rank),
                                (owner.kv_a_layernorm,owner.kv_lora_rank))))
    if ok:
        from vllm import ir
        ok=all(ir.ops.rms_norm.dispatch(x,n.weight,n.variance_epsilon,None).provider=='vllm_c'
               for n in (owner.q_a_layernorm,owner.kv_a_layernorm))
    if ok:
        hit('norms',x.shape[0],owner)
    else:
        COUNTS['norms_fallback']+=1
    return ok


def norms(owner,q,k):
    import torch
    qo=torch.empty_like(q,memory_format=torch.contiguous_format)
    ko=torch.empty_like(k,memory_format=torch.contiguous_format)
    torch.ops.glm_attnfront.dual_rms(q,k,owner.q_a_layernorm.weight,
        owner.kv_a_layernorm.weight,owner.q_a_layernorm.variance_epsilon,
        owner.kv_a_layernorm.variance_epsilon,qo,ko)
    return qo,ko


def rope_eligible(owner, q, k, x):
    rotary=owner.rotary_emb
    # This piece covers only the measured native CUDA op, never FlashInfer
    # or DeepseekScaling's out-of-place native Python arithmetic.
    # Read the dispatch CustomOp chose at construction; enabled() needs the
    # vLLM config context, which is not set during forward.
    native=(getattr(getattr(rotary,'_forward_method',None),'__name__','')=='forward_cuda' and (type(rotary).__name__=='RotaryEmbedding' and not rotary.use_flashinfer
            or type(rotary).__name__=='DeepseekV4ScalingRotaryEmbedding'))
    return (CONFIG[2] and quantized_owner(owner) and native and q.is_cuda
        and str(q.dtype)=='torch.bfloat16' and k.dtype==q.dtype and x.dtype==q.dtype
        and 1<=q.shape[0]<=16 and q.shape[1:]==(owner.num_heads,owner.qk_rope_head_dim)
        and k.shape==(q.shape[0],1,owner.qk_rope_head_dim)
        and x.shape==(q.shape[0],owner.kv_lora_rank)
        and q.stride(0)==owner.num_heads*owner.qk_head_dim
        and q.stride(1)==owner.qk_head_dim
        and k.stride(0) in (owner.kv_lora_rank+owner.qk_rope_head_dim,
                            owner.q_lora_rank+owner.kv_lora_rank+owner.qk_rope_head_dim)
        and x.stride(0)==owner.kv_lora_rank
        and all(t.stride(-1)==1 for t in (q,k,x))
        and rotary.rotary_dim==owner.qk_rope_head_dim and not owner.dcp_q_replicate
        and not owner.mla_attn.use_pcp
        and not getattr(owner.mla_attn.impl,'_glm_recent_bank',None))


def rope(owner,positions,q,k,x):
    import torch
    rotary=owner.rotary_emb
    ok=rope_eligible(owner,q,k,x)
    if ok:
        hit('rope_pack',q.shape[0],owner)
        cache=rotary.cos_sin_cache
        if type(rotary).__name__=='RotaryEmbedding':
            cache=rotary._match_cos_sin_cache_dtype(q)
        dep=torch.ops.glm_attnfront.rope_update(q,k,x,positions,cache,
            owner.mla_attn.layer_name,rotary.is_neox_style)
        return q,k,dep
    COUNTS['rope_fallback']+=1
    q,k=rotary(positions,q,k)
    return q,k,None


def register_ops():
    import torch
    from vllm.model_executor.layers.attention.attention import get_attention_context
    @torch.library.custom_op('glm_attnfront::rope_update',mutates_args=('q','k'))
    def update(q:torch.Tensor,k:torch.Tensor,x:torch.Tensor,p:torch.Tensor,c:torch.Tensor,
               layer_name:str,neox:bool)->torch.Tensor:
        md,layer,cache,slots=get_attention_context(layer_name)
        if (layer.use_pcp or layer.impl.dcp_world_size!=1 or
                not getattr(layer.impl.do_kv_cache_update,'_glm_fp4x',False)):
            raise RuntimeError('attention front cache writer/layout drift')
        from glm_attnfront_kernel import rope_pack
        if cache.numel()==0:
            cache=torch.empty((0,368),device=q.device,dtype=torch.uint8)
            slots=None
        rope_pack(q,k,x,cache,p,c,slots,layer._k_scale,neox)
        return torch.empty(0,device=q.device,dtype=q.dtype)
    @update.register_fake
    def fake(q,k,x,p,c,layer_name,neox):
        return torch.empty(0,device=q.device,dtype=q.dtype)
    if CONFIG[1]:
        from attnfront_native.build import verify
        verify()
        torch.ops.load_library(str(Path(__file__).parent/'attnfront_native/build/attnfront.so'))
        @torch.library.register_fake('glm_attnfront::dual_rms')
        def fake_rms(q,k,qw,kw,qe,ke,qo,ko):
            return None


def check_pins():
    for name,pin in json.loads(Path(__file__).with_name('attnfront_local_pins.json').read_text()).items():
        source=Path(__file__).parent/name
        if hashlib.sha256(source.read_bytes()).hexdigest()!=pin:
            raise RuntimeError('attention front local source drift: '+name)
    spec=importlib.util.find_spec('vllm')
    root=Path(next(iter(spec.submodule_search_locations)))
    for name,pin in PINS.items():
        path=root/(name.removeprefix('vllm.').replace('.','/')+'.py')
        if hashlib.sha256(path.read_bytes()).hexdigest()!=pin:
            raise RuntimeError('attention front upstream source drift: '+name)


def install(mod):
    if mod.__name__ not in (FRONT,COMMON) or getattr(mod,'_glm_attnfront',False):
        return
    raw=Path(mod.__file__).read_text()
    if hashlib.sha256(raw.encode()).hexdigest()!=PINS[mod.__name__]:
        raise RuntimeError('attention front source drift '+mod.__name__)
    if mod.__name__==FRONT:
        import torch
        from glm_nvfp4_attn import apply_partitions
        owner=mod.MultiHeadLatentAttentionWrapper
        original=owner.forward
        transformed=transform_front(raw)
        mod.__dict__.update(_af_partitions=partitions,_af_norm_eligible=norm_eligible,
                            _af_norms=norms,_af_rope=rope,
                            _af_apply_parts=apply_partitions,_af_rope_cached=rope_cached,
                            _af_bf16=torch.bfloat16,_af_debug=DEBUG,
                            _af_capturing=torch.cuda.is_current_stream_capturing,_af_record=record)
    else:
        owner=mod.MLAAttention
        transformed=transform_common(raw)
    ns={}
    exec(compile(transformed,mod.__file__,'exec'),mod.__dict__,ns)
    if mod.__name__==FRONT:
        bind_forward(owner,original,ns['forward'])
    else:
        owner.forward=ns['forward']
    mod._glm_attnfront=True


def register(env=None):
    global CONFIG, DEBUG
    selected=options(env)
    if selected is None:
        return False
    if CONFIG is not None:
        if CONFIG!=selected:
            raise RuntimeError('attention front is boot-fixed; restart to switch')
        return True
    check_pins()
    CONFIG=selected
    DEBUG=(os.environ if env is None else env).get('GLM_ATTN_FRONT_DEBUG','0')=='1'
    register_ops()
    class Finder(importlib.abc.MetaPathFinder):
        busy=set()
        def find_spec(self,name,path=None,target=None):
            if name not in (FRONT,COMMON) or name in self.busy:
                return None
            self.busy.add(name)
            try:
                spec=importlib.util.find_spec(name)
            finally:
                self.busy.remove(name)
            original=spec.loader.exec_module
            def execute(mod):
                original(mod)
                install(mod)
            spec.loader.exec_module=execute
            return spec
    sys.meta_path.insert(0,Finder())
    for name in (COMMON,FRONT):
        if name in sys.modules:
            install(sys.modules[name])
    sys.stderr.write('ATTNFRONT_READY pieces='+repr(CONFIG)+'; boot-fixed E0 candidate\n')
    return True
