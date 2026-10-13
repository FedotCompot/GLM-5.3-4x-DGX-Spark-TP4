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


def hit(piece, rows, owner):
    COUNTS[piece] += 1
    key = (piece, id(owner))
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
    if env.get('GLM_ATTN_WEIGHTS')!='nvfp4' or env.get('GLM_KV_FORMAT')!='fp4x' or env.get('GLM_FULL_MLA')!='triton':
        raise ValueError('attention front requires release NVFP4/FP4x/Triton MLA')
    for key in ('GLM_NVFP4_MERGED','GLM_FP4_RECENT_WINDOW','GLM_FP4_RECENT_AB','GLM_NVFP4_WSIM'):
        if env.get(key,'0') not in ('','0'):
            raise ValueError('attention front does not compose with '+key)
    if flags[1]=='1' and env.get('VLLM_BATCH_INVARIANT','0') not in ('','0'):
        raise ValueError('attention front stock norms require batch-invariant OFF')
    if env.get('VLLM_MARLIN_USE_ATOMIC_ADD','0') not in ('','0'):
        raise ValueError('attention front requires stock deterministic FP32 Marlin reduction')
    return tuple(v=='1' for v in flags)


def once(source, old, new):
    if source.count(old)!=1:
        raise RuntimeError('attention front source anchor drift: '+old[:80])
    return source.replace(old,new)


def method(raw, owner, name):
    cls=next(n for n in ast.parse(raw).body if isinstance(n,ast.ClassDef) and n.name==owner)
    node=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name==name)
    return textwrap.dedent(''.join(raw.splitlines(keepends=True)[node.lineno-1:node.end_lineno]))


def transform_front(raw):
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
    return s


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
    if (CONFIG[0] and x.is_cuda and x.ndim==2 and x.dtype.__str__()=='torch.bfloat16'
            and 1<=x.shape[0]<=16 and x.shape[-1]==6144
            and not getattr(layer,'_use_min_latency_gemm',False)
            and getattr(layer,'tp_size',0)==1 and getattr(layer,'bias',None) is None
            and type(getattr(layer,'scheme',None)).__name__=='AttentionNVFP4'
            and hasattr(layer,'_glm_nvfp4_parts')
            and [p.output_size_per_partition for p in layer._glm_nvfp4_parts]==[1536,576]
            and owner.q_lora_rank==1536 and owner.kv_lora_rank==512 and owner.qk_rope_head_dim==64):
        from glm_nvfp4_attn import apply_partitions
        hit('parts',x.shape[0],owner)
        return apply_partitions(layer,x)
    COUNTS['parts_fallback']+=1
    value=layer(x)[0]
    return value.split([owner.q_lora_rank,owner.kv_lora_rank+owner.qk_rope_head_dim],dim=-1)


def norm_eligible(owner,x):
    ok=(CONFIG[1] and x.is_cuda and x.ndim==2 and x.shape[-1]==6144 and x.dtype.__str__()=='torch.bfloat16'
        and 1<=x.shape[0]<=16 and owner.q_lora_rank==1536 and owner.kv_lora_rank==512
        and not owner.fuse_qkv_rmsnorm
        and type(owner.q_a_layernorm).__name__=='RMSNorm'
        and type(owner.kv_a_layernorm).__name__=='RMSNorm'
        and all(n.pass_weight and n.variance_size_override is None
                and n.weight.dtype==x.dtype for n in (owner.q_a_layernorm,owner.kv_a_layernorm)))
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


def rope(owner,positions,q,k,x):
    import torch
    rotary=owner.rotary_emb
    # This piece covers only the measured native CUDA op, never FlashInfer
    # or DeepseekScaling's out-of-place native Python arithmetic.
    # Read the dispatch CustomOp chose at construction; enabled() needs the
    # vLLM config context, which is not set during forward.
    native=(getattr(rotary._forward_method,'__name__','')=='forward_cuda' and (type(rotary).__name__=='RotaryEmbedding' and not rotary.use_flashinfer
            or type(rotary).__name__=='DeepseekV4ScalingRotaryEmbedding'))
    ok=(CONFIG[2] and native and q.is_cuda and q.dtype==torch.bfloat16
        and 1<=q.shape[0]<=16 and q.shape[1:]==(16,64) and k.shape[1:]==(1,64)
        and q.stride(0)==4096 and q.stride(1)==256 and k.stride(0) in (576,2112)
        and rotary.rotary_dim==64 and not owner.dcp_q_replicate
        and not owner.mla_attn.use_pcp
        and not getattr(owner.mla_attn.impl,'_glm_recent_bank',None))
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
        owner=mod.MultiHeadLatentAttentionWrapper
        transformed=transform_front(raw)
        mod.__dict__.update(_af_partitions=partitions,_af_norm_eligible=norm_eligible,
                            _af_norms=norms,_af_rope=rope)
    else:
        owner=mod.MLAAttention
        transformed=transform_common(raw)
    ns={}
    exec(compile(transformed,mod.__file__,'exec'),mod.__dict__,ns)
    owner.forward=ns['forward']
    mod._glm_attnfront=True


def register(env=None):
    global CONFIG
    selected=options(env)
    if selected is None:
        return False
    if CONFIG is not None:
        if CONFIG!=selected:
            raise RuntimeError('attention front is boot-fixed; restart to switch')
        return True
    check_pins()
    CONFIG=selected
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
