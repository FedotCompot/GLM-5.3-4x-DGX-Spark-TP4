# SPDX-License-Identifier: Apache-2.0
"""CPU-only host forward timing. No GPU work; stubs preserve tensor metadata.

Stock and both patched methods execute their complete Python control flow.
Tensor math/allocations/native calls are CPU-safe no-ops. Consequently these
are host control-flow measurements, not serving latency or GPU speed claims.
"""
import argparse
import contextlib
import gc
import importlib.util
import io
import json
from pathlib import Path
import platform
import statistics
import sys
import time
import types
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'overlay/bringup'))
import glm_attnfront as current


class Tensor:
    dtype='torch.bfloat16'
    is_cuda=True
    device='cuda:0'
    def __init__(self,shape,strides=None):
        self.shape=tuple(shape);self.ndim=len(shape)
        self.strides=strides or (self.shape[-1],1)
    def stride(self,i): return self.strides[i]
    def split(self,widths,dim=-1):
        return tuple(Tensor((*self.shape[:-1],w),self.strides) for w in widths)
    def unsqueeze(self,i):
        return Tensor((self.shape[0],1,self.shape[1]),(self.strides[0],self.shape[1],1))
    def view(self,rows,heads,width):
        return Tensor((self.shape[0],heads,width),(heads*width,width,1))
    def __getitem__(self,key):
        return Tensor((*self.shape[:-1],64),self.strides)
    def __setitem__(self,key,value): pass


class RMSNorm:
    pass_weight=True
    variance_size_override=None
    variance_epsilon=1e-5
    def __init__(self,width): self.weight=Tensor((width,))
    def __call__(self,x): return Tensor(x.shape)


class AttentionNVFP4: pass


class Linear:
    tp_size=1
    bias=None
    scheme=AttentionNVFP4()
    def __init__(self,width): self.width=width
    def __call__(self,x): return Tensor((x.shape[0],self.width)),None


class RotaryEmbedding:
    rotary_dim=64
    use_flashinfer=False
    is_neox_style=False
    cos_sin_cache=Tensor((4096,64))
    def __init__(self): self._forward_method=self.forward_cuda
    def forward_cuda(self,*args): pass
    def __call__(self,p,q,k): return q,k
    def _match_cos_sin_cache_dtype(self,q): return self.cos_sin_cache


class Attention:
    layer_name='model.layers.1.self_attn.attn'
    use_pcp=False
    impl=types.SimpleNamespace(dcp_world_size=1)
    def __call__(self,q,k,r,**kwargs): return k


def fixture():
    class Owner: pass
    o=Owner()
    for k,v in dict(hidden_size=6144,q_lora_rank=2048,kv_lora_rank=512,
            qk_rope_head_dim=64,num_heads=16,qk_nope_head_dim=192,v_head_dim=256,
            qk_head_dim=256,fuse_qkv_rmsnorm=False,dcp_q_replicate=False,
            indexer=None,is_sparse=True,skip_topk=False,g_proj=None).items(): setattr(o,k,v)
    o.fused_qkv_a_proj=Linear(2624)
    o.fused_qkv_a_proj._glm_nvfp4_parts=[types.SimpleNamespace(output_size_per_partition=n) for n in (2048,576)]
    o.q_a_layernorm=RMSNorm(2048);o.kv_a_layernorm=RMSNorm(512)
    o.q_b_proj=Linear(4096);o.o_proj=Linear(6144)
    o.rotary_emb=RotaryEmbedding();o.mla_attn=Attention()
    return o


_RMS_IMPL=types.SimpleNamespace(provider='vllm_c',supported=True)


def dispatch(x,w,e,v=None):
    # Same pinned vllm_c priority-loop/support test, without vLLM imports.
    for impl in (_RMS_IMPL,):
        if impl.supported and v is None and (w is None or w.dtype==x.dtype): return impl


def stubs(capturing=False):
    torch=types.SimpleNamespace(Tensor=Tensor,bfloat16=Tensor.dtype,contiguous_format=0,
        empty_like=lambda x,**kw:Tensor(x.shape),
        empty=lambda *a,**kw:Tensor((0,)),
        empty_strided=lambda shape,strides,**kw:Tensor(shape,strides),
        cuda=types.SimpleNamespace(is_current_stream_capturing=lambda:capturing),
        ops=types.SimpleNamespace(glm_attnfront=types.SimpleNamespace(
            dual_rms=lambda *a:None,rope_update=lambda *a:Tensor((0,)))))
    nv=types.SimpleNamespace(apply_partitions=lambda layer,x:tuple(
        Tensor((x.shape[0],p.output_size_per_partition)) for p in layer._glm_nvfp4_parts))
    vllm=types.SimpleNamespace(ir=types.SimpleNamespace(ops=types.SimpleNamespace(
        rms_norm=types.SimpleNamespace(dispatch=dispatch))))
    return {'torch':torch,'glm_nvfp4_attn':nv,'vllm':vllm}


def functions(module,raw,stub,cached):
    ns=dict(torch=stub['torch'],_af_norm_eligible=module.norm_eligible,
        _af_partitions=module.partitions,_af_norms=module.norms,_af_rope=module.rope,
        _af_apply_parts=stub['glm_nvfp4_attn'].apply_partitions,
        _af_bf16=Tensor.dtype,_af_debug=False,
        _af_capturing=stub['torch'].cuda.is_current_stream_capturing,
        _af_record=current.record,_af_rope_cached=current.rope_cached)
    scope={}
    source=module.transform_front(raw) if not cached else current.transform_front(raw)
    exec(compile(source,'patched-forward','exec'),ns,scope)
    fused=scope['forward']
    exec(compile(current.method(raw,'MultiHeadLatentAttentionWrapper','forward'),'stock-forward','exec'),ns,scope)
    return scope['forward'],fused


def run(before,image,out,reps=15,iterations=20000):
    old=types.ModuleType('attnfront_before');old.__file__=current.__file__
    exec(compile(before.read_text(),str(before),'exec'),old.__dict__)
    raw=(image/'vllm/model_executor/layers/mla.py').read_text()
    records=[];stub=stubs()
    with patch.dict(sys.modules,stub),patch.object(current,'CONFIG',(True,True,True)),contextlib.redirect_stderr(io.StringIO()):
        old.CONFIG=(True,True,True)
        stock,after=functions(current,raw,stub,True)
        _,prior=functions(old,raw,stub,False)
        for m in (1,4,64,2048):
            x=Tensor((m,6144));p=Tensor((m,))
            owners=[fixture() for _ in range(3)]
            owners[0].forward=stock.__get__(owners[0],type(owners[0]))
            owners[1].forward=prior.__get__(owners[1],type(owners[1]))
            current.bind_forward(type(owners[2]),stock,after)
            calls=[lambda o=o:o.forward(p,x) for o in owners]
            for call in calls:
                for _ in range(1000): call()
            times=[[],[],[]]
            enabled=gc.isenabled();gc.disable()
            try:
                for rep in range(reps):
                    for i in ((rep+j)%3 for j in range(3)):
                        start=time.perf_counter_ns()
                        for _ in range(iterations): calls[i]()
                        times[i].append((time.perf_counter_ns()-start)/iterations/1000)
            finally:
                if enabled: gc.enable()
            med=[statistics.median(t) for t in times]
            records.append(dict(m=m,stock_us=med[0],before_us=med[1],after_us=med[2],
                before_overhead_us=med[1]-med[0],after_overhead_us=med[2]-med[0],
                before_x78_ms=(med[1]-med[0])*78/1000,after_x78_ms=(med[2]-med[0])*78/1000,
                saved_x78_ms=(med[1]-med[2])*78/1000,samples_us=times))
    result=dict(kind='CPU fake tensor host control-flow only; no GPU/serving measurement',
        platform=platform.platform(),python=sys.version,reps=reps,iterations=iterations,records=records)
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({**result,'records':[{k:v for k,v in r.items() if k!='samples_us'} for r in records]},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before',type=Path,required=True)
    parser.add_argument('--image',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--reps',type=int,default=15)
    parser.add_argument('--iterations',type=int,default=20000)
    args=parser.parse_args();run(**vars(args))
