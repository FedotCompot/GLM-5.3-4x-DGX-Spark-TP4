#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Coordinator-only E0 byte gate and L2-scrubbed CUDA-event timings; no profiler."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'overlay/bringup'),str(ROOT/'overlay/overlay')]

def bytes_equal(a,b,label):
    import torch
    if a.shape!=b.shape or a.dtype!=b.dtype or not torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8)):
        raise AssertionError('E0 byte mismatch: '+label)

def bench_pair(stock,fused,reset,check,scrub,reps):
    import torch
    result={}
    # Equality eager and seven sequential graph replays, including changed input.
    reset();stock();fused();check()
    stream=torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):reset();stock();fused()
    torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
    reset()
    gs,gf=torch.cuda.CUDAGraph(),torch.cuda.CUDAGraph()
    with torch.cuda.graph(gs):stock()
    with torch.cuda.graph(gf):fused()
    for i in range(7):
        reset(i);gs.replay();gf.replay();check()
    for mode,fs,ff in (('eager',stock,fused),('graph',gs.replay,gf.replay)):
        times={'stock':[],'fused':[]}
        for i in range(reps):
            for name,fn in (('stock',fs),('fused',ff)) if i%2==0 else (('fused',ff),('stock',fs)):
                reset(i)
                scrub.add_(1)  # 64 MiB > GB10 L2, outside event interval.
                begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                begin.record();fn();end.record();end.synchronize()
                times[name].append(begin.elapsed_time(end)*1000)
        result[mode]={k:statistics.median(v) for k,v in times.items()}
        result[mode]['saved_us']=result[mode]['stock']-result[mode]['fused']
        result[mode]['samples_us']=times
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--reps',type=int,default=31)
    p.add_argument('--real-inputs',type=Path)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    if args.reps<7:raise ValueError('at least 7 timing samples')
    import torch
    import vllm._custom_ops as ops
    from attnfront_native.build import verify
    from glm_attnfront_kernel import rope_pack
    from glm_fp4_kv_kernel import pack_store
    import glm_nvfp4_attn as nv
    verify();torch.ops.load_library(str(ROOT/'overlay/bringup/attnfront_native/build/attnfront.so'))
    if torch.cuda.get_device_capability()!=(12,1):raise RuntimeError('requires GB10 sm_121')
    props=torch.cuda.get_device_properties(0)
    if props.multi_processor_count!=48:raise RuntimeError('requires GB10 48 SMs')
    torch.manual_seed(20261013);torch.set_num_threads(2)
    scrub=torch.zeros(64<<20,device='cuda',dtype=torch.uint8)
    records=[];started=time.monotonic()
    angle=torch.randn(262144,32,device='cuda',dtype=torch.float32)
    full_cache=torch.cat((angle.cos(),angle.sin()),-1)
    caches={torch.float32:full_cache,torch.bfloat16:full_cache.to(torch.bfloat16)}
    position_cases=torch.tensor([0,1,127,2047,32767,65535,95000,129000,262000],device='cuda',dtype=torch.int64)
    del angle
    real=[] if args.real_inputs is None else torch.load(args.real_inputs,weights_only=True,map_location='cuda')
    # Finite random and edge activations in the exact live layouts. Each M is
    # checked with separate weights/eps and both input row strides.
    for m in range(1,17):
        for layout in ('partition','cat'):
            src=torch.randn(m,2112,device='cuda',dtype=torch.bfloat16)
            q=src[:,:1536] if layout=='cat' else src[:,:1536].contiguous()
            kv=src[:,1536:] if layout=='cat' else src[:,1536:].contiguous()
            k=kv[:,:512]
            qw=torch.randn(1536,device='cuda',dtype=torch.bfloat16);kw=torch.randn(512,device='cuda',dtype=torch.bfloat16)
            qs,ks=torch.empty_like(q,memory_format=torch.contiguous_format),torch.empty_like(k,memory_format=torch.contiguous_format)
            qf,kf=torch.empty_like(qs),torch.empty_like(ks)
            def stock():ops.rms_norm(qs,q,qw,1e-6);ops.rms_norm(ks,k,kw,1e-5)
            def fused():torch.ops.glm_attnfront.dual_rms(q,k,qw,kw,1e-6,1e-5,qf,kf)
            def check():bytes_equal(qs,qf,'q norm');bytes_equal(ks,kf,'kv norm')
            stock();fused();check()
            for vals in (0.,-0.,1.,256.,2**-20):
                q.fill_(vals);k.fill_(vals);stock();fused();check()
            q.copy_(torch.randn_like(q));k.copy_(torch.randn_like(k))
            for item in real:
                if item['q_a'].shape[0]==m:
                    q.copy_(item['q_a']);k.copy_(item['kv_a']);stock();fused();check()
            timing=bench_pair(stock,fused,lambda i=0:None,check,scrub,args.reps)
            records.append(dict(piece='norms',m=m,layout=layout,byte_equal=True,timing=timing))
        for cache_dtype in (torch.bfloat16,torch.float32):
            for neox in (False,True):
                for rs in (576,2112):
                    baseq=torch.randn(m,16,256,device='cuda',dtype=torch.bfloat16)
                    basek=torch.randn(m,rs,device='cuda',dtype=torch.bfloat16)
                    baseq[...,192:]=torch.randn(m,16,64,device='cuda',dtype=torch.bfloat16)
                    x=torch.randn(m,512,device='cuda',dtype=torch.bfloat16)
                    qbs,qbf=baseq.clone(),baseq.clone();kbs,kbf=basek.clone(),basek.clone()
                    qs,qf=qbs[...,192:],qbf[...,192:]
                    offset=512 if rs==576 else 2048
                    ks,kf=kbs[:,offset:offset+64].unsqueeze(1),kbf[:,offset:offset+64].unsqueeze(1)
                    c=caches[cache_dtype]
                    basepos=position_cases[torch.arange(m,device='cuda')%position_cases.numel()]
                    pos=basepos.clone()
                    slots=torch.arange(m,device='cuda',dtype=torch.int64)
                    slots[0]=-1 # unmapped row still rotates q/k, cache sentinel stays.
                    cache_s=torch.full((1,32,368),0xa5,device='cuda',dtype=torch.uint8);cache_f=cache_s.clone()
                    scale=torch.tensor(.75,device='cuda',dtype=torch.float32)
                    def reset(i=0):
                        qbs.copy_(baseq);qbf.copy_(baseq);kbs.copy_(basek);kbf.copy_(basek)
                        cache_s.fill_(0xa5);cache_f.fill_(0xa5);pos.copy_(basepos+i)
                    def stock():
                        ops.rotary_embedding(pos,qs,ks,64,c,neox)
                        pack_store(x,ks,cache_s,slots,scale)
                    def fused():rope_pack(qf,kf,x,cache_f,pos,c,slots,scale,neox)
                    def check():
                        bytes_equal(qbs,qbf,'full q incl untouched nope');bytes_equal(kbs,kbf,'full k allocation');bytes_equal(cache_s,cache_f,'all cache bytes incl padding and unmapped')
                    # Fully mapped, truncated, absent and out-of-range maps.
                    for trial in ('mapped','negative','outside','short','none'):
                        saved=slots
                        if trial=='mapped':slots=torch.arange(m,device='cuda',dtype=torch.int64)
                        if trial=='outside':slots=torch.full((m,),32,device='cuda',dtype=torch.int64)
                        if trial=='short':slots=slots[:max(0,m-1)]
                        if trial=='none':slots=None
                        reset();ops.rotary_embedding(pos,qs,ks,64,c,neox)
                        if slots is not None:pack_store(x,ks,cache_s,slots,scale)
                        fused();check();slots=saved
                    timing=bench_pair(stock,fused,reset,check,scrub,args.reps)
                    records.append(dict(piece='rope_pack',m=m,rs=rs,cache_dtype=str(cache_dtype),neox=neox,byte_equal=True,timing=timing))
                    (args.out/'partial.json').write_text(json.dumps(records,indent=2)+'\n')
    layer=torch.nn.Module();layer.logical_widths=[1536,576];layer.output_size_per_partition=2112
    layer.input_size_per_partition=6144;layer.params_dtype=torch.bfloat16
    # Actual stock repack/Marlin, two deliberately unequal FP32 globals.
    packed=[];scales=[];globals=[.0001,.0007]
    for n in layer.logical_widths:
        # All 16 E2M1 codes are valid; generate directly in the released ABI
        # to bound host memory without a dense float quantization temporary.
        packed.append(torch.randint(0,256,(n,3072),device='cuda',dtype=torch.uint8))
        scales.append(torch.full((n,384),64.,device='cuda',dtype=torch.float32).to(torch.float8_e4m3fn))
    layer.weight_packed=torch.nn.Parameter(torch.cat(packed),requires_grad=False)
    layer.weight_scale=torch.nn.Parameter(torch.cat(scales),requires_grad=False)
    layer.weight_global_scale=torch.nn.Parameter(torch.tensor(globals,device='cuda',dtype=torch.float32),requires_grad=False)
    nv.prepare_layer(layer)
    for m in range(1,17):
        x=torch.randn(m,6144,device='cuda',dtype=torch.bfloat16);results={}
        def stock():results['stock']=nv.apply_layer(layer,x).split([1536,576],-1)
        def fused():results['fused']=nv.apply_partitions(layer,x)
        def check():
            for i in range(2):bytes_equal(results['stock'][i],results['fused'][i],'Marlin partition '+str(i))
        # Graph captures allocate their own outputs; bind them immediately.
        # bench_pair's check dictionary points to each capture's fixed output.
        timing=bench_pair(stock,fused,lambda i=0:None,check,scrub,args.reps)
        records.append(dict(piece='parts',m=m,byte_equal=True,timing=timing))
    summary=dict(status='PASS',device=props.name,sm=props.multi_processor_count,
      torch=torch.__version__,cuda=torch.version.cuda,elapsed_s=time.monotonic()-started,records=records,
      real_input_rows=len(real),real_input_sha256=None if args.real_inputs is None else hashlib.sha256(args.real_inputs.read_bytes()).hexdigest(),
      byte_gate='uint8 exact, every output/cache byte, eager plus seven reset/changed-input graph replays',timing='CUDA events, 64MiB L2 scrub per sample, paired alternating order')
    (args.out/'result.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='records'}))
if __name__=='__main__':main()
