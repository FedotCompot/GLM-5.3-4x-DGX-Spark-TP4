#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Coordinator-only idle GB10 selector kit. Timing uses CUDA stream events."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'overlay/bringup'))
import glm_indexer_stable_topk as S


def event_ms(fn, repeats):
    import torch
    samples = []
    for _ in range(repeats):
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record(); fn(); b.record(); b.synchronize()
        samples.append(a.elapsed_time(b))
    return dict(samples_ms=samples, median_ms=statistics.median(samples))


def run_cell(rows, cols, pattern, repeats):
    import torch
    from vllm import _custom_ops as ops
    if ops.top_k_per_row_prefill is S.top_k_per_row_prefill:
        raise RuntimeError('microbench requires stock ops; unset GLM_INDEXER_STABLE_TOPK')
    torch.cuda.reset_peak_memory_stats()
    x = torch.empty((rows,cols),dtype=torch.float32,device='cuda')
    generator = torch.Generator(device='cuda').manual_seed(39)
    # Construct in small GPU batches; never copy full logits to CPU.
    for begin in range(0,rows,64):
        part = x[begin:begin+64]
        if pattern == 'quantized':
            part.copy_(torch.randint(-32,33,part.shape,device='cuda',generator=generator)).div_(8)
        elif pattern == 'equal': part.fill_(0)
        elif pattern == 'boundary':
            part.fill_(1); part[:,cols-500:].fill_(2)
        elif pattern == 'unique':
            part.copy_(torch.arange(cols,device='cuda',dtype=torch.float32).remainder(cols).flip(0))
        else: raise ValueError(pattern)
    row_id = torch.arange(rows,device='cuda',dtype=torch.int32)
    starts = (row_id % 19).contiguous()
    ends = (cols - row_id % 37).contiguous()
    out = torch.empty((rows,2048),dtype=torch.int32,device='cuda')
    native = torch.empty_like(out)
    def invoke(fn, result, begin=0, end=rows):
        fn(x[begin:end],starts[begin:end],ends[begin:end],result,
           end-begin,x.stride(0),x.stride(1),2048)
    stable = lambda: invoke(S.top_k_per_row_prefill,out)
    stock = lambda: invoke(ops.top_k_per_row_prefill,native)
    for _ in range(3): stable(); stock()
    torch.cuda.synchronize()
    baseline = out.clone()
    native_baseline = native.clone()
    stock_raw_changed, stock_set_changed = 0, 0
    # Full raw-array equality, every row, repeats and several uneven partitions.
    for _ in range(repeats):
        stable()
        if not torch.equal(out,baseline): raise RuntimeError('stable repeated raw mismatch')
        stock()
        stock_raw_changed += int(not torch.equal(native,native_baseline))
        stock_set_changed += int(not torch.equal(native.sort().values,native_baseline.sort().values))
    splits = ((0,rows),(0,rows//4,rows//2,3*rows//4,rows),
              (0,1,rows//3,rows-1,rows),(0,1040,min(2080,rows),min(3120,rows),rows))
    stock_split_raw_changed, stock_split_set_changed = 0,0
    for split in splits:
        for begin,end in zip(split,split[1:]):
            if end <= begin: continue
            invoke(S.top_k_per_row_prefill,out[begin:end],begin,end)
            invoke(ops.top_k_per_row_prefill,native[begin:end],begin,end)
        if not torch.equal(out,baseline): raise RuntimeError('stable split raw mismatch')
        stock_split_raw_changed += int(not torch.equal(native,native_baseline))
        stock_split_set_changed += int(not torch.equal(native.sort().values,native_baseline.sort().values))
    # Stable-sort oracle on deterministic samples; full-row validity/score-set
    # checks cover every row. Sampling limits reference sort scratch memory.
    sampled = sorted(set((0,1,rows//4,rows//2,rows-2,rows-1)))
    for r in sampled:
        start,end = int(starts[r]),int(ends[r])
        expected = torch.argsort(x[r,start:end],descending=True,stable=True)[:2048].int()
        if not torch.equal(baseline[r],expected): raise RuntimeError('stable-sort oracle mismatch')
    if (baseline < 0).any() or (baseline >= (ends-starts)[:,None]).any():
        raise RuntimeError('stable index bounds failed')
    sorted_indices = baseline.sort().values
    if (sorted_indices[:,1:] == sorted_indices[:,:-1]).any(): raise RuntimeError('duplicate stable key')
    stock(); stable()
    stock_scores = x.gather(1,native.long()+starts[:,None]).sort().values
    stable_scores = x.gather(1,baseline.long()+starts[:,None]).sort().values
    if not torch.equal(stock_scores,stable_scores): raise RuntimeError('stock/stable selected score multiset differs')
    tie_set_changed = not torch.equal(native.sort().values,sorted_indices)
    if pattern == 'unique' and tie_set_changed: raise RuntimeError('untied selected set differs')
    # Prewarmed graph replay tests stream capture and fully overwritten outputs.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph): stable()
    for _ in range(3):
        out.fill_(-77); graph.replay()
        if not torch.equal(out,baseline): raise RuntimeError('stable graph replay mismatch')
    del graph, sorted_indices, stock_scores, stable_scores
    # Alternate arms to limit thermal/drift bias. Each sample is one invocation.
    a1 = event_ms(stock,repeats)
    b = event_ms(stable,repeats)
    a2 = event_ms(stock,repeats)
    split_timing = []
    for begin,end in zip(splits[1],splits[1][1:]):
        split_timing.append(event_ms(lambda: invoke(S.top_k_per_row_prefill,out[begin:end],begin,end),repeats))
    digest = hashlib.sha256(baseline[sampled].cpu().numpy().tobytes()).hexdigest()
    return dict(rows=rows,context=cols,pattern=pattern,correct=True,
                stable_sort_sampled_rows=sampled,raw_repeat_checks=repeats,split_checks=len(splits),graph_replays=3,
                stock_raw_repeat_changes=stock_raw_changed,stock_set_repeat_changes=stock_set_changed,
                stock_raw_split_changes=stock_split_raw_changed,stock_set_split_changes=stock_split_set_changed,
                stock_vs_stable_tie_set_changed=tie_set_changed,stock_A=a1,stable_B=b,stock_A2=a2,
                stock_ms=statistics.median(a1['samples_ms']+a2['samples_ms']),
                ratio_stable_stock=b['median_ms']/statistics.median(a1['samples_ms']+a2['samples_ms']),
                stable_quarter_row_ms=split_timing,sampled_output_sha256=digest,
                peak_allocated_bytes=torch.cuda.max_memory_allocated())


def edge_cases():
    import torch
    # Noncontiguous logits/output, nonzero starts, short/empty intervals,
    # signed zeros, infinities, subnormals, NaNs, dirty masked-out tails.
    rows,cols = 8,5003
    backing = torch.full((rows,cols*2),float('nan'),device='cuda')
    x = backing[:,::2]
    starts = torch.tensor([0,3,17,1,0,19,0,4],dtype=torch.int32,device='cuda')
    ends = torch.tensor([0,4,34,2049,3000,4500,5003,4999],dtype=torch.int32,device='cuda')
    values = [-0.,0.,-float('inf'),float('inf'),float('nan'),1e-40,-1e-40,3.]
    raw = []
    for r in range(rows):
        row = [values[i%len(values)] for i in range(cols)]
        raw.append(row); x[r].copy_(torch.tensor(row,device='cuda'))
    storage = torch.full((rows,2176),-77,dtype=torch.int32,device='cuda')
    out = storage[:,:2048]
    S.top_k_per_row_prefill(x,starts,ends,out,rows,*x.stride(),2048)
    expected = torch.tensor([S.reference(raw[r],int(starts[r]),int(ends[r])) for r in range(rows)],dtype=torch.int32)
    if not torch.equal(out.cpu(),expected) or (storage[:,2048:] != -77).any():
        raise RuntimeError('edge/stride/sentinel oracle mismatch')
    return dict(rows=rows,correct=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rows',type=int,nargs='+',default=[2048,4096])
    parser.add_argument('--contexts',type=int,nargs='+',default=[8192,35000,129000])
    parser.add_argument('--patterns',nargs='+',choices=['quantized','equal','boundary','unique'],default=['quantized','equal','boundary','unique'])
    parser.add_argument('--repeats',type=int,default=10)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.repeats < 3 or any(n < 2048 or n > 4096 for n in args.rows) or any(n < 8192 or n > 129000 for n in args.contexts):
        parser.error('3+ repeats; rows 2048..4096 and contexts 8192..129000')
    import torch
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12,1):
        raise RuntimeError('idle GB10 required')
    begin=time.monotonic()
    result=dict(schema=1,torch=torch.__version__,device=torch.cuda.get_device_name(),timer='cuda_events',
                kernel_sha256=hashlib.sha256((ROOT/'overlay/bringup/stable_topk/kernel.cu').read_bytes()).hexdigest(),
                cells=[],gpu_results='RUNNING')
    def save():
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,indent=2)+'\n')
    try:
        S.load(); result['edge_cases']=edge_cases(); save()
        for rows in args.rows:
            for cols in args.contexts:
                for pattern in args.patterns:
                    cell=run_cell(rows,cols,pattern,args.repeats)
                    result['cells'].append(cell);save()
                    print(json.dumps({k:cell[k] for k in ('rows','context','pattern','stock_ms','ratio_stable_stock')}),flush=True)
        result['gpu_results']='PASS'
    except BaseException as exc:
        result['gpu_results']='FAIL';result['error']=str(exc);raise
    finally:
        result['wall_seconds']=time.monotonic()-begin;save()


if __name__ == '__main__':main()
