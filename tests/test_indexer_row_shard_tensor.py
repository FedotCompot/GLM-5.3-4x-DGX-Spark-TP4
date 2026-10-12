"""ARM64 VM CPU tensor tests; all distributed transport calls are simulated."""
import sys
import types
import unittest
from unittest.mock import patch
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_indexer_row_shard import R, P


def emitted(ids, k=7):
    return torch.stack((ids, ids+9, ids+2, ids+8, ids*0, ids*0-1, ids*0-1),dim=1).int()


class Comm:
    disabled = False
    def __init__(self, rows, slices): self.rows, self.slices, self.input = rows,slices,None
    def all_gather(self,out,inp):
        self.input = inp.clone()
        out.fill_(-1)
        for rank,(a,b) in enumerate(self.slices):
            out[rank*inp.shape[0]:rank*inp.shape[0]+b-a].copy_(self.rows[a:b])


def group(rows,slices,rank=0):
    comm = Comm(rows,slices)
    g = types.SimpleNamespace(world_size=len(slices), rank_in_group=rank,
                              device_group=object(),device_communicator=types.SimpleNamespace(pynccl_comm=comm))
    def gather(inp,dim):
        assert dim == 0
        out = torch.empty((inp.shape[0]*len(slices),inp.shape[1]),dtype=inp.dtype)
        comm.all_gather(out,inp)
        return out
    g.all_gather = gather
    return g


def logits_fn(query,key,weights,ks,ke,clean_logits):
    assert clean_logits is False
    return query[0][:,0,0].float().reshape(-1,1).expand(-1,8).contiguous()


def topk(logits,ks,ke,out,rows,s0,s1,k):
    assert rows == out.shape[0]
    out.copy_(emitted(logits[:rows,0].int()))


class Tensor(unittest.TestCase):
    def test_nccl_and_auto_gather_uneven_empty_order(self):
        for n,quantum in ((2051,1024),(2048,1024),(4096,1)):
            rows = emitted(torch.arange(n))
            slices = R.partition(n,quantum=quantum)
            for rank,(a,b) in enumerate(slices):
                for route in ('nccl','auto'):
                    g = group(rows,slices,rank)
                    with patch.object(R,'TRANSPORT',route): actual = R.gather(rows[a:b],slices,g)
                    self.assertTrue(torch.equal(actual,rows))
                    self.assertTrue(torch.equal(g.device_communicator.pynccl_comm.input[:b-a],rows[a:b]))

    def test_missing_nccl_refused(self):
        g = group(emitted(torch.arange(4)),R.partition(4))
        g.device_communicator.pynccl_comm.disabled = True
        with patch.object(R,'TRANSPORT','nccl'),self.assertRaises(RuntimeError):
            R.gather(emitted(torch.arange(1)),R.partition(4),g)

    def test_raw_order_and_remote_failure_vote(self):
        g = group(emitted(torch.arange(1)),R.partition(1))
        ref = emitted(torch.arange(1))
        with patch('torch.distributed.all_reduce'):
            R.require_equal(ref.clone(),ref,g)
            bad = ref.clone(); bad[:,1:3] = bad[:,1:3].flip(1)
            with self.assertRaises(RuntimeError): R.require_equal(bad,ref,g)
        def remote_bad(flag,**kw): flag.fill_(1)
        with patch('torch.distributed.all_reduce',side_effect=remote_bad),self.assertRaises(RuntimeError):
            R.require_equal(ref.clone(),ref,g)

    def test_generic_owner_rows_all_ranks(self):
        n = 2051
        q = torch.arange(n).float().reshape(n,1,1).expand(n,1,128).contiguous()
        expected = emitted(torch.arange(n))
        slices = R.partition(n)
        for rank,(a,b) in enumerate(slices):
            g = group(expected,slices,rank)
            dist_mod = types.ModuleType('vllm.distributed'); dist_mod.get_tp_group = lambda:g
            calls = []
            def scorer(*args,**kw): calls.append(args[0][0].shape[0]); return logits_fn(*args,**kw)
            out = torch.full((n,7),-1,dtype=torch.int32)
            with patch.dict(sys.modules,{'vllm.distributed':dist_mod}),patch.object(R,'MODE','check'),\
                 patch('torch.cuda.is_current_stream_capturing',return_value=False),patch('torch.distributed.all_reduce'):
                R.score_rows(q,None,None,None,torch.ones(n,1),torch.zeros(n,dtype=torch.int32),
                             torch.ones(n,dtype=torch.int32),out,scorer,topk,95000)
            self.assertEqual(calls,[b-a,n])
            self.assertTrue(torch.equal(out,expected))

    def test_fp4_tiles_offsets_padding_zero_ranks_and_mixed_decode(self):
        for spans,qt in (([2051],512),([2048],1024),([4096],512),
                         ([1040,1040,1040,976],512),([1412,1412,1272],512),([3834,262],1024)):
            n = sum(spans)
            begin = 3  # leave preceding decode rows untouched
            q = torch.arange(n+begin).float().reshape(-1,1,1).expand(-1,1,128).contiguous()
            weights = torch.ones(n+begin,1)
            expected = emitted(torch.arange(begin,begin+n))
            chunks=[]
            pos=begin
            for span in spans:
                chunks.append(types.SimpleNamespace(token_start=pos,token_end=pos+span,
                    local_total_seq_lens=95000,block_table=None,local_cu_seq_lens=None,
                    cu_seqlen_ks=None,cu_seqlen_ke=None))
                pos+=span
            bank = dict(q=torch.empty(qt,1,128), weights=torch.empty(qt,1),
                        starts=torch.empty(qt,dtype=torch.int32),ends=torch.empty(qt,dtype=torch.int32),
                        keys=torch.empty(8,128),scales=torch.empty(8))
            kernel = types.ModuleType('glm_fp4_prefill_kernel')
            kernel.gather_index_tile = lambda *args:None
            calls = []
            def prepare(q,w,ks,ke,qs,ws,starts,ends,absolute,offset,count):
                qs.zero_(); qs[:count].copy_(q[absolute:absolute+count]); ws.fill_(1)
                starts.zero_(); ends.fill_(8)
                calls.append((offset,count,qs.shape[0]))
            kernel.prepare_index_queries = prepare
            tasks,owners,slices = R.tile_plan(spans,[qt]*len(spans))
            for rank,(a,b) in enumerate(slices):
                g = group(expected,slices,rank)
                dist_mod = types.ModuleType('vllm.distributed'); dist_mod.get_tp_group = lambda:g
                out = torch.full((n+begin+2,7),77,dtype=torch.int32)
                calls.clear()
                with patch.dict(sys.modules,{'vllm.distributed':dist_mod,'glm_fp4_prefill_kernel':kernel}),\
                     patch.object(R,'MODE','check'),patch.object(P,'TOPK',7),patch.object(P,'LOGITS_BYTES',qt*8*4),\
                     patch.object(P,'geometry',return_value=(qt,8)),patch.object(P,'bank_for',return_value=bank),\
                     patch.object(P,'_row_shard_stock',P.tiled_prefill,create=True),patch('torch.distributed.all_reduce'):
                    R.tiled_prefill(torch.empty(1,64,132,dtype=torch.uint8),q,None,weights,chunks,out,
                                    logits_fn,topk,depth=95000)
                first,last=owners[rank]
                owned = [(o,count,quantum) for c,o,count,quantum in tasks[first:last]]
                full = [(o,count,quantum) for c,o,count,quantum in tasks]
                self.assertEqual(calls,owned+full)
                self.assertTrue(torch.equal(out[begin:begin+n],expected))
                self.assertTrue(torch.all(out[:begin]==77))
                self.assertTrue(torch.all(out[begin+n:]==77))


if __name__ == '__main__': unittest.main()
