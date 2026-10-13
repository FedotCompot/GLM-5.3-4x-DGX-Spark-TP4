"""CPU-only fake transport tests for the real-prefix in-boot panel."""
import copy
import importlib.util
import json
from pathlib import Path
import unittest
from unittest.mock import patch

path = Path(__file__).resolve().parents[1]/'bench/indexer_row_shard_fast_probe.py'
spec = importlib.util.spec_from_file_location('fast_probe',path)
B = importlib.util.module_from_spec(spec); spec.loader.exec_module(B)


def rows(mode='off',every=32):
    return [dict(rank=i,armed=True,fast_supported=True,failed=False,mode=mode,
        sample_every=every,fast_sequence=0,phase_ms=dict(owner=0,gather=0,reference=0,equality=0),
        **dict.fromkeys(B.COUNTERS,0)) for i in range(4)]


class Fake:
    def __init__(self):
        self.paused=False; self.state=rows(); self.resumes=0
        self.generated=0; self.resets=0; self.flip=False; self.cache_ok=True
    def request(self,path):
        if path == '/is_paused': return {'is_paused':self.paused}
        if path == '/v1/models': return {'data':[{'id':'fake'}]}
        raise AssertionError(path)
    def rpc(self,method,args=()):
        if method == 'indexer_stable_topk_status':
            return [dict(rank=i,armed=True,installed=True,order='set_score_desc_index_asc_emit_index_asc') for i in range(4)]
        if method == 'indexer_row_shard_status': return copy.deepcopy(self.state)
        raise AssertionError(method)
    def pause(self): self.paused=True
    def resume(self):
        assert self.paused
        self.paused=False; self.resumes+=1
    def select(self,mode,every):
        assert self.paused
        for r in self.state:r.update(mode=mode,sample_every=every,fast_sequence=0)
        return self.rpc('indexer_row_shard_status')
    def post(self,path):
        assert self.paused and path == '/reset_prefix_cache'
        self.resets+=1
        return dict(success=self.cache_ok)
    def cache_hits(self): return 0
    def verify_cache(self,result,prefix,before):
        assert self.paused
        return prefix-64
    def completion(self,model,prompt):
        assert not self.paused and model == 'fake'
        depth=int(prompt.split('+')[0]); full='+' in prompt
        if full:
            self.generated+=1
            if self.state[0]['mode'] == 'fast':
                for r in self.state:
                    n=66; every=r['sample_every']; sampled=(n-1)//every+1 if every else 0
                    for key,val in dict(chunks=n,rows=12000,fast_chunks=n,checked=sampled,sampled=sampled,
                                        payload_bytes=n*8192).items(): r[key]+=val
                    r['fast_sequence']+=n
                    for key,val in dict(owner=100,gather=10,reference=sampled*10,equality=sampled).items():
                        r['phase_ms'][key]+=val
        return dict(ttft_s=1.0,completion_s=1.01,usage=dict(prompt_tokens=depth+(12000 if full else 0),
                    completion_tokens=1),output_sha256='bad' if self.flip and full and self.generated>1 else 'same',
                    token_sha256='same-token')


class Probe(unittest.TestCase):
    def test_full_matrix_restores_off_sampling_resumed(self):
        client=Fake(); receipts=[]
        panels=[dict(depth=n,prefix=str(n),suffix='+extension') for n in (35000,95000,129000)]
        trials=B.run(client,panels,receipts.append)
        self.assertEqual(len(trials),30)
        self.assertEqual(client.resets,30)
        self.assertFalse(client.paused)
        self.assertEqual(client.state[0]['mode'],'off')
        self.assertEqual(client.state[0]['sample_every'],32)
        self.assertEqual(receipts[-1]['status'],'FAST_AB_PASS_NOT_PRODUCTION_QUALIFIED')
        for summary in receipts[-1]['summary']:
            self.assertEqual((summary['off_repeats'],summary['fast_repeats']),(6,3))
        self.assertEqual(sum(t['stage']=='sampled' for t in trials),3)

    def test_mismatch_stops_panel_and_leaves_paused_off(self):
        client=Fake(); client.flip=True; receipts=[]
        with self.assertRaisesRegex(RuntimeError,'hash mismatch'):
            B.run(client,[dict(depth=35000,prefix='35000',suffix='+extension')],receipts.append)
        self.assertEqual(client.generated,2)
        self.assertTrue(client.paused)
        self.assertEqual(receipts[-1]['cleanup'],'paused_off')
        self.assertEqual(receipts[-1]['status'],'KILL')

    def test_interrupt_leaves_paused_off(self):
        client=Fake();receipts=[]
        with patch.object(client,'completion',side_effect=KeyboardInterrupt),self.assertRaises(KeyboardInterrupt):
            B.run(client,[dict(depth=35000,prefix='35000',suffix='+extension')],receipts.append)
        self.assertTrue(client.paused)
        self.assertEqual(receipts[-1]['cleanup'],'paused_off')

    def test_cache_reset_refuses_before_any_generation(self):
        client=Fake();client.cache_ok=False;receipts=[]
        with self.assertRaisesRegex(RuntimeError,'APC reset'):
            B.run(client,[dict(depth=35000,prefix='35000',suffix='+extension')],receipts.append)
        self.assertEqual(client.generated,0)
        self.assertTrue(client.paused)

    def test_worker_disagreement_refused(self):
        for mutation in (lambda r:r.update(rank=1),lambda r:r.update(failed=True),
                         lambda r:r.update(fast_sequence=1),lambda r:r.update(mode='fast'),
                         lambda r:r.update(fast_supported=False)):
            states=rows();mutation(states[0])
            with self.assertRaises(RuntimeError):B.workers(states,'off',32)

    def test_timing_reference_or_no_work_refused(self):
        before=rows('fast',0);after=copy.deepcopy(before)
        with self.assertRaises(RuntimeError):B.deltas(before,after,'fast',0)
        for r in after:r.update(chunks=1,fast_chunks=1,checked=1)
        with self.assertRaisesRegex(RuntimeError,'reference/equality'):B.deltas(before,after,'fast',0)

    def test_prefix_and_extension_count_validation(self):
        self.assertEqual(B.validate_depth(35000,37147,49466),12319)
        for args in ((35000,10,12010),(35000,35000,36000),(35000,35000,50000)):
            with self.assertRaises(RuntimeError):B.validate_depth(*args)

    def test_loopback_only(self):
        for url in ('https://localhost:8095','http://192.0.2.1:8095','http://user:secret@localhost','http://localhost/x'):
            with self.assertRaises(ValueError):B.Client(url,60)
        B.Client('http://127.0.0.1:8095',60)

    def test_sse_ttft_and_exact_token_id_hash(self):
        lines=[b': keepalive\n',b'data: '+json.dumps(dict(choices=[dict(text='',logprobs=None)])).encode(),
               b'data: '+json.dumps(dict(choices=[dict(text='A',logprobs=dict(tokens=['token_id:42']))])).encode(),
               b'data: '+json.dumps(dict(choices=[],usage=dict(prompt_tokens=35000,completion_tokens=1))).encode(),
               b'data: [DONE]\n']
        with patch.object(B.time,'perf_counter',side_effect=[12,13]):
            result=B.parse_completion(lines,10,B.time.perf_counter)
        self.assertEqual((result['ttft_s'],result['completion_s']),(2,3))
        for bad in (lines[:-1],[b'data: [DONE]'],[b'data: {"error":"failed"}']):
            with self.assertRaises(RuntimeError):B.parse_completion(bad,0,lambda:1)

    def test_real_client_drains_preserving_cache_and_checks_readback(self):
        client=B.Client('http://localhost:8095',60)
        with patch.object(client,'post',return_value={'status':'paused'}) as post,\
             patch.object(client,'request',return_value={'is_paused':True}):
            client.pause()
            post.assert_called_once_with('/pause?mode=wait&clear_cache=false')
        with patch.object(client,'rpc',return_value=rows('fast',0)) as rpc:
            client.select('fast',0)
            self.assertEqual(rpc.call_args_list[0].args,('indexer_row_shard_set',['fast','0']))
        drift=rows('fast',0);drift[0]['rows']=1
        with patch.object(client,'rpc',side_effect=[rows('fast',0),drift]),self.assertRaisesRegex(RuntimeError,'drift'):
            client.select('fast',0)

    def test_failed_metric_cache_proof(self):
        client=B.Client('http://localhost:8095',60)
        with patch.object(client,'cache_hits',return_value=0),patch.object(B.time,'monotonic',side_effect=[0,31]):
            with self.assertRaisesRegex(RuntimeError,'APC hit'):client.verify_cache({'usage':{}},35000,0)
        with self.assertRaisesRegex(RuntimeError,'unrelated traffic'):
            client.verify_cache({'usage':{'prompt_tokens_details':{'cached_tokens':70000}}},35000,0)


if __name__ == '__main__': unittest.main()
