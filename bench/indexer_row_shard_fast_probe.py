#!/usr/bin/env python3
"""Private, in-boot OFF/FAST/OFF real-prefix panel; stdlib client only."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import signal
import statistics
import time
import urllib.parse
import urllib.request

COUNTERS = ('chunks', 'rows', 'checked', 'mismatches', 'payload_bytes', 'fast_chunks', 'sampled')


def workers(rows, mode=None, every=None):
    if len(rows) != 4 or sorted(r['rank'] for r in rows) != list(range(4)):
        raise RuntimeError('require four unique TP ranks')
    rows = sorted(rows, key=lambda r: r['rank'])
    if any(not r['armed'] or not r['fast_supported'] or r['failed'] for r in rows):
        raise RuntimeError('require armed healthy FP4x FAST workers')
    if mode is not None and any(r['mode'] != mode for r in rows):
        raise RuntimeError('TP mode disagreement')
    if every is not None and any(r['sample_every'] != every for r in rows):
        raise RuntimeError('TP sampling disagreement')
    if len({(r['mode'], r['sample_every'], r['fast_sequence']) for r in rows}) != 1:
        raise RuntimeError('TP policy/call sequence disagreement')
    return rows


def deltas(before, after, mode, every):
    out = [{k:b[k]-a[k] for k in COUNTERS} | {
        'rank':b['rank'], 'phase_ms':{k:b['phase_ms'][k]-a['phase_ms'][k] for k in b['phase_ms']}}
        for a,b in zip(workers(before, mode, every), workers(after, mode, every))]
    if any(any(d[k] < 0 for k in COUNTERS) or d['mismatches'] for d in out):
        raise RuntimeError('counter reset or raw mismatch')
    if mode == 'fast':
        if any(d['fast_chunks'] <= 0 or d['chunks'] != d['fast_chunks'] for d in out):
            raise RuntimeError('no eligible FAST work')
        if every == 0:
            if any(d['checked'] or d['sampled'] or d['phase_ms']['reference'] or
                   d['phase_ms']['equality'] for d in out):
                raise RuntimeError('timing FAST executed reference/equality')
        elif any(d['sampled'] <= 0 or d['checked'] != d['sampled'] for d in out):
            raise RuntimeError('sampled FAST did not verify')
    elif any(d[k] for d in out for k in COUNTERS):
        raise RuntimeError('OFF executed row-shard work')
    if len({tuple(d[k] for k in COUNTERS) for d in out}) != 1:
        raise RuntimeError('TP work disagreement')
    return out


def parse_completion(lines, started, clock=time.perf_counter):
    text, tokens, usage, first, done = '', [], None, None, False
    for line in lines:
        if not line.startswith(b'data: '):
            continue
        raw = line[6:].strip()
        if raw == b'[DONE]':
            done = True
            break
        event = json.loads(raw)
        if 'error' in event:
            raise RuntimeError('completion stream error')
        for choice in event.get('choices', []):
            ids = (choice.get('logprobs') or {}).get('tokens', [])
            if ids and first is None:
                first = clock()-started
            tokens.extend(ids)
            text += choice.get('text', '')
        if event.get('usage') is not None:
            usage = event['usage']
    elapsed = clock()-started
    if not done or first is None or len(tokens) != 1 or usage is None or usage['completion_tokens'] != 1:
        raise RuntimeError('require complete SSE and exactly one generated token ID')
    if not re.fullmatch(r'token_id:\d+', tokens[0]):
        raise RuntimeError('server did not return token IDs')
    return dict(ttft_s=first, completion_s=elapsed, usage=usage,
                output_sha256=hashlib.sha256(text.encode()).hexdigest(),
                token_sha256=hashlib.sha256(tokens[0].encode()).hexdigest())


class Client:
    def __init__(self, endpoint, timeout):
        url = urllib.parse.urlsplit(endpoint)
        if (url.scheme != 'http' or url.hostname not in ('127.0.0.1','localhost','::1')
                or url.username or url.password or url.query or url.fragment or url.path not in ('','/')):
            raise ValueError('private loopback HTTP endpoint required')
        self.endpoint, self.timeout = endpoint.rstrip('/'), timeout

    def request(self, path, body=None, raw=False):
        req = urllib.request.Request(self.endpoint+path,
            data=None if body is None else json.dumps(body).encode(),
            headers={'Content-Type':'application/json'}, method='GET' if body is None else 'POST')
        with urllib.request.urlopen(req, timeout=self.timeout) as response:
            data = response.read()
        return data.decode() if raw else (json.loads(data) if data else {})

    def post(self, path, body=None):
        return self.request(path, {} if body is None else body)

    def rpc(self, method, args=()):
        return self.post('/collective_rpc',dict(method=method,args=list(args),timeout=60))['results']

    def pause(self):
        if self.post('/pause?mode=wait&clear_cache=false').get('status') != 'paused':
            raise RuntimeError('drain failed')
        if self.request('/is_paused').get('is_paused') is not True:
            raise RuntimeError('drained pause readback failed')

    def resume(self):
        if self.post('/resume').get('status') != 'resumed':
            raise RuntimeError('resume failed')

    def select(self, mode, every):
        # Caller pauses first; no graph recapture, cache clear or boot required.
        result = workers(self.rpc('indexer_row_shard_set',[mode, str(every)]), mode, every)
        readback = workers(self.rpc('indexer_row_shard_status'), mode, every)
        if result != readback:
            raise RuntimeError('policy readback drift')
        return readback

    def completion(self, model, prompt):
        body = dict(model=model,prompt=prompt,temperature=0,max_tokens=1,seed=39,
                    stream=True,stream_options={'include_usage':True},
                    logprobs=1,return_tokens_as_token_ids=True,ignore_eos=True)
        req = urllib.request.Request(self.endpoint+'/v1/completions',
            data=json.dumps(body).encode(),headers={'Content-Type':'application/json'},method='POST')
        started = time.perf_counter()
        with urllib.request.urlopen(req, timeout=self.timeout) as response:
            return parse_completion(response, started)

    def cache_hits(self):
        values = [float(line.rsplit(' ',1)[1]) for line in self.request('/metrics',raw=True).splitlines()
                  if re.match(r'^vllm:prefix_cache_hits_total(?:\{| )', line)]
        if not values:
            raise RuntimeError('prefix cache token-hit metric missing')
        return sum(values)

    def verify_cache(self, result, prefix_tokens, before):
        details = result['usage'].get('prompt_tokens_details')
        if details is not None:
            cached = details['cached_tokens']
        else:
            # Pinned metrics flush periodically; poll outside the measured call.
            deadline = time.monotonic()+30
            while True:
                cached = self.cache_hits()-before
                if cached >= prefix_tokens-512 or time.monotonic() >= deadline:
                    break
                time.sleep(1)
        if cached < prefix_tokens-512 or cached > prefix_tokens+512:
            raise RuntimeError('prefix APC hit proof failed or unrelated traffic')
        return cached


def validate_depth(nominal, prefix_tokens, full_tokens):
    extension = full_tokens-prefix_tokens
    # Boot f labels were nominal corpus sizes. Report measured tokenizer counts.
    if not nominal*.9 <= prefix_tokens <= nominal*1.1 or not 8192 <= extension <= 12500:
        raise RuntimeError('require nominal depth +/-10% and 8K..12.5K token extension')
    return extension


def run(client, panels, save, sample_every=32):
    trials = []
    def receipt(status, **kw):
        save(dict(status=status,trials=trials,**kw))
    try:
        if client.request('/is_paused').get('is_paused') is not False:
            raise RuntimeError('diagnostic engine must initially be resumed')
        model = client.request('/v1/models')['data'][0]['id']
        stable = client.rpc('indexer_stable_topk_status')
        if (len(stable) != 4 or sorted(r['rank'] for r in stable) != list(range(4)) or
                any(not r['armed'] or not r['installed'] or r['order'] != 'set_score_desc_index_asc_emit_index_asc' for r in stable)):
            raise RuntimeError('stable selector not installed on all four ranks')
        workers(client.rpc('indexer_row_shard_status'), 'off')
        for panel in panels:
            prefix, suffix = panel['prefix'], panel['suffix']
            # Mandatory sampled qualification first, then 3 OFF/FAST/OFF triplets.
            for stage, repeat, mode, every in [('sampled',0,'fast',sample_every)]+[
                    ('timing',i,mode,0) for i in range(1,4) for mode in ('off','fast','off')]:
                client.pause()
                client.select('off',every)
                if client.post('/reset_prefix_cache').get('success') is not True:
                    raise RuntimeError('APC reset failed')
                client.resume()
                warm = client.completion(model,prefix)
                client.pause()
                # Snapshot cumulative cache hits outside the timed request.
                cache_before = client.cache_hits()
                before = client.select(mode,every)
                client.resume()
                result = client.completion(model,prefix+suffix)
                client.pause()
                after = workers(client.rpc('indexer_row_shard_status'), mode, every)
                delta = deltas(before,after,mode,every)
                prefix_tokens = warm['usage']['prompt_tokens']
                extension = validate_depth(panel['depth'],prefix_tokens,result['usage']['prompt_tokens'])
                cached = client.verify_cache(result,prefix_tokens,cache_before)
                row = dict(depth=panel['depth'],stage=stage,repeat=repeat,mode=mode,sample_every=every,
                           prefix_tokens=prefix_tokens,extension_tokens=extension,cached_tokens=cached,
                           warm_prefix_s=warm['completion_s'],workers=delta,**result)
                trials.append(row)
                receipt('RUNNING')
                peers = [r for r in trials if r['depth'] == panel['depth']]
                if len({(r['output_sha256'],r['token_sha256']) for r in peers}) != 1:
                    raise RuntimeError('one-token T=0 output/token-ID hash mismatch')
        summary = []
        for panel in panels:
            rows = [r for r in trials if r['depth'] == panel['depth'] and r['stage'] == 'timing']
            off,fast = [[r['ttft_s'] for r in rows if r['mode']==mode] for mode in ('off','fast')]
            off_completion,fast_completion = [[r['completion_s'] for r in rows if r['mode']==mode] for mode in ('off','fast')]
            pairs = [dict(repeat=i,off_before_s=rows[(i-1)*3]['ttft_s'],
                          fast_s=rows[(i-1)*3+1]['ttft_s'],off_after_s=rows[(i-1)*3+2]['ttft_s'])
                     for i in range(1,4)]
            summary.append(dict(depth=panel['depth'],pairs=pairs,
                off_completion_median_s=statistics.median(off_completion),
                fast_completion_median_s=statistics.median(fast_completion),
                off_ttft_median_s=statistics.median(off),
                fast_ttft_median_s=statistics.median(fast),ttft_reduction_pct=100*(1-statistics.median(fast)/statistics.median(off)),
                off_repeats=len(off),fast_repeats=len(fast)))
        client.select('off',sample_every)
        client.resume()
        receipt('FAST_AB_PASS_NOT_PRODUCTION_QUALIFIED',summary=summary,final_mode='off')
        return trials
    except BaseException as error:
        # Any error leaves this coordinator-owned diagnostic engine paused.
        # If API died, coordinator performs stop/restore using its existing owner.
        cleanup = 'unknown'
        try:
            client.pause()
            client.select('off',sample_every)
            cleanup = 'paused_off'
        except Exception:
            cleanup = 'restore_required'
        receipt('KILL',reason=str(error),cleanup=cleanup)
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--endpoint',default='http://127.0.0.1:8095')
    p.add_argument('--manifest',type=Path,required=True,
                   help='JSON list: depth (35000/95000/129000), prefix_file, suffix_file')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--timeout',type=int,default=900)
    p.add_argument('--sample-every',type=int,default=32)
    a = p.parse_args()
    if a.sample_every <= 0: p.error('qualification sampling must be enabled')
    if a.output.exists(): p.error('receipt already exists')
    manifest = json.loads(a.manifest.read_text())
    if [x['depth'] for x in manifest] != [35000,95000,129000]:
        p.error('require ordered 35K/95K/129K matrix')
    panels = [dict(depth=x['depth'],prefix=(a.manifest.parent/x['prefix_file']).read_text(),
                   suffix=(a.manifest.parent/x['suffix_file']).read_text()) for x in manifest]
    def save(value):
        temporary = a.output.with_suffix(a.output.suffix+'.tmp')
        temporary.write_text(json.dumps(value,indent=2)+'\n')
        temporary.replace(a.output)
    def interrupted(signum, frame):
        raise RuntimeError('probe interrupted by signal '+str(signum))
    previous = {s:signal.signal(s,interrupted) for s in (signal.SIGTERM,signal.SIGINT)}
    try:
        run(Client(a.endpoint,a.timeout),panels,save,a.sample_every)
    finally:
        for signum, handler in previous.items(): signal.signal(signum,handler)


if __name__ == '__main__': main()
