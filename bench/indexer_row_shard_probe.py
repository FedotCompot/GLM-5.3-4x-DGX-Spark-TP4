#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Coordinator-only drained, loopback A/B index-equality probe on real prompts.

Receipts contain counts, timings, usage and output hashes; never prompt/output
text. CHECK still runs the full stock reference and is not a speed benchmark.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import urllib.parse
import urllib.request


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--endpoint', default='http://127.0.0.1:8095')
    p.add_argument('--prefix-file', type=Path, required=True)
    p.add_argument('--suffix-file', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--timeout', type=int, default=900)
    a = p.parse_args()
    url = urllib.parse.urlsplit(a.endpoint)
    if url.scheme != 'http' or url.hostname not in ('127.0.0.1','localhost','::1'):
        p.error('private loopback endpoint required')
    if a.output.exists(): p.error('receipt already exists')
    prefix, suffix = a.prefix_file.read_text(), a.suffix_file.read_text()
    receipts = []
    def request(path, body=None):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(a.endpoint + path, data=data,
            headers={'Content-Type':'application/json'}, method='POST' if body is not None else 'GET')
        with urllib.request.urlopen(req,timeout=a.timeout) as response:
            raw = response.read()
            return json.loads(raw) if raw else {}
    def post(path, body=None): return request(path, {} if body is None else body)
    def rpc(method,args=()):
        result = post('/collective_rpc',dict(method=method,args=list(args)))['results']
        if len(result) != 4 or not all(x['armed'] for x in result):
            raise RuntimeError('all four workers must be armed')
        return result
    def select(mode):
        result = rpc('indexer_row_shard_set',[mode])
        if not all(r['mode'] == mode for r in result): raise RuntimeError('TP mode disagreement')
    model = request('/v1/models')['data'][0]['id']
    def completion(prompt):
        start = time.perf_counter()
        r = post('/v1/completions',dict(model=model,prompt=prompt,temperature=0,max_tokens=1,seed=39))
        return time.perf_counter()-start,r
    try:
        for mode in ('off','self','check','off'):
            select('off')
            # The coordinator must drain the diagnostic server first. Clearing
            # APC is deliberate here; always rewarm the identical real prefix.
            post('/reset_prefix_cache')
            warm_s,warm = completion(prefix)
            select(mode)
            before = rpc('indexer_row_shard_status')
            elapsed,result = completion(prefix+suffix)
            after = rpc('indexer_row_shard_status')
            delta = [{k:b[k]-x[k] for k in ('chunks','rows','checked','mismatches','payload_bytes')}
                     | {'phase_ms':{k:b['phase_ms'][k]-x['phase_ms'][k] for k in b['phase_ms']}}
                     for x,b in zip(before,after)]
            if mode in ('self','check') and any(d['checked'] <= 0 or d['mismatches'] for d in delta):
                raise RuntimeError('invalid panel: no eligible rows or mismatch')
            row = dict(mode=mode,completion_s=elapsed,warm_prefix_s=warm_s,
                       prefix_usage=warm['usage'],usage=result['usage'],workers=delta,
                       output_sha256=hashlib.sha256(result['choices'][0]['text'].encode()).hexdigest(),
                       timing_scope='one-token wall proxy; CHECK includes replicated reference')
            receipts.append(row)
            a.output.write_text(json.dumps(dict(status='RUNNING',trials=receipts),indent=2)+'\n')
            print(json.dumps(row),flush=True)
        if len({r['output_sha256'] for r in receipts}) != 1:
            raise RuntimeError('one-token T=0 A/B output mismatch')
        a.output.write_text(json.dumps(dict(status='CHECKED_PROBE_PASS_NOT_SPEED_QUALIFIED',trials=receipts),indent=2)+'\n')
    except Exception as error:
        a.output.write_text(json.dumps(dict(status='KILL',reason=str(error),trials=receipts),indent=2)+'\n')
        raise
    finally:
        # If the API died, the coordinator owns cleanup/restoration. Preserve
        # the original exception; no remote lifecycle action from this client.
        try: select('off')
        except Exception: pass


if __name__ == '__main__': main()
