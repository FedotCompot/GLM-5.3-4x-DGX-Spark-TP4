#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Exclusive c1 in-boot A/B/A' using cycle.py prompts and GateMetrics.

Same T=0/no-thinking/512 streaming probe; adds actual returned token IDs.
No lifecycle actions. Always attempt drained OFF restore on completion/error.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import time
import urllib.request
from cycle import PROSE, CODE
from gate_metrics import GateMetrics


def post(base, path, payload):
    req = urllib.request.Request(base + path, data=json.dumps(payload).encode(),
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=600) as response:
        return json.load(response)


def rpc(base, value=None):
    method = 'glm_l2hint_status' if value is None else 'glm_l2hint_switch'
    reply = post(base, '/collective_rpc', dict(method=method, args=[] if value is None else [str(value)]))
    rows = reply.get('results', [])
    if len(rows) != 4 or any(not isinstance(r, dict) or not r.get('enabled') or 'busy' in r for r in rows):
        raise RuntimeError('RPC not drained/armed on four ranks')
    expected = rows[0]['value'] if value is None else value
    if any(r['value'] != expected or r['config_sha256'] != rows[0]['config_sha256']
           or not r['graphs'] or any(len(g) != 2 or min(g) <= 0 or g[0] != g[1] for g in r['graphs']) for r in rows):
        raise RuntimeError('RPC bank/config/descriptor disagreement')
    # Every target descriptor warmup/capture must dispatch both target GEMMs.
    for row in rows:
        for banks in row['captures']:
            if len(banks) != 2 or not banks[1]:
                raise RuntimeError('missing ON capture receipt')
            for capture in banks[1]:
                calls = capture['calls']
                if not all(any(shape in k and v >= 75 for k,v in calls.items())
                           for shape in ('6144x1024', '512x6144')):
                    raise RuntimeError('incomplete target gate_up/down ON capture coverage')
    return rows


def stream(base, prompt, probe):
    # cycle.stream's timing boundaries and payload, plus seed and token IDs.
    body = dict(model='GLM-5.3', messages=[{'role':'user','content':prompt}], max_tokens=512,
                temperature=0, seed=0, stream=True, return_token_ids=True,
                stream_options={'include_usage':True}, chat_template_kwargs={'enable_thinking':False})
    req = urllib.request.Request(base+'/v1/chat/completions',data=json.dumps(body).encode(),
                                 headers={'Content-Type':'application/json'})
    first = last = None; usage = None; text = []; ids = []; finish = None
    with urllib.request.urlopen(req,timeout=600) as response:
        for raw in response:
            if not raw.startswith(b'data: ') or raw.strip() == b'data: [DONE]':continue
            event=json.loads(raw[6:])
            if 'error' in event:raise RuntimeError('stream error')
            for choice in event.get('choices') or []:
                delta=choice.get('delta') or {}
                piece=delta.get('content') or delta.get('reasoning_content') or delta.get('reasoning')
                if piece:
                    now=time.monotonic();first=first or now;last=now;text.append(piece)
                tokens=choice.get('token_ids') or []
                if not isinstance(tokens,list) or any(type(t) is not int for t in tokens):
                    raise RuntimeError('invalid returned token IDs')
                ids.extend(tokens)
                finish=choice.get('finish_reason') or finish
            usage=event.get('usage') or usage
    n=(usage or {}).get('completion_tokens')
    if type(n) is not int or not ids or len(ids)!=n or finish is None or not first or not last or last<=first:
        raise RuntimeError('missing complete token sequence, usage or decode wall')
    if probe is not None:probe.set_decode_wall(last-first,delivered_tokens=n)
    return dict(token_ids=ids,completion_tokens=n,text=''.join(text),finish_reason=finish)


def assess(rows):
    result={}; equal=True
    for kind in ('prose','code'):
        keyed={}
        for row in rows:
            if row['kind']==kind:keyed.setdefault(row['i'],{})[row['phase']]=row
        if set(keyed)!=set(range(10)) or any(set(p)!=set(('A','B','Aprime')) for p in keyed.values()):
            raise ValueError('incomplete paired panel')
        savings=[]; drift=[]
        for triple in keyed.values():
            a,b,ap=[triple[v] for v in ('A','B','Aprime')]
            for key in ('token_ids','text','finish_reason','completion_tokens'):
                equal = equal and a[key]==b[key]==ap[key]
            values=[r['cycle_ms'] for r in (a,b,ap)]
            if any(not math.isfinite(v) or v<=0 for v in values):raise ValueError('invalid cycle timing')
            baseline=(values[0]+values[2])/2
            savings.append(baseline-values[1]);drift.append(abs(values[2]-values[0])/baseline)
        mean=statistics.mean(savings);se=statistics.stdev(savings)/math.sqrt(len(savings))
        result[kind]=dict(n=len(savings),savings_ms=mean,lower_3SE_ms=mean-3*se,
                          paired_savings_ms=savings,median_control_drift=statistics.median(drift))
    keep=equal and all(r['lower_3SE_ms']>0 and r['savings_ms']>=.3 and r['median_control_drift']<=.01 for r in result.values())
    return dict(verdict='KEEP' if keep else 'KILL',token_equality=equal,panel=result,
                scope='one bounded in-boot c1 panel; not promotion or multi-day qualification')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--endpoint',default='http://127.0.0.1:18095')
    parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=False);rows=[];error=None;restored=False
    try:
        initial=rpc(args.endpoint);(args.out/'initial.json').write_text(json.dumps(initial,indent=2)+'\n')
        for phase,value in (('A',0),('B',1),('Aprime',0)):
            before=rpc(args.endpoint,value)
            (args.out/(phase+'-before.json')).write_text(json.dumps(before,indent=2)+'\n')
            stream(args.endpoint,'Say hello in one short sentence.',None)
            gate=GateMetrics(args.endpoint+'/metrics',args.out/phase,labels={'model_name':'GLM-5.3'},metadata={'boot':'l2hint-inboot','phase':phase})
            for kind,prompts in (('prose',PROSE),('code',CODE)):
                for i,prompt in enumerate(prompts):
                    if post(args.endpoint,'/reset_prefix_cache',{}).get('success') is not True:
                        raise RuntimeError('prefix cache reset did not drain')
                    meta=dict(pair_key=f'{kind}-{i}',prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),seed=0,
                              k_mode='boot-default',target_M='boot-default',kind=kind)
                    with gate.request(f'{phase}-{kind}-{i}',exclusive=True,metadata=meta) as probe:
                        output=stream(args.endpoint,prompt,probe)
                    row=dict(phase=phase,kind=kind,i=i,cycle_ms=probe.record['cycle_ms'],
                             committed=probe.record['committed_per_cycle'],**output)
                    rows.append(row)
                    with (args.out/'rows.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                    if phase != 'A':
                        control = next(r for r in rows if r['phase']=='A' and r['kind']==kind and r['i']==i)
                        if any(row[k] != control[k] for k in ('token_ids','text','finish_reason','completion_tokens')):
                            raise RuntimeError(f'token equality failed {phase}-{kind}-{i}')
            after=rpc(args.endpoint)
            (args.out/(phase+'-after.json')).write_text(json.dumps(after,indent=2)+'\n')
            if any(a['replays'][value]<=b['replays'][value] for a,b in zip(after,before)):
                raise RuntimeError('phase did not replay selected target graph bank on every rank')
            print(phase,'complete',flush=True)
        result=assess(rows)
    except Exception as exc:
        error=str(exc);result=dict(verdict='KILL',error=error)
    finally:
        try:
            restored=all(r['value']==0 for r in rpc(args.endpoint,0))
        except Exception as exc:
            error=(error or '')+'; OFF restore failed: '+str(exc)
    result.update(restored_off=restored)
    if error or not restored:result.update(verdict='KILL',error=error)
    (args.out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
    return 0 if result['verdict']=='KEEP' else 1


if __name__=='__main__':raise SystemExit(main())
