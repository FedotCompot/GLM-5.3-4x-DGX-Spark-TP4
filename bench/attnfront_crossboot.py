#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Cross-boot c1 panel using existing cycle prompts, token stream and GateMetrics.

No RPC, graph recapture, device instrumentation or lifecycle actions. The mode
label is supplied by the coordinator; confirm it with DRY and boot logs.
Request-wall cycle_ms is not a CUDA-event measurement.
"""
import argparse
import hashlib
import json
from pathlib import Path
from cycle import PROSE, CODE
PROSE = PROSE + [
    'Describe how ocean tides arise from the Moon and Sun in clear flowing prose.',
    'Explain how tree rings form and what researchers can learn from them.',
    'Describe how an orchestra coordinates its players during a performance.',
    'Explain how a bridge carries the weight of traffic across a river.',
    'Write a short essay about how reading habits change throughout a lifetime.',
]
from gate_metrics import GateMetrics, paired
from l2hint_aba import post, stream
import math
import statistics
import struct
from release_probe import Probes

TOKEN_KEYS = ('token_ids', 'text', 'finish_reason', 'completion_tokens')


def teacher_bits(rows):
    if not isinstance(rows,list) or not rows:raise ValueError('missing teacher panel')
    result=[]
    for row in rows:
        lp=row['prompt_lp']
        if len(lp)!=len(row['tokens']) or len(lp)<2 or lp[0] is not None:raise ValueError('invalid teacher panel')
        bits=[]
        for pos in lp[1:]:
            if not isinstance(pos,dict) or not pos:raise ValueError('teacher support absent')
            values=[]
            for token,value in sorted(pos.items()):
                if not isinstance(value,(int,float)) or not math.isfinite(value):raise ValueError('nonfinite logprob')
                values.append((token,struct.pack('!d',value).hex()))
            bits.append(values)
        result.append((row['id'],row['tokens'],bits))
    return result


def assess(rows):
    result={};equal=True
    for kind in ('prose','code'):
        keyed={}
        for row in rows:
            if row['kind']==kind:keyed.setdefault(row['i'],{})[row['phase']]=row
        if set(keyed)!=set(range(15 if kind=='prose' else 10)) or any(set(p)!=set(('A','B','Aprime')) for p in keyed.values()):
            raise ValueError('incomplete paired panel')
        savings=[];drift=[]
        for triple in keyed.values():
            a,b,ap=[triple[v] for v in ('A','B','Aprime')]
            equal=equal and all(a[k]==b[k]==ap[k] for k in TOKEN_KEYS)
            vals=[r['cycle_ms'] for r in (a,b,ap)]
            if any(not math.isfinite(v) or v<=0 for v in vals):raise ValueError('invalid timing')
            baseline=(vals[0]+vals[2])/2
            savings.append(baseline-vals[1]);drift.append(abs(vals[2]-vals[0])/baseline)
        mean=statistics.mean(savings);se=statistics.stdev(savings)/math.sqrt(len(savings))
        result[kind]=dict(n=len(savings),savings_ms=mean,lower_3SE_ms=mean-3*se,
                          paired_savings_ms=savings,median_control_drift=statistics.median(drift))
    keep=equal and all(r['lower_3SE_ms']>0 and r['savings_ms']>=.3 and r['median_control_drift']<=.01 for r in result.values())
    return dict(verdict='KEEP' if keep else 'KILL',token_equality=equal,panel=result)


def read_panel(path):
    result = json.loads(Path(path).read_text())
    rows = result['rows']
    if result['status'] != 'COMPLETE' or result['mode'] not in ('off', 'on') or len(rows) != 25:
        raise ValueError('incomplete/invalid cross-boot panel')
    keys = {(r['kind'], r['i']) for r in rows}
    if keys != {(k,i) for k in ('prose','code') for i in range(15 if k=='prose' else 10)}:
        raise ValueError('missing/duplicate cross-boot pairs')
    for r in rows:
        ids, n = r['token_ids'], r['completion_tokens']
        if (type(n) is not int or n <= 0 or len(ids) != n
                or any(type(t) is not int for t in ids) or not r['finish_reason']):
            raise ValueError('missing actual completion token IDs')
    if len(result.get('teacher',[]))!=2:raise ValueError('missing two teacher repetitions')
    if teacher_bits(result['teacher'][0])!=teacher_bits(result['teacher'][1]):raise ValueError('teacher A/A bit mismatch')
    return result


def matching(a, b):
    # Existing attribution/provenance checks, including prompt/seed/M/K/labels.
    pair = paired(a['metrics'], b['metrics'])
    if any(a[k] != b[k] for k in TOKEN_KEYS):
        raise ValueError('cross-boot actual token/text/finish/count mismatch')
    if a['cycle_ms'] != a['metrics']['cycle_ms'] or b['cycle_ms'] != b['metrics']['cycle_ms']:
        raise ValueError('cross-boot timing receipt drift')
    return pair


def compare(paths):
    panels = [read_panel(p) for p in paths]
    if [p['mode'] for p in panels] != ['off','on','off']:
        raise ValueError('comparison requires OFF baseline, ON candidate, OFF control')
    if len({p['boot'] for p in panels}) != 3:
        raise ValueError('comparison requires three distinct boot labels')
    reference = {(r['kind'],r['i']):r for r in panels[0]['rows']}
    pairs, rows = [], []
    for phase,panel in zip(('A','B','Aprime'),panels):
        for r in panel['rows']:
            pairs.append(matching(reference[r['kind'],r['i']],r))
            rows.append(r|{'phase':phase})
    expected=teacher_bits(panels[0]['teacher'][0])
    if any(teacher_bits(t)!=expected for panel in panels for t in panel['teacher']):
        raise ValueError('teacher token/support/logprob bit mismatch')
    result = assess(rows)
    result['teacher_bit_identity']=True
    result.update(scope='bounded cross-boot c1 A/B/A-prime; not multi-day qualification',
                  boots=[p['boot'] for p in panels], pairs=pairs,
                  timing='request decode wall / draft opportunities; not CUDA-event cycle median')
    return result


def collect(args):
    args.out.mkdir(parents=True,exist_ok=False)
    rows = []; reference = None
    if args.reference:
        panel = read_panel(args.reference)
        if panel['mode'] != 'off': raise ValueError('reference must be baseline OFF')
        reference = {(r['kind'],r['i']):r for r in panel['rows']}
    result = dict(status='RUNNING', boot=args.boot, mode=args.mode, rows=rows,
                  timing='request decode wall / draft opportunities; not CUDA-event cycle median')
    try:
        teacher_panel=json.loads((Path(__file__).parent/'release/panel.json').read_text())
        probes=Probes(args.endpoint,args.out/'teacher',lambda:None)
        result['teacher']=[probes.teacher(teacher_panel,args.boot+'-t1a',1),
                           probes.teacher(teacher_panel,args.boot+'-t1b',1)]
        if teacher_bits(result['teacher'][0])!=teacher_bits(result['teacher'][1]):raise ValueError('teacher same-boot bit mismatch')
        if reference is not None and teacher_bits(result['teacher'][0])!=teacher_bits(panel['teacher'][0]):raise ValueError('teacher cross-boot bit mismatch')
        stream(args.endpoint,'Say hello in one short sentence.',None)
        gate = GateMetrics(args.endpoint+'/metrics', args.out/'metrics', labels={'model_name':'GLM-5.3'},
                           metadata={'boot':args.boot,'attnfront_mode':args.mode})
        for kind,prompts in (('prose',PROSE),('code',CODE)):
            for i,prompt in enumerate(prompts):
                if post(args.endpoint,'/reset_prefix_cache',{}).get('success') is not True:
                    raise RuntimeError('prefix cache reset did not drain')
                meta = dict(pair_key=f'{kind}-{i}',prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                            seed=0,k_mode='boot-default',target_M='boot-default',kind=kind)
                with gate.request(f'{kind}-{i}',exclusive=True,metadata=meta) as probe:
                    output = stream(args.endpoint,prompt,probe)
                row = dict(kind=kind,i=i,cycle_ms=probe.record['cycle_ms'],
                           committed=probe.record['committed_per_cycle'],metrics=probe.record,**output)
                rows.append(row)
                with (args.out/'rows.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                if reference is not None: matching(reference[kind,i],row)
                print(kind,i,'complete',flush=True)
        result['status'] = 'COMPLETE'
    except (Exception, KeyboardInterrupt) as exc:
        result.update(status='KILL',error=f'{type(exc).__name__}: {exc}')
    finally:
        (args.out/'panel.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return 0 if result['status']=='COMPLETE' else 1


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    cmds=parser.add_subparsers(dest='command',required=True)
    c=cmds.add_parser('collect');c.add_argument('--endpoint',default='http://127.0.0.1:8095')
    c.add_argument('--out',type=Path,required=True);c.add_argument('--boot',required=True)
    c.add_argument('--mode',choices=('off','on'),required=True);c.add_argument('--reference',type=Path)
    c=cmds.add_parser('compare');c.add_argument('panels',nargs=3,type=Path)
    c.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.command=='collect': return collect(args)
    try: result=compare(args.panels)
    except (ValueError,KeyError,TypeError) as exc: result=dict(verdict='KILL',error=str(exc))
    with args.out.open('x') as f: f.write(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result),flush=True)
    return 0 if result['verdict']=='KEEP' else 1


if __name__=='__main__':raise SystemExit(main())
