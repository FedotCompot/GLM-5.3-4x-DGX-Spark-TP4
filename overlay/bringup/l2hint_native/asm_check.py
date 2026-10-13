# SPDX-License-Identifier: Apache-2.0
"""Compare sm_121a instruction words, ignoring names/addresses/ELF metadata."""
import hashlib,json,os,re,subprocess
from pathlib import Path

def dump(binary,out):
    env=dict(os.environ,NVDISASM_PATH='/usr/local/lib/python3.12/dist-packages/triton/backends/nvidia/bin')
    result={}
    for kind in ('ptx','sass','resource-usage'):
        p=subprocess.run(['cuobjdump','--dump-'+kind,str(binary)],capture_output=True,text=True,env=env,timeout=30)
        (out/('bench.'+kind)).write_text(p.stdout);(out/('dump-'+kind+'.stderr')).write_text(p.stderr);p.check_returncode();result[kind]=p.stdout
    return result

def kernels(ptx):
    entries=list(re.finditer(r'\.visible \.entry\s+(\S+)\s*\(',ptx));return {e[1]:ptx[e.start():entries[i+1].start() if i+1<len(entries) else None] for i,e in enumerate(entries) if '6MarlinI' in e[1]}

def check_hints(ptx,reference=False):
    items=kernels(ptx);assert len(items)==(6 if reference else 12),len(items)
    on=off=0
    for name,body in items.items():
        assert 'discard.global' not in body
        hinted='createpolicy.fractional.L2::evict_first.b64' in body
        if hinted:
            assert name.endswith('Lb1EEEvPK4int4S2_PS0_S3_S2_PKfS2_S5_S2_PKiPKiS9_S9_S5_ibiiiiPibbb') or 'Lb1E' in name
            assert 'cp.async.cg.shared.global.L2::cache_hint' in body
            # B is the second kernel parameter. All hinted sources must derive from it.
            tagged=set()
            for _ in range(12):
                for line in body.splitlines():
                    regs=re.findall(r'%rd\d+',line)
                    if not regs:continue
                    if 'ld.param.u64' in line and '_param_1]' in line:tagged.add(regs[0])
                    elif re.search(r'\b(?:add|sub|mov|cvta|cvt)\.',line) and any(r in tagged for r in regs[1:]):tagged.add(regs[0])
            hints=re.findall(r'cp.async[^;]*L2::cache_hint[^;]*;',body)
            assert hints and all(re.findall(r'\[(%rd\d+)\]',line)[-1] in tagged for line in hints),'hint not B-derived'
            on+=1
        else:
            assert 'L2::cache_hint' not in body;off+=1
    assert off==6 and on==(0 if reference else 6)
    return dict(off=off,on=on,B_only=True)

def sass_words(sass):
    pieces=re.split(r'Function\s*:\s*',sass)[1:];result={}
    for p in pieces:
        name=p.splitlines()[0].strip()
        if '6MarlinI' not in name:continue
        words=re.findall(r'/\* (0x[0-9a-fA-F]+) \*/',p)
        assert words,'missing encoded SASS'
        # Match tile template args, drop namespace and final hint bool.
        key=name[name.index('6MarlinI')+8:name.index('EEEv')]
        if 'l2hint_marlin' in name:key=key[:-4] # final Lb0E / Lb1E
        result.setdefault(key,[]).append((name,words))
    return result

def compare(candidate,stock,out):
    out=Path(out);a=dump(candidate,candidate.parent);b=dump(stock,stock.parent)
    checks=check_hints(a['ptx']);check_hints(b['ptx'],True)
    ca,st=sass_words(a['sass']),sass_words(b['sass']);rows=[]
    for key,base in st.items():
        assert len(base)==1 and key in ca,(key,list(ca))
        matching=[(n,w) for n,w in ca[key] if 'Lb0EEEv' in n]
        assert len(matching)==1,key
        n,w=matching[0];assert w==base[0][1],'OFF instruction bytes differ: '+key
        hinted=[(n,w) for n,w in ca[key] if 'Lb1EEEv' in n]
        assert len(hinted)==1 and hinted[0][1]!=w,'ON policy vanished from encoded instructions'
        rows.append(dict(on_instruction_stream_differs=True,tile=key,instruction_bytes_equal=True,words=len(w),sha256=hashlib.sha256(''.join(w).encode()).hexdigest()))
    assert len(rows)==6
    # Retain ON encoded instructions too; PTX establishes the explicit policy.
    report=dict(status='STATIC_PASS',hint=checks,off_stock=rows,candidate_sha256=hashlib.sha256(candidate.read_bytes()).hexdigest(),stock_sha256=hashlib.sha256(stock.read_bytes()).hexdigest(),scope='all six OFF encoded SASS instruction streams equal pinned stock; ON B-only PTX cache hint assembled for sm_121a')
    (out/'assembly.json').write_text(json.dumps(report,indent=2)+'\n');return report
