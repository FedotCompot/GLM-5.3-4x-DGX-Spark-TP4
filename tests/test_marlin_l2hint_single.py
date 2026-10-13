"""CPU contracts for immutable single-bank L2 hint serving."""
import contextlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tests'))
from test_marlin_l2hint import L, inputs, stock_function
from l2hint_native.selfcheck import check_schemas

ENV = dict(GLM_MARLIN_L2HINT='on', VLLM_USE_V2_MODEL_RUNNER='1', GLM_MOE_DET_ALIGN='1')


class Schema:
    def __init__(self, text, n): self.text, self.arguments = text, [None]*n
    def __str__(self): return self.text


def op(schema): return NS(default=NS(_schema=schema))


@contextlib.contextmanager
def boot_fixture():
    with tempfile.TemporaryDirectory() as tmp, contextlib.ExitStack() as stack:
        root = Path(tmp)/'vllm'; root.mkdir()
        binary = Path(tmp)/'l2hint-serving/candidate/marlin.so'
        binary.parent.mkdir(parents=True); binary.write_bytes(b'candidate')
        provider = root/'_moe_C_stable_libtorch.abi3.so'; provider.write_bytes(b'provider')
        pins = {}
        for name in L.PINS:
            path = root/(name.removeprefix('vllm.').replace('.', '/')+'.py')
            path.parent.mkdir(parents=True, exist_ok=True); path.write_text(name)
            pins[name] = L.sha256(path)
        native = ROOT/'overlay/bringup/l2hint_native'
        receipt = dict(reference=False, binary_sha256=L.sha256(binary),
                       source_sha256={str(p.relative_to(native)):L.sha256(p) for p in native.rglob('*')
                                      if p.is_file() and '__pycache__' not in p.parts},
                       stock_binary_name=provider.name, stock_binary_sha256=L.sha256(provider))
        (binary.with_name('build.json')).write_text(json.dumps(receipt))
        assembly = binary.parent.parent/'assembly.json'
        assembly.write_text(json.dumps(dict(status='STATIC_PASS',candidate_sha256=L.sha256(binary))))
        a = Schema('_moe_C::moe_wna16_marlin_gemm(Tensor a) -> Tensor', 29)
        b = Schema('_glm_l2hint_C::moe_wna16_marlin_gemm(Tensor a, bool l2_hint=False) -> Tensor',30)
        selfcheck = binary.parent.parent/'selfcheck.json'
        selfcheck.write_text(json.dumps(dict(status='SELFCHECK_PASS',cuda_initialized=False,
                                             stock_schema=str(a),hint_schema=str(b))))
        calls, loaded = [], []
        class Hooks:
            def after_import(self, name, installer): calls.append((name, installer))
        torch = NS(ops=NS(load_library=loaded.append,
                          _moe_C=NS(moe_wna16_marlin_gemm=op(a)),
                          _glm_l2hint_C=NS(moe_wna16_marlin_gemm=op(b))))
        stack.enter_context(patch.dict(sys.modules,{'torch':torch,
            'vllm.scalar_type':NS(scalar_types=NS(uint4b8=NS(id=7))),
            'glm_skip_mla_plan':NS(Hooks=Hooks)}))
        stack.enter_context(patch.object(L.importlib.util,'find_spec',return_value=NS(submodule_search_locations=[str(root)])))
        for name, value in dict(BINARY=binary,PINS=pins,_REGISTERED=False,_mode='off',
                                _hint_op=None,_u4b8=None,_binary_hash=None,_config_hash=None).items():
            stack.enter_context(patch.object(L,name,value))
        stack.enter_context(patch.object(sys,'meta_path',list(sys.meta_path)))
        yield NS(root=root,binary=binary,receipt=receipt,assembly=assembly,selfcheck=selfcheck,
                 stock=a,hint=b,calls=calls,loaded=loaded,torch=torch)


class SingleBank(unittest.TestCase):
    def test_default_and_explicit_off_do_not_inspect_or_load(self):
        with patch.object(L.importlib.util,'find_spec',side_effect=AssertionError('OFF dependency')):
            for env in ({}, {'GLM_MARLIN_L2HINT':'off'}, {'GLM_MARLIN_L2HINT_AB':'0'}):
                self.assertFalse(L.register(env))

    def test_options_no_private_dev_mode_needed(self):
        self.assertEqual(L.mode(ENV),'on')
        self.assertEqual(L.mode(ENV|{'VLLM_SERVER_DEV_MODE':'0'}),'on')
        for delta in ({'GLM_MARLIN_L2HINT':'1'}, {'GLM_MARLIN_L2HINT':'bad'},
                      {'GLM_MARLIN_L2HINT_AB':'bad'}, {'GLM_MARLIN_L2HINT_AB':'1'},
                      {'GLM_MOE_DET_ALIGN':'0'}, {'VLLM_USE_V2_MODEL_RUNNER':'0'}):
            with self.subTest(delta=delta), self.assertRaises(ValueError): L.mode(ENV|delta)

    def test_single_installs_only_operator_hook_and_live_selfcheck(self):
        with boot_fixture() as f:
            self.assertTrue(L.register(ENV)); self.assertEqual(L._mode,'on')
            self.assertEqual([name for name,_ in f.calls],[L.OPS])
            self.assertEqual(f.loaded,[str(f.root/f.receipt['stock_binary_name']),str(f.binary)])
            self.assertTrue(L.register(ENV)); self.assertEqual(len(f.calls),1)
            with self.assertRaises(RuntimeError): L.register(ENV|{'GLM_MARLIN_L2HINT':'off','GLM_MARLIN_L2HINT_AB':'1','VLLM_SERVER_DEV_MODE':'1'})

    def test_ab_still_installs_all_four_hooks_starts_off(self):
        with boot_fixture() as f, patch.object(L,'_runtime',0):
            L.register(ENV|{'GLM_MARLIN_L2HINT':'off','GLM_MARLIN_L2HINT_AB':'1','VLLM_SERVER_DEV_MODE':'1'})
            self.assertEqual([name for name,_ in f.calls],[L.OPS,L.CG,L.WORKER,L.CORE])
            self.assertEqual(L.current(),0)

    def test_single_dispatch_independent_of_capture_runtime_and_env(self):
        calls, hints = [], []
        with patch.object(L,'_mode','on'), patch.object(L,'_u4b8',7), \
             patch.object(L,'_hint_op',lambda *a:hints.append(a) or a[1]), \
             patch.object(L,'current',side_effect=AssertionError('runtime selector in single mode')):
            mod=NS(moe_wna16_marlin_gemm=stock_function(calls)); L.install_ops(mod)
            for down in (False,True):
                for block in (8,16):
                    with patch.dict(os.environ,GLM_MARLIN_L2HINT='off'),L.capturing(0):
                        p=inputs(down)|{'moe_block_size':block}
                        self.assertIs(mod.moe_wna16_marlin_gemm(**p),p['output'])
            # NVFP4 native draft always uses stock, even in a target ON capture.
            mod.moe_wna16_marlin_gemm(**(inputs()|{'b_q_type':NS(id=8)}))
        self.assertEqual(len(hints),4); self.assertEqual(len(calls),1)

    def test_single_never_allows_runtime_switch(self):
        with patch.object(L,'_mode','on'),self.assertRaises(RuntimeError): L.validate_switch('0')

    def test_readonly_overlay_path(self):
        self.assertEqual(str(L.BINARY),'/overlay/l2hint-serving/candidate/marlin.so')
        self.assertNotIn('/cache/l2hint-serving',(ROOT/'docs/marlin-l2hint-serving.md').read_text())

    def test_missing_binary_fails_before_hooks(self):
        with boot_fixture() as f:
            f.binary.unlink()
            with self.assertRaises(FileNotFoundError): L.register(ENV)
            self.assertFalse(f.calls); self.assertFalse(L._REGISTERED)

    def test_drift_fails_before_hooks(self):
        for what in ('source','binary','provider','build_source','assembly','selfcheck','abi','saved_schema'):
            with self.subTest(what=what),boot_fixture() as f:
                if what=='source': (f.root/'_custom_ops.py').write_text('drift')
                elif what=='binary': f.binary.write_bytes(b'drift')
                elif what=='provider': (f.root/f.receipt['stock_binary_name']).write_bytes(b'drift')
                elif what=='build_source':
                    f.receipt['source_sha256']['selfcheck.py']='0'*64
                    f.binary.with_name('build.json').write_text(json.dumps(f.receipt))
                elif what=='assembly': f.assembly.write_text('{}')
                elif what=='selfcheck': f.selfcheck.write_text('{}')
                elif what=='abi': f.hint.text+=' drift'
                elif what=='saved_schema':
                    d=json.loads(f.selfcheck.read_text());d['hint_schema']='wrong';f.selfcheck.write_text(json.dumps(d))
                with self.assertRaises((RuntimeError,KeyError)): L.register(ENV)
                self.assertFalse(f.calls); self.assertFalse(L._REGISTERED)

    def test_schema_counts_names_defaults_aliases_and_returns(self):
        with boot_fixture() as f:
            self.assertEqual(check_schemas(op(f.stock),op(f.hint)),(str(f.stock),str(f.hint)))
            for text,n in ((str(f.hint),29),(str(f.hint).replace('False','True'),30),
                           (str(f.hint).replace('Tensor a','Tensor! a'),30),
                           (str(f.hint).replace('-> Tensor','-> ()'),30)):
                with self.subTest(text=text),self.assertRaises(RuntimeError):check_schemas(op(f.stock),op(Schema(text,n)))

    def test_site_startup_is_fail_closed(self):
        text=(ROOT/'overlay/bringup/sitecustomize.py').read_text()
        self.assertIn("os.environ.get('GLM_MARLIN_L2HINT', 'off')",text)
        self.assertIn('except BaseException:',text);self.assertIn('os._exit(78)',text)


class DryAndBuild(unittest.TestCase):
    def test_off_base_and_on_only_one_env_pair_four_full_layout_vectors(self):
        from test_launcher import C
        code=subprocess.check_output(['git','show','7492ae7:scripts/cluster.py'],cwd=ROOT,text=True)
        scope={'__name__':'base_cluster'};exec(compile(code,'base_cluster','exec'),scope)
        # 10-13 profile keys are launcher-validated, not passthrough; the 7492ae7 base never saw them.
        scope['PROFILE_KEYS']=[x for x in scope['PROFILE_KEYS'] if x not in ('GLM_INDEXER_STABLE_TOPK','GLM_INDEXER_ROW_SHARD','GLM_INDEXER_ROW_SHARD_INIT','GLM_MARLIN_L2HINT','GLM_DECODE_FAIR_REARM','GLM_ATTN_FRONT_FUSE')]
        profile={k:'v-'+k for k in C.PROFILE_KEYS}
        profile.update(GLM_INDEXER_STABLE_TOPK='0',GLM_INDEXER_ROW_SHARD='0',GLM_MARLIN_L2HINT='off',GLM_ATTN_FRONT_FUSE='0')
        profile.update(GLM_MTP_KSTOP='1',GLM_MTP_KSTOP_UNIFORM_BATCH='k2',GLM_PAD_HYGIENE='0',
            GLM_SPEC_SAMPLE='0',GLM_INDEXER_SHORTCUT='0',GLM_KV_FORMAT='fp4x',GLM_FULL_MLA='triton',
            GLM_MTP_KSTOP_CAPTURE_LAYOUT='reuse',VLLM_USE_V2_MODEL_RUNNER='1',
            GLM_MLA_SPLIT_K='32',GLM_MLA_SPLIT_MAX_ROWS='36',
            GLM_DRAFT_HEAD='0',GLM_DRAFT_EHPROJ='0',GLM_DECODE_FAIR='0',GLM_ATTN_WEIGHTS='int8',
            GLM_NVFP4_GROUPS='attn',GLM_MOE_CANON_ALIGN='0',GLM_MOE_DET_ALIGN='1',
            GLM_MARLIN_L2HINT_AB='0',GLM_MARLIN_L2HINT='off',RECIPE_DISPRAM='require',
            RECIPE_MAX_MODEL_LEN='262144',RECIPE_KV_HEAD_BYTES='6318718976',GLM_SKIP_MLA_PLAN='0',
            VLLM_SERVER_DEV_MODE='0',GLM_PARAM_HASH='0',GLM_DSA_SWA_POOL='0',GLM_TRITON_MLA_PREFILL='0',GLM_GLUE_DSA_IDX_CACHE='0')
        with patch.dict(C.ENV,profile),patch.object(C,'dispram',return_value=None),patch.dict(scope,dispram=lambda:None):
            base=[scope['docker_command'](r,'single-test') for r in range(4)]
            off=[C.docker_command(r,'single-test') for r in range(4)]
            self.assertEqual(off,base)
            with patch.dict(C.ENV,GLM_MARLIN_L2HINT='on'):
                on=[C.docker_command(r,'single-test') for r in range(4)]
            for a,b in zip(off,on):
                aa,bb=a,b
                j=bb.index('GLM_MARLIN_L2HINT=on');self.assertEqual(bb[j-1],'-e')
                self.assertEqual(aa,bb[:j-1]+bb[j+1:])
            if os.environ.get('L2HINT_DRY_PROOF_OUT'):
                Path(os.environ['L2HINT_DRY_PROOF_OUT']).write_text(json.dumps(dict(
                    status='DRY_PASS',base_commit='7492ae7',ranks=4,
                    recipe_max_model_len=profile['RECIPE_MAX_MODEL_LEN'],
                    recipe_kv_head_bytes=profile['RECIPE_KV_HEAD_BYTES'],
                    scope='actual Docker vector renderer; synthetic profile, dispram no-I/O boundary mocked equally',
                    off_equals_base=True,on_only_adds=['-e','GLM_MARLIN_L2HINT=on'],
                    base=base,off=off,on=on),indent=2)+'\n')

    def test_named_build_containers_retained_on_success_and_failure(self):
        # Execute the actual shell lifecycle with a fake Docker boundary.
        for fail in (False,True):
            with self.subTest(fail=fail),tempfile.TemporaryDirectory() as tmp:
                tmp=Path(tmp); bindir=tmp/'bin';bindir.mkdir();log=tmp/'calls.jsonl';out=tmp/'out'
                docker=bindir/'docker'
                docker.write_text('''#!/usr/bin/env python3
import json,os,pathlib,sys
args=sys.argv[1:]
with open(os.environ['DOCKER_TEST_LOG'],'a') as f:f.write(json.dumps(args)+'\\n')
if 'inspect' in args: print('sha256:4def0ef644cb2e9814136dcffd5e385e21bc594f48f3b292234051904abe85a6')
if 'run' in args:
 if os.environ['DOCKER_TEST_FAIL']=='1':sys.exit(9)
 mount=next(a for a in args if a.endswith(':/out'));out=pathlib.Path(mount[:-5])
 (out/('selfcheck.json' if 'l2hint_native.selfcheck' in ' '.join(args) else 'assembly.json')).write_text('{}')
''');docker.chmod(0o755)
                run=subprocess.run(['bash',str(ROOT/'scripts/build_marlin_l2hint.sh'),str(out)],
                    env=os.environ|{'PATH':str(bindir)+os.pathsep+os.environ['PATH'],
                        'DOCKER_TEST_LOG':str(log),'DOCKER_TEST_FAIL':str(int(fail))},capture_output=True,text=True)
                self.assertEqual(run.returncode,9 if fail else 0,run.stderr)
                cmds=[json.loads(row) for row in log.read_text().splitlines()]
                runs=[a for a in cmds if 'run' in a]
                self.assertEqual(len(runs),1 if fail else 2)
                self.assertTrue(all('--name' in a and '--rm' not in a for a in runs))
                self.assertTrue(any('stop' in a for a in cmds));self.assertFalse(any('rm' in a for a in cmds))
                names=out.joinpath('containers.txt').read_text().splitlines()
                self.assertEqual(len(set(names)),2)
                self.assertTrue(all(a[a.index('--name')+1] in names for a in runs))




class CrossBoot(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0,str(ROOT/'bench'))
        import l2hint_crossboot
        cls.X=l2hint_crossboot

    def panels(self,tmp):
        paths=[]
        for phase,mode in (('A','off'),('B','on'),('Aprime','off')):
            rows=[]
            for kind in ('prose','code'):
                for i in range(10):
                    ms=9.5 if phase=='B' else 10
                    metrics=dict(status='OK',request_id=f'{phase}-{kind}-{i}',
                        metadata=dict(pair_key=f'{kind}-{i}',prompt_sha256=f'prompt-{kind}-{i}',seed=0,
                                      k_mode='boot-default',target_M='boot-default'),
                        time_source='client_first_to_last_output_token',labels={'model_name':'GLM-5.3'},
                        cycle_ms=ms,committed_per_cycle=3,cycle_ms_kind='request wall')
                    rows.append(dict(kind=kind,i=i,cycle_ms=ms,token_ids=[1,2],text='ok',
                                     finish_reason='stop',completion_tokens=2,metrics=metrics))
            path=Path(tmp)/(phase+'.json');path.write_text(json.dumps(dict(status='COMPLETE',mode=mode,boot=phase,rows=rows)))
            paths.append(path)
        return paths

    def mutate(self,path,fn):
        data=json.loads(path.read_text());fn(data);path.write_text(json.dumps(data))

    def test_crossboot_known_gain(self):
        with tempfile.TemporaryDirectory() as tmp:
            result=self.X.compare(self.panels(tmp));self.assertEqual(result['verdict'],'KEEP')
            self.assertEqual(result['panel']['prose']['savings_ms'],.5)

    def test_mismatched_ids_or_input_metadata_fails(self):
        for key in ('token_ids','prompt_sha256','seed','time_source','cycle_ms'):
            with self.subTest(key=key),tempfile.TemporaryDirectory() as tmp:
                paths=self.panels(tmp)
                def change(d):
                    r=d['rows'][0]
                    if key=='token_ids':r[key]=[1,3]
                    elif key in ('prompt_sha256','seed'):r['metrics']['metadata'][key]='wrong'
                    elif key=='time_source':r['metrics'][key]='wrong'
                    else:r[key]=1
                self.mutate(paths[1],change)
                with self.assertRaises(ValueError):self.X.compare(paths)

    def test_missing_duplicate_invalid_token_or_mode_or_boot_fails(self):
        for case in ('missing','duplicate','token','mode','boot','status'):
            with self.subTest(case=case),tempfile.TemporaryDirectory() as tmp:
                paths=self.panels(tmp)
                def change(d):
                    if case=='missing':d['rows'].pop()
                    elif case=='duplicate':d['rows'][1]=d['rows'][0]
                    elif case=='token':d['rows'][0]['token_ids']=[]
                    elif case=='mode':d['mode']='off'
                    elif case=='boot':d['boot']='A'
                    else:d['status']='KILL'
                self.mutate(paths[1],change)
                with self.assertRaises(ValueError):self.X.compare(paths)

    def test_drift_or_no_gain_kills(self):
        for phase,value in ((1,10),(2,10.4)):
            with tempfile.TemporaryDirectory() as tmp:
                paths=self.panels(tmp)
                def change(d):
                    for row in d['rows']:row['cycle_ms']=row['metrics']['cycle_ms']=value
                self.mutate(paths[phase],change)
                self.assertEqual(self.X.compare(paths)['verdict'],'KILL')

    def test_collect_mismatch_stops_first_pair_and_keeps_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths=self.panels(tmp);out=Path(tmp)/'candidate';calls=[]
            baseline=self.X.read_panel(paths[0])['rows'][0]
            class Gate:
                def __init__(self,*a,**kw):pass
                @contextlib.contextmanager
                def request(self,*a,**kw):yield NS(record=baseline['metrics'])
            def stream(*a):
                calls.append(1);return {k:baseline[k] for k in self.X.TOKEN_KEYS}|{'token_ids':[3,4]}
            with patch.object(self.X,'GateMetrics',Gate),patch.object(self.X,'stream',stream), \
                 patch.object(self.X,'post',return_value={'success':True}):
                args=NS(out=out,reference=paths[0],endpoint='unused',boot='B',mode='on')
                self.assertEqual(self.X.collect(args),1)
            result=json.loads((out/'panel.json').read_text())
            self.assertEqual(result['status'],'KILL');self.assertEqual(len(result['rows']),1)
            self.assertEqual(len(calls),2) # warmup plus one scored pair, no retries


if __name__=='__main__':unittest.main()
