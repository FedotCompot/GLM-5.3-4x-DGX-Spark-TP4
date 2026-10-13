"""CPU contracts: stock identity, guarded metadata, graph capture and TP RPC."""
import importlib.util
import inspect
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'overlay/bringup'))
import glm_marlin_l2hint as L
from l2hint_native import build


class Tensor:
    def __init__(self, shape, dtype='torch.bfloat16'):
        self.shape, self.dtype = shape, dtype
    def numel(self):
        import math
        return math.prod(self.shape)


def stock_function(calls):
    signature = ', '.join(L.PARAMETERS[:26]) + ', thread_k=-1, thread_n=-1, blocks_per_sm=-1'
    scope = {'calls': calls}
    exec(f'def stock({signature}):\n calls.append(locals().copy())\n return output', scope)
    return scope['stock']


def inputs(down=False):
    k, n = (512, 6144) if down else (6144, 1024)
    p = dict.fromkeys(L.PARAMETERS, None)
    p.update(input=Tensor((4, k)), output=Tensor((32, n)), b_qweight=Tensor((256, k//16, n*2), 'torch.int32'),
             b_scales=Tensor((256, k//128, n)), b_q_type=NS(id=7), size_m=4,
             size_n=n, size_k=k, moe_block_size=8, top_k=8, mul_topk_weights=False,
             is_k_full=True, use_atomic_add=False, use_fp32_reduce=True, is_zp_float=False,
             thread_k=-1, thread_n=-1, blocks_per_sm=-1)
    return p


class Dispatch(unittest.TestCase):
    def setUp(self):
        L._counts.clear()
        self.stock_calls, self.hints = [], []
        self.flag = 0
        self.stock = stock_function(self.stock_calls)
        self.call = L.dispatch(self.stock, lambda *a: self.hints.append(a) or a[1], lambda: self.flag, 7)

    def test_off_passthrough_identity(self):
        p = inputs(); self.assertIs(self.call(**p), p['output'])
        self.assertEqual(len(self.stock_calls), 1); self.assertFalse(self.hints)
        self.assertFalse(L._counts)

    def test_off_skips_guard_and_binding(self):
        with patch.object(L, 'qualified', side_effect=AssertionError('guard on OFF')):
            self.call(**inputs())

    def test_aba_with_same_workspace_objects(self):
        p = inputs(); p['workspace'] = object(); self.call(**p)
        self.flag = 1; self.call(**p); self.flag = 0; self.call(**p)
        self.assertEqual(len(self.stock_calls), 2); self.assertEqual(len(self.hints), 1)
        self.assertIs(self.hints[0][10], p['workspace'])
        self.assertIs(self.hints[0][11], p['sorted_token_ids'])
        self.assertEqual(self.hints[0][18], 7); self.assertIs(self.hints[0][-1], True)
        self.assertEqual(self.hints[0][26:29], (-1, -1, -1))

    def test_both_target_calls_and_block16(self):
        self.flag = 1
        for down in (False, True):
            for block in (8, 16):
                p = inputs(down); p['moe_block_size'] = block
                self.assertIs(self.call(**p), p['output'])
        self.assertEqual(len(self.hints), 4); self.assertFalse(self.stock_calls)

    def test_defaults_and_positional_argument_forwarding(self):
        self.flag = 1; p = inputs()
        self.call(*(p[n] for n in L.PARAMETERS[:26]))
        self.assertEqual(self.hints[0][-4:], (-1, -1, -1, True))

    def test_metadata_fallbacks(self):
        self.flag = 1
        changes = [dict(b_q_type=NS(id=8)), dict(size_n=2048), dict(moe_block_size=32),
                   dict(is_k_full=False), dict(is_zp_float=True), dict(use_atomic_add=True),
                   dict(use_fp32_reduce=False), dict(a_scales=object()), dict(global_scale=object()),
                   dict(input=Tensor((4,6144),'torch.float16')), dict(b_scales=Tensor((256,48,1024),'torch.float16')),
                   dict(b_scales=Tensor((256,24,1024))), dict(b_scales=Tensor((128,48,1024))),
                   dict(b_qweight=Tensor((768,2048))), dict(thread_k=64,thread_n=256),
                   dict(output=Tensor((32,1024),'torch.float16')),
                   dict(b_qweight=Tensor((256,384,2048),'torch.float32')),
                   dict(b_qweight=Tensor((256,383,2048),'torch.int32'))]
        for field in ('b_qzeros','g_idx','perm'):
            changes.append({field:Tensor((1,))})
        for delta in changes:
            with self.subTest(delta=delta):
                self.call(**(inputs() | delta))
        self.assertFalse(self.hints); self.assertEqual(len(self.stock_calls), len(changes))

    def test_empty_optional_tensors(self):
        self.flag = 1; p = inputs()
        for field in ('b_qzeros', 'g_idx', 'perm'): p[field] = Tensor((0,))
        self.call(**p); self.assertEqual(len(self.hints), 1)

    def test_three_stock_explicit_tiles(self):
        self.flag = 1
        for k,n in ((128,128),(64,128),(128,64)):
            self.call(**(inputs() | dict(thread_k=k, thread_n=n)))
        self.assertEqual(len(self.hints), 3)

    def test_signature_drift_refused(self):
        with self.assertRaises(RuntimeError): L.dispatch(lambda a: a, None, None, 7)

    def test_ops_install_idempotent(self):
        mod = NS(moe_wna16_marlin_gemm=self.stock)
        L.install_ops(mod); original=mod.moe_wna16_marlin_gemm
        L.install_ops(mod); self.assertIs(mod.moe_wna16_marlin_gemm, original)


def manager_module():
    class CudaGraphManager:
        def __init__(self):
            self.graphs = {}; self._graphs_captured = False
            self.compilation_config = NS(mode=0)
            self.use_breakable_cg = False
            self._capture_descs = (NS(name='FULL'),)
            self.events = []
        def capture(self, factory, label=''):
            self.events.append(L.current())
            for d in (1, 3):
                forward=factory(d,warmup=True)
                forward(0)
                self.graphs[d] = NS(bank=L.current())
            self._graphs_captured = True
        def run_fullgraph(self, desc): return self.graphs[desc].bank
        def profile_memory(self, factory):
            self.capture(factory)
            memory = sum(len(t) for t in self._glm_l2hint_sets) * 100
            self.graphs.clear(); self._graphs_captured=False
            return memory
    class ModelCudaGraphManager(CudaGraphManager): pass
    return NS(CudaGraphManager=CudaGraphManager, ModelCudaGraphManager=ModelCudaGraphManager)


class GraphBanks(unittest.TestCase):
    def setUp(self):
        L._managers.clear(); L._runtime=0; L._capture=None; L._profiling=False; L._replays[:]=[0,0]
        self.mod=manager_module(); L.install_graphs(self.mod)
        self.mgr=self.mod.ModelCudaGraphManager()

    def test_two_banks_capture_flag_and_aba_replay(self):
        self.mgr.capture(lambda d, **kw: lambda *a: None)
        self.assertEqual(self.mgr.events,[0,1]); self.assertEqual(len(L._managers),1)
        a=self.mgr.graphs[1]
        self.assertEqual(self.mgr.run_fullgraph(1),0)
        L._runtime=1; self.mgr.graphs=self.mgr._glm_l2hint_sets[1]
        self.assertEqual(self.mgr.run_fullgraph(1),1)
        L._runtime=0; self.mgr.graphs=self.mgr._glm_l2hint_sets[0]
        self.assertIs(self.mgr.graphs[1],a); self.assertEqual(self.mgr.run_fullgraph(1),0)
        self.assertEqual(L._replays,[2,1])

    def test_profile_counts_both_and_drops_both(self):
        self.assertEqual(self.mgr.profile_memory(lambda d, **kw: lambda *a:None),400)
        self.assertNotIn('_glm_l2hint_sets',self.mgr.__dict__)
        self.assertFalse(L._managers); self.assertFalse(L._profiling)
        self.mgr.capture(lambda d, **kw: lambda *a:None); self.assertEqual(len(L._managers),1)

    def test_profile_error_restores_state(self):
        with self.assertRaises(ValueError):
            self.mgr.profile_memory(lambda d, **kw: (_ for _ in ()).throw(ValueError('test')))
        self.assertFalse(L._profiling); self.assertIsNone(L._capture)

    def test_draft_remains_one_stock_bank(self):
        draft=self.mod.CudaGraphManager(); draft.capture(lambda d, **kw: lambda *a:None)
        self.assertEqual(draft.events,[0]); self.assertFalse(L._managers)

    def test_non_full_compile_breakable_refused(self):
        for field,value in (('compilation_config',NS(mode=1)),('use_breakable_cg',True),
                            ('_capture_descs',(NS(name='PIECEWISE'),))):
            with self.subTest(field=field):
                m=self.mod.ModelCudaGraphManager(); setattr(m,field,value)
                with self.assertRaises(RuntimeError):m.capture(lambda d:None)

    def test_wrong_replay_bank_refused(self):
        self.mgr.capture(lambda d, **kw: lambda *a:None); L._runtime=1
        with self.assertRaises(RuntimeError): self.mgr.run_fullgraph(1)

    def test_capture_context_nests_and_restores(self):
        with L.capturing(1):
            self.assertEqual(L.current(),1)
            with L.capturing(0):self.assertEqual(L.current(),0)
            self.assertEqual(L.current(),1)
        self.assertEqual(L.current(),0)

    def test_switch_validation_before_ready_and_during_capture(self):
        with self.assertRaises(RuntimeError):L.validate_switch('1')
        self.mgr.capture(lambda d, **kw: lambda *a:None)
        for bad in (1,True,'2','bad'):
            with self.assertRaises(ValueError):L.validate_switch(bad)
        with L.capturing(0), self.assertRaises(RuntimeError):L.validate_switch('1')
        self.assertEqual(L.validate_switch('1'),1)

    def test_switch_tp_vote_prevents_partial_change(self):
        self.mgr.capture(lambda d, **kw: lambda *a:None)
        def gather(votes,ticket,group):
            votes[:]=[ticket,ticket.copy()]; votes[1][1]=0
        torch=NS(cuda=NS(synchronize=lambda:None),distributed=NS(all_gather_object=gather))
        dist=NS(get_tp_group=lambda:NS(world_size=2,cpu_group=object()))
        with patch.dict(sys.modules,{'torch':torch,'vllm.distributed':dist}), self.assertRaises(RuntimeError):
            L.worker_switch('1')
        self.assertEqual(L._runtime,0); self.assertIs(self.mgr.graphs,self.mgr._glm_l2hint_sets[0])

    def test_tp_aba_switch(self):
        self.mgr.capture(lambda d, **kw: lambda *a:None); sync=[]
        torch=NS(cuda=NS(synchronize=lambda:sync.append(1)),distributed=NS(
            all_gather_object=lambda votes,ticket,group:votes.__setitem__(slice(None),[ticket,ticket])))
        dist=NS(get_tp_group=lambda:NS(world_size=2,cpu_group=object()))
        with patch.dict(sys.modules,{'torch':torch,'vllm.distributed':dist}):
            for v in ('0','1','0'):
                self.assertEqual(L.worker_switch(v)['value'],int(v))
        self.assertEqual(len(sync),3)


class StartupAndBuild(unittest.TestCase):
    def test_off_register_has_no_dependencies(self):
        with patch.object(L.importlib.util,'find_spec',side_effect=AssertionError('OFF import')):
            self.assertFalse(L.register({})); self.assertFalse(L.register({'GLM_MARLIN_L2HINT_AB':'0'}))

    def test_options_require_private_deterministic_v2(self):
        env=dict(GLM_MARLIN_L2HINT_AB='1',VLLM_SERVER_DEV_MODE='1',VLLM_USE_V2_MODEL_RUNNER='1',GLM_MOE_DET_ALIGN='1')
        self.assertTrue(L.options(env))
        for key in env:
            with self.subTest(key=key), self.assertRaises(ValueError):L.options(env|{key:'bad'})

    def test_core_drain_blocks_broadcast(self):
        calls=[]
        class EngineCore:
            def collective_rpc(self,*args):calls.append(args);return ['ok']
        L.install_core(NS(EngineCore=EngineCore)); core=EngineCore()
        core.scheduler=NS(get_num_unfinished_requests=lambda:1,running=[])
        self.assertEqual(core.collective_rpc('glm_l2hint_switch',args=('1',)),[{'busy':1}]);self.assertFalse(calls)
        core.scheduler=NS(get_num_unfinished_requests=lambda:0,running=[])
        self.assertEqual(core.collective_rpc('glm_l2hint_switch',args=('1',)),['ok'])
        core.scheduler.running=[object()]
        self.assertEqual(core.collective_rpc('glm_l2hint_switch',args=('0',)),[{'busy':1}])
        self.assertEqual(core.collective_rpc('glm_l2hint_status'),['ok'])

    def test_sources_generated_with_isolated_registration(self):
        with tempfile.TemporaryDirectory() as tmp:
            tu=build.generate(Path(tmp)/'on'); text=tu.read_text()
            ops=(tu.parent/'csrc/libtorch_stable/moe/marlin_moe_wna16/ops.cu').read_text()
            self.assertIn('STABLE_TORCH_LIBRARY_FRAGMENT(_glm_l2hint_C',ops)
            self.assertNotIn('_moe_C,',ops);self.assertNotIn('consumer',text)
            kernel=(tu.parent/'csrc/libtorch_stable/moe/marlin_moe_wna16/marlin_template.h').read_text()
            self.assertEqual(kernel.count('createpolicy.fractional.L2::evict_first.b64'),1)
            self.assertIn('"l"(&B[b_gl_idx])',kernel)
            self.assertIn('cp_async4(&sh_b_stage',kernel)
            self.assertEqual(text.count('template __global__ void Marlin<'),12)
            ref=build.generate(Path(tmp)/'off',True)
            self.assertEqual(ref.read_text().count('template __global__ void Marlin<'),6)

    def test_source_pins_match_local_extracted_image(self):
        image=Path(os.environ.get('VLLM_SOURCE_ROOT','/tmp/glm-adaptive-image/vllm'))
        for name,pin in L.PINS.items():
            self.assertEqual(L.sha256(image/(name.removeprefix('vllm.').replace('.','/')+'.py')),pin)

    def test_f2_workspace_wrapper_composes_below_align(self):
        import glm_glue_lite as G
        calls=[]; hint=[]; workspace=object(); hidden=NS(device='cuda:0')
        stock=stock_function(calls)
        gemm=L.dispatch(stock,lambda *a: hint.append(a) or a[1],lambda:1,7)
        def fused(hidden_states, workspace=None):
            p=inputs(); p['workspace']=workspace
            # Same tensors already produced by the align path, passed through.
            p['sorted_token_ids']=object();p['expert_ids']=object()
            return gemm(**p)
        mod=NS(fused_marlin_moe=fused)
        G.install_moe_ws(mod,capturing=lambda:False,stream_id=lambda device:11)
        with patch.object(G,'lever',return_value=True),patch.object(G,'tick'), \
             patch.dict(G._WS,{'cuda:0':workspace},clear=True),patch.dict(G._HOME,{'cuda:0':11},clear=True):
            mod.fused_marlin_moe(hidden)
        self.assertIs(hint[0][10],workspace);self.assertFalse(calls)
        self.assertTrue(mod.fused_marlin_moe._glm_glue)

    def test_default_off_four_docker_vectors_match_base(self):
        # Actual base launcher text, same environment and root/mount paths.
        import subprocess
        sys.path.insert(0,str(ROOT/'tests'))
        import test_launcher
        C=test_launcher.C
        code=subprocess.check_output(['git','show','7492ae7:scripts/cluster.py'],cwd=ROOT,text=True)
        scope={'__name__':'base_cluster'};exec(compile(code,'base_cluster','exec'),scope)
        base=NS(**scope)
        # 10-13 profile keys are launcher-validated, not passthrough; the 7492ae7 base never saw them.
        scope['PROFILE_KEYS']=[x for x in scope['PROFILE_KEYS'] if x not in ('GLM_INDEXER_STABLE_TOPK','GLM_INDEXER_ROW_SHARD','GLM_INDEXER_ROW_SHARD_INIT','GLM_MARLIN_L2HINT','GLM_DECODE_FAIR_REARM','GLM_ATTN_FRONT_FUSE')]
        profile={key:'v-'+key for key in C.PROFILE_KEYS}
        profile.update(GLM_INDEXER_STABLE_TOPK='0',GLM_INDEXER_ROW_SHARD='0',GLM_MARLIN_L2HINT='off',GLM_ATTN_FRONT_FUSE='0')
        profile.update(GLM_MTP_KSTOP='0',GLM_MTP_KSTOP_UNIFORM_BATCH='0',GLM_PAD_HYGIENE='0',
                       GLM_DRAFT_HEAD='0',GLM_DRAFT_EHPROJ='0',GLM_DECODE_FAIR='0',GLM_ATTN_WEIGHTS='int8',
                       GLM_NVFP4_GROUPS='attn',GLM_MOE_CANON_ALIGN='0',GLM_MOE_DET_ALIGN='0',
                       GLM_MARLIN_L2HINT_AB='0')
        with patch.dict(C.ENV,profile),patch.object(C,'dispram',return_value=None), \
             patch.dict(scope,dispram=lambda:None):
            self.assertEqual([C.docker_command(r,'l2-test') for r in range(4)],
                             [base.docker_command(r,'l2-test') for r in range(4)])

    def test_optin_only_adds_ab_flag_to_all_rank_envs(self):
        import test_launcher
        C=test_launcher.C
        profile={key:'v-'+key for key in C.PROFILE_KEYS}
        profile.update(GLM_INDEXER_STABLE_TOPK='0',GLM_INDEXER_ROW_SHARD='0',GLM_MARLIN_L2HINT='off',GLM_ATTN_FRONT_FUSE='0')
        profile.update(GLM_MTP_KSTOP='0',GLM_MTP_KSTOP_UNIFORM_BATCH='0',GLM_PAD_HYGIENE='0',
                       GLM_DRAFT_HEAD='0',GLM_DRAFT_EHPROJ='0',GLM_DECODE_FAIR='0',GLM_ATTN_WEIGHTS='int8',
                       GLM_NVFP4_GROUPS='attn',GLM_MOE_CANON_ALIGN='0',GLM_MOE_DET_ALIGN='1',
                       GLM_MARLIN_L2HINT_AB='0',VLLM_SERVER_DEV_MODE='1',VLLM_USE_V2_MODEL_RUNNER='1',GLM_PARAM_HASH='1')
        with patch.dict(C.ENV,profile),patch.object(C,'dispram',return_value=None):
            before=[C.rank_env(r) for r in range(4)]
            with patch.dict(C.ENV,GLM_MARLIN_L2HINT_AB='1'):
                self.assertEqual([C.rank_env(r) for r in range(4)],
                                 [r|dict(GLM_MARLIN_L2HINT_AB='1') for r in before])


class PanelGate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0,str(ROOT/'bench'))
        import l2hint_aba
        cls.P=l2hint_aba

    def rows(self):
        return [dict(kind=kind,i=i,phase=phase,cycle_ms=10 if phase!='B' else 9.5,
                     token_ids=[1,2],text='hello',completion_tokens=2,finish_reason='stop')
                for kind in ('prose','code') for i in range(10) for phase in ('A','B','Aprime')]

    def test_known_paired_gain(self):
        result=self.P.assess(self.rows())
        self.assertEqual(result['verdict'],'KEEP');self.assertTrue(result['token_equality'])
        self.assertEqual(result['panel']['prose']['lower_3SE_ms'],.5)

    def test_actual_ids_mismatch_even_equal_text_kills(self):
        rows=self.rows();rows[1]['token_ids']=[1,3]
        self.assertEqual(self.P.assess(rows)['verdict'],'KILL')

    def test_control_drift_kills(self):
        rows=self.rows()
        for row in rows:
            if row['phase']=='Aprime':row['cycle_ms']=10.3
        self.assertEqual(self.P.assess(rows)['verdict'],'KILL')

    def test_incomplete_panel_refused(self):
        with self.assertRaises(ValueError):self.P.assess(self.rows()[:-1])

    def test_no_gain_or_small_gain_kills(self):
        for value in (10,9.9):
            rows=self.rows()
            for row in rows:
                if row['phase']=='B':row['cycle_ms']=value
            self.assertEqual(self.P.assess(rows)['verdict'],'KILL')

    def test_ci_includes_zero_kills(self):
        rows=self.rows()
        for row in rows:
            if row['phase']=='B':row['cycle_ms']=8 if row['i']<5 else 11
        self.assertEqual(self.P.assess(rows)['verdict'],'KILL')


if __name__=='__main__': unittest.main()
