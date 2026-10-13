# SPDX-License-Identifier: Apache-2.0
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import types
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'overlay/bringup'),str(ROOT/'overlay/overlay')]
import glm_attnfront as a
NATIVE=ROOT/'overlay/bringup/attnfront_native'
IMAGE=Path(os.environ.get('GLM_IMAGE_SRC','/tmp/glm-adaptive-image'))/'vllm'
ON=dict(GLM_ATTN_FRONT_FUSE='1',GLM_ATTN_WEIGHTS='nvfp4',GLM_KV_FORMAT='fp4x',GLM_FULL_MLA='triton')

class Config(unittest.TestCase):
    def test_off_no_subflag_effect(self):
        for env in ({},{'GLM_ATTN_FRONT_FUSE':'0'},{'GLM_ATTN_FRONT_PARTS':'bad'},dict(ON,GLM_ATTN_FRONT_FUSE='0',GLM_ATTN_FRONT_NORMS='bad')):
            self.assertIsNone(a.options(env))
    def test_default_all_and_each_piece(self):
        self.assertEqual(a.options(ON),(True,True,True))
        for mask in range(1,8):
            env=dict(ON,**{k:str((mask>>i)&1) for i,k in enumerate(a.FLAGS)})
            self.assertEqual(a.options(env),tuple(bool((mask>>i)&1) for i in range(3)))
    def test_invalid_modes(self):
        for key in ('GLM_ATTN_FRONT_FUSE',*a.FLAGS):
            for val in ('','yes','2','-1'):
                with self.subTest(key=key,val=val), self.assertRaises(ValueError):
                    a.options(dict(ON,**{key:val}))
        with self.assertRaises(ValueError):a.options(dict(ON,**dict.fromkeys(a.FLAGS,'0')))
    def test_release_dependencies(self):
        for key in ('GLM_ATTN_WEIGHTS','GLM_KV_FORMAT','GLM_FULL_MLA'):
            with self.subTest(key=key),self.assertRaises(ValueError):a.options(dict(ON,**{key:'other'}))
    def test_nonqualified_composition_refused(self):
        for key in ('GLM_NVFP4_MERGED','GLM_FP4_RECENT_WINDOW','GLM_FP4_RECENT_AB','GLM_NVFP4_WSIM'):
            with self.subTest(key=key),self.assertRaises(ValueError):a.options(dict(ON,**{key:'1'}))
    def test_off_register_no_import_or_hooks(self):
        before=list(sys.meta_path)
        with patch.object(a,'check_pins',side_effect=AssertionError),patch.object(a,'register_ops',side_effect=AssertionError):
            self.assertFalse(a.register({}))
        self.assertEqual(before,sys.meta_path)

class Reachability(unittest.TestCase):
    def test_hit_logs_each_attention_owner_and_m_once(self):
        import io,contextlib
        target=types.SimpleNamespace(mla_attn=types.SimpleNamespace(layer_name='target.attn'))
        draft=types.SimpleNamespace(mla_attn=types.SimpleNamespace(layer_name='draft.attn'))
        out=io.StringIO()
        with patch.object(a,'_HITS',set()),patch.object(a,'COUNTS',dict(norms=0)),contextlib.redirect_stderr(out):
            a.hit('norms',4,target);a.hit('norms',1,target);a.hit('norms',1,target);a.hit('norms',1,draft)
            self.assertEqual(a.COUNTS['norms'],4)
        self.assertEqual(len(out.getvalue().splitlines()),3)
        self.assertIn('target.attn',out.getvalue());self.assertIn('draft.attn',out.getvalue())

class HitCoverage(unittest.TestCase):
    def test_requires_every_target_layer_piece_and_capture_size(self):
        import importlib.util
        spec=importlib.util.spec_from_file_location('hits',ROOT/'bench/attnfront_hits.py')
        hits=importlib.util.module_from_spec(spec);spec.loader.exec_module(hits)
        lines=['ATTNFRONT_READY']+[f'ATTNFRONT_HIT {p} M{m} model.layers.{i}.self_attn.attn'
              for p in hits.PIECES for m in range(1,17) for i in range(1,78)]
        self.assertEqual(hits.check('\n'.join(lines))['required_hits'],3696)
        for incomplete in (lines[:-1],lines[1:],lines+['ATTNFRONT_HIT norms M1 model.layers.0.self_attn.attn']):
            with self.assertRaises(ValueError):hits.check('\n'.join(incomplete))

class Eligibility(unittest.TestCase):
    # Config fields match the serving checkpoint; TP4 gives sixteen local heads.
    REAL_CONFIG = dict(hidden_size=6144, q_lora_rank=2048, kv_lora_rank=512,
                       qk_rope_head_dim=64, qk_nope_head_dim=192, v_head_dim=256,
                       num_attention_heads=64, rms_norm_eps=1e-5, num_hidden_layers=78)

    def fixture(self, m, rank=None):
        import torch
        cfg = dict(self.REAL_CONFIG)
        if rank is not None: cfg['q_lora_rank'] = rank
        class Tensor:
            def __init__(self, shape, strides=None):
                self.shape=tuple(shape); self.ndim=len(shape)
                self.dtype=torch.bfloat16; self.is_cuda=True
                self.strides=strides or (shape[-1],1)
            def stride(self, dim): return self.strides[dim]
        class RMSNorm:
            def __init__(self, width):
                self.pass_weight=True; self.variance_size_override=None
                self.weight=Tensor((width,)); self.variance_epsilon=cfg['rms_norm_eps']
        class AttentionNVFP4: pass
        class Linear:
            tp_size=1; bias=None
            scheme=AttentionNVFP4()
            def __call__(self, x):
                return torch.zeros((m,cfg['q_lora_rank']+576),dtype=torch.bfloat16),None
        class RotaryEmbedding:
            rotary_dim=64; use_flashinfer=False
            def forward_cuda(self): pass
            def __call__(self,p,q,k): return q,k
        layer=Linear()
        layer._glm_nvfp4_parts=[types.SimpleNamespace(output_size_per_partition=n)
                               for n in (cfg['q_lora_rank'],576)]
        rotary=RotaryEmbedding(); rotary._forward_method=rotary.forward_cuda
        owner=types.SimpleNamespace(**{k:cfg[k] for k in (
            'hidden_size','q_lora_rank','kv_lora_rank','qk_rope_head_dim',
            'qk_nope_head_dim','v_head_dim')},num_heads=cfg['num_attention_heads']//4,
            qk_head_dim=cfg['qk_nope_head_dim']+cfg['qk_rope_head_dim'],
            fused_qkv_a_proj=layer,q_a_layernorm=RMSNorm(cfg['q_lora_rank']),
            kv_a_layernorm=RMSNorm(cfg['kv_lora_rank']),fuse_qkv_rmsnorm=False,
            rotary_emb=rotary,dcp_q_replicate=False,
            mla_attn=types.SimpleNamespace(layer_name='model.layers.1.self_attn.attn',
                use_pcp=False,impl=types.SimpleNamespace()))
        h=Tensor((m,cfg['hidden_size']))
        q=Tensor((m,owner.num_heads,64),(4096,256,1))
        x=Tensor((m,512))
        ks=[Tensor((m,1,64),(rs,64,1)) for rs in (576,cfg['q_lora_rank']+576)]
        return owner,h,q,ks,x

    def test_real_and_legacy_all_capture_sizes_admitted(self):
        provider=types.SimpleNamespace(ir=types.SimpleNamespace(ops=types.SimpleNamespace(
            rms_norm=types.SimpleNamespace(dispatch=lambda *args:types.SimpleNamespace(provider='vllm_c')))))
        with patch.object(a,'CONFIG',(True,True,True)), patch.object(a,'hit'), patch.dict(sys.modules,vllm=provider):
            for rank in (self.REAL_CONFIG['q_lora_rank'],1536):
                for m in range(1,17):
                    owner,h,q,ks,x=self.fixture(m,rank)
                    with self.subTest(rank=rank,m=m):
                        self.assertTrue(a.parts_eligible(owner,h))
                        self.assertTrue(a.norm_eligible(owner,h))
                        for k in ks: self.assertTrue(a.rope_eligible(owner,q,k,x))
                        # All three independently remain admitted with parts OFF.
                        with patch.object(a,'CONFIG',(False,True,True)):
                            self.assertTrue(a.norm_eligible(owner,h))
                            self.assertTrue(a.rope_eligible(owner,q,ks[1],x))

    def test_layer_zero_bf16_falls_back_every_piece_all_m(self):
        with patch.object(a,'CONFIG',(True,True,True)):
            for m in range(1,17):
                owner,h,q,ks,x=self.fixture(m)
                owner.fused_qkv_a_proj.scheme=None
                del owner.fused_qkv_a_proj._glm_nvfp4_parts
                self.assertFalse(a.parts_eligible(owner,h))
                self.assertFalse(a.norm_eligible(owner,h))
                for k in ks:
                    self.assertFalse(a.rope_eligible(owner,q,k,x))
                    self.assertEqual(a.rope(owner,None,q,k,x),(q,k,None))
                parts=a.partitions(owner,h)
                self.assertEqual([tuple(t.shape) for t in parts],[(m,2048),(m,576)])

    def test_int8_mtp_keeps_independent_norms_and_rope(self):
        provider=types.SimpleNamespace(ir=types.SimpleNamespace(ops=types.SimpleNamespace(
            rms_norm=types.SimpleNamespace(dispatch=lambda *args:types.SimpleNamespace(provider='vllm_c')))))
        with patch.object(a,'CONFIG',(True,True,True)), patch.object(a,'hit'), patch.dict(sys.modules,vllm=provider):
            owner,h,q,ks,x=self.fixture(4)
            owner.fused_qkv_a_proj.scheme=types.SimpleNamespace()
            del owner.fused_qkv_a_proj._glm_nvfp4_parts
            self.assertFalse(a.parts_eligible(owner,h))
            self.assertTrue(a.norm_eligible(owner,h))
            self.assertTrue(a.rope_eligible(owner,q,ks[1],x))

    def test_unknown_shape_and_wrong_partition_sizes_refused(self):
        provider=types.SimpleNamespace(ir=types.SimpleNamespace(ops=types.SimpleNamespace(
            rms_norm=types.SimpleNamespace(dispatch=lambda *args:types.SimpleNamespace(provider='vllm_c')))))
        with patch.object(a,'CONFIG',(True,True,True)),patch.object(a,'hit'),patch.dict(sys.modules,vllm=provider):
            for rank in (1024,2048):
                owner,h,q,ks,x=self.fixture(4,rank)
                if rank==2048: owner.fused_qkv_a_proj._glm_nvfp4_parts[0].output_size_per_partition=1536
                self.assertFalse(a.parts_eligible(owner,h))
                self.assertEqual(a.norm_eligible(owner,h),rank==2048)
                self.assertEqual(a.rope_eligible(owner,q,ks[1],x),rank==2048)

    def test_stock_host_selection_and_valid_item_tree_all_m(self):
        import math
        stock=(NATIVE/'vendor/layernorm_kernels.cu').read_text()
        self.assertIn('std::gcd(16 / sizeof(scalar_t), hidden_size)',stock)
        self.assertIn('std::min(hidden_size / calculated_vec_size, max_block_size)',stock)
        self.assertIn('((num_tokens < 256) ? 1024 : 256)',stock)
        for m in range(1,17):
            for width,threads in ((2048,256),(1536,192),(512,64)):
                vec=math.gcd(16//2,width)
                self.assertEqual(vec,8)
                self.assertEqual(min(width//vec,1024 if m<256 else 256),threads)
                # Stock thread t consumes vector [8t..8t+7], exactly once.
                self.assertEqual([i for t in range(threads) for i in range(8*t,8*t+8)],list(range(width)))
                self.assertEqual(threads//32,{2048:8,1536:6,512:2}[width])

class Sources(unittest.TestCase):
    def test_vendor_pins(self):
        pins=json.loads((NATIVE/'vendor/BASE.json').read_text())
        self.assertTrue(pins['commit'].startswith('487ecf187'))
        for name,digest in pins['files'].items():
            self.assertEqual(hashlib.sha256((NATIVE/'vendor'/Path(name).name).read_bytes()).hexdigest(),digest)
    def test_python_pins(self):
        for name,pin in a.PINS.items():
            p=IMAGE/(name.removeprefix('vllm.').replace('.','/')+'.py')
            self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(),pin,name)
    def test_transforms_compile(self):
        for name,fn in ((a.FRONT,a.transform_front),(a.COMMON,a.transform_common)):
            raw=(IMAGE/(name.removeprefix('vllm.').replace('.','/')+'.py')).read_text()
            compile(fn(raw),name,'exec')
    def test_anchor_drift_refused(self):
        with self.assertRaises(RuntimeError):a.once('xx','x','y')
        with self.assertRaises(RuntimeError):a.once('','x','y')
    def test_default_apply_bytecode_unchanged(self):
        old=subprocess.check_output(['git','show','261cbdb:overlay/overlay/glm_nvfp4_attn.py'],cwd=ROOT).decode()
        new=(ROOT/'overlay/overlay/glm_nvfp4_attn.py').read_text()
        def body(s):
            return ast.dump(next(n for n in ast.parse(s).body if isinstance(n,ast.FunctionDef) and n.name=='apply_layer'),include_attributes=False)
        self.assertEqual(body(old),body(new))
    def test_rms_arithmetic_replica(self):
        stock=(NATIVE/'vendor/layernorm_kernels.cu').read_text()
        original=stock[stock.index('  auto vec_op = [&variance]'):stock.index('\n}\n',stock.index('  auto vec_op = [&variance]'))]
        replica=(NATIVE/'dual_rms.cu').read_text()
        start=replica.index('  auto vec_op = [&variance]');end=replica.index('\n}\n',start)
        replica=replica[start:end].replace('logical_threads','blockDim.x').replace('row_idx * hidden_size','blockIdx.x * hidden_size')
        replica=replica.replace('  if (threadIdx.x < blockDim.x) {\n    vllm::vectorize_read_with_alignment<VEC_SIZE>(\n        input_row, hidden_size, threadIdx.x, blockDim.x, vec_op, scalar_op);\n  }',
            '  vllm::vectorize_read_with_alignment<VEC_SIZE>(\n      input_row, hidden_size, threadIdx.x, blockDim.x, vec_op, scalar_op);')
        self.assertEqual(replica,original)
    def test_rms_logical_thread_guard(self):
        source=(NATIVE/'dual_rms.cu').read_text()
        self.assertIn('<<<2*m,q.size(1)/VEC_SIZE,0',source)
        self.assertIn('hidden_size / VEC_SIZE',source)
        self.assertIn('if (threadIdx.x < logical_threads)',source)
        self.assertIn('.Reduce(variance, CubAddOp{}, logical_threads)',source)
        self.assertIn('float(qe),float(ke)',source)
    def test_pack_body_is_stock(self):
        stock=(ROOT/'overlay/bringup/glm_fp4_kv_kernel.py').read_text()
        start=stock.index('        d = tl.arange(0, 512)',stock.index('def _pack_store('));end=stock.index('\n        if CAP:',start)
        expected=stock[start:end].replace('rope = tl.load(R + t * RS + rd).to(tl.float32)','rope = rotated_k.to(tl.bfloat16).to(tl.float32)')
        source=(ROOT/'overlay/bringup/glm_attnfront_kernel.py').read_text()
        start=source.index('            d = tl.arange(0, 512)');end=source.index('\n\n\ndef rope_pack',start)
        actual='\n'.join(line[4:] for line in source[start:end].splitlines()).rstrip()
        self.assertEqual(actual,expected.rstrip())
    def test_rope_rounding_and_race_barriers(self):
        source=(ROOT/'overlay/bringup/glm_attnfront_kernel.py').read_text()
        self.assertIn('xr = tl.fma(x, c, -(y * s))',source)
        self.assertIn('yr = tl.fma(x, s, y * c)',source)
        self.assertEqual(source.count('tl.debug_barrier()'),2)
        self.assertIn('rotated_k.to(tl.bfloat16).to(tl.float32)',source)
        self.assertIn('enable_fp_fusion=False',source)
    def test_stock_rope_device_code_unchanged(self):
        stock=(NATIVE/'vendor/pos_encoding_kernels.cu').read_text()
        block=stock[stock.index('namespace vllm {'):stock.index('void rotary_embedding(\n')]
        self.assertIn(block,(NATIVE/'stock_rope.cu').read_text())
    def test_cache_dependency_and_conditional_update(self):
        raw=(IMAGE/'model_executor/layers/attention/mla_attention.py').read_text()
        result=a.transform_common(raw)
        self.assertEqual(result.count('if attnfront_dep is None:'),1)
        self.assertEqual(result.count('if kv_cache_dummy_dep is None:'),1)
        self.assertIn('kv_cache_dummy_dep=kv_cache_dummy_dep',result)
        front=a.transform_front((IMAGE/'model_executor/layers/mla.py').read_text())
        self.assertIn('attnfront_dep=_front_dep',front)
        self.assertLess(front.index('_af_rope_cached('),front.index('self.indexer(hidden_states'))
    def test_switch_off_startup_cannot_register(self):
        tree=ast.parse((ROOT/'overlay/bringup/sitecustomize.py').read_text())
        guards=[n for n in ast.walk(tree) if isinstance(n,ast.If) and any(isinstance(v,ast.Import) and any(alias.name=='glm_attnfront' for alias in v.names) for v in n.body)]
        self.assertEqual(len(guards),1)
        self.assertIn("'GLM_ATTN_FRONT_FUSE'",ast.unparse(guards[0].test))

class Partitions(unittest.TestCase):
    def test_two_calls_unequal_globals_all_m(self):
        import torch
        import glm_nvfp4_attn as nv
        for m in range(1,17):
            parts=[types.SimpleNamespace(weight=i,weight_scale=i+3,weight_global_scale=torch.tensor(g),workspace=i+7,
                output_size_per_partition=n,input_size_per_partition=6144) for i,(g,n) in enumerate(((.01,2048),(.07,576)))]
            layer=types.SimpleNamespace(_glm_nvfp4_parts=parts)
            x=torch.zeros((m,6144),dtype=torch.bfloat16)
            calls=[];outs=[]
            def apply(**kw):
                calls.append(kw);out=torch.full((m,kw['size_n']),float(kw['weight_global_scale']),dtype=torch.bfloat16);outs.append(out);return out
            fused=nv.apply_partitions(layer,x,apply)
            self.assertEqual(len(calls),2)
            self.assertIs(fused[0],outs[0]);self.assertIs(fused[1],outs[1])
            reference=nv.apply_layer(layer,x,apply=apply).split([2048,576],-1)
            for actual,expected in zip(fused,reference):
                self.assertTrue(torch.equal(actual.view(torch.uint8),expected.contiguous().view(torch.uint8)))
            self.assertEqual([float(c['weight_global_scale']) for c in calls[:2]],[float(p.weight_global_scale) for p in parts])
            self.assertTrue(all(c['use_fp32_reduce'] and c['bias'] is None and c['input_dtype'] is None for c in calls))
    def test_partitions_refuse_unprepared_or_dtype(self):
        import torch
        import glm_nvfp4_attn as nv
        for x in (torch.zeros(1,6144),torch.zeros(1,6144,dtype=torch.bfloat16)):
            with self.assertRaises(ValueError):nv.apply_partitions(types.SimpleNamespace(),x)

class Dry(unittest.TestCase):
    def test_switch_off_docker_vectors_equal_exact_base(self):
        import io,tarfile,tempfile
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp)/'base';base.mkdir()
            data=subprocess.check_output(['git','archive','261cbdb'],cwd=ROOT)
            with tarfile.open(fileobj=io.BytesIO(data)) as t:t.extractall(base,filter='data')
            for mode in ('int8','nvfp4'):
                cfg=Path(tmp)/(mode+'.env')
                cfg.write_text((ROOT/'tests/fixtures/legacy-dry.env').read_text()+
                    '\nGLM_MARLIN_L2HINT=off\nGLM_ATTN_FRONT_FUSE=0\nGLM_ATTN_FRONT_NORMS=bad\nGLM_ATTN_FRONT_CAPTURE_SWEEP=bad\nGLM_ATTN_WEIGHTS='+mode+'\nGLM_NVFP4_GROUPS=attn\n')
                env=dict(os.environ,DRY='1',CTN='glm53full-attnfront-dry',RECIPE_CONFIG=str(cfg),GLM_MARLIN_L2HINT='off')
                commands=[]
                for tree in (base,ROOT):
                    result=subprocess.run([str(tree/'start.sh'),'serve'],env=env,text=True,capture_output=True,check=True)
                    commands.append([line for line in result.stdout.splitlines() if line.startswith('docker run ')])
                self.assertEqual(len(commands[0]),8)
                self.assertEqual(commands[0],commands[1])

class OnDry(unittest.TestCase):
    def test_one_master_pair_on_every_serving_and_jit_vector(self):
        import tempfile,shlex,shutil
        with tempfile.TemporaryDirectory() as tmp:
            tree=Path(tmp)/'tree'
            shutil.copytree(ROOT,tree,ignore=shutil.ignore_patterns('.git','__pycache__','build','*.so'))
            # DRY only: existence fixture for the separately supplied copy guard.
            (tree/'overlay/guard/libdispram_copy_guard.so').write_bytes(b'dry-existence-fixture')
            cfg=Path(tmp)/'fixture.env'
            text=(ROOT/'tests/fixtures/legacy-dry.env').read_text()+"\nGLM_MARLIN_L2HINT=off\nGLM_ATTN_WEIGHTS=nvfp4\nGLM_NVFP4_GROUPS=attn\nGLM_KV_FORMAT=fp4x\nGLM_ATTN_FRONT_PARTS=1\nGLM_ATTN_FRONT_NORMS=1\nGLM_ATTN_FRONT_ROPE_PACK=1\nRECIPE_DISPRAM=require\n"
            vectors=[]
            for mode in ('0','1'):
                cfg.write_text(text+'GLM_ATTN_FRONT_FUSE='+mode+'\n')
                env=dict(os.environ,DRY='1',CTN='glm53full-attnfront-dry',RECIPE_CONFIG=str(cfg),GLM_MARLIN_L2HINT='off')
                r=subprocess.run([str(tree/'start.sh'),'serve'],env=env,text=True,capture_output=True,check=True)
                vectors.append([shlex.split(line) for line in r.stdout.splitlines() if line.startswith('docker run ')])
            self.assertEqual(len(vectors[1]),8)
            for off,on in zip(*vectors):
                delta={token for token in on if token.startswith('GLM_ATTN_FRONT_')}
                self.assertEqual(delta,{'GLM_ATTN_FRONT_FUSE=1','GLM_ATTN_FRONT_PARTS=1','GLM_ATTN_FRONT_NORMS=1','GLM_ATTN_FRONT_ROPE_PACK=1'})
                filtered=[];i=0
                while i<len(on):
                    if on[i]=='-e' and on[i+1].startswith('GLM_ATTN_FRONT_'):i+=2
                    else:filtered.append(on[i]);i+=1
                self.assertEqual(off,filtered)

class Flow(unittest.TestCase):
    def test_stock_and_transformed_tensor_flow_all_m_and_pieces(self):
        import torch
        class CPUCuda(torch.Tensor):
            @property
            def is_cuda(self): return True
        raw=(IMAGE/'model_executor/layers/mla.py').read_text()
        old=a.method(raw,'MultiHeadLatentAttentionWrapper','forward')
        new=a.transform_front(raw)
        for m in range(1,17):
            torch.manual_seed(m)
            h=torch.randn((m,6144),dtype=torch.bfloat16).as_subclass(CPUCuda)
            projected=torch.randn((m,2624),dtype=torch.bfloat16)
            weightq=torch.randn(2048,dtype=torch.bfloat16);weightk=torch.randn(512,dtype=torch.bfloat16)
            def norm(x,w,eps):
                f=x.float();return (f*torch.rsqrt((f*f).mean(-1,keepdim=True)+eps)*w.float()).bfloat16()
            for mask in range(8):
                observed={}
                def linear(x):observed['q_input']=x.clone();return torch.zeros((m,4096),dtype=x.dtype),None
                def attention(q,k,r,**kw):
                    observed['cache_norm']=k.clone();observed['k_rope']=r.clone();observed['dependency']=kw.get('attnfront_dep')
                    return k
                owner=types.SimpleNamespace(hidden_size=6144,q_lora_rank=2048,kv_lora_rank=512,qk_rope_head_dim=64,
                    fused_qkv_a_proj=lambda x:(projected,None),q_a_layernorm=lambda x:norm(x,weightq,1e-5),
                    kv_a_layernorm=lambda x:norm(x,weightk,1e-5),q_b_proj=linear,fuse_qkv_rmsnorm=False,
                    num_heads=16,dcp_q_replicate=False,qk_head_dim=256,qk_nope_head_dim=192,v_head_dim=256,
                    rotary_emb=lambda p,q,k:(q,k),indexer=lambda h,q,p,r:observed.update(indexer_q=q.clone()),
                    is_sparse=True,skip_topk=False,indexer_rope_emb=None,mla_attn=attention,g_proj=None,o_proj=lambda x:(x,None))
                ns={'torch':torch};fns={};exec(compile(old,'stock','exec'),ns,fns)
                expected=fns['forward'](owner,torch.arange(m),h).clone();baseline=dict(observed);observed.clear()
                ns.update(_af_norm_eligible=lambda owner,x:bool(mask&2),
                    _af_partitions=lambda owner,x: projected.split([2048,576],-1) if not mask&1 else (projected[:,:2048].contiguous(),projected[:,2048:].contiguous()),
                    _af_norms=lambda owner,q,k:(norm(q,weightq,1e-5),norm(k,weightk,1e-5)),
                    _af_rope=lambda owner,p,q,k,x:(q,k,torch.empty(0) if mask&4 else None))
                owner._glm_attnfront_stock=lambda *args: self.fail('cached path unexpectedly fell back')
                owner._glm_attnfront_pieces=tuple(bool(mask&(1<<i)) for i in range(3))
                ns.update(_af_bf16=torch.bfloat16,_af_debug=False,_af_capturing=lambda:False,
                    _af_record=lambda *args:self.fail('eager bookkeeping'),
                    _af_apply_parts=lambda layer,x:(projected[:,:2048].contiguous(),projected[:,2048:].contiguous()),
                    _af_rope_cached=lambda owner,p,q,k,x:(q,k,torch.empty(0)))
                exec(compile(new,'fused','exec'),ns,fns)
                result=fns['forward'](owner,torch.arange(m),h)
                self.assertTrue(torch.equal(result.view(torch.uint8),expected.view(torch.uint8)))
                for key in ('q_input','cache_norm','k_rope','indexer_q'):
                    self.assertTrue(torch.equal(observed[key].contiguous().view(torch.uint8),baseline[key].contiguous().view(torch.uint8)),key)
                self.assertEqual(observed['dependency'] is not None,bool(mask&4))

if __name__=='__main__':unittest.main()
