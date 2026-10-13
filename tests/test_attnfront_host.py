# SPDX-License-Identifier: Apache-2.0
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('attnfront_host',ROOT/'bench/attnfront_host.py')
h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
a=h.current


class CachedForward(unittest.TestCase):
    def setUp(self):
        import os
        image=Path(os.environ.get('GLM_IMAGE_SRC','/tmp/glm-adaptive-image'))
        self.raw=(image/'vllm/model_executor/layers/mla.py').read_text()
        self.stub=h.stubs()
        self.ctx=patch.dict(sys.modules,self.stub);self.ctx.start();self.addCleanup(self.ctx.stop)
        for name,value in (('CONFIG',(True,True,True)),('_RMS_PROVIDERS',{})):
            ctx=patch.object(a,name,value);ctx.start();self.addCleanup(ctx.stop)
        self.stock,self.fast=h.functions(a,self.raw,self.stub,True)

    def owner(self):
        o=h.fixture();a.bind_forward(type(o),self.stock,self.fast);return o

    def call(self,o,m=4): return o.forward(h.Tensor((m,)),h.Tensor((m,6144)))

    def test_static_once_across_cached_calls_and_rows(self):
        o=self.owner()
        with patch.object(a,'static_pieces',wraps=a.static_pieces) as static:
            self.call(o)
            self.assertEqual(o._glm_attnfront_pieces,(True,True,True))
            self.assertIs(o.forward.__func__,self.fast)
            for m in (1,4,16,64,2048): self.call(o,m)
            self.assertEqual(static.call_count,1)
        with patch.object(a,'static_pieces',side_effect=AssertionError),patch.object(a,'owner_shape',side_effect=AssertionError),patch.object(a,'norm_eligible',side_effect=AssertionError),patch.object(a,'rope_eligible',side_effect=AssertionError):
            self.call(o)

    def test_large_m_routes_stock_before_capture_or_predicates(self):
        o=self.owner();self.call(o)
        calls=[];o._glm_attnfront_stock=lambda *args:calls.append(args) or 'stock'
        with patch.dict(self.fast.__globals__,_af_capturing=lambda: (_ for _ in ()).throw(AssertionError('large-M capture query')),
                        _af_record=lambda *args: self.fail('large-M bookkeeping')):
            for m in (0,17,64,2048): self.assertEqual(self.call(o,m),'stock')
        self.assertEqual(len(calls),4)

    def test_layer_zero_installs_original_without_repeated_checks(self):
        o=self.owner();o.fused_qkv_a_proj.scheme=None
        self.call(o)
        self.assertIs(o.forward.__func__,self.stock)
        self.assertEqual(o._glm_attnfront_pieces,(False,False,False))
        with patch.object(a,'static_pieces',side_effect=AssertionError),patch.object(a,'hit',side_effect=AssertionError):
            for m in (1,4,64,2048): self.call(o,m)

    def test_ineligible_shape_and_disabled_pieces_install_stock(self):
        for shape in (False,True):
            o=self.owner()
            if shape: o.hidden_size=4096
            else: a.CONFIG=(False,False,False)
            self.call(o);self.assertIs(o.forward.__func__,self.stock)
            a.CONFIG=(True,True,True)

    def test_rms_provider_once_each_supported_signature_across_owners(self):
        d=self.stub['vllm'].ir.ops.rms_norm
        with patch.object(d,'dispatch',wraps=d.dispatch) as dispatch:
            o=self.owner();self.call(o)
            self.assertEqual(dispatch.call_count,32)
            for m in (1,4,16): self.call(o,m)
            self.call(self.owner())
            self.assertEqual(dispatch.call_count,32)
            a.CONFIG=(False,True,True)
            self.call(self.owner())
            self.assertEqual(dispatch.call_count,64)
            self.assertEqual({args[0].shape[-1] for args,_ in dispatch.call_args_list},{2048,512})

    def test_provider_rejection_cached_and_independent_of_parts_rope(self):
        d=self.stub['vllm'].ir.ops.rms_norm
        with patch.object(d,'dispatch',return_value=types.SimpleNamespace(provider='native')) as dispatch:
            o=self.owner();self.call(o)
            self.assertEqual(o._glm_attnfront_pieces,(True,False,True))
            self.call(o);self.call(self.owner())
            self.assertEqual(dispatch.call_count,1)

    def test_eager_never_records_and_capture_keeps_hit_lines(self):
        o=self.owner()
        with patch.object(a,'hit') as hit,patch.dict(a.COUNTS,{k:0 for k in a.COUNTS}):
            from unittest.mock import Mock
            capturing=Mock(return_value=False)
            with patch.dict(self.fast.__globals__,_af_capturing=capturing):
                for _ in range(3): self.call(o)
                self.assertEqual(capturing.call_count,3)
            hit.assert_not_called();self.assertFalse(any(a.COUNTS.values()))
            stub=h.stubs(capturing=True)
            _,capture=h.functions(a,self.raw,stub,True)
            capture(o,h.Tensor((4,)),h.Tensor((4,6144)))
            self.assertEqual([c.args[0] for c in hit.call_args_list],['parts','norms','rope_pack'])
            hit.reset_mock()
            ns=capture.__globals__;ns['_af_debug']=True;ns['_af_capturing']=lambda:False
            capture(o,h.Tensor((4,)),h.Tensor((4,6144)))
            self.assertEqual(hit.call_count,3)

    def test_dtype_device_mismatch_bypass(self):
        o=self.owner();self.call(o)
        o._glm_attnfront_stock=lambda *args:'stock'
        for attr,value in (('dtype','torch.float32'),('is_cuda',False)):
            x=h.Tensor((4,6144));setattr(x,attr,value)
            self.assertEqual(o.forward(None,x),'stock')

    def test_int8_mtp_preserves_independent_norm_and_rope(self):
        o=self.owner();o.fused_qkv_a_proj.scheme=object()
        del o.fused_qkv_a_proj._glm_nvfp4_parts
        self.call(o);self.assertEqual(o._glm_attnfront_pieces,(False,True,True))

    def test_static_matches_previous_predicates_all_rows_layouts_and_widths(self):
        for rank in (1536,2048):
            for parts in (False,True):
                o=self.owner();o.q_lora_rank=rank
                o.q_a_layernorm=h.RMSNorm(rank)
                o.fused_qkv_a_proj._glm_nvfp4_parts[0].output_size_per_partition=rank
                a.CONFIG=(parts,True,True)
                cached=a.static_pieces(o,h.Tensor((4,6144)))
                for m in range(1,17):
                    x=h.Tensor((m,6144));q=h.Tensor((m,16,64),(4096,256,1))
                    k=h.Tensor((m,1,64),(576 if parts else rank+576,64,1))
                    with patch.object(a,'hit'):
                        expected=(a.parts_eligible(o,x),a.norm_eligible(o,x),
                            a.rope_eligible(o,q,k,h.Tensor((m,512))))
                    self.assertEqual(cached,expected)

    def test_rotary_static_dispatch_rejection_keeps_other_pieces(self):
        for attr,value in (('use_flashinfer',True),('rotary_dim',32),('_forward_method',lambda:None)):
            o=self.owner();setattr(o.rotary_emb,attr,value)
            self.call(o);self.assertEqual(o._glm_attnfront_pieces,(True,True,False))

    def test_local_host_and_unchanged_kernel_pins(self):
        import hashlib,json
        parent=ROOT/'overlay/bringup'
        pins=json.loads((parent/'attnfront_local_pins.json').read_text())
        self.assertIn('glm_attnfront.py',pins)
        for name,digest in pins.items():
            self.assertEqual(hashlib.sha256((parent/name).read_bytes()).hexdigest(),digest)

    def test_state_dict_post_hook_invalidates_parent_recursive_reload(self):
        o=self.owner();hooks=[]
        o.register_load_state_dict_post_hook=lambda fn:hooks.append(fn)
        self.call(o);self.assertEqual(len(hooks),1)
        o.fused_qkv_a_proj.scheme=None
        hooks[0](o,None)
        self.assertNotIn('forward',o.__dict__)
        self.call(o);self.assertIs(o.forward.__func__,self.stock)
        self.assertEqual(len(hooks),1)


if __name__=='__main__': unittest.main()
