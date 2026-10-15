"""Overlay TP admission: which runtime hooks accept TP8 and which stay TP4-only."""
import ast
import hashlib
from pathlib import Path
import sys
import types
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'overlay/bringup'), str(ROOT / 'overlay/overlay')]
import glm_dsa_short
import glm_moe_det
import glm_mtp_fix

GLM = types.SimpleNamespace(architectures=['GlmMoeDsaForCausalLM'], index_topk=2048, index_head_dim=128,
                            index_n_heads=32, qk_rope_head_dim=64)


def parallel(tp, dcp=1, pcp=1):
    return types.SimpleNamespace(tensor_parallel_size=tp, decode_context_parallel_size=dcp,
                                 prefill_context_parallel_size=pcp)


class Gates(unittest.TestCase):
    def test_dsa_shortcut_admits_tp8_callers_keep_tp4(self):
        self.assertTrue(glm_dsa_short.supported(GLM, parallel(4)))
        self.assertFalse(glm_dsa_short.supported(GLM, parallel(8)))
        self.assertTrue(glm_dsa_short.supported(GLM, parallel(8), (4, 8)))
        self.assertFalse(glm_dsa_short.supported(GLM, parallel(8, dcp=2), (4, 8)))
        self.assertFalse(glm_dsa_short.supported(GLM, parallel(16), (4, 8)))
        source = (ROOT / 'overlay/bringup/glm_dsa_short.py').read_text()
        self.assertIn('supported(self.model_config.hf_config, self.parallel_config, (4, 8))', source)
        for name in ('glm_indexer_row_shard.py', 'glm_indexer_stable_topk.py'):
            text = (ROOT / 'overlay/bringup' / name).read_text()
            self.assertIn('supported(config.model_config.hf_config, config.parallel_config)', text, name)

    def test_draft_tp_follows_target(self):
        self.assertEqual(glm_mtp_fix.speculative_config()['draft_tensor_parallel_size'], 4)
        self.assertEqual(glm_mtp_fix.speculative_config(tp=8)['draft_tensor_parallel_size'], 8)

    def test_moe_det_pins_the_edited_mtp_source(self):
        digest = hashlib.sha256((ROOT / 'overlay/overlay/glm_nvfp4_mtp.py').read_bytes()).hexdigest()
        self.assertEqual(glm_moe_det.PINS[glm_moe_det.MTP], digest)

    def test_mtp_experts_and_kstop_admit_tp4_and_tp8_only(self):
        mtp = (ROOT / 'overlay/overlay/glm_nvfp4_mtp.py').read_text()
        self.assertIn('moe.tp_size not in (4, 8) or moe.ep_size != 1', mtp)
        tree = ast.parse((ROOT / 'overlay/kstop/kstop_runtime.py').read_text())
        checks = [n for n in ast.walk(tree) if isinstance(n, ast.Compare)
                  and isinstance(n.left, ast.Attribute) and n.left.attr == 'tensor_parallel_size']
        self.assertEqual(len(checks), 1)
        self.assertIsInstance(checks[0].ops[0], ast.NotIn)
        self.assertEqual(ast.literal_eval(checks[0].comparators[0]), (4, 8))


if __name__ == '__main__':
    unittest.main(verbosity=2)
