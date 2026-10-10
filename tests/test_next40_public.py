"""CPU contracts for the statically enabled, measured next40 bundle."""
import os
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
os.environ.update(RECIPE_ROOT=str(ROOT), RECIPE_HOSTS='s1 s2 s3 s4',
                  RECIPE_IPS='192.0.2.1 192.0.2.2 192.0.2.3 192.0.2.4',
                  IMAGE='image:local', MODEL_DIR='/models/target', DRAFT_DIR='/models/draft',
                  NCCL_HOST_DIR='/nccl', OVERLAY_REMOTE='/runtime', FABRIC_IFACE='eth1',
                  IB_HCA='hca0,hca1')
sys.path[:0] = [str(ROOT/'scripts'), str(ROOT/'overlay/bringup'), str(ROOT/'overlay/kstop')]
import cluster as C
import glm_screen_failsafe as fs
import glm_draft_head as dh
import kstop_guard_fuse as guard
import kstop_runtime as rt


class PublicVector(unittest.TestCase):
    def test_start_profile_selects_static_measured_state(self):
        profile = (ROOT/'profiles/current.env').read_text()
        for line in ("GLM_SCREEN_FAILSAFE='1'", "GLM_GREEDY_SAMPLE_FUSE='1'",
                     "GLM_KSTOP_GUARD_FUSE='1'", "GLM_SKIP_MLA_PLAN='1'",
                     "GLM_SKIP_MLA_PLAN_KSTOP='1'", "export GLM_GLUE_DSA_IDX_CACHE='1'",
                     "export GLM_GLUE_IDX_EXPECT='57'"):
            self.assertIn(line, profile)
        self.assertNotIn('GLM_SKIP_MLA_PLAN_AB_INIT=', profile)
        self.assertNotIn('GLM_SKIP_MLA_PLAN_KSTOP_AB_INIT=', profile)

    def test_launcher_validates_and_exports_bundle(self):
        base = dict(C.ENV, GLM_SCREEN_FAILSAFE='1', GLM_GREEDY_SAMPLE_FUSE='1',
                    GLM_KSTOP_GUARD_FUSE='1', GLM_SKIP_MLA_PLAN='1',
                    GLM_SKIP_MLA_PLAN_KSTOP='1', GLM_FULL_MLA='triton',
                    GLM_MTP_KSTOP='1', GLM_SPEC_SAMPLE='0')
        with patch.dict(C.ENV, base, clear=True):
            out = C.optional_env()
        for key in ('GLM_SCREEN_FAILSAFE', 'GLM_GREEDY_SAMPLE_FUSE',
                    'GLM_KSTOP_GUARD_FUSE', 'GLM_SKIP_MLA_PLAN',
                    'GLM_SKIP_MLA_PLAN_KSTOP'):
            self.assertEqual(out[key], '1')

    def test_launcher_refuses_invalid_compositions(self):
        cases = (
            dict(GLM_GREEDY_SAMPLE_FUSE='yes'),
            dict(GLM_GREEDY_SAMPLE_FUSE='1', GLM_MTP_KSTOP='0'),
            dict(GLM_SKIP_MLA_PLAN='0', GLM_SKIP_MLA_PLAN_KSTOP='1'),
            dict(GLM_KSTOP_GUARD_FUSE='1', GLM_MTP_KSTOP_CONTROL_MODE='broadcast'),
        )
        for delta in cases:
            env = dict(C.ENV)
            env.update(GLM_SCREEN_FAILSAFE='0', GLM_GREEDY_SAMPLE_FUSE='0',
                       GLM_KSTOP_GUARD_FUSE='0', GLM_SKIP_MLA_PLAN='0',
                       GLM_SKIP_MLA_PLAN_KSTOP='0')
            env.update(delta)
            with self.subTest(delta=delta), patch.dict(C.ENV, env, clear=True):
                with self.assertRaises(ValueError):
                    C.optional_env()


class Failsafe(unittest.TestCase):
    def setUp(self):
        self.registry = patch.dict(fs.REGISTRY, {}, clear=True)
        self.registry.start()
        self.addCleanup(self.registry.stop)

    @staticmethod
    def refuse(payload, valid):
        if not valid:
            raise dh.VoteRefused(dict(votes=[], payload_mismatch=False))

    def test_failed_cold_contract_rolls_back_and_latches(self):
        state = NS(stock=True)
        journal = fs.Journal()
        journal.save(state, 'stock')
        journal.save(state, 'armed')
        item = fs.register('glue-F3', journal.restore, {'GLM_SCREEN_FAILSAFE': '1'})

        def fail():
            state.stock = False
            state.armed = True
            journal.mark()
            raise RuntimeError('simulated F3 refusal')

        with patch.object(dh, 'gather_receipts', side_effect=lambda row: [row]):
            self.assertIsNone(item.run(fail, self.refuse))
        self.assertTrue(state.stock)
        self.assertFalse(hasattr(state, 'armed'))
        self.assertIn('simulated F3 refusal', item.reason)


class GreedyFuse(unittest.TestCase):
    def test_exact_tokens_and_confidence(self):
        class Speculator:
            def __init__(self, *args): pass
            def propose(self, *args): pass
            def _configure_fused_multi_step_decode(self): pass

        rt.install_mtp(Speculator)
        logits = torch.tensor([[3., 1., 0.], [2., 2., 0.], [-float('inf'), 4., 1.]],
                              dtype=torch.bfloat16)
        owner = NS(model=NS(compute_logits=lambda hidden: logits), use_fp64_gumbel=False,
                   max_num_reqs=3, device='cpu',
                   speculative_config=NS(draft_sample_method='greedy'))
        with patch.dict(os.environ, GLM_GREEDY_SAMPLE_FUSE='1'):
            owner._kstop = rt.Runtime(owner)
        got = Speculator.sample_draft(owner, None, None, None, None, None, None, None)
        z = logits.float()
        self.assertTrue(torch.equal(got, logits.argmax(-1)))
        self.assertTrue(torch.equal(owner._kstop.confidence,
                                    (z.amax(-1)-z.logsumexp(-1)).exp()))


class GuardFuse(unittest.TestCase):
    def test_reference_packet_matches_legacy_for_all_widths(self):
        words = torch.tensor([17, 23, 1], dtype=torch.int64)
        confidence = torch.tensor([-.0, float('nan'), -.75, .5])
        for n in range(5):
            conf = torch.zeros(4, dtype=torch.int64)
            if n:
                conf[:n] = confidence[:n].view(torch.int32).to(torch.int64)
            legacy = torch.cat((words[:2], conf, words[2:], -words[:2], -conf))
            self.assertTrue(torch.equal(guard.reference_packet(words, confidence, n), legacy))
            self.assertFalse(bool(guard.reference_verdict(legacy)))


if __name__ == '__main__':
    unittest.main(verbosity=2)
