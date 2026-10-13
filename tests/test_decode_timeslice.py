# SPDX-License-Identifier: Apache-2.0
"""Time-slice policy through the complete pinned scheduler, CPU only."""
import copy
import importlib.util
import json
import math
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import test_adaptive_chunk as T
from test_adaptive_chunk import A, W, request, scheduler, Pause
import glm_decode_fair as F


class Timeslice(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        T.PinnedSchedule.setUpClass()
        cls.source = T.PinnedSchedule.source
        ns = dict(T.PinnedSchedule.namespace, _glm_budget_factory=lambda s:
                  F.step_budget(s, A.THRESHOLD, s.test_chunk, s.test_decode_steps,
                                   getattr(s, 'test_rearm', 'legacy')))
        exec('from __future__ import annotations\n' + A.transform(cls.source, True), ns)
        cls.schedule = staticmethod(ns['schedule'])

    def test_boot_n40_without_sidecar_through_real_installer(self):
        with tempfile.TemporaryDirectory() as tmp:
            mod = NS(__file__=str(Path(tmp)/'scheduler.py'), Scheduler=NS(schedule=None))
            Path(mod.__file__).write_text(self.source)
            mod.__dict__.update(T.PinnedSchedule.namespace)
            mod.Scheduler = NS(schedule=None)
            env = dict(GLM_KV_FORMAT='fp4x', GLM_W2_PREFILL_CONTROL='/cache/prefill.json',
                       GLM_DECODE_FAIR='1', GLM_DECODE_FAIR_CHUNK='4096',
                       GLM_DECODE_FAIR_DECODE_STEPS='40')
            self.assertTrue(A.install(mod, env))
            s = self.make(n=40)
            def emit():
                out = mod.Scheduler.schedule(s)
                for d in s.running:
                    if d.request_id == 'd':
                        d.num_tokens = d.num_computed_tokens + 1
                        d.num_tokens_with_spec = d.num_tokens + 2
                        d.spec_token_ids = [1, 2]
                s.current_step += 1
                return out
            self.assertIn('p', emit().num_scheduled_tokens)
            for _ in range(40):
                self.assertEqual(emit().num_scheduled_tokens, {'d': 3})
            self.assertIn('p', emit().num_scheduled_tokens)
            self.assertFalse(hasattr(s, '_glm_decode_fair_control'))
            s.running = [r for r in s.running if r.request_id == 'p']
            self.assertEqual(emit().num_scheduled_tokens['p'], 4096)
            for bad in ('-1', 'bad', '1.5'):
                with self.assertRaises(ValueError):
                    F.settings(dict(env, GLM_DECODE_FAIR_DECODE_STEPS=bad))

    def make(self, chunk=4096, n=20, reverse=False, waiting=False, slots=0):
        p, d = request('p', 100048, 4096), request('d', 60043, 60043, decode=True)
        s = scheduler(running=[d] if waiting else ([d, p] if reverse else [p, d]),
                      waiting=[request('p', 100048)] if waiting else [], draft_slots=slots)
        s.test_chunk, s.test_decode_steps = chunk, n
        return s

    def step(self, s):
        out = self.schedule(s)
        for d in s.running:
            if d.request_id == 'd':
                d.num_tokens = d.num_computed_tokens + 1
                d.num_tokens_with_spec = d.num_tokens + 2
                d.spec_token_ids = [1, 2]
        s.current_step += 1
        return out

    def test_exact_n_pure_steps_both_orders_waiting_and_draft_reservations(self):
        for chunk, n in ((4096, 20), (4096, 40), (2048, 10)):
            for reverse, waiting, slots in ((False, False, 0), (True, False, 0),
                                            (False, True, 0), (False, False, 2)):
                s = self.make(chunk, n, reverse, waiting, slots)
                for cycle in range(3):
                    out = self.step(s)
                    self.assertEqual(out.num_scheduled_tokens['d'], 3)
                    self.assertEqual(out.num_scheduled_tokens['p'], min(chunk, 4096-3-2*slots))
                    p = next(r for r in s.running if r.request_id == 'p')
                    computed = p.num_computed_tokens
                    for _ in range(n):
                        out = self.step(s)
                        self.assertEqual(out.num_scheduled_tokens, {'d': 3})
                        self.assertEqual(out.scheduled_spec_decode_tokens, {'d': [1, 2]})
                        self.assertEqual(p.num_computed_tokens, computed)
                        self.assertIn(p, s.running)
                self.assertTrue(all(tokens > 0 for _, tokens, _ in s.allocations))

    def test_no_eligible_decoder_restores_adaptive_immediately(self):
        for mode in ('removed', 'async', 'complete', 'no_work'):
            s = self.make()
            self.step(s)
            d = s.running[-1]
            if mode == 'removed': s.running.remove(d)
            if mode == 'async': d.next_decode_eligible_step = s.current_step + 3
            if mode == 'complete':
                d.num_computed_tokens = d.num_prompt_tokens + d.max_tokens + 1
                d.num_output_placeholders = 1
            if mode == 'no_work': d.num_tokens_with_spec = d.num_computed_tokens
            self.assertEqual(self.step(s).num_scheduled_tokens['p'], 4096)
            self.assertEqual(s._glm_df_since_prefill, 20)

    def test_pause_and_kv_refusal_do_not_consume_floor(self):
        s = self.make(waiting=True)
        allocate = s.kv_cache_manager.allocate_slots
        s.kv_cache_manager.allocate_slots = lambda r, *a, **kw: None if r.request_id == 'p' else allocate(r, *a, **kw)
        self.assertEqual(self.step(s).num_scheduled_tokens, {'d': 3})
        self.assertEqual(s._glm_df_since_prefill, 20)
        s.kv_cache_manager.allocate_slots = allocate
        self.assertIn('p', self.step(s).num_scheduled_tokens)
        s._pause_state = Pause.PAUSED_ALL
        self.assertEqual(self.step(s).num_scheduled_tokens, {})
        self.assertEqual(s._glm_df_since_prefill, 0)
        s._pause_state = Pause.UNPAUSED
        self.assertEqual(self.step(s).num_scheduled_tokens, {'d': 3})
        self.assertEqual(s._glm_df_since_prefill, 1)

    def test_aggregate_prefills_and_preempted_emitted_map(self):
        s = self.make(2048, 10)
        s.running.insert(1, request('q', 100048, 4096))
        out = self.step(s)
        self.assertEqual(sum(v for k,v in out.num_scheduled_tokens.items() if k != 'd'), 2048)
        for _ in range(10): self.assertEqual(self.step(s).num_scheduled_tokens, {'d': 3})
        s = self.make()
        budget = F.FairStepBudget(s, A.THRESHOLD, 4096, 20)
        # Native preemption removed the prefill before output construction.
        budget.finish({'d': 3})
        self.assertEqual(s._glm_df_since_prefill, 20)
        s._glm_df_since_prefill = 0
        budget = F.FairStepBudget(s, A.THRESHOLD, 4096, 20)
        budget.finish({})
        self.assertEqual(s._glm_df_since_prefill, 0)

    def test_c1_boot_off_and_n_zero_match_existing_schedule(self):
        for chunk, n in ((0, 40), (4096, 40), (2048, 10)):
            for rows, wait, cap in (([request('d', 100, 100, decode=True)], [], 2048),
                                    ([], [request('p', 100048)], 2048),
                                    ([request('d', 100, 100, decode=True)], [request('p', 100048)], 512)):
                left = scheduler(running=copy.deepcopy(rows), waiting=copy.deepcopy(wait), cap=cap)
                right = scheduler(running=copy.deepcopy(rows), waiting=copy.deepcopy(wait), cap=cap)
                right.test_chunk, right.test_decode_steps = chunk, n
                self.assertEqual(vars(T.PinnedSchedule.adaptive(left)), vars(self.schedule(right)))
                self.assertEqual(left.allocations, right.allocations)
        for chunk in F.CHUNKS:
            s = self.make(chunk, 0)
            for _ in range(5): self.assertIn('p', self.step(s).num_scheduled_tokens)

    def test_rank_invariance_multistep_and_no_eligibility_credit(self):
        traces = []
        for rank in range(4):
            s = self.make()
            s.rank_local_mem_available = rank * 100
            trace = []
            for i in range(50):
                if i == 2:
                    s.running[-1].next_decode_eligible_step = s.current_step + 1
                    s._pause_state = Pause.PAUSED_ALL
                if i == 3: s._pause_state = Pause.UNPAUSED
                before = getattr(s, '_glm_df_since_prefill', 20)
                out = self.step(s)
                if i == 2: self.assertEqual(s._glm_df_since_prefill, before)
                trace.append(vars(out))
            traces.append(trace)
        self.assertTrue(all(t == traces[0] for t in traces))

    def test_real_installer_schema2_switch_off_on_n_change_and_legacy(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'fair.json'
            module = NS(__file__=str(T.PinnedSchedule.path),
                        Scheduler=type('Scheduler', (), {}), **T.PinnedSchedule.namespace)
            env = dict(GLM_KV_FORMAT='fp4x', GLM_W2_PREFILL_CONTROL='/cache/prefill.json',
                       GLM_DECODE_FAIR='1', GLM_DECODE_FAIR_CONTROL=str(path))
            with patch.dict(os.environ, env): A.install(module)
            s = self.make()
            for seq, (chunk, n) in enumerate(((4096, 20), (4096, 40), (0, 40), (2048, 10))):
                path.write_text(json.dumps(dict(schema=2, chunk=chunk, decode_steps=n, sequence=seq)))
                out = module.Scheduler.schedule(s)
                self.assertIn('p', out.num_scheduled_tokens)
                self.assertEqual(s._glm_df_since_prefill, 0 if chunk else n)
                d = s.running[-1]
                d.num_tokens = d.num_computed_tokens + 1
                d.num_tokens_with_spec = d.num_tokens + 2
                d.spec_token_ids = [1, 2]
                if chunk:
                    self.assertEqual(module.Scheduler.schedule(s).num_scheduled_tokens, {'d': 3})
                    self.assertEqual(s._glm_df_since_prefill, 1)
                    d.num_tokens = d.num_computed_tokens + 1
                    d.num_tokens_with_spec = d.num_tokens + 2
                    d.spec_token_ids = [1, 2]
            path.write_text(json.dumps(dict(schema=1, chunk=1024, sequence=4)))
            self.assertEqual(module.Scheduler.schedule(s).num_scheduled_tokens['p'], 1024)
            self.assertEqual(s._glm_df_since_prefill, 0)

    def staged(self, mode, residual=256):
        s = scheduler(running=[request('d', 60000, 60000, decode=True)],
                      waiting=[request('p', 60000 + residual, hits=60000)], draft_slots=0)
        s.test_chunk, s.test_decode_steps, s.test_rearm = 4096, 40, mode
        if residual == 1: s.num_spec_tokens = 0  # fixture has no worker to roll back padded draft rows
        self.assertEqual(self.step(s).num_scheduled_tokens, {'d': 3, 'p': min(residual, 4093)})
        # Arrival AFTER the mixed step: q cannot share p's spare chunk budget.
        s.waiting.append(request('q', 80000 + 256, hits=80000))
        return s

    def test_staged_cached_completion_admits_second_prefill_earlier(self):
        for mode, expected in (('legacy', 40), ('complete', 0), ('scaled', 3)):
            s = self.staged(mode)
            p = next(r for r in s.running if r.request_id == 'p')
            self.assertEqual(p.num_computed_tokens, p.num_prompt_tokens)
            for _ in range(expected):
                self.assertEqual(self.step(s).num_scheduled_tokens, {'d': 3})
                self.assertEqual([r.request_id for r in s.waiting], ['q'])
            self.assertEqual(self.step(s).num_scheduled_tokens, {'d': 3, 'q': 256})
            self.assertFalse(s.waiting)

    def test_long_chunk_protection_same_n40_in_all_modes_and_orders(self):
        for mode in F.REARM:
            for reverse, waiting, slots in ((False, False, 0), (True, False, 0),
                                            (False, True, 0), (False, False, 2)):
                s = self.make(n=40, reverse=reverse, waiting=waiting, slots=slots)
                s.test_rearm = mode
                for _ in range(2):
                    self.assertIn('p', self.step(s).num_scheduled_tokens)
                    if not s.waiting: s.waiting.append(request('q', 80256, hits=80000))
                    # 4089..4093 tokens still ceil to 40 in scaled mode.
                    for _ in range(40):
                        self.assertEqual(self.step(s).num_scheduled_tokens, {'d': 3})

    def test_completed_and_unfinished_in_same_map_still_arm_complete(self):
        s = self.make(n=40, slots=0)
        s.running.insert(0, request('short', 60256, 60000))
        s.test_rearm = 'complete'
        out = self.step(s)
        self.assertEqual(out.num_scheduled_tokens, {'short': 256, 'p': 3837, 'd': 3})
        self.assertEqual(s._glm_df_since_prefill, 0)
        for _ in range(40):
            self.assertEqual(self.step(s).num_scheduled_tokens, {'d': 3})

    def test_scaled_counts_aggregate_emitted_tokens_and_rounds_up(self):
        for residual in (1, 256, 3000, 4093):
            s = self.staged('scaled', residual)
            hold = math.ceil(40 * residual / 4096)
            self.assertEqual(s._glm_df_since_prefill, 40 - hold)
            for _ in range(hold): self.assertEqual(self.step(s).num_scheduled_tokens, {'d': 3})
            self.assertIn('q', self.step(s).num_scheduled_tokens)
        s = self.make(n=40, slots=0)
        s.test_rearm = 'scaled'
        s.running = [request('a', 60200, 60000), request('b', 80312, 80000), s.running[-1]]
        self.assertEqual(self.step(s).num_scheduled_tokens, {'a': 200, 'b': 312, 'd': 3})
        self.assertEqual(s._glm_df_since_prefill, 35)  # ceil(40 * 512 / 4096)

    def test_variant_pause_refusal_preemption_no_synthetic_credit(self):
        for mode in ('complete', 'scaled'):
            s = self.make(n=40, waiting=True, slots=0)
            s.test_rearm = mode
            alloc = s.kv_cache_manager.allocate_slots
            s.kv_cache_manager.allocate_slots = lambda r, *a, **kw: None if r.request_id == 'p' else alloc(r, *a, **kw)
            self.assertEqual(self.step(s).num_scheduled_tokens, {'d': 3})
            self.assertEqual(s._glm_df_since_prefill, 40)
            s.kv_cache_manager.allocate_slots = alloc
            self.assertIn('p', self.step(s).num_scheduled_tokens)
            s._pause_state = Pause.PAUSED_ALL
            self.assertEqual(self.step(s).num_scheduled_tokens, {})
            self.assertEqual(s._glm_df_since_prefill, 0)
            # An attempted short prefill was removed by native preemption.
            s._pause_state = Pause.UNPAUSED
            short = request('short', 256)
            budget = F.step_budget(s, A.THRESHOLD, 4096, 40, mode)
            budget.request(short, 0, 4096, 4096, {})
            budget.finish({})
            self.assertEqual(s._glm_df_since_prefill, 0)
            budget.finish({'d': 3})
            self.assertEqual(s._glm_df_since_prefill, 1)

    def test_boot_variants_without_control_use_real_installer(self):
        for mode, expected in (('complete', 40), ('scaled', 37)):
            module = NS(__file__=str(T.PinnedSchedule.path),
                        Scheduler=type('Scheduler', (), {}), **T.PinnedSchedule.namespace)
            env = dict(GLM_KV_FORMAT='fp4x', GLM_W2_PREFILL_CONTROL='/cache/prefill.json',
                       GLM_DECODE_FAIR='1', GLM_DECODE_FAIR_REARM=mode,
                       GLM_DECODE_FAIR_DECODE_STEPS='40')
            A.install(module, env)
            s = scheduler(running=[request('d', 60000, 60000, decode=True)],
                          waiting=[request('p', 60256, hits=60000)], draft_slots=0)
            self.assertEqual(module.Scheduler.schedule(s).num_scheduled_tokens, {'d': 3, 'p': 256})
            self.assertEqual(s._glm_df_since_prefill, expected)
            self.assertFalse(hasattr(s, '_glm_decode_fair_control'))

    def test_variant_rank_invariance_staged_multi_step(self):
        for mode in ('complete', 'scaled'):
            traces = []
            for rank in range(4):
                s = self.staged(mode)
                s.rank_local_mem_available = rank * 100
                traces.append([vars(self.step(s)) for _ in range(45)])
            self.assertTrue(all(trace == traces[0] for trace in traces))

    def test_rearm_runtime_ab_and_missing_field_legacy_with_boot_variant(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'fair.json'
            module = NS(__file__=str(T.PinnedSchedule.path),
                        Scheduler=type('Scheduler', (), {}), **T.PinnedSchedule.namespace)
            env = dict(GLM_KV_FORMAT='fp4x', GLM_W2_PREFILL_CONTROL='/cache/prefill.json',
                       GLM_DECODE_FAIR='1', GLM_DECODE_FAIR_CONTROL=str(path),
                       GLM_DECODE_FAIR_REARM='complete', GLM_DECODE_FAIR_DECODE_STEPS='40')
            A.install(module, env)
            s = scheduler(draft_slots=0)
            for seq, mode in enumerate(('legacy', 'complete', 'scaled', None, 'complete')):
                control = dict(schema=2, chunk=4096, decode_steps=40, sequence=seq)
                if mode is not None: control['rearm'] = mode
                path.write_text(json.dumps(control))
                s.running = [request('d', 60000, 60000, decode=True)]
                s.waiting = T.Queue([request('p', 60256, hits=60000)])
                self.assertEqual(module.Scheduler.schedule(s).num_scheduled_tokens, {'d': 3, 'p': 256})
                expected = {'legacy': 0, 'complete': 40, 'scaled': 37, None: 0}[mode]
                self.assertEqual(s._glm_df_since_prefill, expected)
                # Unchanged controls MUST NOT reset a hold.
                self.assertEqual(F.control_policy(s, str(path), 512), (4096, 40))
                self.assertEqual(s._glm_df_since_prefill, expected)
            path.write_text(json.dumps(dict(control, rearm='scaled')))
            with self.assertRaisesRegex(RuntimeError, 'stale'):
                module.Scheduler.schedule(s)

    def test_simulated_frontier_matches_cost_model(self):
        rows = []
        for chunk, n, expected_a, expected_b in ((4096, 20, 7.1, 183),
                                                (4096, 40, 10.9, 231), (2048, 10, 7.0, 190)):
            s = self.make(chunk, n)
            p = s.running[0]
            p.num_computed_tokens = 0
            elapsed, committed, mixed, pure = 0.0, 0.0, 0, 0
            while p.num_computed_tokens < p.num_prompt_tokens:
                out = self.step(s)
                if out.num_scheduled_tokens.get('p'):
                    mixed += 1
                    elapsed += .443 + .001164 * out.num_scheduled_tokens['p']
                    committed += 2.3
                else:
                    self.assertEqual(out.num_scheduled_tokens, {'d': 3})
                    pure += 1
                    elapsed += .095
                    committed += 2.4
            self.assertEqual(mixed, math.ceil(100048/chunk))
            self.assertEqual(pure, (mixed-1)*n)
            cycle = .443 + .001164*chunk + .095*n
            model_a = (2.3 + 2.4*n)/cycle
            model_b = mixed*cycle + 5
            self.assertAlmostEqual(model_a, expected_a, delta=.06)
            self.assertAlmostEqual(model_b, expected_b, delta=1)
            self.assertAlmostEqual(committed/elapsed, model_a, delta=.15)
            self.assertAlmostEqual(elapsed+5, model_b, delta=.03*model_b)
            rows.append(dict(chunk=chunk, decode_steps=n, mixed_steps=mixed, pure_steps=pure,
                             model_a_tps=model_a, model_b_ttft_s=model_b,
                             simulated_a_tps=committed/elapsed, simulated_b_ttft_s=elapsed+5))
        print('TIMESLICE SIMULATION ' + json.dumps(rows, sort_keys=True))


class Controls(unittest.TestCase):
    def test_rearm_validation_and_writer_compatibility(self):
        root = Path(__file__).resolve().parents[1]
        spec = importlib.util.spec_from_file_location('writer', root/'scripts/decode_fair_control.py')
        writer = importlib.util.module_from_spec(spec); spec.loader.exec_module(writer)
        env = dict(GLM_DECODE_FAIR='1', GLM_KV_FORMAT='fp4x', GLM_W2_PREFILL_CONTROL='/cache/prefill.json')
        self.assertEqual(F.boot_rearm(env), 'legacy')
        for bad in ('', 'yes', 'COMPLETE'):
            with self.assertRaises(ValueError): F.settings(dict(env, GLM_DECODE_FAIR_REARM=bad))
        with self.assertRaises(ValueError): F.settings(dict(GLM_DECODE_FAIR_REARM='complete'))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'fair.json'
            for mode in F.REARM:
                path.unlink(missing_ok=True)
                payload = writer.write(path, 4096, 0, 40, mode)
                state = NS()
                self.assertEqual(F.control_policy(state, str(path), 512), (4096, 40))
                self.assertEqual(F.control_rearm(state, str(path), 'scaled'), mode)
                for bad in ('bad', None, True, [], {}):
                    path.write_text(json.dumps(dict(payload, rearm=bad)))
                    with self.assertRaises(RuntimeError): F.control_policy(state, str(path), 512)
                    with self.assertRaises(ValueError): writer.write(path, 4096, 1, 40, bad if bad is not None else 'bad')
                path.write_text(json.dumps(dict(schema=1, chunk=4096, sequence=1, rearm=mode)))
                with self.assertRaises(RuntimeError): F.control_policy(state, str(path), 512)
                with self.assertRaises(ValueError): writer.write(path, 4096, 2, rearm=mode)

    def test_schema2_validation_and_atomic_writer(self):
        root = Path(__file__).resolve().parents[1]
        spec = importlib.util.spec_from_file_location('writer', root/'scripts/decode_fair_control.py')
        writer = importlib.util.module_from_spec(spec); spec.loader.exec_module(writer)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'fair.json'
            for n in (-1, True, 1.5, '20'):
                with self.assertRaises(ValueError): writer.write(path, 4096, 0, n)
                path.write_text(json.dumps(dict(schema=2, chunk=4096, decode_steps=n, sequence=0)))
                with self.assertRaises(RuntimeError): F.control_policy(NS(), str(path), 4096)
                path.unlink()
            payload = writer.write(path, 4096, 0, 20)
            self.assertEqual(payload, dict(schema=2, chunk=4096, decode_steps=20, sequence=0))
            s = NS()
            self.assertEqual(F.control_policy(s, str(path), 512), (4096, 20))
            s._glm_df_since_prefill = 3
            self.assertEqual(F.control_policy(s, str(path), 512), (4096, 20))
            self.assertEqual(s._glm_df_since_prefill, 3)
            for bad in (dict(payload, extra=1), {k:v for k,v in payload.items() if k != 'decode_steps'},
                        dict(payload, schema=True), dict(payload, schema=1)):
                path.write_text(json.dumps(bad))
                with self.assertRaises(RuntimeError): F.control_policy(s, str(path), 4096)
            path.write_text(json.dumps(dict(payload, decode_steps=40)))
            with self.assertRaisesRegex(RuntimeError, 'stale'): F.control_policy(s, str(path), 4096)
            writer.write(path, 0, 1, 0)
            self.assertEqual(F.control_policy(s, str(path), 4096), (0, 0))
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            self.assertEqual(list(Path(tmp).iterdir()), [path])


if __name__ == '__main__': unittest.main()
