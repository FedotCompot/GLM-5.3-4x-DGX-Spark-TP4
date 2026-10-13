# SPDX-License-Identifier: Apache-2.0
"""Central, opt-in mixed-step budget; TP workers consume SchedulerOutput only."""
import json
from pathlib import Path
import sys

from glm_adaptive_chunk import StepBudget, LARGE

CHUNKS = (512, 1024, 2048, 4096)
REARM = ('legacy', 'complete', 'scaled')


def settings(env):
    mode = env.get('GLM_DECODE_FAIR', '0')
    if mode not in ('0', '1'):
        raise ValueError('GLM_DECODE_FAIR must be 0 or 1')
    chunk = int(env.get('GLM_DECODE_FAIR_CHUNK', '4096'))
    if chunk not in CHUNKS:
        raise ValueError('GLM_DECODE_FAIR_CHUNK must be 512/1024/2048/4096')
    boot_decode_steps(env)
    boot_rearm(env)
    path = env.get('GLM_DECODE_FAIR_CONTROL', '')
    if path and (mode != '1' or not Path(path).is_absolute()):
        raise ValueError('GLM_DECODE_FAIR_CONTROL requires enabled mode and an absolute path')
    if mode == '1':
        from glm_adaptive_chunk import settings as adaptive_settings
        if not adaptive_settings(env)[0] or not env.get('GLM_W2_PREFILL_CONTROL'):
            raise ValueError('decode fairness requires adaptive prefill and GLM_W2_PREFILL_CONTROL')
    return mode == '1', chunk, path


def boot_decode_steps(env):
    value = int(env.get('GLM_DECODE_FAIR_DECODE_STEPS', '0'))
    if value < 0:
        raise ValueError('GLM_DECODE_FAIR_DECODE_STEPS must be nonnegative')
    return value


def boot_rearm(env):
    value = env.get('GLM_DECODE_FAIR_REARM', 'legacy')
    if value not in REARM:
        raise ValueError('GLM_DECODE_FAIR_REARM must be legacy/complete/scaled')
    if value != 'legacy' and env.get('GLM_DECODE_FAIR', '0') != '1':
        raise ValueError('GLM_DECODE_FAIR_REARM requires decode fairness')
    return value


def control_rearm(scheduler, path, fallback):
    # Old schemas/files always retain the old cadence, even with a boot variant.
    return scheduler._glm_decode_fair_control.get('rearm', 'legacy') if path else fallback


def control_chunk(scheduler, path, fallback):
    if not path:
        return fallback
    # Only EngineCore's scheduler reads this file. Atomic replace selects the
    # whole next step; workers never read it, even if their mounts differ.
    control = json.loads(Path(path).read_text())
    if (not isinstance(control, dict)
            or type(control.get('schema')) is not int or control['schema'] not in (1, 2)
            or (set(control) - ({'rearm'} if control['schema'] == 2 else set())) != (
                {'schema', 'chunk', 'sequence'} |
                ({'decode_steps'} if control['schema'] == 2 else set()))
            or ('rearm' in control and control['rearm'] not in REARM)
            or type(control['chunk']) is not int or control['chunk'] not in (0, *CHUNKS)
            or (control['schema'] == 2 and (type(control['decode_steps']) is not int
                                          or control['decode_steps'] < 0))
            or type(control['sequence']) is not int or control['sequence'] < 0):
        raise RuntimeError('decode fair control schema/chunk/decode_steps/rearm/sequence')
    prior = getattr(scheduler, '_glm_decode_fair_control', None)
    if control != prior:
        if prior is not None and control['sequence'] <= prior['sequence']:
            raise RuntimeError('decode fair stale sequence')
        scheduler._glm_decode_fair_control = dict(control)
        # A new arm starts with a mixed chunk; subsequent chunks have an N-step floor.
        scheduler._glm_df_since_prefill = control.get('decode_steps', 0)
        print('GLM_DECODE_FAIR ' + json.dumps(control, sort_keys=True), file=sys.stderr, flush=True)
    return control['chunk']


def control_policy(scheduler, path, fallback, decode_steps=0):
    chunk = control_chunk(scheduler, path, fallback)
    return chunk, scheduler._glm_decode_fair_control.get('decode_steps', 0) if path else decode_steps


class FairStepBudget(StepBudget):
    def __init__(self, scheduler, threshold, chunk, decode_steps=0):
        super().__init__(scheduler, threshold)
        self.scheduler = scheduler
        self.chunk = chunk
        self.decode_steps = decode_steps
        self.since_prefill = getattr(scheduler, '_glm_df_since_prefill', decode_steps)
        self.prefills = {r.request_id for r in scheduler.running
                         if r.num_computed_tokens < r.num_prompt_tokens}
        spec = scheduler.vllm_config.speculative_config
        self.draft_slots = spec.max_num_new_slots_for_drafting if spec is not None else 0
        self.decodes = {}
        if self.serving and chunk:
            for r in scheduler.running:
                if r.num_computed_tokens < r.num_prompt_tokens:
                    continue
                # Match native async eligibility/max-output guards. No worker
                # flags, memory readings, device values or acceptance estimates.
                if scheduler.current_step < r.next_decode_eligible_step:
                    continue
                if (r.num_output_placeholders > 0 and
                        r.num_computed_tokens + 2 - r.num_output_placeholders >=
                        r.num_prompt_tokens + r.max_tokens):
                    continue
                need = min(r.num_tokens_with_spec + r.num_output_placeholders - r.num_computed_tokens,
                           scheduler.max_model_len - r.num_computed_tokens - scheduler.num_sampled_tokens_per_step)
                threshold_cap = scheduler.scheduler_config.long_prefill_token_threshold
                if threshold_cap > 0:
                    need = min(need, threshold_cap)
                if need > 0:
                    self.decodes[r.request_id] = need
        has_prefill = bool(self.prefills) or any(
            r.num_computed_tokens < r.num_prompt_tokens
            for queue in (scheduler.waiting, scheduler.skipped_waiting) for r in queue)
        self.active = bool(self.decodes) and has_prefill
        if self.active:
            # Keep allocation/capture config untouched. The aggregate prefill
            # allowance below bounds mixed work; use existing 4096 capacity to
            # leave room for decode rows and every native draft reservation.
            self.limit = self.input_limit = LARGE

    def request(self, request, computed, token_budget, input_budget, scheduled):
        if not self.active:
            return super().request(request, computed, token_budget, input_budget)
        if computed >= request.num_prompt_tokens:
            return token_budget, input_budget, LARGE
        self.prefills.add(request.request_id)
        if self.since_prefill < self.decode_steps:
            return token_budget, input_budget, 0
        used = sum(n for rid, n in scheduled.items() if rid in self.prefills)
        # Native preemption removes victims and refunds budgets. Recompute
        # reservations from live running + emitted rows, so both compose.
        pending = [self.decodes[r.request_id] for r in self.scheduler.running
                   if r.request_id in self.decodes and r.request_id not in scheduled]
        reserve = sum(pending)
        cap = max(0, min(self.chunk - used, token_budget - reserve,
                         input_budget - reserve - (len(pending) + 1) * self.draft_slots))
        return token_budget, input_budget, cap

    def finish(self, scheduled):
        # Count actual emitted steps, not eligibility or allocation attempts.
        # KV refusals/preemption/pause must not reset the floor or create credit.
        if not self.active:
            self.scheduler._glm_df_since_prefill = self.decode_steps
        elif any(scheduled.get(rid, 0) > 0 for rid in self.prefills):
            self.scheduler._glm_df_since_prefill = 0
        elif any(scheduled.get(rid, 0) > 0 for rid in self.decodes):
            self.scheduler._glm_df_since_prefill = min(self.decode_steps, self.since_prefill + 1)


class RearmStepBudget(FairStepBudget):
    """Opt-in re-arm only; legacy budget methods and transform stay unchanged."""
    def __init__(self, scheduler, threshold, chunk, decode_steps, rearm):
        super().__init__(scheduler, threshold, chunk, decode_steps)
        self.rearm = rearm
        self.prefill_remaining = {}

    def request(self, request, computed, token_budget, input_budget, scheduled):
        if self.active and computed < request.num_prompt_tokens:
            # Native APC/remote-KV lookup already ran; finish precedes the
            # native _update_after_schedule advancement of computed tokens.
            self.prefill_remaining[request.request_id] = request.num_prompt_tokens - computed
        return super().request(request, computed, token_budget, input_budget, scheduled)

    def finish(self, scheduled):
        if not self.active or not any(scheduled.get(rid, 0) > 0 for rid in self.prefills):
            return super().finish(scheduled)
        if self.rearm == 'complete':
            # Credit qu0b / SGLang #43844: protect unfinished chunks only.
            unfinished = any(0 < scheduled.get(rid, 0) < remaining
                             for rid, remaining in self.prefill_remaining.items())
            self.scheduler._glm_df_since_prefill = 0 if unfinished else self.decode_steps
        else:
            tokens = sum(scheduled.get(rid, 0) for rid in self.prefills)
            hold = (self.decode_steps * tokens + self.chunk - 1) // self.chunk
            self.scheduler._glm_df_since_prefill = self.decode_steps - hold


def step_budget(scheduler, threshold, chunk, decode_steps=0, rearm='legacy'):
    if rearm == 'legacy':
        return FairStepBudget(scheduler, threshold, chunk, decode_steps)
    return RearmStepBudget(scheduler, threshold, chunk, decode_steps, rearm)
