#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""CPU pinned-scheduler estimate, NOT a fleet measurement; needs GLM_IMAGE_SRC."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tests'))
import test_adaptive_chunk as T
import glm_decode_fair as F


def simulate(mode, concurrent, prompt, decode_s=.070):
    T.PinnedSchedule.setUpClass()
    ns = dict(T.PinnedSchedule.namespace, _glm_budget_factory=lambda s:
              F.step_budget(s, T.A.THRESHOLD, 4096, 40, mode))
    exec('from __future__ import annotations\n' + T.A.transform(T.PinnedSchedule.source, True), ns)
    s = T.scheduler(running=[T.request('d', 60000, 60000, decode=True)], draft_slots=0)
    # First newcomer at t=0; remaining arrivals DURING its first mixed step.
    arrivals = [(i * decode_s, f'p{i}') for i in range(concurrent - 1)]
    sent, ttft = {}, {}
    elapsed = 0.0
    hits = int(prompt * .95) // 64 * 64  # model block-aligned APC hits
    mixed = pure = 0
    while len(ttft) < concurrent - 1:
        while arrivals and arrivals[0][0] <= elapsed:
            arrival, rid = arrivals.pop(0)
            s.waiting.append(T.request(rid, prompt, hits=hits))
            sent[rid] = arrival
        prefill_ids = {r.request_id for r in (*s.running, *s.waiting)
                       if r.num_computed_tokens < r.num_prompt_tokens}
        out = ns['schedule'](s)
        tokens = sum(n for rid, n in out.num_scheduled_tokens.items() if rid in prefill_ids)
        # Historical mixed-step fit; assume 70 ms MTP pure-decode steps.
        elapsed += .443 + .001164 * tokens if tokens else decode_s
        mixed += bool(tokens)
        pure += not bool(tokens)
        for r in s.running:
            if r.num_computed_tokens >= r.num_prompt_tokens:
                if r.request_id != 'd' and r.request_id not in ttft:
                    ttft[r.request_id] = elapsed - sent[r.request_id]
                r.num_tokens = r.num_computed_tokens + 1
                r.num_tokens_with_spec = r.num_tokens + 2
                r.spec_token_ids = [1, 2]
        s.current_step += 1
        if s.current_step > 10000: raise RuntimeError('simulation stalled')
    return dict(mode=mode, concurrent=concurrent, prompt_tokens=prompt, cache_hits=hits,
                residual_tokens=prompt-hits, ttft_s=ttft, mixed_steps=mixed, pure_steps=pure)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    # Fixture setup emits no workload and the scheduler never touches a GPU.
    with contextlib.redirect_stderr(io.StringIO()):
        rows = [simulate(mode, c, prompt) for c in (2, 3, 4)
                for prompt in (60000, 100000, 150000) for mode in F.REARM]
    result = dict(kind='ESTIMATE', pure_decode_step_s=.070,
                  mixed_cost='0.443 + 0.001164 * scheduled_prefill_tokens seconds',
                  arrivals='0, 0.070, 0.140 seconds; one persistent decoder; 95% block-aligned APC', rows=rows)
    text = json.dumps(result, indent=2, sort_keys=True) + '\n'
    if a.output: a.output.write_text(text)
    else: print(text, end='')


if __name__ == '__main__': main()
