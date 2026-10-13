# Decode/prefill time-slicing

On by default: `GLM_DECODE_FAIR=1`, `GLM_DECODE_FAIR_CHUNK=4096`, `GLM_DECODE_FAIR_DECODE_STEPS=40`
(`profiles/current.env`). Source: `overlay/bringup/glm_decode_fair.py`, wired into the adaptive prefill
scheduler transform in `overlay/bringup/glm_adaptive_chunk.py`.

## Problem

vLLM's chunked prefill mixes a prefill chunk into every scheduler step while another request decodes.
On this cluster a mixed step with a 4096-token chunk takes about 5.2 s (`0.443 + 0.001164 x C` seconds,
C = chunk tokens), so a request that is decoding while another agent sends a cold 100K-token prompt
gets one token batch every ~5 s, about 0.44 tok/s, for the whole ~131 s of that prefill.

## Policy

Only when there is both prefill work and at least one eligible decoding request:

1. The **sum** of prefill tokens in a step is capped at the chunk (4096). All outstanding decode rows
   and their native draft slots are reserved first; running order, queue order, KV admission and
   native preemption are unchanged.
2. After a step that scheduled prefill, the next **40 steps are pure decode** (prefill gets a zero
   budget). Pure-decode steps keep the captured CUDA graph shapes (~0.08-0.09 s per step).
3. Steps that emit nothing, or are paused or refused, do not count towards the 40.

With no eligible decoding request, the unchanged adaptive rule applies (2048 below 16,384 total prompt
tokens, otherwise 4096) and a lone prefill runs at full speed. Decode-only and c1 steps schedule exactly
what the adaptive scheduler schedules. The policy lives in the central EngineCore scheduler only; its decision is
broadcast to all tensor-parallel ranks with the normal `SchedulerOutput`, so ranks cannot disagree.

## Optional re-arm experiment

Unset `GLM_DECODE_FAIR_REARM` retains the current policy and serving DRY vectors.
The original budget class and scheduler transform are unchanged. With fairness enabled,
`GLM_DECODE_FAIR_REARM=complete|scaled` selects scheduling-only variants:

- `complete`: arm N decode steps only if at least one emitted prefill row leaves its
  own prompt unfinished. Completing every emitted prefill releases the hold. An
  unfinished request that emitted nothing neither arms nor extends the hold.
- `scaled`: arm `ceil(N * sum(emitted prefill tokens) / chunk)` steps, using integer
  arithmetic. APC hits are excluded; actual scheduled rows, including any native
  padding, count. A full chunk retains N steps; a 256-token residual at 4096/N40
  arms 3 steps. This variant also scales partial chunks caused by reservations.
- `legacy`: explicit current behavior, useful for A/B/A'.

Native queue order, decode/draft reservations, preemption/refunds, admission and KV
refusals stay unchanged. Completion uses each request's **post-APC** residual before
native computed-token advancement, and the final emitted map after preemption. Empty,
paused or refused work creates no credit. With no eligible decoder the adaptive rule
still applies. Only EngineCore reads policy state.

Credit: **qu0b / SGLang contributors**, [SGLang #43844](https://github.com/sgl-project/sglang/pull/43844),
for diagnosing the completed cached/short-prefill re-arm delay and the chunk-only
protection idea. This is a local vLLM implementation; no upstream code is copied.

## Measured (boot T2, October 8, 2026)

| Check | Result |
|---|---|
| A decodes (60,043-token prefix-cached prompt) while B prefills a cold ~100K prompt | A **12.46 tok/s** (14.15 / 10.77); policy off on the same October 7 window boot: 0.44 tok/s |
| B time to first token | **212.0 s** (216.1 / 208.0); policy off: 130.9 s, so **+62 %** |
| A's token gaps during B's prefill | median 0.08-0.09 s; 25 gaps of ~5.3 s, one per prefill chunk |
| c1 decode, same boot, policy ON vs OFF (ABBA x2, 48 pairs) | **+0.83 %** [-0.36, +2.05], consistent with unchanged; prose +1.51 %, code +0.29 % |
| Preemptions / errors in the scenario | 0 / 0 |

After B's first token, A returns to its solo rate. In real agent use, A pauses for tool calls, and B's
prefill runs at full speed during those pauses, so the realized TTFT cost may be below the synthetic +62 %.
On the October 7 boot, N=20 instead of 40 gave A about 7.1 tok/s for +30 % TTFT; 4096/N40 was chosen for
multi-agent use.

## Runtime control without a reboot

The profile boots the policy with no file dependency. To be able to change or switch it off while serving,
boot with `GLM_DECODE_FAIR_CONTROL=/cache/decode-fair.json`. The file lives on the rank-0 host at
`$OVERLAY_REMOTE/cache/decode-fair.json` (mounted as `/cache/decode-fair.json`) and must exist before the
first request is scheduled. The launcher requires a fresh `$OVERLAY_REMOTE`, creates `$OVERLAY_REMOTE/cache` during
`./start.sh serve` and writes `d2w2-prefill-control.json` there just before the containers start. Write the sidecar
once that file appears; the engine then loads weights for about five minutes before health 200. Use one writer:

```bash
# during the boot: policy ON, 4096 / 40, sequence 0
python3 scripts/decode_fair_control.py "$OVERLAY_REMOTE/cache/decode-fair.json" --chunk 4096 --decode-steps 40 --sequence 0

# emergency OFF while serving
SEQ=1   # one more than the sequence currently in the file
python3 scripts/decode_fair_control.py "$OVERLAY_REMOTE/cache/decode-fair.json" --chunk 0 --decode-steps 0 --sequence "$SEQ"
```

The writer replaces the file atomically and refuses a sequence that is not higher than the one in the file.
The scheduler reads the file every step: an unchanged file is accepted silently, and a changed file takes effect
on the next whole scheduler step, with no drain, only if its sequence is higher than the last accepted one.
Rank 0's log acknowledges each accepted change with a `GLM_DECODE_FAIR {...}` line carrying the new values.
A missing, malformed or stale (changed but not higher sequence) file fails closed: the central scheduler raises
an error, which stops the engine; it does not switch the policy off. Without a sidecar, switch it off by booting with `GLM_DECODE_FAIR=0`.

### Schema-2 re-arm A/B

On an engine with this patch, schema 2 accepts one optional field:
`"rearm": "legacy" | "complete" | "scaled"`. Schema 1 still accepts exactly its old
keys, with zero decode steps. Schema 2 without `rearm` always selects `legacy`,
including when a boot env selects a variant. Unknown fields, invalid modes and stale
sequences fail closed as before. The atomic writer emits byte-identical old payloads
unless `--rearm` is supplied, and requires `--decode-steps` for that option.

```bash
# Only on a patched diagnostic engine, one writer, strictly increasing sequence.
python3 scripts/decode_fair_control.py "$OVERLAY_REMOTE/cache/decode-fair.json" --chunk 4096 --decode-steps 40 --rearm complete --sequence 1
python3 scripts/decode_fair_control.py "$OVERLAY_REMOTE/cache/decode-fair.json" --chunk 4096 --decode-steps 40 --rearm legacy --sequence 2
```

Each accepted update starts with a mixed-step opportunity, as it did before this
patch. Drain probe requests before switching arms to avoid counting that reset as a
performance gain. An older engine rejects the new field; never write it there.
Without a control file, select the env variant at boot.

## Simulator and staged-arrival probe (unqualified on fleet)

```bash
GLM_IMAGE_SRC=/path/to/extracted-image python3 -B scripts/decode_fair_sim.py --output estimate.json
python3 -B scripts/decode_fair_staged_probe.py --help
```

The simulator executes the pinned scheduler with CPU fixtures: one persistent decoder,
1--3 newcomers (2--4 concurrent total), 60K/100K/150K prompts, 95% block-aligned APC
hits, arrivals 0/70/140 ms, 70 ms pure MTP steps and the historical mixed-step cost fit.
The last newcomer estimates are below; these are **estimates**, not current fleet timings.

| Concurrent | Prompt | Legacy TTFT (s) | Complete (s) | Scaled (s) |
|---|---:|---:|---:|---:|
| 2 | 60K | 3.982 | 3.982 | 3.982 |
| 3 | 60K | 10.693 | 7.893 | 9.993 |
| 4 | 60K | 17.405 | 14.605 | 16.705 |
| 3 | 100K | 18.555 | 18.555 | 18.555 |
| 4 | 150K | 42.834 | 42.834 | 42.834 |

At 100K/150K, the residual itself spans chunks and the completing step also schedules
an unfinished next request: `complete` correctly retains protection. Neither variant
removes prefill compute or predicts owner queue p99. For a separately staged 256-token
completion, the hold-only estimate is 2.80 s legacy, 0 s complete, 0.21 s scaled.
For 2.4 accepted tokens/70 ms step, pure decode is an assumed 34.3 tok/s; removing a
hold can reduce the running stream's decode rate during bursts of short prefills.

The probe targets loopback `http://127.0.0.1:8095/v1/completions`. Supply a JSON array
of common-prefix token IDs plus three distinct valid suffix token IDs. It warms only
the common prefix in a fresh per-trial `cache_salt` namespace (so earlier B/C tails
cannot become full APC hits), starts A and waits for its first token, submits B, then C after
50 ms. It records C TTFT and A exact token counts from streaming `logprobs.tokens`
over B-submission-to-C-first-token wall time, plus full A decode tok/s and maximum
overlap gap. MTP batches count tokens, not SSE messages. Logprobs add measurement
work equally to all arms. Raw receipts contain timings/usage/counts, not text/prompts.

A run fails if C arrives after B's first token, A finishes before C's first token,
usage/token counts disagree, the stream lacks DONE/usage/logprobs, or overlap has fewer
than two emissions. Check cached-token usage (unknown if absent), rank-0 request/APC
logs and preemption metrics independently; network submission alone cannot prove the
engine's queue order. Use exactly the same prefix, suffixes, output limits and probe
settings for A/B/A'. The coordinator's fleet plan accompanies the offline result.

## Tests

`tests/test_decode_fair.py` and `tests/test_decode_timeslice.py` run the complete pinned vLLM scheduler on CPU:
both running orders, many-request permutations, the aggregate cap, decode reservations, APC deltas, KV refusal,
priority preemption refunds, four state replicas, unchanged c1/decode-only outputs, the default-off path,
OFF/ON/OFF/ON sidecar changes with running requests, and the 4096/N40 boot policy through the real installer. The re-arm tests add staged
cached completions, simultaneous complete/unfinished rows, aggregate scaled accounting,
unchanged long-chunk holds, pause/refusal/preempted-map credit checks and runtime
legacy/complete/scaled/legacy switches. `tests/test_decode_fair_probe.py` checks batched
stream counting, invalid staging and incomplete receipts entirely offline.
