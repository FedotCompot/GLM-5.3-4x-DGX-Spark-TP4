# Release gate result: eh_proj + glue-lite F1/F2 (2026-10-10, gen 12)

**Verdict: PASS under the owner-approved reduced release scope.** The exact configuration is serving as
`glm53full-cand3win7-w4stack-12` under `glm-serving-watch`: deterministic align, FP8 draft `eh_proj`, and glue-lite
F1/F2 on; F3, cache trim, async kstop v2 and K4 off; tau 0.74; decode fairness 4096/N40; 262,144-token context.
Measured private serving-source commit: `6f5235dc14549677704d10e6564277883686e60d`. The public branch is a portable
export whose effective release profile and DRY launch vectors were checked for equivalence; that private object is
not presented as the public branch commit.

The [full gate procedure](docs/release-1010-gate.md) remains the stricter reference protocol. At 15:05/15:07 Europe/Warsaw the owner approved
a reduced gate with fresh boot/receipts, T=0, sparkDash, RigMark, memory stress, 15-minute soak, one 128K needle,
collection, handover and verification. The 116-item quality panel, paired 16K/128K needles and glue exactness were
carried with hashes. Exact-config 16K x2 and 250K x2 subsequently passed a read-only S-class supplement
on the same release boot after the gate. qeval x3 was not run; the carried g6 qeval diagnostic keeps
its original informational/KILL label and is not represented as a PASS.

| Check | Release result |
|---|---|
| Quality admission; qeval status | Carried qpanel/B (glm53full-cand3win7-w4ehproj-6): 110/116, 2 truncations; decision PASS. carried w4-ehproj-g5: 71/75, primary 51/55, 1 truncation, 4 failed tasks; decision KILL; carried w4-ehproj-g6: 71/75, primary 51/55, 1 truncation, 4 failed tasks; decision INFORMATIONAL (release_decision KILL). Qeval x3 not run on release boot. |
| Needle 16K, fresh twice | Fresh same-boot S-class after the gate: 8/10, 8/10 (16154 prompt tokens); missed `alias_result` and `matching_count`; all registry fields correct; both finish=stop; TTFT 20.6 / 0.6 s; PASS. |
| Needle 128K, one run | 1 run, 9/10; missed `matching_count`; TTFT 170.9 s; PASS under the reduced rule. |
| Needle 250K, fresh twice | Fresh same-boot S-class after the gate: 8/10, 8/10 (249837 prompt tokens); missed `alias_result` and `matching_count`; all registry fields correct; both finish=stop; TTFT 383.3 / 1.2 s; PASS. |
| c4 shared-pool and single maximum-context stress | PASS; c4 65024+1024 each, single 261120+1024; rank 0/1/2/3 minima 5.81/6.77/7.46/7.35 GiB; swap maxima 0/0/0/0 KiB; preemptions 0. |
| Prefix-cache first/repeat and retained-cache stress | 8K cold/replay prefill 905/16114 tok/s; stress APC hits/queries 64896/65024 tokens; retained-cache quiet minima 5.89/6.84/7.51/7.40 GiB (ranks 0/1/2/3). |
| Mixed-load soak | PASS; requested 15 min, observed 1071.1 s; 27 requests, 0 errors, 0 preemptions; quiet drift -0.02/0.01/-0.02/0.02 GiB (ranks 0/1/2/3). |
| Draft-head + eh_proj INIT | 4/4 ranks ON/ready, combined FP8 eh_proj qualified; 16 cases and 342/342/342/342 rows per rank; 0 failed; target head bit-exact on every rank. |
| Temperature-0 sequential repeats on this boot | 30/30 prompts identical across 3 repeats; 30/30 native-ID equality to carried stack-g2/glue-t0. |
| RigMark 1.0.0, thinking on, effort low | prose 30.05 tok/s; code 42.37 tok/s; structured 47.34 tok/s; code c4 aggregate 67.90 tok/s; 8K cold prefill 905 tok/s; all workload gates passed. |

APC stayed on: repeat 2 at 16K (0.6 s TTFT) and 250K (1.2 s TTFT) is a prefix-cache-hit replay, not an independent prefill.

The aggregate `matching_count` field has been missed in every surveyed needle run since 2026-10-05 (0/63 correct across 16K–250K; 0/67 including this supplement), a long-standing model/probe limitation rather than a regression of this release; `alias_result` is also missed at 16K and 250K on this boot, as at both lengths on d1 (det-align) and at 16K on g6.

Portable source run: `diagnostics/glm53-full-20261009-night/relgate/run-1010d/`.
`COMPLETE.json` records `all_pass=true` for the reduced scope. Full receipt paths and SHA256s
are in [the gate provenance](docs/release-1010-gate.md#recorded-reduced-gate-and-cell-provenance).

Supplementary source run: `diagnostics/glm53-full-20261009-night/relgate/run-1010d-s250k/` (same boot, after the reduced gate).
