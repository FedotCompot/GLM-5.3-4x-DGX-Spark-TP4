# Glue-lite F1+F2+F3

Glue-lite removes repeated launches around existing vLLM kernels without changing model weights.
**F1 router BF16, F2 MoE workspace reuse and F3 DSA index caching default on.**
The adapter is registered after the other bringup hooks so it wraps the composed serving path.

F1 feeds BF16 router logits directly to the CUDA grouped-topk kernel. Its BF16 instantiation widens
each logit before arithmetic, eliminating an extra full router-logit cast. It checks the pinned
GateLinear/grouped-topk sources and expected target:75,mtp:1 arming counts.
F2 retains the Marlin MoE workspace instead of zeroing it on every invocation. The existing kernel's
lock-release path resets the used locks; source pins and workspace lifecycle checks enforce that contract.
Changed routing or workspace semantics require fresh qualification.

The F1+F2 measured panel was **token-identical**, with **+0.43 % prose** and **+1.0 % code** throughput.
Token identity applies to that tested panel and configuration; it is not a universal proof for every workload.
The frozen candidate combines these switches with det-align and FP8 eh_proj. Its exact gen-12 boot passed the
owner-approved reduced release gate: sparkDash prose c1 39.38 tok/s, T=0 30/30 across three runs with byte/native-ID
equality to stack-2, and the 4.5 GiB stress gate passed. Lever percentages are not added together or substituted for
the measured table. The README and `GATE-RESULT.md` distinguish fresh phases from carried evidence.

The current profile uses these settings. Each switch is boot-time; use a fresh boot after changes.

```bash
GLM_GLUE_ROUTER_BF16=1
GLM_GLUE_MOE_WS=1
GLM_GLUE_DSA_IDX_CACHE=1
GLM_GLUE_ROUTER_EXPECT=target:75,mtp:1
GLM_GLUE_IDX_EXPECT=57
GLM_SCREEN_FAILSAFE=1
```

F3 reuses the native MLA backend's shared integer conversion buffer only across the 57 pinned target skip
layers; MTP and computing layers continue to refresh it. Missing or inconsistent buffer ownership refuses the
lever. The screen fail-safe votes cold contract results across all TP ranks and restores the stock path before
capture if any rank refuses.

`tests/test_glue_lite.py` covers source drift, router exactness, workspace reset/reuse and composition.
`tests/gpu/gpu_bitexact_f1.py` and `tests/gpu/gpu_gate_f2.py` provide GPU checks for the pinned image.
The FP4x launcher accepts F3 only with deterministic alignment, canonical alignment off, and the pinned
75-target/1-MTP router and 57-layer index geometry.

Credit: **knapcio** for the launch ledger, source-pinned adapter, tests and measurements;
**vLLM grouped-topk and Marlin contributors**, **NVIDIA CUDA** and **PyTorch contributors** for the kernels
and runtime. No external source is vendored by this adapter. Local runtime code is Apache-2.0-labelled.
[CREDITS](../CREDITS.md), [NOTICE](../NOTICE), [licence boundaries](../LICENSES/README.md).
