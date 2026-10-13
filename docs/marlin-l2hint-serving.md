# Marlin MoE weight eviction hints

Release default since 2026-10-13: `GLM_MARLIN_L2HINT=on` installs the fixed hint dispatcher in
ordinary serving. Build the variant once per clone with `./start.sh l2hint-setup` (runs
`scripts/build_marlin_l2hint.sh` in the pinned image, installs `overlay/l2hint-serving/`, keeps any previous build
under `logs/`); `serve` refuses to start with the hints on and no build present. Measured on the release boot:
every per-stream sparkDash cell 0.2–2.9 % above public main on the same day, teacher-forced prompt logprobs bit-identical
on the short panel, temperature-0 30/30 ([result](results/release-1013.json)). `off` (or unset) installs no hooks. The single-bank mode
patches only the Marlin wrapper: stock graph capture, memory profiling and
replay are unchanged; no second bank, worker RPC or EngineCore RPC is installed.
It requires V2 and deterministic alignment, and does not enable dev mode.
Target gate_up/down use the same qualified metadata below; MTP NVFP4 and
unsupported metadata use stock. The dispatcher does not read an environment
variable or runtime bank selector after boot. Rollback requires a fresh OFF
boot. Single ON and AB cannot be combined.

Experimental A/B remains available: `GLM_MARLIN_L2HINT_AB=1` installs a narrow
dispatcher and two target FULL graph banks. There is no initial-ON option:
every boot starts on bank 0. `glm_l2hint_switch("0"|"1")` selects the bank
through a drained EngineCore collective RPC. Every TP rank votes on readiness,
current/selected bank, graph counts and source/binary configuration hash before
mutation. The worker synchronizes its GPU before selecting a new bank.

Bank 0 calls the original `vllm._custom_ops.moe_wna16_marlin_gemm` wrapper,
which dispatches into the image's `_moe_C` binary. It does not execute the
extension's unhinted copy. With the experiment disabled, startup installs no
hooks and the four serving Docker argument vectors match base 7492ae7.

Bank 1 selects `_glm_l2hint_C::moe_wna16_marlin_gemm` only for BF16 activation,
BF16 output/scales, U4B8 group128, target gate_up K6144/N1024 or down
K512/N6144, block8/16, full K, FP32 reduction without atomic add, act order,
activation quantization or zero points. The extension contains all three stock
small-batch tiles with both block sizes; it retains stock configuration
selection using unhinted kernel resource attributes. Unsupported metadata
falls back to stock. Native MTP uses NVFP4 and remains stock, sharing its
original single graph set. This kit does not implement prefetch or L1 hints.

The only CUDA source change is a compile-time B-stream copy policy:
`createpolicy.fractional.L2::evict_first` and
`cp.async.cg.shared.global.L2::cache_hint`. A, scales, locks, reductions,
arithmetic, launch heuristics and synchronization retain stock source. The
dispatcher sits below `glm_moe_det`'s alignment interception and glue F2's
workspace wrapper: it forwards their sorted IDs, expert IDs and workspace
unchanged. Neither overlay is modified. ON adds host argument binding; this
cost belongs in the measured effect.

The dedicated graph hook wraps the stock `CudaGraphManager.capture` loop,
once with flag 0 and once with flag 1. Profiling also captures both banks and
discards both afterward, so the actual extra graph memory is budgeted. Shared
input/output buffers and the global graph pool remain owned by vLLM, with
serialized replay. Only mode0 FULL graphs are supported; torch.compile,
PIECEWISE and breakable graphs are refused. Each factory forward produces a
capture/warmup dispatch receipt, and graph replay counters identify the active
bank. RPC status exposes counts, source hashes and both graph tables. A failed
second capture fails boot rather than serving a partial bank.

## Build and checks

`scripts/build_marlin_l2hint.sh NEW_DIRECTORY` builds inside the locally
present pinned ARM64 image ID
`4def0ef644cb2e9814136dcffd5e385e21bc594f48f3b292234051904abe85a6`.
It never pulls an image. Two sequential nvcc builds use CUDA13, sm_121a plus
compute_121a PTX, nice10, at most two compiler threads, two CPUs, 1536 MiB
RAM without swap and no network/GPU/model mounts. Each compiler call is
bounded at 150 seconds; the complete build container at 480 seconds plus
10 seconds TERM grace. A separate toolkit-stub registration selfcheck is
bounded at 45 seconds plus 10 seconds grace. Maximum total is 545 seconds
plus Docker startup/inspect overhead. Both containers receive unique names
recorded in `containers.txt`; success/failure/interrupt stops and retains them.
No build container is auto-removed. No serving image binary is overwritten.

The checker requires six ON B-derived PTX hints, six hint-free copies, six
unhinted SASS instruction streams equal to a freshly compiled pinned stock
source reference, and distinct encoded ON instructions. The actual serving
stock provider's SHA256 is recorded and rechecked at boot. Static instruction
equality is with that stock source reference; it is not a claim of instruction
comparison with the image's full precompiled library. A toolkit libcuda stub
allows both libraries to register without CUDA initialization; their real
schemas must match except namespace and the final bool hint argument.

Put the build output at `overlay/l2hint-serving/` before normal overlay
sync, preserving `candidate/{marlin.so,build.json}`, `assembly.json` and
`selfcheck.json`. Runtime verifies source hashes, actual stock provider hash,
candidate binary hash and receipts, then compares the live stock/hint schemas
with each other and the saved selfcheck. Any mismatch fails boot with exit78
through sitecustomize. The fixed runtime library is
`/overlay/l2hint-serving/candidate/marlin.so` on the read-only overlay; never
pre-create `/cache`, which the launcher creates fresh. The other logs/reference binary are
diagnostic artifacts and may be retained outside the runtime overlay.

Mac CPU checks:

```bash
GLM_IMAGE_SRC=/tmp/glm-adaptive-image python3 -B -m unittest \
  tests.test_marlin_l2hint tests.test_launcher tests.test_recipe tests.test_release_1010
```

## In-boot panel

In an exclusive coordinator-owned B boot with the usual release guards:

```bash
python3 -B bench/l2hint_aba.py --endpoint http://127.0.0.1:8095 --out NEW_PANEL_DIRECTORY
```

The panel uses `bench/cycle.py`'s ten prose and ten code prompts and existing
GateMetrics per-request cycle attribution. It runs A=OFF, B=ON, A'=OFF,
one discarded warmup per phase, T=0, seed0, thinking OFF, 512-token limit,
and resets prefix cache before every measured request. It requires returned
integer token IDs covering the full completion, equal token IDs/text/finish
reasons/counts across phases, selected-bank replay on every rank and both
target calls in every ON capture receipt (at least 75 each). It stops on
the first token mismatch, retains receipts and attempts drained OFF restore.

For each prompt the paired saving is `(cycle_A + cycle_Aprime)/2 - cycle_B`.
KEEP requires both prose/code mean saving >=0.3 ms/cycle, mean-3SE >0 and
median absolute A/A' drift <=1%. Mean-3SE is a conservative approximate
normal interval, not proof that prompt samples are independent; retain raw
pairs and replicate a KEEP panel in a new output directory. Any missing
metric attribution, incomplete tokens/pairs, graph coverage failure, timing
failure, restoration failure or numerical mismatch is KILL. Never reuse the
synthetic consumer saving as a serving estimate. Fleet numerics/performance,
extra graph memory, schema loading with the real driver and down-kernel
byte equality remain unverified until the coordinator runs the B window.

## Single-bank cross-boot panel

At identical full context/KV/layout/guards, collect one panel per boot:

```bash
python3 -B bench/l2hint_crossboot.py collect --mode off --boot baseline \
  --endpoint http://127.0.0.1:8095 --out NEW_BASELINE_DIRECTORY
python3 -B bench/l2hint_crossboot.py collect --mode on --boot candidate \
  --endpoint http://127.0.0.1:8095 --out NEW_CANDIDATE_DIRECTORY \
  --reference NEW_BASELINE_DIRECTORY/panel.json
python3 -B bench/l2hint_crossboot.py collect --mode off --boot control \
  --endpoint http://127.0.0.1:8095 --out NEW_CONTROL_DIRECTORY \
  --reference NEW_BASELINE_DIRECTORY/panel.json
python3 -B bench/l2hint_crossboot.py compare NEW_BASELINE_DIRECTORY/panel.json \
  NEW_CANDIDATE_DIRECTORY/panel.json NEW_CONTROL_DIRECTORY/panel.json \
  --out NEW_COMPARISON.json
```

The coordinator owns stop/boot/recovery, exclusive traffic and admission.
Each collection reuses cycle prompts, l2hint_aba's seed0/512-token stream and
GateMetrics, resets APC for scored c1 prompts, discards one warmup and retains
actual IDs plus raw metrics. Mode/boot names are operator labels, so bind them
to DRY vectors, per-rank boot logs and the coordinator's admission receipts.
No collection calls a runtime switch or graph RPC. Candidate and control
stop on the first mismatched token/text/finish/count or input/timing metadata.
Comparison requires 20 unique pairs per boot, three distinct boot labels,
OFF/ON/OFF order, both prose/code mean saving >=0.3 ms/cycle, mean-3SE >0 and
median control drift <=1%. It reuses the A/B evaluator and gate_metrics.paired.

Run the existing `bench/t0_probe.py --apc-warm --det-align ON` on each boot
and explicitly check COMPLETE plus 30 eligible/identical rows, zero divergence
and identical warm request/output sequences across boots. Its built-in
`det_align_on_cold_pass` is deliberately not a warm-mode verdict; exit0 alone
is insufficient. Require actual IDs as an additional cross-boot c1 gate.

All cycle numbers here are **request decode wall / draft opportunities**,
not GPU-event cycle medians. No profiler, CUPTI or device instrumentation is
used. Any additional device timing must use CUDA events only; the display
memory lender and monitor must remain running with their existing guards.
