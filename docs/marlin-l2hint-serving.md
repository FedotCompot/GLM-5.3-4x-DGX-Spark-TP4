# Queue 36 stage B: in-model Marlin weight eviction hints

Experimental and default OFF: `GLM_MARLIN_L2HINT_AB=1` installs a narrow
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
plus Docker startup/inspect overhead. No serving image binary is overwritten.

The checker requires six ON B-derived PTX hints, six hint-free copies, six
unhinted SASS instruction streams equal to a freshly compiled pinned stock
source reference, and distinct encoded ON instructions. The actual serving
stock provider's SHA256 is recorded and rechecked at boot. Static instruction
equality is with that stock source reference; it is not a claim of instruction
comparison with the image's full precompiled library. A toolkit libcuda stub
allows both libraries to register without CUDA initialization; their real
schemas must match except namespace and the final bool hint argument.

Put the build output at `overlay/cache/l2hint-serving/` before normal overlay
sync, preserving `candidate/{marlin.so,build.json}`, `assembly.json` and
`selfcheck.json`. Runtime verifies source hashes, actual stock provider hash,
candidate binary hash and receipts. The other logs/reference binary are
diagnostic artifacts and may be retained outside the runtime cache.

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
