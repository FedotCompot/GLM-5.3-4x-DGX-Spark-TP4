# Stable prefill top-k (queue #39b)

Experimental, default OFF: unset or `GLM_INDEXER_STABLE_TOPK=0` installs no
hook and adds no serving Docker argument. `=1` selects the first 2048 keys of
a stable sort by `(score descending, relative index ascending)`. Output is
also in that canonical order, with `-1` padding. No score perturbation, new
quantization, scorer change, decode selector change or unchecked sharding mode.
Only the pinned full GLM TP4/PCP1/DCP1, GB10 sm_121, FP32-logits/int32-index ABI
is supported. Active NaNs sort before +infinity; signed zeros tie; subnormals
and infinities retain numeric ordering. Uninitialized values outside each
`[rowStart,rowEnd)` are never read. Indices remain relative to `rowStart`.

## Source diagnosis

The extracted image has Python sources, but no csrc tree. The C++/CUDA audit
uses the local upstream Git object `487ecf187d3dfe74d2cf6119a92881dba403c219`,
`csrc/libtorch_stable/sampler.cu`, SHA256
`f0adb3a9e260f62a194ec83fd2b2c4ad2b0548c1868d6cf4910004d68d1fef43`.
The image's `_custom_ops.py`, sparse indexer and worker are hash-pinned.

- `topKPerRowPrefill` (:545-564) assigns **one CTA per row**; there is no
  cross-row reduction. `topKPerRowJob` (:332-542) shortcuts intervals <=K
  to ascending indices with padding, without scoring order.
- `processHistogramStep` (:157-326) counts score bins with exact integer
  atomics. Histogram counts alone do **not** cause nondeterministic selection.
  Above-cutoff candidates are appended via `smemFoundTopKValues` atomicAdd
  (:267), so even their raw output order has no index-order guarantee.
- The final candidate bucket is appended through `smemFinalDstIdx` atomicAdd
  (:282). Its insertion sort (:503-525) breaks equal scores using staging
  positions `i < j`, **not key indices**. The CUB alternative sorts score-only
  keys, so any staging-order stability also cannot restore index order. If
  the remaining quota cuts through such a tied group, the final-sort/truncation
  path can change the tied selected set even when the bucket fits in 2048.
- When the cutoff score bucket exceeds 2048 entries, refinement reaches the
  last score bits, then atomicAdd on `histo.data[binIdx]` (:295) accepts only
  destinations `<topK`. Exactly equal score keys race for the remaining
  slots; this changes the **selected set**, not just its permutation.
- The host prefill dispatch (:725-758) changes insertion/radix mode at
  **numRows >12288**, not context length. Both requested 2048/4096-row shapes
  always use insertion mode, however long their contexts. Different row
  partitions may also cross that dispatch boundary outside this kit.

Thus changing row grids, row addresses/alignment, vectorized traversal or GPU
scheduling can change tie outcomes. Nondeterminism is possible on repeated
identical launches on one GPU; it is **not restricted to row splitting**.
Source establishes the absence of a guarantee, not the frequency of changes
on this fleet. Replicated TP ranks independently run these atomics and can
already disagree in raw order or tied sets given even identical logits.
There is no index-selection collective in the DCP1 replicated path. Differences
in upstream logits are a separate possibility; stable selection guarantees
agreement only for identical scores/bounds. Boot e's 35K mismatch is
user-reported evidence, not a GPU measurement performed by this job.

## Existing alternatives and design proof

`torch.topk` has no stable tie-order contract, including `sorted=True` and
deterministic-algorithm mode. [PyTorch documents this limitation](https://docs.pytorch.org/docs/2.14/generated/torch.topk.html).
Pinned PyTorch 2.13.0 CPU `topk(ones(32),8)` returns
`[20,23,22,21,18,17,24,19]`, versus stable `[0,1,2,3,4,5,6,7]`.
This is a CPU counterexample; CUDA topk repeat frequency remains unmeasured.
`torch.argsort(...,stable=True)` is the GPU kit's sampled oracle, not the
serving implementation (a full sort allocates context-sized scratch).

The image's `dcp_indexer_cutedsl.py` (SHA256
`aff489e6505b495db182bfaf95017e31e2ceab0cf1685b3e3a94976fec02bfce`)
has a 64-bit `(float-flipped score, complement token ID)` stable-set selector.
However `_radix_pass` appends selected keys using atomicAdd on
`committed_count`, so raw output order still varies. Its input is already
packed DCP candidates, assumes a multiple of 512, uses float token IDs, and
the stable key distinguishes signed zeros. Packing all context keys would
double logits storage and stress register use. It cannot recover locally
discarded ties. The third-party Triton `streaming_topk` also includes an
index tiebreak, but its public wrapper requires columns <32768, uses int16
indices and `BLOCK_N=32`; it is an expert-routing interface and cannot directly
cover K=2048 at 35K/129K. Extending it would require a new kernel qualification.

The new original CUDA kernel uses the same key idea with int32 indices:
`(ordered FP32 score <<32) | (UINT32_MAX - relative_index)`. Every valid key
is unique and greater than padding key zero. Numeric scores map monotonically;
zero/NaN normalization gives an explicit total order. A descending 8-bit
radix histogram refines only the bucket containing the Kth key. Integer
counts and descending bucket scans determine its prefix regardless of atomic
arrival order. A bucket containing exactly the remaining quota ends refinement
early; otherwise unique keys guarantee termination by the eighth byte.
Keys >= that threshold prefix are exactly the stable top-K set.

Warp match aggregates histogram atomics. Atomic compaction writes only the
exact selected set into 2048 shared slots. A CUB radix sort of the **unique
64-bit keys** then removes all compaction-order dependence. One CTA owns each
row; neither row number nor row count enters a key or decision. Query padding,
row slicing and metadata spans therefore cannot affect selection when input
scores/bounds are unchanged. Input/output strides are explicit. Scratch is
33,300 static shared bytes per CTA; no global workspace or context-sized copy.
Up to eight full-row scans plus compaction and a 2048-key sort; speed is
unknown and could lose badly at long context. No full-sort speed claim.

The `vllm._custom_ops.top_k_per_row_prefill` hook covers stock replicated,
FP4x tiled replicated and #39's owner/reference/SELF paths because each
looks up/passes that same callable at invocation time. Register before vLLM
imports; the hook builds/loads a source-hashed standalone library through
one niced nvcc job. No Torch C++ ABI extension is needed. CUDA stream and
device are passed explicitly; no per-call synchronize, scratch allocation or
host copy. Source drift, wrong geometry/device/type, failed build/launch and
invalid intervals fail closed. The startup flag stays fixed throughout a boot.
Worker `indexer_stable_topk_status` exposes installation and invocation counts.

## Qualification limits

This proves deterministic selection for identical inputs, not bit-identical
attention or tokens versus historical stock. Stock does not sort all selected
indices globally; canonical output order can alter floating-point reduction
order even if the selected set is unchanged. Cutoff ties intentionally change
the selected set to the requested policy. Establish a new stable-replicated
baseline, compare sharding against it, and separately run existing model quality
and exact-token gates. The user's requested tie policy is the explicit exception
to historical stock output identity; it is not general permission for numerical
changes. No GPU correctness, graph replay, timing, TP-rank agreement or model
quality result is claimed from CPU/static receipts.

## Tests and benchmark

Mac: `python3 -B -m unittest tests.test_indexer_stable_topk tests.test_launcher tests.test_release_1010`.
The CPU sort oracle is independent of the radix model. Tests cover cutoff ties,
nonzero starts, masked tails, empty/short rows, signed zero, NaNs/infinities,
subnormals, random FP32 bit patterns, row splits, source pins and opt-in hooks.

Coordinator kit: `python3 -B bench/indexer_stable_topk_micro.py --output result.json`.
It uses CUDA events only. No profiling APIs or external profiler are invoked.
24 cells: 2048/4096 rows x 8192/35000/129000 columns x quantized/equal/boundary/
unique scores. It checks every row's full raw array across repeats and uneven
splits (including 1040-row metadata-style splits), samples independent stable
sorts, verifies bounds/uniqueness/stock score multisets, and tests graph replay.
The edge panel compares against the CPU oracle with noncontiguous logits and
padded output strides. Stock order/set instability is counted, not assumed.
Reported full stable/stock and quarter-row timings are selector-only events,
not end-to-end sharding, scorer, gather or TTFT measurements. Inputs are synthetic
FP32, with many exact ties; they are not captured real MQA logits.
