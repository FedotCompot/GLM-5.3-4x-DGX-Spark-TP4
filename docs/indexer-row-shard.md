# Checked TP indexer row ownership (queue #39)

Default OFF: `GLM_INDEXER_ROW_SHARD=0` or unset. OFF installs no hook and
adds no Docker environment argument. The base is vLLM g487ecf187 plus the
release overlay. Credit: vLLM #54951/#54394 contributors (Apache-2.0), idea
adapted independently from the offline survey; no external source vendored.

**Status: diagnostic port, performance promotion blocked by stock tie order.**
`docs/fp4-kv.md` records stock prefill cutoff selection through unordered
atomics. Reusing the same selector does not prove byte equality between
invocations or ranks. This implementation therefore offers no unchecked ON
mode. It cannot honestly deliver an accelerated path that is exact by
construction against this baseline. Sorting or a stable selector would change
the released path and is outside this patch.

Boot controls (all ranks identical):

```bash
GLM_INDEXER_ROW_SHARD=1
GLM_INDEXER_ROW_SHARD_INIT=off
GLM_INDEXER_ROW_SHARD_TRANSPORT=nccl
VLLM_SERVER_DEV_MODE=1
```

Private collective RPC `indexer_row_shard_set` accepts `off`, `self`, `check`.
`indexer_row_shard_status` returns counters and CUDA event phase totals.
Drain requests before switching; require four agreeing replies. Prefill is
eager. Decode, verify and CUDA graphs keep their original forward method.
SELF checks two replicated scorer/selector invocations. CHECK gathers owner
indices and compares every int32 against a replicated reference, including
order and `-1` padding. Any local mismatch is voted over the TP device group;
all workers raise before attention consumes that layer's indices. The stock
reference writes the authoritative buffer; candidate data never replaces it.
This is a falsification tool, not a guarantee that future unchecked calls pass.

The CPU gate requires scheduled prefill rows 2048..4096, CPU max depth >16384,
full GLM TP4, PCP1/DCP1, original E4M3 index cache, head dimension 128 and
top-2048. Mixed decode rows are left untouched. APC residuals below 2048 stay
replicated. Source and constructor checks refuse other models/topologies.

The pinned metadata builder separately splits its logits workspace at 512 MiB:
at 129000 keys, a 4096-row forward has metadata spans 1040/1040/1040/976.
Gating these spans individually would leave the feature dead. Generic E4M3
uses the aggregate scheduled-row gate and gathers each metadata span; changed
scorer shapes are explicitly unproven and checked. FP4x plans original query
tiles across **all** metadata spans and performs **one gather per computing
indexer layer**. Each original tile keeps its original Q padding, key bucket,
query offset, causal bounds and kernel shape. Partial/empty owner ranks take
part using sentinel-padded equal shards. Tile balancing may remove less than
3/4 of work; it avoids changing scorer geometry merely to improve balance.

Transport `nccl` directly invokes TP PyNCCL, bypassing the b12x adapter;
the existing fabric NCCL configuration is retained. Missing/disabled PyNCCL
is a hard error. `auto` uses TP all_gather and the existing RoCE/NCCL size gate.
No custom transport or scratch limit is changed. The 16 MiB b12x limit is
**per input shard**, not the concatenated output. Even row partitioning of
4096 x 2048 int32 gives 8 MiB input and 32 MiB gathered output. Tile ownership
can pad that to 16 MiB/rank at 35K; heterogeneous request tiles can be larger,
in which case AUTO takes the existing NCCL fallback. Confirm live route and
RDMA counters; source inspection cannot prove that NCCL used the fabric.

Overlay interactions:

- `glm_dsa_short`: only decode/verify is short; the original method and short
  graph bank remain in use. The generic transformed prefill op retains the
  short source transformation in its decode fallback.
- F3 `GLM_GLUE_DSA_IDX_CACHE`: the original shared top-k tensor is written in
  place before index conversion. Native computing/unarmed MLA clears F3's
  conversion cache; static skip layers see the same authoritative buffer.
  No cache entry spans a computing layer. No new index cache is introduced.
- FP4x `glm_fp4_prefill`: constructor banks/capacity remain unchanged. New
  owner storage is transient. Approximate extra gather peak is <=80 MiB for
  an even 4096-row case, potentially larger with padded tile owners. No memory
  floor is lowered. Original metadata and KV/cache insertion stay in place.
- MTP: native skip/compaction/shared-index lifecycle is unchanged. Only a real
  MTP prefill of >=2048 scheduled rows can reach the probe. Draft decode and
  subsequent index sharing keep native paths. No target/MTP duplicate gather
  is added on a layer that does not execute its indexer.

`scripts/indexer_row_shard_cost.py` emits estimates for 35K/95K/129K, both
chunk sizes, and 78 versus 21 layer budgets. The profile measures 8.5K new
tokens: 35K indexer 0.918 s, 129K logits+top-k 3.13 s; 95K is interpolation.
The profile's rounded 25% would give 3.35 s; the model uses the named kernel
sum instead. 78 gathers is the requested pessimistic budget; the saved model
survey reports 21 computing target indexers (57 static skip layers), so
confirm actual executed count rather than gathering on every MLA layer.

Assuming 0.5..3.0 ms per gather (unmeasured), 78 gathers cost 39..234 ms per
forward; 21 cost 10.5..63 ms. With original metadata splits and tile balancing:

| Depth | Rows | Saved indexer ms | Net ms, 78 gathers | Reduction in GPU-busy time |
|---|---:|---:|---:|---:|
| 35K | 2048 | 111 | -123..72 | -4.7..2.8% |
| 35K | 4096 | 265 | 31..226 | 0.6..4.4% |
| 95K | 2048 | 337 | 103..298 | 3.4..9.9% |
| 95K | 4096 | 748 | 514..709 | 8.6..11.8% |
| 129K | 2048 | 452 | 218..413 | 6.8..12.8% |
| 129K | 4096 | 1097 | 863..1058 | 13.4..16.4% |

Ideal 3/4 savings at 4096 rows are 332/842/1131 ms at these depths. The
shape-preserving planner instead removes 60%, 66.7%, 72.7% for decimal
35000/95000/129000 depths. Exact token depths change metadata splits and
these fractions. No overlap/launch/gather-key/host-scheduling gain is assumed.
These are phase estimates, not cold TTFT or owner-request speedups. APC hits
with short new suffixes get no benefit. Checked mode adds owner work and a
gather to the replicated reference, so its expected wall result is slower.

CUDA `phase_ms` divides owner, gather, reference, equality. FP4x owner/reference
include their respective key gathers; report this when estimating a future
fast path. Use max-rank phase deltas. Passing real-prompt comparisons is
necessary but cannot prove atomic tie determinism for all inputs. Keep the
performance item parked unless a baseline-compatible deterministic selector
contract is established separately.
