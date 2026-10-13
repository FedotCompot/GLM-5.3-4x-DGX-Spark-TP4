# TP indexer row ownership (queue #39)

Default OFF: `GLM_INDEXER_ROW_SHARD=0` or unset. OFF installs no hook and
adds no Docker environment argument. The base is vLLM g487ecf187 plus the
release overlay. Credit: vLLM #54951/#54394 contributors (Apache-2.0), idea
adapted independently from the offline survey; no external source vendored.

**Status: FAST prototype over stable top-k; fleet FAST qualification pending.**
The coordinator's boot f summary reports 264 checked chunks, zero raw
mismatches, at nominal 35K/95K/129K. These historical results qualify the
geometry for the requested accelerated experiment, not its new code or speed.
Stock atomic cutoff tie order remains nondeterministic. FAST therefore refuses
without the installed `GLM_INDEXER_STABLE_TOPK=1` selector. The comparison
baseline is stable replicated OFF, not historical stock selection.

Boot controls (all ranks identical):

```bash
GLM_INDEXER_ROW_SHARD=1
GLM_INDEXER_ROW_SHARD_MODE=off
GLM_INDEXER_ROW_SHARD_SAMPLE_EVERY=32
GLM_INDEXER_STABLE_TOPK=1
GLM_INDEXER_ROW_SHARD_TRANSPORT=nccl
VLLM_SERVER_DEV_MODE=1
```

Private collective RPC `indexer_row_shard_set(mode, sample_every=None)` accepts
`off`, `self`, `check` (`checked` alias), `fast`. `MODE` replaces the legacy
`INIT` selector; conflicting non-OFF INIT values are rejected. Arming remains
explicit with `GLM_INDEXER_ROW_SHARD=1`. Unset controls stay OFF.
`GLM_INDEXER_ROW_SHARD_MODE=fast` selects FAST at boot; starting OFF and
switching via RPC is recommended. `SAMPLE_EVERY` defaults to 32, zero disables
sampling explicitly for timing. RPC interval arguments may be decimal strings.
`indexer_row_shard_status` returns counters and CUDA event phase totals.
Drain requests before switching; require four agreeing replies. Prefill is
eager. Decode, verify and CUDA graphs keep their original forward method.
SELF checks two replicated scorer/selector invocations. CHECK gathers owner
indices and compares every int32 against a replicated reference, including
order and `-1` padding. Any local mismatch is voted over the TP device group;
all workers raise before attention consumes that layer's indices. The stock
reference writes the authoritative buffer in SELF/CHECK; their candidate data
never replaces it.
FAST computes only original owned FP4x tiles plus the padded gather, then
copies the gathered raw indices into the existing shared buffer. It does no
replicated reference or equality work on unsampled calls. Generic E4M3's changed
scorer launch dimensions remain checked-only: FAST requires FP4x installation.
First eligible layer invocation after a mode/interval change and every Nth
eligible invocation thereafter gets the full replicated reference and all-rank
raw equality vote. Sampling counts eligible computing-layer invocations, not
requests, metadata spans or static skip layers; a global sample does not cover
every layer of every request. Any sampled failure latches FAILED on every rank,
raises before attention, and rejects subsequent switches/forwards. Kill and
restore the candidate; OFF cannot erase the failure. Sampling is a falsifier,
not a proof for unsampled inputs. Short/ineligible prefill remains replicated.

With sampling zero, status deltas must report `checked=sampled=0` and zero
reference/equality CUDA-event time; `fast_chunks=chunks>0` on all four ranks.
Events use the active stream and nonblocking completed-event reaping. Drained
status/switch RPCs synchronize the final pending event and harvest totals;
FAST adds no per-chunk event synchronization. The final shared-buffer copy is
outside owner/gather phase totals and included in end-to-end TTFT. No profiler,
CUPTI or nsys is used.

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

**Historical stock-profile estimate (not a FAST result):**
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
necessary; new FAST GPU results remain unknown. Stable canonical ties can
change historical stock outputs, so production promotion additionally needs
the coordinator's stock-versus-stable quality panel.

## In-boot accelerated A/B probe

`bench/indexer_row_shard_fast_probe.py --manifest prompts.json --output fast.json`
uses loopback HTTP only, an exclusively owned diagnostic engine and real local
text files. Manifest is an ordered list of three objects with `depth` values
35000/95000/129000, `prefix_file`, `suffix_file` (paths relative to manifest).
Actual prefix tokenizer count must be within 10% of the nominal label; extension
8192..12500 tokens, allowing boot f's measured 12.3K. Counts are saved, not guessed.
The client requires stable installation, healthy FP4x workers, four unique ranks,
identical mode/sampling/eligible-call sequence, and initial OFF/resumed state.

Each trial drains through `/pause?mode=wait&clear_cache=false`, selects OFF,
resets APC (must succeed), resumes and warms the identical prefix, drains again,
selects the trial mode, resumes and times its one-token extension. Prefix-cache
hits are proven by usage details when enabled or the pinned token-hit metric
(delta, polled after timing). Other traffic invalidates the panel. No graph
recapture, boots or weight lifecycle actions occur in the client.

At each depth: one mandatory FAST sample-enabled qualification, then three
OFF/FAST/OFF triplets with sampling zero. Total 30 prefix warms + 30 extensions;
3 FAST and 6 bracketing OFF scored repetitions per depth. Streamed T=0,
max_tokens=1, seed=39, ignore_eos=true, logprobs=1 with token-ID representation.
TTFT is wall time to the first generated-token SSE event (not headers or an
empty event); completion time runs through DONE. Every extension at that depth
must have identical text and exact token-ID hashes. Receipts include per-rank
counter/event deltas, paired TTFTs, completion medians and cached-token counts.
Prompts, tokens and raw indices are not saved in receipts. CUDA events provide
internal GPU phases; stdlib wall clock provides HTTP TTFT/completion.

PASS returns OFF with sampling restored and resumes the engine. Any failure
writes KILL and leaves it paused; attempts OFF readback, and reports
`restore_required` if cleanup fails. The coordinator owns stop/restore and
must inspect final state before admitting traffic. This is not a production
quality qualification or a guarantee of historical-stock output identity.
