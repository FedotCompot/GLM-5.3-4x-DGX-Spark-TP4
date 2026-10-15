# Scaling from 4x to 8x DGX Spark (TP8)

Status: **plan plus launcher/overlay support; nothing here has booted on eight Sparks.** Every number
below is arithmetic from the pinned model config, the TP4 loader report and `scripts/fp4_kv_layout.py`;
none is a measurement. `python3 scripts/topology.py 8` prints the same figures.

## What TP8 changes

| Area | TP4 assumption | TP8 handling |
|---|---|---|
| Launcher (`scripts/cluster.py`) | exactly 4 hosts/IPs/images/NCCL hashes, `range(4)` everywhere, vector says TP4/nnodes 4/draft TP4 | `RECIPE_TP_SIZE` (4 or 8) must equal the host count; the vector's `--tensor-parallel-size`, `--nnodes` and `draft_tensor_parallel_size` are rewritten from it |
| Profile | `profiles/current.env` only | `profiles/tp8.env` is layered on top when `RECIPE_TP_SIZE=8`; `.env` still wins |
| Attention heads | 16 per rank; every Triton MLA wrapper refuses `H != 16` | 8 per rank; the kernels already mask a 16-wide head tile, so the wrappers admit H8 and size the reduce grid by `H` |
| K-stop runtime | `(TP,PP,DP,DCP,PCP) == (4,1,1,1,1)` | admits TP 4 or 8 |
| NVFP4 MTP experts | `moe.tp_size != 4` refused | admits 4 or 8 (expert intermediate 256/rank, Marlin-legal) |
| DSA short-context shortcut | `supported()` requires TP4 | indexer is replicated, so TP 4 or 8 admitted |
| Draft NVFP4 LM head | per-rank vocab 38,720 | 19,360 rows: `19360 % 64 == 32`, the Marlin bank refuses. **Off at TP8**, so FP8 eh_proj (which requires it) is off too |
| Attention-front fusion | admitted shape has 16 local heads; kernel is H16-only | **off at TP8** (refused by the launcher) |
| Marlin MoE L2 hints | specialised for gate_up 6144x1024 / down 512x6144 | TP8 shapes are 6144x512 / 256x6144: **off at TP8** until a new specialisation is built and qualified |
| Indexer row shard / stable top-k | `world == 4` eligibility, TP4-only init | **refused at TP8** (both are already opt-in) |
| KV pool | 6,318,718,976 B head = 4,135 blocks | 31,404,851,200 B head = 16,394 blocks (see below) |
| CPU boot preflight | fakes TP4 group metadata, needs DRY ranks 0..3 | world size and ranks read from the DRY vector |
| RoCEnante one-shot | already `SUPPORTED_WORLD_SIZES = 2..16`, `ROCE_MAX_PEERS 16` | no code change; pinned host memory grows (below) |
| Gate/bench tools (`integ_1006`, `stress_step`, `release_gate`, `bench/*`) | 4 ranks hard-coded | **unchanged, TP4-only**; a TP8 gate needs them generalised |

## Divisibility at TP8

All TP-sharded GEMMs in GLM-5.3 split cleanly by 8 and stay Marlin-legal (N % 64, K % 128, expert group 128):

| Partition | Total | TP4 / rank | TP8 / rank | TP8 |
|---|---:|---:|---:|---|
| attention heads | 64 | 16 | 8 | ok (kernel tile 16, masked) |
| q_b N | 16,384 | 4,096 | 2,048 | ok |
| kv_b N | 28,672 | 7,168 | 3,584 | ok |
| o_proj K | 16,384 | 4,096 | 2,048 | ok |
| dense gate_up N / down K (layers 0-2) | 24,576 / 12,288 | 6,144 / 3,072 | 3,072 / 1,536 | ok |
| routed expert gate_up N / down K | 4,096 / 2,048 | 1,024 / 512 | 512 / 256 | ok (2 groups of 128) |
| shared expert down K | 2,048 | 512 | 256 | ok |
| vocab (embedding, BF16 head) | 154,880 | 38,720 | 19,360 | ok for stock layers |
| vocab (draft NVFP4 Marlin bank) | 154,880 | 38,720 | 19,360 | **not N % 64** |

Not sharded by TP (identical on every rank at any TP): `fused_qkv_a` (6144 -> 2048+512+64), the DSA indexer
(32 heads x 128, top-k 2048, 21 full + 57 shared layers plus MTP), router gates, norms, the MTP `eh_proj`
(6144x12288) and the MLA latent KV cache itself. The MoE stays TP-sliced (EP1): all 256 experts live on
every rank, each with 1/8 of its intermediate dimension.

## Per-node memory and context

| | TP4 (released) | TP8 (projected) |
|---|---:|---:|
| Weights per rank | 96.83 GiB (loader report) | ~49.35 GiB |
| Replicated part (estimate) | ~1.86 GiB | ~1.86 GiB |
| Freed per node vs TP4 | | ~47.5 GiB |
| Ordinary KV head | 6,318,718,976 B | 31,404,851,200 B |
| KV pool (head + 2,046 MiB carveout) | 4,135 blocks, 264,640 tokens, 7.88 GiB | 16,394 blocks, 1,049,216 tokens, 31.25 GiB |
| Max-context (262,144) requests resident at once | 1 | 4 (all four slots) |
| Largest single context the pool holds | 264,256 | 1,048,576 (model `max_position_embeddings`) |
| Net extra headroom vs TP4 before new overheads | | ~24 GiB |

Weights: `replicated + (96.83 - replicated) x 4 / TP`; the replicated estimate counts `fused_qkv_a` at 1 B/param,
indexer and router at 2 B/param and eh_proj at 1 B/param. A 1-3 GiB error in it moves the TP8 figure by
under 1 GiB. MLA latent KV is replicated per rank, so **TP8 does not shrink KV per token**; the gain is
weight memory reinvested in a 4x larger shared pool. The TP8 head is the smallest 2 MiB-aligned head that
gives each of the four slots one full 262,144-token request plus the K3 lookahead, with the release's
six-block reserve. About 24 GiB per node stays deliberately unallocated: the memory floors
(live 3.5, stress 4.5, capture 6.0, admission 6.5, pre-capture 7.5 GiB) were measured at TP4 only.

`RECIPE_MAX_MODEL_LEN` stays 262,144 at TP8. A 1M-token single context fits the pool arithmetically but is
unqualified (needles, prefill time at ~0.9K tok/s would be ~19 min, indexer and split-kernel behaviour), so the
launcher caps contexts at `max_position_embeddings` and the profile keeps 262,144. Slots stay four: the K-stop
graph layouts `[1, 4, 12, 16]` are admitted for four slots only.

New per-node costs that eat into the headroom (estimates, unmeasured):

- RoCEnante pinned host memory `(2 x world + 2) x 16 MiB`: 160 MiB at TP4, 288 MiB at TP8; the GPU gather
  scratch grows from 64 to 128 MiB.
- NCCL pinned connection buffers: 96 connections (~0.58 GiB) at TP4; if they scale with peers, ~224
  (~1.35 GiB) at TP8. `RECIPE_NCCL_NO_LL128=1` drops the LL128 share if this matters.
- MLA attention cost per rank does **not** halve: each rank still gathers the same 2,048 selected latent rows
  per token per layer; only the 16-row head tile is half empty.

## RoCE fabric for eight nodes

Each Spark's ConnectX-7 reaches the GB10 over two PCIe Gen5 x4 links (about 13.6 GB/s each), and every 200G
port appears once per link. The release's two "rails" `rocep1s0f0,roceP2p1s0f0` are the two PCIe twins of
**port 0**; port 1 is unused (layout documented in @ajclark's
[NETWORK.md](https://github.com/ajclark/GLM-5.3-DCP4-DFlash2-NVMe-KV-Offload-4x-DGX-Spark/blob/main/docs/NETWORK.md)).
So the TP4 fleet uses one cable per node, and TP8 needs **one 200G switch port per node: an 8-port (or larger)
200G RoCE switch**, each node's port 0 cabled to it and both twins addressed as today. The port (25 GB/s) and
the two x4 links (~27 GB/s) are nearly matched, so cabling port 1 to a second switch would add little bandwidth.
The fabric must stay lossless RoCE v2 (PFC/ECN as on the TP4 fabric); the eight fabric addresses go into `IPS`
in rank order.

A switchless ring is possible at any node count (both ports to the two neighbours; @ajclark ran four Sparks
that way, [issue #2](https://github.com/tonyd2wild/GLM-5.3-Int4-Int8Mix-TP4-4x-DGX-Spark/issues/2)), but it
needs NCCL ring-only settings (`NCCL_IB_SUBNET_AWARE_ROUTING=1`, merged NICs, `NCCL_RUNTIME_CONNECT=1` with
PAT/NVLS/CollNet off, one channel, rank order equal to cable order) and it cannot carry RoCEnante: the one-shot
all-reduce writes to every peer, and a ring has no path between non-neighbours. An 8-hop ring all-reduce also
adds latency on every decode collective. This recipe does not support a ring at TP8.

The one-shot all-reduce writes each rank's payload to every peer, so per-rank egress per collective grows from
3x to 7x the payload. Decode payloads are small (12 KiB per token row for hidden 6144 in BF16, 192 KiB at M16),
so c1 stays latency-bound; at c4/M16 the bandwidth floor rises from about 0.56 MiB to 1.31 MiB per rank per
all-reduce, roughly 24 us to 55 us at 25 GB/s (arithmetic, not measured). With about two all-reduces per layer
and 79 layers, that floor is a few milliseconds per c4 step and matters at c4 and above. The default
`GLM_ROCE_MAX_SIZE` 4 MiB and 16 MiB gather cap still cover decode; the logits all-gather shard halves to
19,360 columns.

Two per-node checks now run in `./start.sh preflight` because they get likelier with eight nodes: every
`IB_HCA` device must carry an IPv4 RoCE v2 GID at `NCCL_IB_GID_INDEX` (the valid index can differ per node
and otherwise fails as an opaque NCCL init error, issue #2), and a node whose idle SM clock sits under 60 % of
the fleet median is reported, because one slow rank gates the whole TP group (a post-crash GB10 stuck at
721-890 MHz is recorded in tony's README).

## TP8 versus TP4xPP2 versus EP

- **TP8 (this change).** Halves weight streaming per rank (the dominant decode cost), keeps one pipeline stage
  so c1 latency can improve, and quadruples the KV pool. Costs: attention and indexer work per rank do not
  shrink, collectives span 8 ranks, and three TP4-shaped speed levers are off until requalified.
- **TP4 x PP2.** Each stage holds half the layers on four nodes, so weights and KV per rank both halve
  (a larger pool than TP8 for the same floors), and every TP4 kernel, graph and qualification stays valid
  inside a stage. But a single request walks both stages serially, so c1 decode speed is at best TP4's, and
  every overlay here (K-stop, native MTP, short-DSA, graph layouts) refuses `PP != 1`. Better for many
  concurrent long contexts; worse for single-stream latency.
- **EP (experts across nodes).** 32 experts per node with full-width GEMMs (better Marlin efficiency), but
  two all-to-alls per MoE layer over RoCE and load imbalance at c1-c4, where a token touches 8 of 256 experts.
  MLA attention would still be TP or DP. Not supported by this stack's MoE hooks; not a first step.
- **DCP (decode context parallel).** Shards the MLA latent KV across ranks instead of replicating it.
  Measured elsewhere on 4 Sparks with TP4 + DCP4 (tonyd2wild's README and @ajclark's issue #4, different vLLM
  base, fp8/NVFP4 KV, DFlash2): at 8 GB/rank the pool grew 4.00x (131,072 to 524,288 tokens); decode about
  -5 % on a switch (-22 % on a ring); prefill fell to roughly 340-415 tok/s at every length. TP4 + DCP4 on the
  existing four nodes may therefore beat TP8 on capacity per node, but not on speed. Here the Triton MLA
  kernels, K-stop and the DSA shortcut all refuse DCP, so it is a port, not a switch. TP8 + DCP8 would hold
  one KV copy over eight nodes, at the same prefill cost.

Recommendation: TP8 for single-stream latency and context capacity, with TP4xPP2 as the fallback if
measured TP8 collectives eat the weight-streaming gain.

## Items from tonyd2wild issue #8

[@ajclark's report](https://github.com/tonyd2wild/GLM-5.3-Int4-Int8Mix-TP4-4x-DGX-Spark/issues/8) lists speedups
from a vLLM 0.29.0, TP4/DCP2, switchless-ring, DFlash2 stack. Evaluated against this recipe:

| Item | Verdict here |
|---|---|
| NCCL on both PCIe links | Already done: `IB_HCA` names both twins of port 0. Their port-1 twin addressing exists for ring NIC merging; not needed on a switch. TP8 keeps the same layout. |
| Draft-aware verify cap (K 1/3/5/7 of DFlash2's 7) | Same idea as this recipe's K-stop confidence stop on native MTP (K <= 3). No change. |
| Int8 drafter `fc` split over TP, deterministic conv Marlin | DFlash2-specific. `VLLM_MARLIN_USE_ATOMIC_ADD` is not set here and MoE alignment is deterministic. No change. |
| FP8 draft head | This recipe's NVFP4 draft head is the same lever. At TP8 both hit the Marlin N % 64 tile on the 19,360-row vocab shard; padding the bank to 19,392 rows would restore either. Follow-up. |
| RoCEnante for DCP collectives, DCP combine, prefill gathers | Only meaningful with DCP, which this stack does not run (see DCP above). |
| L2 prefetch during collectives | Not in this recipe. TP8 spends more time in collectives, so overlap has more to hide. Follow-up, needs GPU work. |
| Vocab-parallel greedy argmax | Relevant to TP8: K-stop all-gathers 154,880 logit columns per draft pass. Max and logsumexp decompose over vocab shards (local max/sum-exp, then two small all-reduces), replacing the gather. The FP32 probabilities would change reduction order, so K-stop's guard and tau need requalification. Follow-up. |
| Prefix-hit fix (all-groups EAGLE fallback in the KV coordinator) | Concerns hybrid sliding-window drafter groups. Native MTP here has no such group, and the release measured 64,896/65,024 APC hit tokens. No change. |
| Voice-first scheduler, NVMe KV tier | Out of scope for TP8. |

Taken into this branch: the corrected fabric layout above, and the per-node GID and clock checks in preflight.

## Risks

- No TP8 boot, memory, quality or speed evidence exists. The H8 MLA path, K-stop, MTP NVFP4 experts and
  DSA shortcut run at a new TP size; their logic is size-agnostic, but none is GPU-qualified at TP8.
- Different reduction partitions make TP8 outputs differ from TP4 bit-for-bit; temperature-0 baselines,
  T0 repeatability and needles must be re-established, not compared.
- Memory floors and the pre-capture floor are TP4 measurements; the large head is a projection.
- Draft-head, eh_proj, L2-hint and attention-front levers are off, so TP8 starts without their measured TP4
  gains (L2 hints +0.2-2.9 % per stream, attention front about 0.4 ms per c1 cycle) and with the BF16 draft head.
- Twice the nodes double the failure domain, the boot-time weight verification and the fabric to configure.
- The gate and bench tooling is TP4-only; a TP8 release gate needs it generalised first.
- A passing NCCL probe does not prove vLLM will start (issue #2: lazy connect hides unreachable graph edges);
  only a real boot is evidence. Never run NCCL probe containers while a ~95 GB model loads (global OOM).
- One slow or misconfigured node out of eight gates every step; the preflight warns on clocks and GIDs only.

## Using it

```bash
# .env: RECIPE_TP_SIZE=8, eight HOSTS, eight IPS (rank order), optional eight IMAGES / NCCL_SHA256
DRY=1 ./start.sh serve       # eight rank vectors, TP 8, nnodes 8, 16,394-block K-stop layout
python3 scripts/topology.py 8
```

## Verification status

Verified on a CPU workstation (no Spark attached):

- `tests/test_topology.py`: model constants match the pinned config, partition legality at TP4/TP8/TP16/TP32,
  the TP4 pool reproduces 4,135 blocks, the TP8 head is the smallest 2 MiB head for four full contexts.
- `tests/test_launcher_tp8.py`: `start.sh` layering, eight rendered rank vectors, TP4 vector byte-identical,
  host/NCCL-hash counts, TP4-only lever refusals, context cap at `max_position_embeddings`.
- `tests/test_mla_heads.py` (Triton interpreter): the H8 dispatch equals the first eight heads of the released
  H16 kernels bit for bit (FP8/BF16 split and unsplit, FP4x split and unsplit) and routes like the release.
  The interpreter's BF16 `tl.dot` is not numerically faithful, so this proves indexing, not GB10 numerics.
- `tests/test_tp8_overlay.py`: TP admission in K-stop, MTP experts, DSA shortcut; row-shard/stable top-k stay TP4.
- `tests/test_boot_preflight.py`: world size and ranks from the DRY vector.
- `roce/tests/test_glm_roce_cpu.py`: RoCEnante vote, fail-together and routing over eight gloo ranks.

Untested: anything on GPUs or the fabric — TP8 boot, memory admission, H8 MLA numerics on GB10, NCCL and
RoCEnante at eight peers, K-stop guards across eight ranks, quality and speed. The gate tools stay TP4-only.

## Hardware questions to answer

1. Does the TP8 boot pass admission with the 31.4 GB head, and what are the per-rank MemAvailable minima
   under the c4 x 262,144 stress? Can the head grow further, or must it shrink?
2. NCCL connection count and pinned memory at TP8; is `RECIPE_NCCL_NO_LL128=1` needed?
3. c1/c4 decode and prefill versus TP4 on the same day; how much do 8-rank collectives cost per step?
4. Does the H8 split MLA match the stock FlashInfer sparse MLA within the TP4 tolerance on GB10?
5. Switch choice (8+ ports at 200G) and RoCE QoS (PFC/ECN); RoCEnante timeouts or retransmits at 8 peers?
6. Is a padded (19,392-row) draft head bank worth building to restore the NVFP4 draft head and eh_proj?
7. New Marlin L2-hint specialisations for 6144x512 / 256x6144 and an H8 attention-front kernel: worth it?
8. Does TP4 + DCP4 on the existing nodes beat TP8 for long-context capacity once its prefill cost is measured here?
9. How many milliseconds per step does the full-vocab logits all-gather cost at TP8, i.e. is vocab-parallel argmax worth it?
