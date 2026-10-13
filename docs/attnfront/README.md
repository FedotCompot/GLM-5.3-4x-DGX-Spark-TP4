# Default-OFF E0 attention front

Base `261cbdb`, pinned ARM64 vLLM `g487ecf187`; no GPU qualification yet.
`GLM_ATTN_FRONT_FUSE=1` is boot-fixed. Independently disable any piece with
`GLM_ATTN_FRONT_PARTS=0`, `GLM_ATTN_FRONT_NORMS=0`, or
`GLM_ATTN_FRONT_ROPE_PACK=0`; all default to 1 under the master switch.
OFF installs no hooks, loads no native library, and preserves the original
`apply_layer` and stock upstream functions. Launcher passes no new env pairs OFF.

Only released NVFP4/FP4x/Triton MLA is admitted. Recent-KV, merged NVFP4,
weight simulation, Marlin atomic reduction, and batch-invariant norms are
refused. Dimensions/layouts outside the M1..16/H16/D512/R64 path use stock.
Admission enumerates `(hidden,q_rank,kv_rank,rope,local_heads,nope,value)` as
`(6144,2048,512,64,16,192,256)` and legacy q1536. Predicates derive partition
widths, q strides and k strides from module attributes. Real partitions are
`[2048,576]`; stock cat k stride is2624, tuple k stride576. Layer0's unquantized
BF16 projection keeps all three pieces stock. Quantized INT8 MTP keeps its stock
projection but may use independently admitted norms/RoPE.
`GLM_ATTN_FRONT_CAPTURE_SWEEP=1` with the master ON selects capture sizes1..16
for a diagnostic B boot only, retaining compilation mode0/FULL_DECODE_ONLY,
KV capacity, context and K-stop. OFF ignores this control. The usual four-size
capture config cannot establish every-M serving reachability. Run
`bench/attnfront_hits.py --worker r0=LOG0 ... --worker r3=LOG3 --out FILE` to
require all77 target layers, all16 sizes and all3 pieces on each worker.

The optional indexer fusion is not implemented: no independent byte proof
has been established for its layernorm/quantization sequence.

Build once in the local pinned image using `./start.sh attnfront-setup`.
Each enabled worker logs `ATTNFRONT_READY`; the first eligible use of each
piece per attention owner and row count logs `ATTNFRONT_HIT parts|norms|rope_pack M<rows> <layer-name>`. Missing expected hits is a HOLD.
No GPU/network/build container removal; the binary, source hashes, ABI and
stock provider hashes are checked before enabled native loading. The launcher
checks source/binary receipts before taking the serving path. The usual overlay
staging carries the extension to workers; never copy a Mac-built artifact to
serving. Rebuild in the coordinator's pinned image.

## Exactness by construction

The pinned stock provider embeds compatible `sm_120` kernels, which GB10 runs;
the new package targets `sm_121a`. Its provider SHA and five stock RoPE/RMS SASS
functions are retained by the build. Floating expression order is checked from
that provider as well as the freshly compiled reference.

All line references below use the immutable files in
`overlay/bringup/attnfront_native/vendor` or the pinned image modules recorded in
`overlay/bringup/attnfront_source_pins.json`.

1A: `glm_nvfp4_attn.apply_layer` retains its original body. `apply_partitions`
uses the same `apply_fp4_marlin_linear` keyword arguments, partition weights,
scales, workspace, FP32 reduction and launch order. The supported caller is
`DeepSeekV2FusedQkvAProjLinear`, a disabled-TP, biasless merged linear. Its
stock `ColumnParallelLinear.forward` only calls `scheme.apply_weights` and
returns `(output, None)`. Eliminating cat+split changes storage and row stride,
not the BF16 elements. Both widths and offsets are multiples of 8; norm input
alignment/vectorization is unchanged. Unsupported schemes, bias, TP, min-latency
GEMM, prefill and unknown shapes keep the original linear call. The two globals
remain distinct; no re-quantization, combined Marlin grid or accumulation change.

Norms: vendor `layernorm_kernels.cu:14–102` is reproduced from the first vector
operation to the final vector store, checked textually by the CPU suite. Stock
host lines 253–293 select BF16, vector width 8, rank 2, HasWeight=true and block
sizes 256 (2048/8) and 64 (512/8) at M1..16. The admitted legacy q1536
uses 192 threads. One physical q_width/8-thread block per row launches across
2*M independent rows. The KV row's *logical* thread stride
and CUB valid-items parameter remain 64. Threads 64–255 (legacy: 64–191) contribute no input and
participate in barriers; their aggregates are ignored. CUB's pinned
`block_reduce_warp_reductions.cuh`, `ApplyWarpAggregates` lines 138–195, reads a
successor warp only when `warp*32 < num_valid`; only warps 0 and 1 contribute
for KV, exactly as stock. Q2048 uses all eight warps (legacy: six). Within each warp the pinned
shuffle tree uses offsets1,2,4,8,16. Thread0 then folds warp aggregates in
ascending order1..7 for q2048,1..5 for legacy q1536, and only1 for KV; the
valid-items guard skips all later aggregates, including padded KV warps.
Same per-thread eight-term FP32
sum, `cub::BlockReduce<float,1024>`, CubAddOp, warp shuffles/warp aggregate order,
`rsqrtf(variance/hidden_size+epsilon)`, FP32 x*inverse*weight, and one BF16 RNE
cast. Q and KV have separate epsilon/weight pointers. Hidden widths are runtime
kernel arguments, retaining stock dynamic FP32 division semantics. The selected IR provider
must be `vllm_c`, with weights and no variance override; other norm paths stay
stock. Grid scheduling cannot affect independent row reductions.

RoPE/pack: vendor `pos_encoding_kernels.cu:8–32` supplies FP32 conversion,
cache lookup at `[position, pair]`/`[position,32+pair]`, NeoX offsets (pair,pair+32)
or interleaved offsets (2*pair,2*pair+1). Retained stock sm_121a SASS establishes
`xr=fma(x,cos,-round(y*sin))` and `yr=fma(x,sin,round(y*cos))`, followed by BF16
RNE. Explicit Triton fma plus `enable_fp_fusion=False` reproduce these operations.
Full-CTA barriers prevent cross-warp in-place NeoX pair load/store races. Both
q and k are rotated at the adjacent stock call. The k result is converted to
BF16 then back to FP32 before quantization, preserving the stock launch boundary.
The `_pack_store` body through RoPE store is copied exactly, replacing only the
load of already-rounded k by that identical BF16 value. `_parts`/`_e4m3_rne` are
imported from the locally hash-pinned released writer. Latent reduction is max
of finite nonnegative magnitudes, so its tree cannot change its value. Row pow2
scale, 32 block bytes, FP4 ties/signs/nibble order, FP32 row-scale bytes, 12 zero
padding bytes, E4M3 RoPE scale/division and clamp remain stock. Unmapped rows
still rotate but leave the cache untouched, including truncated slot maps.

The fusion receives the current layer's cache/slot mapping inside a custom op,
then returns the same empty dependency tensor ABI used by stock MLA. Patched
MLA.forward skips its stock update only when this explicit dependency is passed,
and forwards it to unified attention. No tensor attributes, host state, cached
metadata or replay-time eligibility decisions communicate the write. Profile
runs rotate normally with an empty disposable uint8 cache; large rows, unsupported
RoPE class/dispatch, PCP/DCP and recent banks keep stock. The indexer retains its
stock q_c and operations; the moved MLA cache write has no indexer-cache aliases.

These are source/operation-order arguments. GPU eager and replay byte gates
remain mandatory; compilation alone does not establish fleet E0 qualification.

## Coordinator kit

`bench/attnfront_micro.py --out NEW_DIR` checks every output/cache byte, both
strided layouts, every M1..16, random/edge norms with separate eps/weights,
both cache dtypes and both RoPE styles, masked/truncated/absent slots, eager
and seven CUDA-graph replays with changed input/positions. It also prepares
actual NVFP4 weights with unequal globals, runs both stock Marlin partitions,
and compares the direct tuple with stock cat+split for every M. Optional
`--real-inputs FILE.pt` accepts a list of `q_a`/`kv_a` activation dictionaries
for additional norm checks. It records 31 alternating paired CUDA-event samples
per cell with a 64 MiB L2 scrub outside every event interval. No profiler/CUPTI.

`bench/attnfront_crossboot.py` collects 15 prose and 10 code requests per boot,
plus duplicate teacher panels. The comparator requires exact token/support and
API logprob IEEE bits (including signed zero), complete matched A/B/A' panels,
mean cycle savings >=0.3 ms for both categories, positive lower three-SE bound,
and median control drift <=1%. Cycle measurements are attributed request decode
wall / draft opportunities, not CUDA-event device timings or full-vocab logits.
Teacher identity is API-visible logprobs; require a device logits byte dump if
full-vocabulary identity is needed.
