# Credits

Marlin: Elias Frantar / IST-DASLab, Neural Magic and vLLM contributors.
The pinned vLLM g487ecf187 sources keep their Apache-2.0 notices and original
SOURCE_PINS. The source recovery, generator and PTX/SASS checker come from
the G-screen kit b77256c (knapcio), derived from G3 at 7cfc8ef. The serving
copy includes no synthetic consumer, cache flush, NVML guard or scratch discard.

Queue #36 credits sudoingX/dgx-spark-deepseek-v41 for the idea of evict-first
expert-weight streams protecting following small reused weights. No code from
that project was copied. The generated B-only policy adapter and serving
dispatcher are local code. NVIDIA PTX createpolicy/cache_hint semantics are used
without changing arithmetic or synchronization.
