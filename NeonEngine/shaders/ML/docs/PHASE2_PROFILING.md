# Phase 2 profiling checklist

The consolidated cross-phase checklist is `PROFILING.md`; this file retains the
detailed Phase 2 shape matrix.

This is the consolidated benchmark list for elementwise, reduction, softmax,
and LayerNorm kernels. Run each optimized entry point beside its reference
counterpart using identical BDA descriptors and canonicalization flags.

## Required shape/dtype matrix

| Family | Shapes | Dtypes | Cases |
| --- | --- | --- | --- |
| Elementwise | `1`, `31`, `256`, `4096`, `1M`, `16M`; `[1024,4096]`; `[4096,1024]` | fp32 | contiguous, rank-2, rank-3, innermost stride 1, broadcast scalar |
| Elementwise extended | `[1024,4096]`, `[4096,1024]`; GLU `[1024,4096]` | fp32 | pow/atan2/mod/comparisons/logical, GLU/SwiGLU/GeGLU |
| Generate/copy | `1`, `31`, `256`, `4096`, `1M` | fp32 | copy/fill/arange/linspace/eye |
| Reduce | `[1,1]`, `[1024,31]`, `[1024,256]`, `[1024,4096]`, `[64,1M]` | fp32 | sum/mean/min/max/prod/any/all, subgroup 8/16/32/64 |
| Softmax | `[1,32]`, `[1024,128]`, `[1024,512]`, `[256,8192]`, `[64,32768]` | fp32 | scale 1/.125, softcap off/on, forward/backward |
| LayerNorm | `[1024,32]`, `[1024,128]`, `[1024,768]`, `[1024,2048]`, `[4096,4096]` | fp32 | epsilon 1e-5/1e-6, forward/backward |
| RMSNorm/L2 | `[1024,128]`, `[1024,768]`, `[4096,2048]` | fp32 | forward/backward, epsilon 1e-5/1e-6 |

Add odd sizes, zero-sized outputs, size-one dimensions, non-tile multiples,
negative/base offsets, and non-contiguous canonical views to the correctness
run. Route exotic cases to the slower reference/strided path rather than
penalizing the fast path.

## Healthy signals

* Large memory-bound elementwise/reduction/norm kernels: >=85% of measured
  sustainable global-memory bandwidth.
* Softmax: >=75% bandwidth on long rows while maintaining high special-function
  utilization; short rows are expected to be launch/barrier bound.
* Masked softmax should remain within 10% of unmasked softmax for the same row
  shape; larger loss indicates mask addressing or divergence is dominating.
* Fused elementwise should beat the equivalent unfused sequence by at least the
  removed global-memory round-trip cost.
* Comparison/logical elementwise should reach at least 85% of add bandwidth for
  the same contiguous shape; extra predicate conversion should be negligible.
* Gated activations should beat separate activation-plus-multiply dispatches on
  total elapsed time, even if their arithmetic throughput is lower.
* Product reductions should be evaluated separately from bandwidth-bound sums;
  compare achieved FLOP/s and occupancy as well as GB/s.
* No unexpected float atomics, no divergent barriers, and no occupancy cliff
  from register use above the family budget.

Record kernel time, achieved bandwidth, FLOP/s, tensor utilization, global
load/store efficiency, subgroup utilization, shared-memory bank conflicts,
registers/thread, occupancy, barrier stalls, and launch overhead.
