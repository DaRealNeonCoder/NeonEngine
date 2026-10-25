# SlangML profiling checklist

This is the consolidated profiling checklist. `PHASE2_PROFILING.md` remains
the detailed Phase 2 matrix; this file adds the Phase 3 GEMM requirements and
is the preferred top-level checklist for the engine's benchmark harness.

## GEMM shapes and dtypes

```text
fp32 portable:
  128x128x128, 256x256x256, 1024x1024x1024, 4096x4096x4096
  4096x1024x4096, 1024x4096x1024, 1x4096x4096, 4096x1x4096

batch:
  1 and 16

epilogues:
  none, bias, bias+relu, bias+gelu, residual+scale

split-K:
  K=4096 and 16384 with splitCount=2,4,8

quantized:
  M,N,K = 256x4096x4096 and 1024x4096x4096
  int8 per-channel, int8 group=64, int4 per-channel, int4 group=64

cooperative:
  fp16 input/fp32 accumulate at 256x256x256 and 1024x1024x1024

portable low-precision storage:
  f16->f32 and bf16->f32 at 256x256x256 and 1024x1024x1024

einsum/contraction:
  matrix (128,128,128), (256,256,512), (1024,1024,1024)
  triangle (I,J,K,C) = (64,64,64,32), (128,128,256,64)
```

## Healthy signals

* Large GEMM should approach the best measured fraction of the device's fp32
  FMA or cooperative-matrix peak. A result more than 2x slower than the
  vendor-tuned baseline indicates a tile/occupancy/reuse problem.
* Split-K must be compared end-to-end, including workspace writes, reduction,
  and barriers; it is unhealthy if it only wins when workspace traffic is
  omitted.
* Fused epilogues should beat separate GEMM-plus-activation dispatches by the
  cost of the removed global-memory round trip.
* Quantized kernels should report effective dequantized FLOP/s, bytes moved,
  scale/zero-point bandwidth, and accuracy against fp32 reference GEMM.
* Low-precision storage kernels should report conversion overhead and error
  against fp32 reference GEMM separately from input bandwidth and FMA rate.
* Triangle contractions should be compared with an index-equivalent fp32
  reference and report K-reduction cost, output traffic, and launch overhead.
* GEMV and grouped-small kernels should be evaluated on latency and useful
  FLOP/s, not only peak throughput.

Always record kernel time, achieved FLOP/s, tensor/FMA utilization, global-load
efficiency, shared-memory bank conflicts, registers/thread, occupancy, barrier
stalls, workspace traffic, and launch overhead. Compare every optimized kernel
with the matching reference kernel on identical canonical descriptors.
