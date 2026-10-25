# SlangML tuning notes

This document records starting points for pipeline-cache/autotuning. Values are
not promises; the host should benchmark them on the actual device and retain
the complete variant key from `CONTRACT.md`.

## GEMM starting variants

| Vendor family | Portable BM | Portable BN | BK | Workgroup | First cooperative shapes |
| --- | ---: | ---: | ---: | ---: | --- |
| NVIDIA | 16 | 16 | 16 | 256 | 16x16x16, 16x16x32 |
| AMD | 16 | 16 | 16 | 256 | 16x16x16, 32x32x16 if supported |
| Intel | 16 | 16 | 16 | 256 | 16x16x16 |

Try 32x8 and 8x32 for skinny matrices, but do not compile every combination
for every pipeline. Use shape buckets and retain only variants that improve
the measured roofline.

## Register/shared-memory guidance

The portable default uses approximately 2 KiB shared memory and one scalar
accumulator per thread. A register-blocked variant should be rejected when
registers/thread causes an occupancy cliff or spills. Multi-stage buffering
multiplies shared memory by the stage count and should be used only when K is
large enough to hide global-memory latency.

## Subgroup sizes

Test required subgroup sizes 16, 32, and 64 where supported. Cooperative-matrix
variants must use the subgroup size required by the device's matrix shape.

## Variant-key discipline

Cache by source hash, entry point, device architecture, dtypes, BDA mode,
compile-time tile/generic values, specialization values, capabilities, and
debug/index/determinism flags. Do not create a new pipeline for runtime alpha,
beta, or dimensions when they are ordinary hot scalar fields.
