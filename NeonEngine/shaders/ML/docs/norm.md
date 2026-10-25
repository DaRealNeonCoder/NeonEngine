# LayerNorm kernels

## Entry points

Implemented in `slangml/norm/layernorm_f32.slang`:

```text
sml_norm_layernorm_fwd_f32_subgroup
sml_norm_layernorm_bwd_f32_subgroup
```

Additional Phase 2 normalization modules provide:

```text
sml_norm_rmsnorm_fwd_f32_subgroup
sml_norm_rmsnorm_bwd_f32_subgroup
sml_norm_l2_normalize_fwd_f32_subgroup
sml_norm_l2_normalize_bwd_f32_subgroup
```

## Parameter block and tensor table

`headerAddress` points to `LayerNormParams`:

| Offset | Field |
| ---: | --- |
| 0..47 | `KernelHeader` |
| 48 | `rows : uint` |
| 52 | `cols : uint` |
| 56 | `epsilon : float` |
| 60 | `reserved : float` |

Forward tensors: `0=input`, `1=gamma`, `2=beta`, `3=output`.

Backward tensors: `0=input`, `1=gamma`, `2=upstream`, `3=dx`, `4=dGammaPartial`,
`5=dBetaPartial`. The partial tensors have shape `[rows, cols]`; a follow-up
reduction sums them into parameter gradients without float atomics.

Workgroup size is `256x1x1`, one workgroup per row:

```text
dispatch.x = rows
dispatch.y = 1
dispatch.z = 1
```

Shared memory is 1 KiB. No global workspace is needed for forward. Backward
requires the two partial-gradient output tensors and the host must initialize
or recycle them according to its gradient-reduction schedule.

## Numerical contract

Mean, variance, inverse standard deviation, and all reductions are fp32.
Forward computes `normalized * gamma + beta`. Backward recomputes statistics to
avoid saving per-row mean/invstd and writes exact per-element partial parameter
gradients.

## Performance notes

LayerNorm is bandwidth-bound for large hidden dimensions and reduction/
synchronization-bound for small ones. Target >=85% bandwidth for large rows.
The recompute strategy trades extra input reads for lower global-memory traffic
and storage; profile that trade for AlphaFold pair/MSA widths. Expected register
pressure is below 40 scalar registers/thread; shared memory is 1 KiB.

## Profiling checklist

Benchmark fp32:

```text
rows x cols = 1024 x 32, 1024 x 128, 1024 x 768,
              1024 x 2048, 4096 x 4096
epsilon = 1e-5 and 1e-6
forward and backward with partial-gradient outputs
```

Healthy large rows should approach >=85% measured bandwidth; hidden sizes below
one subgroup are latency-bound. Profile input/output bandwidth, barrier stalls,
register occupancy, and the cost of the backward recomputation first.
