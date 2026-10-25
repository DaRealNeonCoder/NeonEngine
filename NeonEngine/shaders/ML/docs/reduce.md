# Reduction kernels

## Entry points

Implemented in `slangml/reduce/reduce_f32.slang`:

```text
sml_reduce_sum_fwd_f32_subgroup
sml_reduce_mean_fwd_f32_subgroup
sml_reduce_min_fwd_f32_subgroup
sml_reduce_max_fwd_f32_subgroup
sml_reduce_sum_bwd_f32_subgroup
sml_reduce_mean_bwd_f32_subgroup
```

`slangml/reduce/reduce_stats_f32.slang` adds:

```text
sml_reduce_{logsumexp,var,std,l1,l2}_fwd_f32_subgroup
```

`slangml/reduce/reduce_extra_f32.slang` adds:

```text
sml_reduce_{prod,any,all}_fwd_f32_subgroup
sml_reduce_prod_bwd_f32_subgroup
```

The current slice reduces the last dimension of a row-major canonical tensor.
Two-pass large-domain and deterministic variants will use a workspace in a
later reduction extension; the single-pass row kernels are the fast common
case.

The host canonicalizer is responsible for folding an arbitrary reduction-axis
set into the row/column form used by these fast kernels. A separate slow
strided-axis kernel remains the correct route when that fold is not legal.

## Parameter block and bindings

`headerAddress` points to `ReduceParams`:

| Offset | Field |
| ---: | --- |
| 0..47 | `KernelHeader` |
| 48 | `rows : uint` |
| 52 | `cols : uint` |
| 56 | `axis : uint` |
| 60 | `mode : uint` |

Tensor table: `0=input`, `1=output`. Both are fp32. No extra bindings or
workspace are required. The host must ensure the output is writable; kernels
fully overwrite one output value per row.

## Dispatch

The workgroup is `256x1x1`. One workgroup owns one row:

```text
dispatch.x = ceil_div(rows - dispatchBase.x, 1)
dispatch.y = 1
dispatch.z = 1
```

Each lane walks `col = lane; col < cols; col += 256`. Subgroup arithmetic
reduces lane partials, then one subgroup reduces subgroup partials. The portable
path uses the shared-memory reduction helper when subgroup operations are
disabled.

## Performance notes

Reductions are memory-bound for large `cols` and target >=85% bandwidth. The
arithmetic intensity is low, so use the subgroup path by default and keep the
shared-memory footprint at 1 KiB (`256 * sizeof(float)`). The expected
occupancy cliff is register pressure above roughly 32 registers/thread or
subgroup-size/workgroup combinations that leave too few active workgroups.
The first bottleneck to profile is memory coalescing; the second is the final
subgroup barrier.

## Profiling checklist

Benchmark fp32 input/output with:

```text
rows x cols = 1 x 1, 1024 x 31, 1024 x 256, 1024 x 4096,
               4096 x 16384, 64 x 1_048_576
subgroup sizes = 8, 16, 32, 64 where supported
ops = sum, mean, min, max
ops = prod, any, all, logsumexp, var, std, l1, l2
product/boolean edge cases: cols = 0, 1, 2, 31, 256, 4096
```

Healthy large-row reductions should approach >=85% device memory bandwidth;
very short rows are launch/barrier bound. Record achieved bandwidth,
subgroup utilization, active warps/waves, shared-memory transactions, and
register occupancy. Product is latency/math bound for long rows and can be
less bandwidth-efficient than sum. Boolean reductions should be compared
against a branchless integer reduction and must not introduce divergent
barriers.
