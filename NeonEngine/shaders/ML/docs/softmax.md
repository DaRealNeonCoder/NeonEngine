# Softmax kernels

## Entry points

Implemented in `slangml/softmax_loss/softmax_f32.slang`:

```text
sml_softmax_softmax_fwd_f32_online
sml_softmax_log_softmax_fwd_f32_online
sml_softmax_softmax_bwd_f32_online
sml_softmax_softmax_masked_fwd_f32_online
```

## Parameter block and dispatch

`headerAddress` points to `SoftmaxParams`:

| Offset | Field |
| ---: | --- |
| 0..47 | `KernelHeader` |
| 48 | `rows : uint` |
| 52 | `cols : uint` |
| 56 | `scale : float` |
| 60 | `logitCap : float`, zero disables softcapping |

Forward tensor table: `0=logits`, `1=output`. Backward tensor table:
`0=probability`, `1=upstream`, `2=output`. All tensors are fp32 in this slice.

Masked forward uses `0=logits`, `1=mask`, `2=output`. `maskMode=1` means
additive mask; any other nonzero mode treats zero as masked and nonzero as
valid. Masked output remains fully overwritten.

Workgroup size is `256x1x1`, one group per row:

```text
dispatch.x = rows
dispatch.y = 1
dispatch.z = 1
```

The kernel computes row max, online exponent sum, and output without an `N x N`
temporary. Workspace is 1 KiB shared memory per workgroup and zero global
workspace. The host barrier is output write-to-read before a dependent kernel.

## Numerical contract

Statistics are fp32. Forward subtracts the row maximum before exponentiation.
Backward uses `dx = p * (dy - sum(p*dy)) * scale`. `logitCap > 0` applies a
tanh softcap before the reduction. The current slice has no mask tensor; masked
softmax belongs in the next fused attention-compatible variant.

## Performance notes

Softmax is usually bandwidth- and synchronization-bound for short rows and
exponential-throughput-bound for long rows. It targets >=85% bandwidth for
large rows and the highest available exp/FMA throughput for rows that fit in a
single workgroup. Shared memory is 1 KiB; subgroup arithmetic removes most
reduction traffic. Profile special-function throughput, barriers, occupancy,
and row length before changing tile size.

## Profiling checklist

Benchmark fp32:

```text
rows x cols = 1 x 32, 1024 x 128, 1024 x 512, 1024 x 2048,
              256 x 8192, 64 x 32768
scale = 1.0 and 0.125; logitCap = 0 and 20.0
forward and backward
```

Healthy long-row kernels should sustain >=75% of the device's measured global
bandwidth while also using high exp throughput; short rows are launch/barrier
bound. Compare max error and sum-to-one error against the reference kernel.
