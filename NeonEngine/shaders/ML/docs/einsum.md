# Einsum and triangle contractions

Phase 3 exposes the contractions used by Evoformer-style pair and triangle
blocks as explicit kernels. This avoids a general rank-N einsum interpreter in
the hot path and gives the host a stable pipeline key, dispatch formula, and
stride ABI for each operation.

## Entry points

```text
sml_einsum_contract_ik_kj_fwd_f32
sml_einsum_triangle_mul_outgoing_fwd_f32
sml_einsum_triangle_mul_incoming_fwd_f32
```

All entry points use BDA tensor descriptors and fp32 storage. They are forward
kernels; backward contractions can reuse the ordinary GEMM/transposed-view
ABI or be added as separate kernels when the host graph needs saved-gradient
accumulation.

## Matrix contraction

`sml_einsum_contract_ik_kj_fwd_f32` computes:

```text
C[i,j] = alpha * sum_k A[i,k] * B[k,j] + beta * C[i,j]
```

The parameter block starts with `KernelHeader`, followed by:

| Offset | Field |
| ---: | --- |
| 0..47 | `KernelHeader` |
| 48 | `I` |
| 52 | `J` |
| 56 | `K` |
| 60 | `strideA0` |
| 64 | `strideA1` |
| 68 | `strideB0` |
| 72 | `strideB1` |
| 76 | `strideC0` |
| 80 | `strideC1` |
| 84 | `alpha` |
| 88 | `beta` |

Tensor table indices are `0=A`, `1=B`, and `2=C`. One 256-thread workgroup
processes a grid-stride range of linear C elements; the dispatch grid is
`ceil_div(I*J - dispatchBase.x, 256)` by 1 by 1. C is fully overwritten after
the optional beta read, so no clear is required. The host must provide a
storage-read-after-write barrier before a consumer dispatch.

## Triangle multiplicative contractions

The outgoing kernel computes the channel-preserving contraction:

```text
O[i,j,c] = alpha * sum_k A[i,k,c] * B[j,k,c] + beta * O[i,j,c]
```

The incoming kernel computes:

```text
O[i,j,c] = alpha * sum_k A[k,i,c] * B[k,j,c] + beta * O[i,j,c]
```

These are the explicit triangle-multiplication index orders. Projection and
gating layers remain separate GEMM/elementwise operations, allowing the host
to fuse them later without changing the contraction ABI.

Their parameter block starts with `KernelHeader`, followed by `I`, `J`, `K`,
`C`, three strides for A, three for B, three for O, then `alpha` and `beta`.
The tensor table is `0=A`, `1=B`, `2=O`. A linear output element uses
`c = linear % C`, `j = (linear / C) % J`, and `i = linear / (C*J)`.
Dispatch is `ceil_div(I*J*C - dispatchBase.x, 256)` by 1 by 1.

The kernels are compute-bound when K is large and bandwidth-bound for small K
or large channel counts. They use one thread per output element and no
workspace or atomics. Profile K reduction cost, global-load efficiency,
register pressure, and launch overhead; use a tiled specialization for large
pair tensors once real model shapes justify its added shared-memory footprint.

## Numerical and aliasing contract

All products and accumulators are fp32. Inputs and outputs must not alias
unless the host supplies a separate output tensor. `beta != 0` requires a
readable output descriptor. Empty dimensions produce no writes. The default
fast variants assume canonical strides and in-bounds dispatch; guarded bounds
remain a separate debug variant.

## Profiling checklist

Measure matrix contractions at `(I,J,K) = (128,128,128), (256,256,512),
(1024,1024,1024)` and triangle contractions at `(I,J,K,C) =
(64,64,64,32), (128,128,256,64)`. Include beta 0 and 1, contiguous and
canonical transposed strides, and compare each result with a CPU fp32
reference. Record kernel time, achieved FLOP/s, bytes moved, occupancy,
registers/thread, and launch overhead.
