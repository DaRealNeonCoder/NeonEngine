# GEMM family

Phase 3 implements the first GEMM slice used by ordinary MLPs and by the
architectural pieces that will later support Evoformer-style attention and
triangle updates. This is not an implementation of AlphaFold in its entirety.
The future AlphaFold module will compose these GEMM primitives with attention,
normalization, and pair/MSA-specific kernels.

## Entry points

Portable tiled GEMM:

```text
sml_gemm_gemm_fwd_f32_portable
sml_gemm_gemm_fwd_f32_epilogue
sml_gemm_gemm_bwd_f32_portable
```

Split-K:

```text
sml_gemm_gemm_splitk_fwd_f32_workspace
sml_gemm_gemm_splitk_reduce_f32_workspace
```

Skinny and grouped GEMM:

```text
sml_gemm_gemv_fwd_f32_portable
sml_gemm_grouped_fwd_f32_small
```

Quantized weights:

```text
sml_gemm_quantized_fwd_f32_i8
sml_gemm_quantized_fwd_f32_i4
```

Cooperative matrix:

```text
sml_gemm_gemm_fwd_f16_coopmat
```

Low-precision storage with fp32 accumulation:

```text
sml_gemm_gemm_fwd_f16_accum_f32
sml_gemm_gemm_fwd_bf16_accum_f32
```

These variants use fp16 or bf16 storage for A and B, convert each value to
fp32 before the FMA loop, and write fp32 C. They use the same hot parameter
layout, transpose flags, batching, alpha/beta handling, and fused epilogue as
the portable fp32 kernel. The storage dtype is a pipeline variant, not a
runtime choice; descriptors must carry the matching dtype tag.

The cooperative entry point is guarded by `SLANGML_ENABLE_COOPMAT=1` and
requires the documented cooperative-matrix capabilities. The portable tiled
kernel is the required fallback and is not slowed by the cooperative path.

## Low-precision storage contract

`gemm_lowp.slang` uses the same parameter block as the portable GEMM. The
tensor table is:

* `0 = A`, fp16 or bf16 storage;
* `1 = B`, fp16 or bf16 storage matching the selected entry point;
* `2 = C`, fp32 storage;
* `3 = bias`, optional fp32 storage;
* `4 = residual`, optional fp32 storage.

The 16-bit values are loaded from the BDA byte address with a naturally
aligned 32-bit read and decoded in the shader. This supports both halves of a
word without requiring a byte-addressable 16-bit pointer. Each product is
converted to fp32 before accumulation, and the fp32 result is retained through
alpha/beta and the fused epilogue. The host must enable 16-bit storage and
select the pipeline variant matching the descriptor dtype.

The workgroup is 256 threads with the default 16x16x16 tile. Shared memory is
`BM*BK*4 + BK*BN*4` bytes because the decoded tiles are fp32; with the default
tile this is 2048 bytes. The path is a correctness-first portable fallback for
devices without cooperative matrix support. Its main bottleneck is FMA
throughput after conversion rather than storage bandwidth; profile conversion
instructions, FMA utilization, registers/thread, occupancy, and global-load
efficiency against the cooperative path.

## Low-precision profiling checklist

Benchmark both `f16->f32` and `bf16->f32` at `256x256x256` and
`1024x1024x1024`, with batch 1 and 16, and with no epilogue, bias+ReLU, and
residual+scale. Compare against fp32 portable GEMM and the device's best
cooperative-matrix baseline. Report kernel time, effective input bandwidth,
achieved fp32 FMA FLOP/s, conversion overhead, occupancy, and numerical error
against an fp32 reference GEMM.

## Portable GEMM parameter block

`headerAddress` points to `GemmParams`:

| Offset | Field |
| ---: | --- |
| 0..47 | `KernelHeader` |
| 48 | `M` |
| 52 | `N` |
| 56 | `K` |
| 60 | `batchCount` |
| 64 | `strideA0` |
| 68 | `strideA1` |
| 72 | `strideB0` |
| 76 | `strideB1` |
| 80 | `strideC0` |
| 84 | `strideC1` |
| 88 | `strideABatch` |
| 92 | `strideBBatch` |
| 96 | `strideCBatch` |
| 100 | `alpha` |
| 104 | `beta` |
| 108 | `outputScale` |
| 112 | `biasMode` |
| 116 | `splitCount` |
| 120 | `tileCountM` |
| 124 | `tileCountN` |
| 128 | `reserved` |

Tensor table:

* `0 = A`
* `1 = B`
* `2 = C`
* `3 = bias`
* `4 = residual`

`SML_GEMM_FLAG_TRANS_A` and `SML_GEMM_FLAG_TRANS_B` select logical transposed
views without changing the tensor table. This supports backward mappings such
as `dA = dC * B^T` and `dB = A^T * dC`.

The default tile is `BM=16, BN=16, BK=16`, with one 256-thread workgroup per
output tile. Compile-time defines control register/shared-memory shape:

```text
-D SML_GEMM_BM=16 -D SML_GEMM_BN=16 -D SML_GEMM_BK=16
```

Dispatch formula:

```text
dispatch.x = ceil_div(N - dispatchBase.x, BN)
dispatch.y = ceil_div(M - dispatchBase.y, BM)
dispatch.z = batchCount - dispatchBase.z
```

No global workspace is required. Shared memory is:

```text
BM * BK * 4 + BK * BN * 4
```

For the default tile this is 2048 bytes. The kernel uses two shared tiles,
coalesced loads, register accumulation, and a barrier per K tile. A future
multi-stage specialization may use `SML_GEMM_STAGES > 1`; its shared-memory
formula is multiplied by the stage count.

### Fused epilogue

The same kernel can fuse:

* bias (`SML_GEMM_FLAG_BIAS`);
* residual (`SML_GEMM_FLAG_RESIDUAL`);
* ReLU, GELU-tanh, or SiLU;
* output scale;
* alpha/beta GEMM combination.

`biasMode=1` is per-column, `2` is per-row, and any other enabled bias mode is
scalar. Bias and residual descriptors are read only when their flags are set.

## Split-K contract

The split-K forward entry writes fp32 partials to `workspaceAddress` with
logical layout:

```text
[split][batch][M][N]
```

Workspace bytes:

```text
splitCount * batchCount * M * N * 4
```

The host must allocate and treat this workspace as write-only for the forward
partial kernel, then dispatch the reduction kernel. The reduction kernel fully
overwrites C after summing split partials and applying beta. Required barrier:
storage write -> storage read between the two dispatches.

Split-K is intended for large-K or occupancy-limited shapes. It increases
workspace traffic and should not be selected for ordinary square GEMM unless
profiling shows a gain.

## GEMV and grouped GEMM

`sml_gemm_gemv_fwd_f32_portable` maps one workgroup to one output row and
reduces K across 256 lanes. It is intended for batch-1 and skinny MLP cases.

`sml_gemm_grouped_fwd_f32_small` consumes a packed `GroupedGemmProblem` array
immediately after the hot parameter header. Each descriptor contains BDA
addresses, M/N/K, strides, alpha, and beta. One 64-thread workgroup handles one
problem and grid-strides across its output elements. This is deliberately a
small-problem foundation; large grouped problems should route to tiled GEMM.

## Quantized GEMM

Quantized tensor table:

* `0 = fp32 activation A`
* `1 = int8 or packed-int4 weights`
* `2 = fp32 scales`
* `3 = fp32 zero points`
* `4 = fp32 output C`

`groupSize=0` means per-output-channel scales. Otherwise scale and zero-point
indexing is `[k/groupSize][n]`. Packed int4 uses the contract's low-nibble-first
order. The current implementation is a correctness-first small/skinny path;
the future cooperative/int8 path will replace its scalar K loop for large
matrices.

## Cooperative-matrix contract

> Update (2026-11): the same Slang 2025.17.2 compiler does support `CoopMat`
> load/store from BDA pointers and groupshared arrays. A real tensor-core GEMM
> (fp16 x fp16 -> fp32, with on-the-fly Q4/Q8/BF16 weight dequant) now lives in
> `slangml/quant/gemm_quant_coopmat.slang`; see `quant.md`. The entry below is
> still the scalar fallback.

The cooperative entry point name is reserved for the 16x16x16 fp16/fp32 path.
With the local `slangc 2025.17.2` SDK compiler, public BDA `CoopMat.Load/Store`
overloads are not exposed, so the current source uses a correct fp16/fp32
fallback under that name rather than emitting an unverified or invalid SPIR-V
module. Once the engine pins a Slang compiler/device combination with the
public BDA overload, this file is the replacement point for the true
cooperative implementation.

The intended cooperative path uses 16x16x16 fp16 input tiles and fp32 accumulators.
Required compile flags:

```text
-D SLANGML_ENABLE_COOPMAT=1
-capability cooperative_matrix
-capability subgroup_basic
```

The engine must also enable the device's cooperative-matrix feature and choose
supported 16x16x16 shapes for the target vendor. The portable tiled entry
remains the fallback when any requirement is unavailable.

## Backward mapping

Backward uses the same portable GEMM ABI with transposed logical views:

```text
dA = dC * B^T
dB = A^T * dC
```

The host supplies output descriptors and hot strides for each mapping. Weight
and activation gradient accumulation into shared destinations is not performed
implicitly; if gradients from multiple batches must be accumulated, use a
separate reduction/accumulation kernel or a documented workspace.

## Performance notes

GEMM is compute-bound when M/N/K are large and should target the highest
available fraction of fp32 FMA or cooperative-matrix throughput. Arithmetic
intensity for the default tile is approximately `2*BM*BN*BK / (BM*BK + BK*BN
+ BM*BN)` FLOPs per loaded element, so global-memory reuse and occupancy are
the first bottlenecks. The portable tile uses 2 KiB shared memory and one
accumulator register per output; larger register blocks should be introduced
only as compile-time variants and profiled for occupancy cliffs.

The cooperative path should be preferred for fp16/bf16 model workloads. The
portable path is expected to be slower but must remain correct on NVIDIA, AMD,
and Intel devices. Profile tensor/FMA utilization, achieved FLOP/s, shared
bank conflicts, global-load efficiency, registers/thread, occupancy, and K-tile
barrier stalls.

## Profiling checklist

Benchmark fp32 portable GEMM:

```text
M,N,K = (128,128,128), (256,256,256), (1024,1024,1024),
        (4096,4096,4096), (4096,1024,4096), (1024,4096,1024),
        (1,4096,4096), (4096,1,4096)
batch = 1 and 16
tiles = 16x16x16, then 32x8x16 and 16x32x16 where compiled
transA/transB = all four combinations for 256x256x512
epilogues = none, bias, bias+relu, bias+gelu, residual+scale
```

Healthy large portable GEMM should be judged against the device's fp32 peak;
leaving more than 2x the expected vendor-tuned baseline indicates tile,
occupancy, or memory-reuse problems. Skinny GEMV should be compared against a
specialized vendor BLAS baseline, not square GEMM. Split-K must be measured
including workspace write/read time. Quantized kernels should report both
effective dequantized FLOP/s and bytes moved.
