# SlangML kernel contract

Status: performance-first ABI for GPU-only Slang compute kernels targeting
Vulkan 1.3 SPIR-V.  The host engine owns device creation, memory allocation,
descriptor binding, pipeline creation, dispatch, synchronization, profiling,
autotuning, and weight loading.  SlangML contains only `.slang` modules and
Markdown contracts.

## 1. Non-negotiable performance rules

These rules bind every current and future phase:

* Every family document starts with a **Performance notes** section containing
  the roofline target, arithmetic-intensity reasoning, expected bottleneck,
  tile choice, occupancy expectation, register/shared-memory budget, known
  bottlenecks, and first profiling counters.
* Memory-bound kernels target at least 85% of sustainable device peak
  bandwidth. GEMM, attention, convolution, IPA, and triangle-update kernels
  target the highest achievable fraction of peak tensor/FMA throughput; their
  documentation must explain whether math, memory, synchronization, or launch
  overhead is limiting.
* Use coalesced 128-bit loads/stores whenever alignment permits, and use the
  host-set `SLANGML_FLAG_ALIGNED16` fast path. Shared-memory tiles must use a
  documented padding or swizzle that avoids bank conflicts.
* Use register tiling, software-pipelined multi-stage shared-memory buffering,
  and the minimum number of barriers. Address computation is hoisted out of
  inner loops wherever possible.
* Fuse aggressively: GEMM/conv epilogues, norm+residual+activation,
  softmax+mask+scale, bias+dropout, and any operation whose unfused form would
  round-trip through global memory.
* Avoid divergence and integer division/modulo in inner loops. Precompute
  reciprocal/magic-number forms or use power-of-two tile loops. Use `rcp`,
  `rsqrt`, `exp2`-based exponentials, and `fma` where the documented accuracy
  budget permits. Every precision trade must be stated.
* Prefer cooperative-matrix paths for matmul-like work. The baseline numeric
  policy is fp16/bf16 input with fp32 accumulation; int8/fp8 paths are added
  when the device and model permit them. Vendor tile choices belong in
  `TUNING.md`.
* Prefer persistent kernels, grouped dispatches, and single-pass algorithms.
  Avoid atomics on hot paths; reduce within a workgroup before using global
  atomics.
* Minimize workspace and global-memory traffic. For AlphaFold-sized pair
  tensors, recompute instead of storing when recomputation costs less than the
  bandwidth and the family document records that choice.
* Do not slow a fast kernel for exotic rank, stride, dtype, or shape cases.
  Route unusual cases to a separate slower kernel.
* Every family Markdown file has a **Profiling checklist** with exact benchmark
  shapes/dtypes and healthy versus underperforming throughput/bandwidth goals.

## 2. Module and compilation model

Source root:

```text
shaders/ML/slangml/slangml-common.slang
shaders/ML/slangml/common/*.slang
shaders/ML/slangml/<family>/*.slang
shaders/ML/reference/*.slang
shaders/ML/docs/*.md
```

Import the common module with:

```slang
import slangml_common;
```

Recommended baseline command for an entry point is:

```text
slangc <shader>.slang -I shaders/ML/slangml \
  -target spirv -profile spirv_1_5 -stage compute -entry <entry> \
  -fvk-use-entrypoint-name -fvk-use-scalar-layout -fspv-reflect -O3 \
  -o <out>.spv
```

Use the exact specialization and capability flags documented by the family.
The engine must cache pipelines by the complete variant key described below.

## 3. Canonical parameter ABI

### 3.1 KernelHeader at parameter-block offset 0

Every parameter block starts with `KernelHeader` from
`common/tensor.slang`. Its scalar-layout fields are:

| Offset | Field | Type |
| ---: | --- | --- |
| 0 | `dispatchBase.xyz` | `uint3` |
| 12 | `numel` | `uint` |
| 16 | `flags` | `uint` |
| 20 | `tensorTableOffset` | `uint` |
| 24 | `tensorCount` | `uint` |
| 28 | `reserved0` | `uint` |
| 32 | `tensorTableAddress` | `uint64_t` |
| 40 | `reserved1` | `uint` |
| 44 | `reserved2` | `uint` |

The fixed header is 48 bytes.  Kernel-specific hot scalars follow it in the
same parameter block: dimensions, reductions, strides used in inner loops,
alpha/beta, epsilon, tile counts, and workspace offsets.  Hot loops must not
reload those values from a descriptor table.

`KernelPushConstants` contains only pointer-sized handles and small flags:

| Offset | Field | Type |
| ---: | --- | --- |
| 0 | `headerAddress` | `uint64_t` |
| 8 | `workspaceAddress` | `uint64_t` |
| 16 | `flags` | `uint` |
| 20 | `reserved` | `uint` |

Push constants do **not** contain tensor descriptors. The host may put the
parameter block in a BDA allocation and pass its address through the push
constant block. If a bound-buffer fallback is selected, the family document
must specify its small-tensor parameter binding; that path is not ABI-equivalent
to the BDA fast path.

### 3.2 TensorDesc table

`TensorDesc` is a 96-byte scalar-layout record in a BDA table. The descriptor
is optimized for 32-bit hot indexing:

| Offset | Field | Type | Meaning |
| ---: | --- | --- | --- |
| 0 | `shape[8]` | `uint[8]` | Canonical logical extents |
| 32 | `stride[8]` | `int[8]` | Strides in elements |
| 64 | `baseOffset` | `int64_t` | One 64-bit base offset |
| 72 | `dataAddress` | `uint64_t` | BDA byte address of storage |
| 80 | `dtype` | `uint` | Storage dtype tag |
| 84 | `rank` | `uint` | Active rank, at most 8 |
| 88 | `flags` | `uint` | Fast-path flags |
| 92 | `bufferIndex` | `uint` | Bound-buffer fallback only |

The default path uses BDA (`SLANGML_USE_BDA=1`). Bound-buffer mode is an
optional small-tensor fallback (`SLANGML_BOUND_BUFFER_FALLBACK=1`), not a
co-equal variant. Do not multiply the fast pipeline matrix merely because a
bound-buffer form exists.

### 3.3 Index width policy

Default hot paths use 32-bit `shape`, `stride`, coordinates, relative element
offsets, loop counters, and dispatch sizes. The only 64-bit operations allowed
in the normal addressing path are:

1. adding the descriptor's `baseOffset`; and
2. forming the final BDA byte address.

`SLANGML_INDEX64=1` enables a separate rare-tensor variant for logical domains
that exceed the 32-bit index contract. It may use 64-bit linearization in its
outer loops, but it is never part of the default fast variant. Never put 64-bit
math in an inner loop of a normal kernel.

## 4. Host canonicalization contract

Before dispatch, the engine always canonicalizes every descriptor:

1. merge adjacent contiguous dimensions;
2. drop size-1 dimensions;
3. collapse broadcast dimensions where legal;
4. compute canonical strides and `baseOffset`;
5. set `SLANGML_FLAG_CANONICAL`;
6. set `SLANGML_FLAG_CONTIGUOUS`, `SLANGML_FLAG_RANK_LE3`,
   `SLANGML_FLAG_INNER_STRIDE1`, and `SLANGML_FLAG_ALIGNED16` when true.

Kernels may assume canonical descriptors. Fast-path selection is:

* fully contiguous: one linear index and vectorized 128-bit access;
* rank <= 3: specialized coordinate math with no general rank loop;
* innermost stride 1: coalesced row/column fast path;
* `ALIGNED16`: vec4 load/store path for 32-bit storage;
* otherwise: a separate strided/broadcast kernel.

The canonicalizer is host work and is not repeated in GPU kernels.

## 5. Entry-point naming

All entry points use:

```text
sml_<family>_<op>_<fwd|bwd>_<dtype>[_<variant>]
```

Examples:

```text
sml_elementwise_add_fwd_f32_contig
sml_reduce_sum_bwd_f32_strided
sml_softmax_softmax_fwd_f16_online
sml_norm_layernorm_bwd_bf16_subgroup
```

Every family document must specify the exact name, workgroup size, grid
formula, bindings, parameter layout, workspace, read/write sets, barriers,
specialization constants, compile-time generics, saved tensors, and backward
contract.

## 6. Tunables, variant keys, and resource budgets

### 6.1 Two tunable classes

**Pipeline-time specialization constants** may change without recompiling the
Slang source:

* workgroup dimensions;
* loop unroll counts that do not change local array shapes;
* subgroup size (8/16/32/64);
* algorithm flags such as causal, deterministic, or guarded bounds;
* reduction split count and small scalar thresholds.

**Compile-time generics/defines** require a new compiled module because they
must stay in registers or define static storage:

* register-block dimensions;
* per-thread array sizes;
* cooperative-matrix tile shapes;
* pipeline stage count;
* vector width and packed-storage layout;
* fused operation-chain type;
* BDA versus bound-buffer ABI.

Each family must document shared-memory usage as a function of these tunables,
for example:

```text
shared_bytes = stages * (BM * (BK + padA) + BK * (BN + padB)) * bytes_per_element
```

It must also state register-pressure expectations, such as an estimated
register count per thread and the occupancy cliff to watch for.

### 6.2 Pipeline variant key

The cache key is the tuple:

```text
(shader source/module hash,
 entry-point name,
 target/profile,
 device vendor/architecture,
 BDA-or-bound ABI,
 storage/compute/accumulation dtypes,
 compile-time generic values,
 specialization-constant values,
 enabled SPIR-V capabilities,
 debug/deterministic/index64 flags)
```

Do not create separate variants for values that are not consumed by the
compiled entry point. Do not merge variants when their resource layout,
capabilities, register arrays, or numerical behavior differ.

## 7. Subgroup policy

Subgroup operations are the default baseline when the device exposes them.
`SLANGML_SUBGROUP_SIZE` is a specialization constant with supported values
8, 16, 32, and 64. The host must use Vulkan subgroup-size control and a
required-subgroup-size pipeline state when a kernel depends on a fixed width.
The relevant host features are `VkPhysicalDeviceSubgroupSizeControlFeatures`
and `VK_EXT_subgroup_size_control`.

The portable shared-memory path remains available through
`SLANGML_ENABLE_SUBGROUP_OPS=0`, but is a fallback for devices or kernels that
cannot use arithmetic/shuffle/ballot operations. It must not be the default
variant when the subgroup capability is present.

## 8. Required Vulkan features and Slang capability flags

The host must query and enable features before selecting a variant:

* `shaderInt64` for address and rare index support;
* `scalarBlockLayout` for the declared scalar-layout parameter ABI;
* `bufferDeviceAddress` for the default data and tensor-table path;
* `storageBuffer16BitAccess`, `shaderFloat16`, and `shaderInt8` for native
  fp16/bf16/int8 storage and arithmetic paths;
* subgroup arithmetic, shuffle, ballot, and required-subgroup-size control;
* Vulkan memory model with device scope for look-back scan, stream-K, and
  cross-workgroup memory-ordering variants;
* float atomics where a kernel explicitly uses them;
* cooperative matrix support for cooperative-matrix variants;
* float-controls features when exact rounding, denormal, or contraction
  behavior is part of the documented numerical contract.

Recommended Slang capability flags for the optional paths are:

```text
# BDA / physical storage buffer pointers
-capability bufferreference -capability bufferreference_int64

# Subgroup arithmetic, shuffle, ballot
-capability subgroup_basic -capability subgroup_arithmetic \
  -capability subgroup_shuffle -capability subgroup_ballot

# Vulkan memory model and device scope
-capability vk_mem_model
# Or, when spelling SPIR-V capabilities explicitly:
-capability spvVulkanMemoryModelKHR \
  -capability spvVulkanMemoryModelDeviceScopeKHR

# Float atomics
-capability atomicfloat
# Add atomicfloat2 for float min/max variants where supported.

# Cooperative matrix
-capability cooperative_matrix
# Add cooperative_matrix_reduction/conversion/block_load as used by a kernel.
```

`shaderInt64`, scalar block layout, 16-bit storage, float16, and int8 are
primarily Vulkan feature/profile requirements rather than portable Slang
capability switches; the engine must still validate the reflected SPIR-V
requirements. Slang's `-fvk-use-scalar-layout` is required for the buffer ABI.

## 9. Aliasing and in-place rules

* Inputs and outputs are non-aliasing by default unless a family explicitly
  documents an in-place entry point.
* An in-place kernel must state whether the operation is elementwise-safe,
  row-safe, or requires a temporary. Arbitrary overlapping views are invalid.
* Descriptors with the same `dataAddress` and overlapping byte ranges are
  considered aliased. The host must not dispatch a kernel whose documented
  read/write sets create a write-after-read hazard without the required
  temporary or barrier.
* Read-only broadcast inputs may alias each other. Scatter, reductions,
  normalization statistics, and backward accumulation must not assume output
  aliasing unless named as an in-place variant.

## 10. Initialization and atomics

The host owns output initialization unless the kernel contract says otherwise:

* `scatter_add`, embedding backward, and atomic reductions require the host to
  zero the destination before dispatch;
* split-K GEMM requires zeroing or initializing the split workspace before the
  first partial write;
* `split`, `concat`, and ordinary elementwise writes fully overwrite their
  documented output and do not require a prior clear;
* deterministic variants use fixed-order workgroup/multi-pass reductions and
  avoid float atomics.

## 11. Bounds and invalid indices

The default fast variant treats out-of-bounds indices as undefined behavior.
It performs no clamp or per-element guard that would add hot-path cost. The
optional guarded variant sets `SLANGML_KERNEL_FLAG_GUARDED_BOUNDS` or compiles
with `SLANGML_DEBUG_BOUNDS=1`; it is a separate variant and must not affect
release code generation. `SLANGML_DEBUG_BOUNDS` and similar debug switches
compile to nothing in release variants.

## 12. Packed int4 and fp8 layouts

* Packed int4 stores two logical values per byte: logical element 0 is the low
  nibble and logical element 1 is the high nibble. Signed int4 uses two's
  complement nibble values `[-8, 7]`.
* Packed-int4 base addresses must be byte aligned; vectorized unpack paths
  operate on at least 16-byte aligned blocks when `ALIGNED16` is set. Odd tails
  use a scalar masked load.
* FP8 E4M3 and E5M2 use one byte per element, with the encoding defined by
  `common/dtypes.slang`. E4M3 is finite-only FN; E5M2 preserves Inf/NaN.
* Quantized kernels document scale and zero-point storage explicitly. The
  default layout is fp16/fp32 scale per output channel, optional zero-point in
  int8/uint8 per channel or group, contiguous in the channel-major dimension.
  Group size and alignment are kernel parameters, not inferred in a hot loop.
* Integer dequantization accumulates in fp32 or int32 as documented. Saturating
  casts clamp before packing; overflow is never silently wrapped in a
  saturating path.

## 13. Dispatch splitting and batch folding

Every kernel accepts `dispatchBase.xyz` and uses grid-stride loops where useful.
If a logical batch or grid dimension exceeds a Vulkan device dispatch limit,
the host folds that dimension into a linear logical index and splits the work
into multiple dispatches. The folded shape and the original logical strides
are passed as hot scalars; the kernel must not infer them from dispatch limits.
Each split dispatch changes only `dispatchBase` and the dispatch grid. Workgroup
barriers never synchronize across split dispatches.

## 14. Numerical and autodiff policy

* fp16/bf16 inputs accumulate in fp32 by default.
* Norm and softmax statistics are fp32.
* Stable online softmax and stable log-sum-exp are required where applicable.
* Deterministic behavior is optional and compile-time selected; it is not a
  default concern or a reason to slow the fast variant.
* Non-trivial operations expose explicit `<fwd>` and `<bwd>` entry points.
  The backward contract lists saved forward tensors and whether gradients are
  assigned or accumulated. Slang `[Differentiable]`/`bwd_diff` may implement
  local elementwise derivatives, but the GPU entry-point ABI stays explicit.

## 15. Required family documentation

For every entry point, document:

* exact name using the naming rule above;
* workgroup size and specialization constants;
* compile-time generics/defines;
* dispatch formula including `dispatchBase` and batch folding;
* bindings, tensor-table indices, and access modes;
* `KernelHeader` plus hot-scalar offsets;
* workspace bytes, alignment, and initialization owner;
* read/write sets and required host barriers;
* canonical fast paths and slow-path routing;
* arithmetic intensity, roofline target, bottleneck, occupancy, register and
  shared-memory budget;
* forward/backward saved tensors and numerical tolerances;
* a **Profiling checklist** with exact benchmark shapes, dtypes, and expected
  throughput/bandwidth.
