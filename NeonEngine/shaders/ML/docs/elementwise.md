# Elementwise and broadcasting kernels

## Entry points

Implemented in `slangml/elementwise/elementwise_f32.slang`:

```text
sml_elementwise_{neg,abs,exp,log,sqrt,rsqrt,reciprocal,sin,cos,tanh,sigmoid,relu,silu,gelu,softplus,mish,
                 leaky_relu,prelu,elu,selu,softsign,hardswish,hardsigmoid,gelu_exact,square,sign}_fwd_f32_contig
sml_elementwise_{neg,abs,exp,log,sqrt,rsqrt,reciprocal,sin,cos,tanh,sigmoid,relu,silu,gelu,softplus,mish,
                 leaky_relu,prelu,elu,selu,softsign,hardswish,hardsigmoid,gelu_exact,square}_bwd_f32_contig
sml_elementwise_fused_relu_silu_fwd_f32_contig
sml_elementwise_{add,sub,mul,div,min,max,where}_fwd_f32_broadcast
sml_elementwise_{add,mul}_bwd_f32_same
```

Additional entry points in `elementwise_extra_f32.slang` cover:

```text
sml_elementwise_{pow,atan2,mod,lt,le,gt,ge,eq,ne,logical_and,logical_or,logical_xor}_fwd_f32_broadcast
sml_elementwise_pow_bwd_f32_same
sml_elementwise_{swiglu,geglu,glu}_fwd_f32_contig
```

`elementwise_generate_f32.slang` provides `copy`, `fill`, `arange`,
`linspace`, and `eye` generation/copy kernels.

The `_contig` entry points accept the canonical contiguous fast path and also
route non-contiguous descriptors through the shared strided helper. The engine
should select a future dedicated `_strided` variant for heavily strided or
rank>3 tensors rather than slowing the common path.

## Parameter block

The BDA push constant is `KernelPushConstants`. `headerAddress` points to an
`ElementwiseParams` block:

| Offset | Field |
| ---: | --- |
| 0..47 | `KernelHeader` |
| 48 | `outputRank : uint` |
| 52 | `opFlags : uint` |
| 56 | `alpha : float` |
| 60 | `beta : float` |

Tensor table entries:

* unary forward: `0=input`, `1=output`;
* unary backward: `0=input`, `1=upstream`, `2=output`;
* fused ReLU-SiLU forward: `0=input`, `1=output`;
* binary broadcast forward: `0=a`, `1=b`, `2=output`;
* binary same-shape backward: `0=a`, `1=b`, `2=upstream`, `3=da`, `4=db`.

All tensors are fp32 for this first implementation. The host canonicalizes
descriptors before dispatch and sets `numel` to the output logical element
count.

## Dispatch and workspace

Workgroup size is `256x1x1`. It is currently a compile-time entry-point
constant; the next tuning pass can split 128/256 variants without changing the
ABI. For a logical output domain `N` and base `B`:

```text
dispatch.x = ceil_div(N - B, 256)
dispatch.y = 1
dispatch.z = 1
```

Each invocation handles `linear = dispatchBase.x + SV_DispatchThreadID.x`.
No workspace is required. Read/write sets are exactly the tensor entries listed
above. The host barrier is a storage-buffer write-to-read or write-to-write
barrier between successive uses of the output.

Broadcast forward uses `outputRank` and canonical shape/stride metadata. Binary
backward is intentionally same-shape only; broadcast gradients must route to a
separate reduction-to-input-shape kernel so the fast same-shape path remains
free of atomics.

## Fused-elementwise framework

`IElementwiseOp` supplies compile-time operation objects with `apply` and
`derivative`. `ReluOp`, `SiluOp`, `GeluTanhOp`, and `ReluSiluChain` demonstrate
the generic chain mechanism. A chain is compiled into one kernel and performs
one global load and one global store.

## Performance notes

The contiguous path is memory-bound. Target at least 85% of sustainable device
bandwidth for large fp32 tensors; arithmetic intensity is approximately one
or a few FLOPs per 8-byte load/store, so launch overhead and address math are
the main small-tensor bottlenecks. The `ALIGNED16` path should use vec4 loads
and stores in the next optimized specialization. Register pressure should stay
below roughly 32 scalar registers per thread; shared memory is zero.

The fused chain is preferred over separate ReLU and SiLU dispatches because it
removes one global-memory round trip. Fast math uses native `exp`, `tanh`, and
reciprocal operations; the documented activation error should be measured
against the reference kernel before enabling approximation-specific variants.

## Profiling checklist

Benchmark these exact cases with fp32 inputs/outputs:

```text
N = 1, 31, 256, 4096, 1_048_576, 16_777_216: neg, exp, relu, silu
[rows, cols] = [1024, 4096], [4096, 1024], [1, 16_777_216]: add/mul broadcast
pow/atan2/mod/comparison/logical kernels: [1024,4096], [4096,1024]
GLU/SwiGLU/GeGLU: [1024,4096], [8192,4096] packed two-gate inputs
copy/fill/arange/linspace/eye: 1, 31, 256, 4096, 1M elements
rank = 2 and 3, contiguous and innermost-stride-1 non-contiguous
```

Healthy results: large unary/fused kernels at >=85% sustained memory bandwidth;
binary broadcast at >=75% because of extra index arithmetic; small tensors
dominated by launch latency. Profile achieved bandwidth, global-load/store
efficiency, branch divergence, registers/thread, and dispatch time first.
Comparison/logical kernels should stay within 15% of add bandwidth on the same
shape. Gated activations are math-bound when sigmoid/GELU dominates; compare
them against the equivalent unfused two-dispatch sequence and require the
fused kernel to win on total time.
