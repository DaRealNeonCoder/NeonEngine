# Linear attention: Gated DeltaNet (linear_attention family)

Qwen3-Next / Qwen3.5 / Qwen3.8 use Gated DeltaNet for 3 of every 4 layers.
Sources: `slangml/linear_attention/{conv1d,gated_delta_rule}_f32.slang`.
The output norm (`rmsnorm(o) * w * silu(z)`) is `sml_llm_rmsnorm_fwd_f32_small`
with `SML_NORM_GATE_SILU` (see `llm_ops.md`).

Block order for one Qwen3.8 DeltaNet layer (decode or prefill):

```text
h  = rmsnorm(x)                             (1+w)
qkv = in_proj_qkv(h)  z = in_proj_z(h)  a = in_proj_a(h)  b = in_proj_b(h)   GEMV/GEMM
qkv = silu(conv1d(qkv))                     sml_linattn_conv1d_fwd_f32
o   = gated_delta_rule(q, k, v, a, b)       sml_linattn_gated_delta_rule_fwd_f32
o   = rmsnorm(o) * w * silu(z)              per V head (rows = T * Hv, cols = dv)
x  += out_proj(o)                           GEMV with residual epilogue
```

## Causal depthwise conv1d (`conv1d_f32.slang`)

`sml_linattn_conv1d_fwd_f32`:

```text
y[t][c] = act( sum_{j<W} w[c][j] * in(t - (W-1) + j)[c] + bias[c] )
in(tau) = x[tau] (tau >= 0) or stateIn[(W-1) + tau][c] (zeros without HAS_STATE)
stateOut[i][c] = in(T - (W-1) + i)
```

Flags: `SILU 1`, `BIAS 2`, `HAS_STATE 4`, `WRITE_STATE 8`. W <= 8 (Qwen: 4).
State layout is `[W-1][C]` (channel-minor, coalesced). HF stores
`conv_state` as `[C][W]`, so transpose on import. HF's weight `[C, 1, W]` maps to `w [C][W]`.

`Conv1dParams`: header, `T` 48, `C` 52, `W` 56, `flags` 60, `ldX` 64, `ldY` 68,
`tokensPerThread` 72. Tensors: 0 x, 1 w, 2 bias, 3 stateIn, 4 y, 5 stateOut.
Grid: x = ceil(C / 128), y = ceil(T / tokensPerThread). One thread per channel
keeps the window in registers (x read once, y written once). `stateOut` may
alias `stateIn` only when grid y = 1 (decode).

Measured: T = 512, C = 10240: 311 us (87% of copy bandwidth). T = 1: 2.5 us.

## Gated delta rule (`gated_delta_rule_f32.slang`)

`sml_linattn_gated_delta_rule_fwd_f32`, decode step and prefill scan:

```text
kh   = h / (Hv / Hk)                         (HF repeat_interleave)
q    = l2norm(q_t[kh]) * qScale,  k = l2norm(k_t[kh])     l2norm: x * rsqrt(sum x^2 + eps)
g    = -exp(A_log[h]) * softplus(a_t[h] + dt_bias[h]),  beta = sigmoid(b_t[h])
S    = exp(g) * S;  delta = beta * (v_t - S^T k);  S += k delta^T;  o_t = S^T q
```

`GATES_PRECOMPUTED 1`: a and b already hold g and beta.

`GdnParams`: header, `T` 48, `numKHeads` 52, `numVHeads` 56, `dv` 60,
`ldQKV` 64, `qOffset` 68, `kOffset` 72, `vOffset` 76, `ldAB` 80, `ldOut` 84,
`qScale` 88, `eps` 92, `flags` 96. `dk` is compile-time `SML_GDN_DK`
(default 128). Tensors: 0 qkv (the conv output; HF split order q, k, v with
heads contiguous), 1 a, 2 b, 3 A_log, 4 dt_bias, 5 state fp32 `[Hv][dk][dv]`
(read and written in place), 6 out `[T][ldOut]`.

Grid: x = dv / 32, y = Hv. A workgroup owns 32 value columns of one head;
each thread holds dk/4 = 32 state values in registers (lane = column + 8 * row
slice), so the dk reduction is two xor-shuffles and the per-token loop has no
barriers. Tokens are staged 16 at a time in shared memory (q, k, v, norms,
gates), so their loads overlap.

Measured (Qwen3.8 layer: Hk 16, Hv 48, dk = dv = 128):

| T | Time | Notes |
| ---: | ---: | --- |
| 1 | 48 us | 6.3 MB state read + write, 84% of copy bandwidth |
| 512 | 2.1 ms | sequential scan; ~4 us per token per layer |

The scan is latency-bound for long prompts. At T = 512 it is ~3% of the layer's
GEMM time (392 GFLOP at 5.3 TFLOP/s = 74 ms), so it is not the prefill bottleneck yet. The chunked WY form
(tensor-core matmuls per 64-token chunk, as in flash-linear-attention) is the
roadmap item for long prompts.

Numerics: fp32 state and math; matches the HF `torch_recurrent_gated_delta_rule`
to ~2e-7 relative over 37 tokens.

Speculative decoding note: the recurrence overwrites the state in place. To
roll back rejected draft tokens, snapshot the state (6.3 MB/layer) before
verification, or run verification with a copy (roadmap item).

## Profiling checklist

```text
gated_delta_rule: T = 1, 16, 512, 4096; Hk/Hv = 16/48 (Qwen3.8), 16/32 (Qwen3-Next)
conv1d: T = 1, 512, 4096; C = 10240 (Qwen3.8), 8192
```
Healthy decode: >= 80% of copy bandwidth on the state traffic. Prefill: below
10% of the layer's GEMM time. Run `python tests/llm_bench.py linear`.
