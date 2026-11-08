# LLM block ops (llm family): norm, RoPE + KV cache, MoE router, sampling

Fused per-token kernels around the projections. Sources:
`slangml/llm/{rmsnorm_fused,rope_qknorm,moe_router,sampling}_f32.slang`.
All math is fp32; tensors are contiguous and 16-byte aligned.

## Performance notes

These kernels move little data per decode step; they are launch- and
latency-bound (a few microseconds each). Their job is to remove dispatches and
global round trips, not to hit peak bandwidth:

* fused residual add + RMSNorm (`1 + w`) + int8 quantize of a 5120 row: 6.5 us,
  replacing add, norm and quantize dispatches;
* QK-norm + RoPE + fp16 KV-cache append in one pass per head;
* argmax over 248320 logits: 6.3 us (100% of copy bandwidth);
* top-64 over 248320 logits: 148 us (iterative extraction; radix select is a
  roadmap item). Use argmax for greedy decoding.

## Fused RMSNorm (`rmsnorm_fused_f32.slang`)

```text
sml_llm_rmsnorm_fwd_f32_row     one 256-thread workgroup per row (cols <= 8192)
sml_llm_rmsnorm_fwd_f32_small   one 32-lane group per row, 4 rows/workgroup (cols <= 1024)
```

```text
h = x (+ r)                         SML_NORM_ADD_RESIDUAL   1
H = h                               SML_NORM_WRITE_SUM      2   (may alias x or r)
y = h * rsqrt(mean(h^2) + eps) * w  w -> 1 + w with SML_NORM_UNIT_OFFSET 4 (Qwen3.5/3.8)
y *= silu(z)                        SML_NORM_GATE_SILU      8   (DeltaNet output norm)
Y = y fp32                          SML_NORM_OUT_F32        16
Yq, Ys = int8 blocks of y           SML_NORM_OUT_Q8         32  (GEMV input format)
```

`RMSNormFusedParams`: header, `rows` 48, `cols` 52 (multiple of 32), `eps` 56,
`flags` 60, `ldX` 64, `ldR` 68, `ldY` 72 (also H), `ldZ` 76.
Tensors: 0 x, 1 r, 2 w, 3 Y, 4 H, 5 z, 6 Yq, 7 Ys. The row stays in registers
between the statistics and output passes, so H may alias x or r.
Dispatch: `_row` x = rows; `_small` x = ceil(rows / 4).

## QK-norm + RoPE + KV-cache append (`rope_qknorm_f32.slang`)

`sml_llm_rope_qknorm_fwd_f32`, grid x = Hq + Hkv (+ Hkv when writing V), y = T,
64 threads per head vector (D <= 256 by default).

* Q heads: `q = rope(norm_q(Qsrc))` -> `Qout` fp32 `[T][Hq][D]`.
* K heads: `k = rope(norm_k(Ksrc))` -> `Kcache` fp16 `[Hkv][maxCtx][D]` at slot(t).
* V heads (`SML_ROPE_WRITE_V`): copy to `Vcache` fp16 at slot(t).

Flags: `NORM_Q 1`, `NORM_K 2`, `UNIT_OFFSET 4` (`1 + w`), `INTERLEAVED 8`
(GPT-J pairs (2i, 2i+1); default is NeoX rotate-half (i, i + rd/2)), `MROPE 16`,
`MROPE_INTERLEAVED 32`, `FREQ_TABLE 64`, `WRITE_V 128`, `SLOT_TABLE 256`.

* Partial rotary: only the first `rotaryDim` dims rotate (Qwen3.8: 64 of 256).
* Frequencies: pass a host table of `inv_freq[i]` computed in double and
  rounded to fp32 (`SML_ROPE_FREQ_TABLE`). This matches HF exactly and covers
  linear / NTK / YaRN / llama3 scaling; `attnFactor` multiplies cos and sin
  (YaRN mscale). The in-shader `theta^(-2i/rd)` fallback has ~3e-7 relative
  error per frequency, which reaches ~2e-4 absolute error by position 5000.
* Angles are `float(pos) * inv_freq` in fp32 (as HF), then range-reduced with a
  two-constant 2*pi so sin/cos stay accurate: measured 1e-6 error at
  position 262K.
* MRoPE: positions are `[3][posStride]` (T, H, W ids). Interleaved (Qwen3-VL,
  Qwen3.5/3.8, sections [11, 11, 10]): frequency i uses H if `i % 3 == 1 && i < 3*s1`,
  W if `i % 3 == 2 && i < 3*s2`, else T. Contiguous (Qwen2-VL): T for i < s0,
  H for i < s0 + s1, else W. For text-only input all three rows are equal.
* Slots: `slots[t]` with `SLOT_TABLE`, else `slotBase + t` (the cache index is
  the sequence index, not the RoPE position, which differs for vision tokens).
* Q source strides let the kernel read Qwen's interleaved `[q, gate]` q_proj
  output directly (`qHeadStride = 2 * D`).

`RopeParams`: header, `T` 48, `numQHeads` 52, `numKVHeads` 56, `headDim` 60,
`rotaryDim` 64, `flags` 68, `eps` 72, `log2Theta` 76, `attnFactor` 80,
`qTokenStride` 84, `qHeadStride` 88, `kTokenStride` 92, `kHeadStride` 96,
`vTokenStride` 100, `vHeadStride` 104, `maxCtx` 108, `slotBase` 112,
`posStride` 116, `section0..2` 120..128.
Tensors: 0 Qsrc, 1 Ksrc, 2 Vsrc, 3 q-norm w, 4 k-norm w, 5 positions i32,
6 Qout, 7 Kcache, 8 Vcache, 9 inv_freq, 10 slots.

## MoE router (`moe_router_f32.slang`)

`sml_llm_moe_router_fwd_f32`: logits `[T][E]` -> expert ids `[T][k]` and
weights `[T][k]`, descending score, ties to the lower index. E <= 512, k <= 32.

* softmax (default): `w = softmax(logits)[topk]`, with `NORM_TOPK 2`
  renormalised over the top-k (= softmax over the selected logits; Mixtral,
  Qwen-MoE, gpt-oss);
* `SIGMOID 1` (DeepSeek-V3 style, no group limits): `s = sigmoid(logits)`,
  selection on `s + bias` with `SELECT_BIAS 4`, weights `s` (optionally
  renormalised);
* every weight is multiplied by `scaleOut`.

`RouterParams`: header, `T` 48, `E` 52, `topK` 56, `flags` 60, `scaleOut` 64,
`ldLogits` 68. Tensors: 0 logits, 1 selection bias, 2 ids, 3 weights.
Dispatch: x = ceil(T / 4) (one 32-lane group per token).

## Sampling (`sampling_f32.slang`)

```text
greedy:    sml_llm_argmax_partial_fwd_f32  (x = chunks, y = rows)
           sml_llm_argmax_final_fwd_f32    (x = rows)
sampling:  sml_llm_topk_partial_fwd_f32    (x = chunks, y = rows)
           sml_llm_topk_merge_fwd_f32      (x = rows)
           sml_llm_sample_topk_fwd_f32     (x = rows)
penalties: sml_llm_logits_penalty_fwd_f32  (x = ceil(pairs / 256), y = rows)
```

A chunk is 4096 logits (`SML_SAMPLE_PER_THREAD = 16`). The merge holds
`numChunks * K <= 4096` candidates, i.e. V <= 262144 at K = 64.

`SampleParams`: header, `R` 48, `V` 52, `ldLogits` 56, `K` 60, `numChunks` 64,
`temperature` 68, `topP` 72, `minP` 76, `seed` 80, `step` 84, `flags` 88
(1 = read `u` from tensor 4 instead of Philox).

Sampling order (matches HF warpers on the top-k set): temperature, softmax,
min-p (`p < minP * p_max` dropped), top-p (smallest prefix with cumulative
probability >= topP), renormalise, inverse CDF with
`u = Philox(seed, counter = step, stream = row)`. `temperature <= 0` picks the
top candidate. Tensor 3 receives the filtered probabilities (useful for MTP
acceptance tests).

Penalties (`PenaltyParams`: header, `R` 48, `numPairs` 52, `ldLogits` 56,
`repetition` 60, `frequency` 64, `presence` 68) take unique (token, count)
pairs: HF repetition penalty, then `l -= frequency * count + presence`.

## Profiling checklist

```text
rmsnorm_row 1 x 5120, 1 x 2880, 512 x 5120 (all flags)   healthy: < 10 us per row at decode
rmsnorm_small 48 x 128 with GATE_SILU + OUT_Q8           healthy: < 10 us
rope_qknorm T = 1 and 512, Qwen3.8 (24 + 4 + 4 heads)    healthy: < 10 us decode
argmax V = 248320                                         healthy: >= 85% copy bandwidth
top-k K = 64, V = 248320                                  current 148 us; target < 30 us
moe_router T = 1 and 512, E = 32 / 128 / 256              healthy: < 10 us decode
```
Run `python tests/llm_bench.py small`.
