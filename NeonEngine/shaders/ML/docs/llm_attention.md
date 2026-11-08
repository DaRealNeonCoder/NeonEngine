# Causal / GQA attention over an fp16 KV cache (llm family)

`slangml/llm/attention_flash_f32.slang`. One kernel serves decode, multi-token
verification (MTP / speculative decoding) and prefill.

```text
sml_llm_attention_fwd_f16kv_d{64,128,256}
sml_llm_attention_combine_fwd_f32          (only when numChunks > 1)
```

## Math

For query token t, q-head h (kv head `h / (Hq / Hkv)`) and visible positions p:

```text
s_p  = (q . k_p) * scale                     (q scaled once at load)
s_p  = softcap * tanh(s_p / softcap)         softcap > 0 (Gemma 2)
visible: p < limit(t) and p >= limit(t) - window (window > 0)
limit(t) = kvLenBase + t + 1 (SML_ATTN_CAUSAL 1), else kvLenBase
out  = sum_p softmax(s)_p v_p, denominator += exp(sink[h] - max)  (SML_ATTN_SINKS 2, gpt-oss)
out *= sigmoid(gate)                                              (SML_ATTN_GATE 4, Qwen3.5/3.8)
```

Online softmax in fp32; finite -1e30 stands in for minus infinity so fully
masked blocks never produce NaN.

## Layouts

* Q fp32 `[T][qTokenStride]`, head h at `h * D` (the RoPE kernel's output).
* K / V cache fp16 `[Hkv][maxCtx][D]` (written by `sml_llm_rope_qknorm_fwd_f32`).
* Gate fp32 at `t * gateTokenStride + h * gateHeadStride + d` (Qwen's q_proj
  output: stride `2 D` per head, tensor base offset `D`).
* Out fp32 `[T][outTokenStride]`.
* Workspace for split-KV: O fp32 `[numChunks][T * Hq][D]`, ML float2
  `[numChunks][T * Hq]` (max, sum). Written only for chunks that start below
  the group's last limit; the combine pass reads only chunks below each token's
  limit, so nothing needs clearing.

`AttnParams`: header, `numQHeads` 48, `numKVHeads` 52, `T` 56,
`tokensPerGroup` 60, `qTokenStride` 64, `maxCtx` 68, `kvLenBase` 72,
`chunkSize` 76, `numChunks` 80, `scale` 84, `softcap` 88, `window` 92,
`flags` 96, `gateTokenStride` 100, `gateHeadStride` 104, `outTokenStride` 108.
The combine pass appends `headDim` at 112.
Tensors: 0 Q, 1 K cache, 2 V cache, 3 Out, 4 sinks, 5 gate, 6 ws O, 7 ws ML.

Grid: x = numChunks, y = ceil(T / tokensPerGroup), z = Hkv. Combine: x = T * Hq.

## Mapping and performance notes

A workgroup (128 threads = 4 groups of 32 lanes) owns one KV head, a group of
tokens and one chunk of positions. Its query vectors are every (token, q-head)
pair of that KV head, at most 8 (`tokensPerGroup * G <= 8`; Qwen3.8 G = 6,
gpt-oss G = 8), so every K/V row is read once per GQA group. Dims are split D/32
per lane. Per 32-position block:

1. QK: each lane forms its partial dot products for the 8 vectors, and one
   reduce-scatter butterfly (9 shuffles) leaves lane L with vector `(L>>2)&7`'s
   score. It keeps positions `j = 4 jj + (L & 3)` in 8 registers.
2. Block max (2 shuffles), rescale O, exponentiate; fully visible blocks skip
   masking (all decode blocks).
3. PV: `p` broadcast by shuffle, 8 x D/32 FMAs per position.

The four groups merge through 16 KiB of shared memory. 32-lane clusters make it
work on 64-wide subgroups too.

Measured on the RTX 3050 Laptop, Qwen3.8 full-attention layer (24/4 heads, D = 256):

| Context | chunk | Time | KV bandwidth |
| ---: | ---: | ---: | ---: |
| 1024 | 256 | 81 us | 52 GB/s |
| 8192 | 1024 | 336 us | 100 GB/s (64% of copy) |
| 32768 | 1024 | 1321 us | 102 GB/s (65% of copy) |

Prefill (T = 512 / 2048 on an empty cache, single chunk): ~1.0 TFLOP/s. That is
SIMT fp32 and ALU-bound; a cooperative-matrix version is on the roadmap. At
4K tokens attention is ~2% of Qwen3.8 prefill FLOPs, so the GEMMs dominate.

Host heuristics:

* Decode: choose `chunkSize` (multiple of 32) so `numChunks * Hkv` is at least
  2-4x the SM count; short contexts need small chunks (ctx 1024 with 4 splits
  only fills 16 workgroups).
* Prefill: `tokensPerGroup = floor(8 / G)`, one chunk covering the whole
  context (the kernel writes the final output directly), or split chunks for
  very long contexts.

## Numerical contract

fp32 accumulation, fp16 K/V (rounding error ~5e-4 relative on the cached
values). Measured max error vs fp64 on the stored fp16 cache: <= 5e-7 relative.

## Profiling checklist

```text
decode T = 1: Hq/Hkv/D = 24/4/256 (Qwen3.8), 64/8/64 (gpt-oss, window 128 + sinks),
              32/8/128 (Llama-style); ctx 1K, 8K, 32K, 128K; chunk 256 and 1024
verify T = 2..4 (MTP)
prefill T = 512, 2048, 8192 from an empty cache and after 4K cached tokens
```
Healthy decode: >= 80% of copy bandwidth on contexts >= 8K (current 65%).
Healthy prefill: >= 50% of fp32 FMA peak for the SIMT path, then move to tensor
cores. Run `python tests/llm_bench.py bench_attention`.
