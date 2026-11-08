# Quantized linear layers (quant family)

Weight-streaming kernels for LLM inference: activation quantization, decode
GEMV (dense and MoE), prefill GEMM on cooperative matrices, and embedding
lookup. Sources: `slangml/quant/*.slang`, helpers in `slangml/slangml-llm.slang`.
Status and to-dos live in `LLM_ROADMAP.md`.

## Performance notes

* Decode (1 token) is pure weight streaming: arithmetic intensity is 2 FLOP
  per weight, so the GEMV must run at DRAM bandwidth. Measured on the RTX 3050
  Laptop (copy bandwidth 153-157 GB/s): Q4 M=1 157-166 GB/s, fused SwiGLU Q4
  159 GB/s, Q8 M=1 145-169 GB/s, F16 M=1 134-160 GB/s. (Above 100% of "copy"
  because a copy also writes.)
* Q4/Q8 use int8 dot products (`OpSDot`, dp4a): ~0.5 instructions per weight,
  versus ~3 for unpack + convert + FMA, which is ALU-bound on small GPUs.
* Multi-row GEMV (M = 2..8, for speculative/MTP verification or small batches)
  reuses each weight load across M activation rows. Measured Q4 M=4 91-143 GB/s,
  Q8 M=4 131-155 GB/s. Above M = 8, use the GEMM.
* Prefill GEMM is tensor-core bound. Measured 5.0-5.4 TFLOP/s (Q4, M >= 512)
  against a measured pure-MMA ceiling of 15.3 TFLOP/s on this GPU. The fetch
  phase (global -> registers -> fp16 shared tiles) is the bottleneck; see
  `LLM_ROADMAP.md` for the next steps (fp16 activations, BK = 64, coopmat2).
* All 128-bit accesses go through `sml_ld_*` / `sml_st_*` (`loadAligned`).
  Under `-fvk-use-scalar-layout`, plain `float4*` loads are emitted `Aligned 4`
  and the driver splits them into 32-bit loads; fixing this made the GEMM 1.4x
  and the M=4 GEMV up to 1.7x faster.

## Weight formats

Every weight is logically `[N][K]` (PyTorch `Linear`: out x in, `y = x W^T`).
`K` must be a multiple of 32; quantization blocks are 32 consecutive `k` of one
row. Codes and metadata are separate arrays, both row-major and 16-byte
aligned. Format tags are `SML_WFMT_*` in `slangml-llm.slang`.

| Format | Codes per 32-block | Metadata | Dequant | Bits/weight (group 32 / 64 / 128) |
| --- | --- | --- | --- | --- |
| `Q4` | 16 bytes (one `uint4`), split-nibble order | `half2 (d, m)` per group | `w = d * q + m`, q in [0, 15] | 5.0 / 4.5 / 4.25 |
| `Q8` | 32 bytes | `half2 (d, m)` per group | `w = d * q + m`, q in [-128, 127] | 9.0 / 8.5 / 8.25 |
| `MXFP4` | 16 bytes (one `uint4`), split-nibble order, e2m1 codes | one e8m0 byte `E` per block | `w = 2^(E-127) * e2m1(q)` | 4.25 |
| `F16` | `[N][K]` halves | none | | 16 |
| `BF16` | `[N][K]` bf16 | none | | 16 |

* Group size is 32 * 2^groupShift along K (`groupShift` hot scalar). Meta
  index: `n * (K / group) + k / group`.
* **Split-nibble order (Q4):** byte j (0..15) of a block holds element j in the
  low nibble and element j + 16 in the high nibble (the llama.cpp Q4_0 order).
  This differs from the generic packed-int4 rule in `CONTRACT.md` section 12 on
  purpose: `w & 0x0F0F0F0F` then lines up with activation bytes 0..15 and
  `(w >> 4) & 0x0F0F0F0F` with bytes 16..31, so the dot product needs no shuffles.
* Repacking (host, at load time; reference packers in `tests/llm_formats.py`):
  GGUF Q4_0 -> `m = -8 d`; Q4_1 -> `(d, m)`; Q4_K -> per-32 sub-block
  `d = d_super * sc_j`, `m = -dmin * m_j` (fp16 rounding of the product);
  GPTQ / AWQ -> `m = -zero * scale`; Q8_0 -> `(d, 0)`.
  MXFP4 (gpt-oss checkpoints): byte j of a checkpoint block holds element 2j
  (low nibble) and 2j + 1 (high); the loader moves the nibbles into split order
  and copies the scale bytes unchanged (`repack_mxfp4`, `src/gpt_oss_model.cpp`).
  IQ4_NL still needs its own table format.

### MXFP4 decode

e2m1 values {0, 0.5, 1, 1.5, 2, 3, 4, 6} doubled are the integers
{0, 1, 2, 3, 4, 6, 8, 12}, so a block is `2^(E-128) * sum(mag2 * sign * x)` and
runs on the same int8 dp4a path as Q4. `sml_e2m1x4_decode` turns four nibbles
into doubled magnitudes with SWAR adds (`mag = m + [m>=5] + [m>=6] + 3[m>=7]`,
no table lookups) and splits them by sign, so each 4 weights cost two dp4a into
a positive and a negative accumulator. `sml_e8m0_half_scale` builds `2^(E-128)`
exactly from the exponent bits. Measured at 2880 x 2880: 139.6 GB/s (88.5% of
copy), 31.6 us versus 36.5 us for Q4 g32 at the same shape (fewer bytes).
Results equal the exact dot of the official dequantized weights with the int8
activations (`gemv_q_ref("mxfp4", ...)`, max error 9e-7 relative).

## Activation int8 blocks

`sml_quant_quantize_q8act_fwd_f32` (`quantize_f32.slang`) and the
`SML_NORM_OUT_Q8` path of the fused RMSNorm produce:

* `Xq` int8 `[M][K]`, `Xs` float2 `[M][K/32]` = `(d, s)`, `d = amax / 127`,
  `s = d * sum(q)`.

`s` is deliberately the sum of the *dequantized* block: then the GEMV's affine
term `m * s` plus its int8 dot equals the exact dot of dequantized weights and
dequantized activations. Using the exact fp32 sum instead leaves the rounding
error multiplied by the un-centred Q4 codes (measured 1-3.6% error versus
0.5%). Measured end-to-end error of Q4/Q8 GEMV versus fp64 on dequantized
weights: ~0.5% relative L2.

Params (`QuantizeActParams`): header, `rows` 48, `cols` 52, `ldX` 56.
Tensors: 0 X fp32, 1 Xq, 2 Xs. Dispatch: x = ceil(cols / 4 / 256), y = rows.
Cost: ~2 us for one 5120 or 17408 row (launch bound).

## Decode GEMV (`gemv_quant.slang`)

```text
sml_quant_gemv_fwd_{q4,q8}_m{1,2,4,8}
sml_quant_gemv_fwd_{mxfp4,f16,bf16}_m{1,4}
sml_quant_gemv_swiglu_fwd_{q4,q8}_m{1,4}
sml_quant_gemv_swiglu_fwd_{f16,bf16}_m1
```

`y[m][n] = alpha * x[m] . W[n] (+ bias[n]) (+ residual[m][n])`; the `_swiglu`
form computes gate and up rows together and writes
`glu(alpha * x.Wg + bg, alpha * x.Wu + bu)` with `actMode` 0 = SiLU, 1 = GELU-tanh,
2 = gpt-oss clamped SwiGLU (`actLimit`, `actAlpha`).

`GemvParams` (scalar layout): header 0..47, `N` 48, `K` 52, `M` 56,
`groupShift` 60, `flags` 64 (1 bias, 2 residual), `alpha` 68, `ldY` 72, `ldR` 76,
`actMode` 80, `actLimit` 84, `actAlpha` 88.

Tensors: 0 W codes, 1 W meta, 2 Xq (Q4/Q8) or X fp32 (F16/BF16), 3 Xs, 4 Y,
5 bias (gate bias), 6 residual (up bias for `_swiglu`), 7 Wu codes, 8 Wu meta.
The residual may alias Y (in-place residual add is element-safe).

Mapping: 128 threads; each subgroup owns ROWS consecutive rows and strides its
lanes across K one 32-block per lane per step. ROWS is 2 / 4 / 4 / 8 for
`_m1 / _m2 / _m4 / _m8` (measured best; `-D SML_GEMV_ROWS=n` forces one value).
No shared memory, no barriers. Dispatch: x = ceil(N / (ROWS * 128 / subgroupSize)).
`M` may be below the variant's MC (extra rows are clamped and not written).

## MoE decode GEMV (`gemv_moe.slang`)

```text
sml_quant_gemv_moe_swiglu_fwd_{q4,q8,bf16,mxfp4}
sml_quant_gemv_moe_down_fwd_{q4,q8,bf16,mxfp4}
```

Experts are stacked `[E][N][K]`; `codeStride` / `metaStride` are bytes between
experts. A slot is one (token, expert) pair from `sml_llm_moe_router_fwd_f32`;
slot s belongs to token `s / slotsPerToken`.

* `_swiglu`: `y[s][n] = glu(x[tok(s)] . Wg[e_s][n] + bg, x[tok(s)] . Wu[e_s][n] + bu)`;
  grid x = row tiles (2 rows per subgroup), y = slots.
* `_down`: `y[tok][n] = sum_s w_s * (x[s] . Wd[e_s][n] + bd[e_s][n]) (+ residual)`;
  grid y = tokens. The weighted sum over a token's experts happens in registers
  before the subgroup reduction, so there is no scatter or atomic.

`MoeGemvParams`: header, `N` 48, `K` 52, `numSlots` 56, `groupShift` 60,
`flags` 64, `alpha` 68, `ldY` 72, `actMode` 76, `actLimit` 80, `actAlpha` 84,
`codeStride` 88, `metaStride` 92, `biasStride` 96, `slotsPerToken` 100.
Tensors: 0..8 as the dense GEMV (6 = residual for `_down`, up bias for
`_swiglu`), 9 expert ids i32 `[numSlots]`, 10 routing weights `[numSlots]`,
11 expert address table.

**Expert address table** (`flags & 4`): tensor 11 is `uint64 [E][4]` =
`{codes, meta, codes2, meta2}` byte addresses per expert (entries 2/3 are the up
matrix of `_swiglu`; `_down` uses 0/1). Tensors 0/1/7/8 and
`codeStride`/`metaStride` are then ignored. Each expert can live anywhere the GPU
can address: a VRAM buffer or host-visible system memory read over PCIe. The
host can move experts between tokens by rewriting the table; the kernels do not
change. Codes must be 16-byte aligned and meta 4-byte aligned.

Measured at gpt-oss-20b shapes (H = I = 2880, top-4, MXFP4, RTX 3050 Laptop):
`_swiglu` 330 us (106.7 GB/s, 68% of copy; each lane does only ~3 blocks per
matrix at this shape, see the roadmap), `_down` 128 us (138 GB/s, 88%), whole MoE
layer with the activation quantize 438 us.

## Prefill GEMM (`gemm_quant_coopmat.slang`)

```text
sml_quant_gemm_fwd_{q4,q8,f16,bf16}_coopmat       BM = 128
sml_quant_gemm_fwd_{q4,q8,bf16}_coopmat_m64       BM = 64 (short prompts)
```

`C[m][n] = alpha * A[m] . W[n] (+ bias[n]) (+ residual[m][n])`, A fp32
`[M][lda]`, C fp32 `[M][ldc]`. Weights in the formats above. A and dequantized
W are rounded to fp16 in shared memory; MMA is fp16 x fp16 -> fp32 16x16x16.
|A| must stay below 65504.

Tile BM x 128 x 32, 256 threads (8 subgroups of 32 lanes). Two shared stages
(40 KiB total, stored as `uint4` so every store is 128-bit; 80-byte row stride
keeps fragment loads conflict-free); the next tile is fetched into registers
while the current one is multiplied, one barrier per K tile. The epilogue reuses
the B stages. Full tiles without flags store accumulators straight to C.

`GemmQuantParams`: header, `M` 48, `N` 52, `K` 56, `groupShift` 60, `lda` 64,
`ldc` 68, `ldr` 72, `flags` 76 (1 bias, 2 residual), `alpha` 80.
Tensors: 0 W codes, 1 W meta, 2 A, 3 C, 4 bias, 5 residual.
Dispatch: x = ceil(N / 128), y = ceil(M / BM). Requires `VK_KHR_cooperative_matrix`
(fp16 16x16x16 -> fp32) and 32-wide subgroups.

## Embedding (`embedding.slang`)

`sml_quant_embedding_fwd_{q4,q8,f16,bf16}`: `out[t] = scale * dequant(table[tok[t]])`.
Same `[V][K]` layouts, so a tied LM head can share the buffer.
`EmbedParams`: header, `T` 48, `K` 52, `groupShift` 56, `ldOut` 60, `scale` 64.
Tensors: 0 codes, 1 meta, 2 tokens i32, 3 out fp32. Dispatch: x = ceil(K/4/256), y = T.

## Profiling checklist

```text
GEMV (decode), M = 1 and 4, Qwen3.8-27B shapes (N, K):
  (12288, 5120) q_proj+gate   (10240, 5120) in_proj_qkv   (6144, 5120)
  (5120, 6144) out_proj       (5120, 17408) down          swiglu 2 x (17408, 5120)
  LM head (248320, 5120) Q8
  formats Q4 g64, Q8 g32, F16
Healthy: M=1 >= 90% of copy bandwidth; M=4 >= 70%.
Underperforming: M=1 < 80% -> check 128-bit loads (`Aligned 16` in SPIR-V),
  rows per subgroup, occupancy.

gpt-oss-20b (python llm_bench.py gptoss): MXFP4 M=1 2880 x 2880 and the MoE pair
  (32 experts, top-4, H = I = 2880). Healthy: >= 85% of copy for both MoE kernels.

GEMM (prefill): M = 64 (_m64), 512, 2048 at the same (N, K); Q4, Q8, F16.
Healthy: >= 50% of the measured pure-MMA rate (bench: mma probe in LLM_ROADMAP.md).
Current: ~35%.
Report kernel time, achieved GB/s or TFLOP/s, and error vs fp64 on dequantized weights.
```
Run `python tests/llm_bench.py gemv` / `gemm`.
