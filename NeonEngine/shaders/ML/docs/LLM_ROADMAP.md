# SlangML LLM roadmap

Working notes for growing SlangML from the AF3/MLP kernel set into a library
that can run a ~20B-parameter decoder LLM. This file is the running to-do list,
decision log, and status tracker. Keep it current when kernels land.

Scope: the library kernels plus the first model host. **gpt-oss-20b is the
model being built** (decision 2026-11-02); its C++ host (weight loading, layer
loop, KV cache, CLI) is documented in `gpt_oss.md`. A tokenizer and batched
prefill come next.

The LLM kernels are inference only. Training (from scratch, gpt-oss architecture) uses its own `train/` kernel family (tensor-core GEMMs with dynamic loss scaling, corpus-trained BPE tokenizer, WSD schedule); see `gpt_oss_training.md`.

Family docs: `gpt_oss.md` (the gpt-oss model host), `quant.md` (formats, GEMV, MoE GEMV, GEMM, embedding),
`llm_ops.md` (norm, RoPE + KV cache, router, sampling), `llm_attention.md`,
`linear_attention.md` (Gated DeltaNet).

---

## 1. Target architectures

### 1.1 Qwen3.8-27B (same text stack as Qwen3.5-27B)

The original primary target; its kernels (Gated DeltaNet, gated attention,
MRoPE) are done, but the model host is gpt-oss first (section 1.2).

Source: the `config.json` on Hugging Face (`Qwen/Qwen3.8-27B`, identical to
`Qwen/Qwen3.5-27B` for the text stack), plus the transformers `qwen3_5` /
`qwen3_next` modelling code.

| Field | Value |
| --- | --- |
| Class | `Qwen3_5ForConditionalGeneration` (vision tower + text decoder) |
| Layers | 64 = 16 x (3 Gated DeltaNet + 1 gated full attention) |
| Hidden | 5120 |
| FFN | dense SwiGLU, intermediate 17408, no bias |
| Full attention | 24 Q heads, 4 KV heads (GQA group 6), head_dim 256 |
| Q projection | outputs `24 * 256 * 2`: per head `[q(256), gate(256)]` interleaved |
| Output gate | `attn_out * sigmoid(gate)` before `o_proj` |
| QK norm | RMSNorm over head_dim, weight applied as `(1 + w)`, before RoPE |
| RoPE | theta 1e7, partial: first 25% of head_dim (64 dims), rotate-half (NeoX) |
| MRoPE | interleaved, sections [11, 11, 10] over the 32 frequencies (T/H/W ids); text-only = plain 1-D RoPE |
| Linear attention (Gated DeltaNet) | 16 QK heads x 128, 48 V heads x 128, conv kernel 4 |
| DeltaNet projections | `in_proj_qkv` (5120 -> 10240, split q,k,v), `in_proj_z` (5120 -> 6144), `in_proj_b`, `in_proj_a` (5120 -> 48 each), `out_proj` (6144 -> 5120) |
| DeltaNet math | causal depthwise conv1d + SiLU on qkv; L2-norm q,k (eps 1e-6); q /= sqrt(128); `beta = sigmoid(b)`; `g = -exp(A_log) * softplus(a + dt_bias)`; QK heads `repeat_interleave`d to V heads (ratio 3) |
| DeltaNet state | fp32 `[48 heads][128][128]` per layer (`mamba_ssm_dtype float32`) |
| DeltaNet output | gated RMSNorm: `rmsnorm(o) * w * silu(z)` (plain `w`, not `1+w`) |
| Layer norms | RMSNorm `(1 + w)` zero-centred, eps 1e-6, computed in fp32 |
| Vocab | 248320, untied embedding / LM head |
| Context | 262144 native |
| Extras | 1 MTP layer (speculative decoding draft head) |

Recurrent decode step, per V head (from `torch_recurrent_gated_delta_rule`):

```text
S   = S * exp(g)                     # S is [dk][dv]
kv  = S^T k
d   = (v - kv) * beta
S   = S + k d^T
o   = S^T q
```

Parameter budget (text stack): full-attn layer 105M + MLP 267M; DeltaNet layer
116M + MLP 267M; embeddings 1.27B; LM head 1.27B. Total about 27B.

### 1.2 Model being built: gpt-oss-20b (see `gpt_oss.md`)

| Field | Value |
| --- | --- |
| Layers | 24, alternating sliding-window (128) and full attention |
| Hidden | 2880; 64 Q heads, 8 KV heads, head_dim 64; QKV bias |
| MoE | 32 experts, top-4, expert FFN 2880, router softmax over the top-k |
| Activation | clamped SwiGLU: `gate = min(gate, 7)`, `up = clamp(up, -7, 7)`, `out = (up + 1) * gate * sigmoid(1.702 * gate)` |
| Attention sinks | learned per-head logit added to the softmax denominator |
| RoPE | YaRN, theta 150000, factor 32, original ctx 4096 |
| Weights | MXFP4 experts (e2m1 + e8m0 scale per 32), bf16 attention/router/embeddings |
| Vocab | 201088 |

### 1.3 Other features worth supporting

Common in other open models (Llama 3, Qwen2/3 dense and MoE, Mistral,
Gemma 2/3, DeepSeek-V3):

* plain RMSNorm (`w`, not `1 + w`), LayerNorm with bias;
* full-head RoPE, GPT-J interleaved RoPE, linear/NTK/YaRN/llama3 frequency
  scaling (all of these are just a modified `inv_freq` table plus an
  attention scale);
* attention logit softcapping (Gemma 2), sliding window (Mistral, Gemma, gpt-oss);
* MoE with softmax or sigmoid routing, shared experts (Qwen-MoE, DeepSeek);
* GeGLU (Gemma), tied embeddings, embedding scaling.

---

## 2. What the existing library already had

Generic fp32 building blocks: elementwise (incl. SwiGLU/GeGLU), reductions,
softmax, LayerNorm/RMSNorm (plain `w`, one row per 256-thread group), a 16x16
portable fp32 GEMM, a naive GEMV, a naive scalar int8/int4 GEMM, and an fp16
"coopmat" entry that was actually a scalar loop. All AF3-specific.

Gaps that this work filled for LLM inference: fast quantized GEMV (decode is
~98% weight streaming), tensor-core GEMM (prefill), RoPE + KV cache, causal/GQA
flash attention, Gated DeltaNet + causal conv1d, sampling, embedding lookup,
MoE routing, and fused norm + residual + quantize.

---

## 3. Design decisions

D1. **Weights are the bottleneck; activations stay fp32.** The residual
stream and all activations are fp32 in global memory. Decode activations are
tiny (20 KB for hidden 5120), so fp32 costs nothing; it avoids fp16 overflow on
residual outliers. Prefill GEMM converts activation tiles to fp16 in shared
memory for the tensor cores.

D2. **Library-native weight formats; the host repacks at load.** Every
quantized weight is logically `[N][K]` (PyTorch `Linear`: out x in, `y = x W^T`),
quantized along K in 32-element blocks. Codes and scales are stored as separate
arrays (structure-of-arrays) so every load is aligned and coalesced. See
`quant.md` for the bit layout. Formats:

* `Q4` affine 4-bit: `w = d * q + m`, q in [0,15], `(d, m)` fp16 pair per group
  (group = 32/64/128 along K). Covers GGUF Q4_0/Q4_1/Q4_K (sub-blocks), GPTQ and
  AWQ (zero points fold into `m`).
* `Q8` affine int8: `w = d * q + m`, q in [-128,127]. Covers Q8_0.
* `F16`, `BF16` unquantized.

D3. **Decode GEMV uses int8 dot products.** Activations are quantized to
int8 blocks of 32 (`d_x = amax/127`) and the weight blocks are consumed with
`OpSDot` (dp4a). Q4 dequant costs about 0.5 instructions per weight this way,
versus ~3 for unpack + convert + FMA, which would be ALU-bound on a small GPU.
The block sum carried with the activations is `d * sum(q)` (the dequantized
sum), so the result equals the exact dot of dequantized weights and dequantized
activations; using the exact fp32 sum was measured 2-6x less accurate for Q4.

D4. **One attention kernel for decode and prefill.** A workgroup handles a set
of up to 8 "query vectors" (token, head) that share one KV head, so K/V are read
once per GQA group. The KV range can be split across workgroups
(flash-decoding) with a separate combine pass. Causal limits, sliding window,
sinks, softcap and the sigmoid output gate are runtime flags.

D5. **KV cache is fp16, layout `[kvHead][maxCtx][headDim]`.** Each head's rows
are contiguous, so a 32-lane subgroup reads one row per 16-byte-per-lane load.

D6. **DeltaNet recurrence keeps the state in registers.** One workgroup owns
32 value-columns of one head. Each thread holds 32 state values; the K-dimension
reduction stays inside a subgroup (shuffles, no barriers), so the per-token loop
has no workgroup barriers. The same kernel handles decode (T=1) and prefill
(sequential scan over T). A chunked (WY) tensor-core form is a later speed-up.

D7. **Fusion targets.** residual-add + RMSNorm + int8 quantize (one pass);
QK-norm + RoPE + KV-cache write (one pass); SwiGLU gate/up in one GEMV;
residual add in the GEMV epilogue; sigmoid output gate in the attention combine.

D8. **Contract compliance.** New kernels use the existing BDA `KernelHeader`
plus tensor table ABI and the `sml_<family>_<op>_<fwd>_<dtype>[_<variant>]`
naming. LLM kernels require contiguous, 16-byte aligned tensors whose rows are
multiples of 32 elements (the host allocates them that way); dims and strides
are hot scalars in the parameter block.

D9. **Explicit alignment on vector loads.** `-fvk-use-scalar-layout` makes
Slang emit `Aligned 4` on `float4*` / `uint4*` loads, so the driver splits them
into 32-bit loads. All LLM hot paths use `sml_ld_*` / `sml_st_*`
(`loadAligned<16>`), which made the prefill GEMM 1.4x and the M=4 GEMV up to
1.7x faster. Any new kernel must do the same (check for `Aligned 16` in
`spirv-dis` output).

---

## 4. Hardware notes (this dev machine)

* NVIDIA GeForce RTX 3050 Laptop GPU (Ampere GA107): **3.87 GiB VRAM**, 16 SMs,
  subgroup 32, 48 KiB shared/workgroup. Supports fp16/bf16, int8 dot product
  (accelerated), `VK_KHR_cooperative_matrix` with fp16 16x16x16 -> fp32,
  bf16 16x16x16 -> fp32, int8 16x16x32 -> int32, and `VK_NV_cooperative_matrix2`.
  Measured: ~155 GB/s device copy bandwidth, 15.3 TFLOP/s pure fp16 MMA.
* AMD Radeon integrated GPU: subgroup 64, no accelerated int8 dot, no coopmat.
* Slang 2025.17.2 (Vulkan SDK 1.4.328.1): `CoopMat` load/store works from both
  groupshared arrays and BDA pointers (the old note in `gemm.md` saying BDA
  loads are missing is out of date). `dot4add_i8packed` lowers to `OpSDot`.

**Capacity warning.** Qwen3.8-27B at ~4.5 bits/weight is ~15 GB; it cannot be
resident in 3.9 GiB. Running it on this GPU needs weight streaming from system
RAM (decode then bound by PCIe, roughly 1 token/s) or a bigger GPU. gpt-oss-20b
(MoE, 3.6B active) is far friendlier: only ~1.3 GB of expert weights are touched
per token, so expert offload is viable. This choice affects the host design,
not the kernels.

Decode budget per token (Qwen3.8-27B, Q4 g64 + Q8 LM head): about 14-15 GB of
weights + 64 KiB per context token of KV + ~300 MB DeltaNet state traffic.
Weight streaming dominates; the GEMV kernel must hit >= 85% of DRAM bandwidth.

---

## 5. Status

Legend: [x] done and tested on GPU, [~] partial, [ ] not started.

### P0: decode path

* [x] Activation int8 block quantize (`quant/quantize_f32.slang`)
* [x] Quantized GEMV Q4/Q8/F16/BF16, 1-8 activation rows, bias/residual
      epilogue, fused SwiGLU / GeGLU / gpt-oss SwiGLU (`quant/gemv_quant.slang`)
* [x] Fused residual-add + RMSNorm (`w` or `1+w`) + SiLU gate + int8 output
      (`llm/rmsnorm_fused_f32.slang`)
* [x] QK-norm + partial / NeoX / GPT-J / MRoPE (interleaved + contiguous) +
      YaRN tables + fp16 KV-cache append (`llm/rope_qknorm_f32.slang`)
* [x] Flash attention (decode + verify + prefill, GQA, causal, window, sinks,
      softcap, sigmoid output gate) + split-KV combine (`llm/attention_flash_f32.slang`)
* [x] Causal conv1d with rolling state, decode + prefill (`linear_attention/conv1d_f32.slang`)
* [x] Gated DeltaNet recurrent step / scan with fused l2norm + gates
      (`linear_attention/gated_delta_rule_f32.slang`)
* [x] Embedding lookup for all weight formats (`quant/embedding.slang`)
* [x] Argmax, top-k (K <= 64 at 248K vocab), temperature / top-p / min-p
      sampling with Philox, repetition / frequency / presence penalties
      (`llm/sampling_f32.slang`)

### P1: prefill and other models

* [x] Tensor-core (coopmat) GEMM with on-the-fly dequant, BM 128 and 64
      (`quant/gemm_quant_coopmat.slang`); ~35% of measured MMA peak so far
* [x] MoE router (softmax / sigmoid, top-k, renormalise, bias, scale)
      (`llm/moe_router_f32.slang`)
* [x] Expert-indexed gate/up + weighted down GEMV for MoE decode (`quant/gemv_moe.slang`)
* [x] MXFP4 weight format (e2m1 + e8m0, 4.25 bits): dense GEMV `_mxfp4_m{1,4}`,
      MoE `_mxfp4`, SWAR decode onto int8 dp4a (`slangml-llm.slang`)
* [x] MoE expert address table (flag 4): per-expert weight pointers, so experts
      can live in VRAM or host memory (PCIe) without kernel changes

### gpt-oss host (`gpt_oss.md`)

* [x] Headless Vulkan compute context, runtime Slang compile + SPIR-V cache (`src/ml_compute.*`)
* [x] Streaming safetensors reader, official checkpoint layout (`src/ml_safetensors.*`)
* [x] gpt-oss model: repack (MXFP4 split-nibble, gate/up split), bf16 or q8
      dense weights, per-layer expert placement (VRAM / host), YaRN table,
      sliding-window KV view, decode forward, greedy argmax (`src/gpt_oss_model.*`)
* [x] CLI `LveEngine.exe --gpt-oss` with scoring + logit dumps (`src/gpt_oss_app.*`)
* [x] End-to-end test vs the official PyTorch model (`tests/gpt_oss_e2e_test.py`)
* [ ] Run the real gpt-oss-20b checkpoint (needs a machine with >= 8 GB VRAM + RAM for offloaded layers)
* [ ] Tokenizer (o200k_harmony) + harmony chat format in C++ (Python helper: `tests/gpt_oss_tokens.py`)
* [ ] Batched prefill, sampling (top-k/top-p kernels exist), VRAM expert cache
* [x] Memorilla latent memory (off by default; `memorilla_plan.md`): memory module
      kernels, frozen-decoder training, `.nmem` stores, `--memorilla` inference
      (`src/memorilla*.cpp`, `tests/memorilla_{train,infer}_test.py`)

### Test and tooling

* [x] `tests/llm_harness.py`: device-local buffers, fp16/int8/coopmat features,
      multi-dispatch command buffers, GPU timestamps with clock warm-up,
      automatic `spirv-val` on every compile.
* [x] `tests/llm_kernel_tests.py`: 104 GPU checks against float64 NumPy references.
* [x] `tests/llm_bench.py`: benchmarks at Qwen3.8-27B shapes.
* [x] `tests/llm_formats.py`: reference weight/activation packers (the loader spec).
* [x] `tests/compile_llm.ps1`: compiles + validates all 52 LLM entry points.

---

## 6. Measured performance

RTX 3050 Laptop, copy bandwidth ~155 GB/s, pure-MMA ceiling 15.3 TFLOP/s.

| Kernel | Shape | Result |
| --- | --- | --- |
| GEMV Q4 M=1 | Qwen3.8 projections | 157-166 GB/s (100-108% of copy) |
| GEMV SwiGLU Q4 M=1 | 5120 -> 2 x 17408 | 159 GB/s |
| GEMV Q4 / Q8 M=4 | same | 91-143 / 131-155 GB/s |
| GEMV F16 M=1 | same | 134-160 GB/s |
| Coopmat GEMM Q4 | M = 512-2048 | 5.0-5.4 TFLOP/s |
| Coopmat GEMM Q4 `_m64` | M = 64 | 4.0-4.5 TFLOP/s |
| Attention decode D=256 | 8K / 32K context | 100 / 102 GB/s (64-65%) |
| Attention prefill D=256 | T = 2048 | ~1.0 TFLOP/s (SIMT) |
| Gated DeltaNet decode | 48 heads, 128x128 | 48 us, 84% of copy |
| Gated DeltaNet prefill | T = 512 | 2.1 ms |
| conv1d prefill | T = 512, C = 10240 | 87% of copy |
| RMSNorm + add + int8 | 5120 | 6.5 us |
| argmax / top-64 | V = 248320 | 6.3 us / 148 us |
| GEMV MXFP4 M=1 | 2880 x 2880 (gpt-oss expert) | 31.6 us, 139.6 GB/s (88.5% of copy) |
| MoE MXFP4 gate/up + down | gpt-oss-20b, top-4 | 330 us (68%) + 128 us (88%) |

Projected Qwen3.8-27B decode on a GPU with this bandwidth and enough memory:
~15 GB of weights per token / ~160 GB/s = ~95 ms, about 10 tokens/s before
attention and launch overhead.

Projected gpt-oss-20b decode, everything in VRAM: about 1.27 GB of expert
weights + 1.28 GB of bf16 attention weights + 1.16 GB LM head per token, so
about 3.7 GB per token (2.6 GB with `--dense q8`). At this GPU's bandwidth
that is ~25 ms (~40 tokens/s); a 16 GB card with 2-3x the bandwidth
proportionally faster.

---

## 7. To-do (prioritised)

1. ~~**Model-level decisions**~~ decided 2026-11-02: gpt-oss-20b on a 16 GB
   GPU (cloud) with an 8 GB local fallback; experts stay MXFP4, attention and LM
   head bf16 (or Q8 with `--dense q8`), experts offloaded per layer to host
   memory when VRAM is short. Next for the model: batched prefill, MoE gate/up
   kernel speed (68% of copy at H = I = 2880), sampling, tokenizer.
2. **Coopmat GEMM speed** (prefill): fp16 activation-input variant (norm /
   SwiGLU kernels write fp16, halving A traffic and removing conversion);
   BK = 64 with fewer barriers; try `VK_NV_cooperative_matrix2` tensor-addressed
   loads on NVIDIA; tile-shape autotune. Target >= 50% of 15.3 TFLOP/s.
3. **Decode attention** to >= 80% bandwidth: shared-memory broadcast of P
   instead of per-position shuffles, 2 positions per iteration, a QG = 6
   variant for Qwen (no wasted vectors); int8 KV-cache option.
4. **Prefill attention on tensor cores** (Q K^T and P V as 16x16x16 MMAs, O kept
   in shared memory for the online rescale).
5. ~~**MXFP4**~~ done (2026-11-02, SWAR decode, no LUT). **IQ4_NL table format** (16-entry LUT + per-block scale) so gpt-oss
   experts stay at 4.25 bits.
6. **Fusions**: SwiGLU GEMV writing int8 activations directly (one dispatch
   less per layer); QKV GEMV writing straight into the RoPE kernel's input
   layout; final norm + LM head.
7. **Chunked (WY) Gated DeltaNet** prefill on tensor cores for prompts > 4K.
8. **Top-k by radix select** (target < 30 us at 248K vocab).
9. **MTP / speculative decoding support**: DeltaNet state snapshot/restore
   kernel, conv1d state rollback.
10. **MoE prefill**: token permutation by expert + grouped GEMM over experts.
11. Vision tower ops if the multimodal path is needed (Conv3d patch embed,
    2-D RoPE, GELU-tanh MLP, LayerNorm with bias).

---

## 8. Findings / gotchas

* Laptop GPU clocks idle between submits; benchmarks must keep the GPU busy for
  ~150 ms first (the harness does). Without it results were up to 25x off.
* `break` inside a `[ForceUnroll]` loop produced SPIR-V with a dominance error
  (`spirv-val` catches it; the harness now runs it on every compile). Use a
  guarded body instead.
* `cond ? ptr : nullptr` does not type-check in Slang 2025.17; read the optional
  descriptor unconditionally and guard the dereference with the flag.
* `CoopMat.Load` from a `groupshared uint4[]` takes the stride in uint4 units,
  which allows 128-bit shared stores of fp16 tiles.
* Computing RoPE `inv_freq` in-shader is not exact enough for long contexts;
  the host table is the exact path.
* GEMM ablation (M=1024, N=K=5120): without the global fetch the loop runs at
  10.4 TFLOP/s, so the fetch/convert phase is the remaining bottleneck. Tile
  swizzling, a deeper register prefetch and hoisting parameters did not help.
* The AMD iGPU (64-wide subgroups, driver 2.0.279) passes all 104 checks except
  the coopmat GEMM, which it cannot run. Two driver rules came out of it, and
  every new kernel must follow them:
  1. no 16-bit BDA stores (they were dropped); pack pairs into 32-bit words;
  2. form addresses as integer byte offsets (`sml_base(...) + bytes`) and
     `reinterpret` once. Casting a pointer that already carries an offset
     (`float* + n` -> `float4*`) lost the offset, and `pointer + uint64`
     indices are avoided too.
* Host dispatch formulas that depend on rows per workgroup must use the real
  subgroup size (the GEMV/MoE kernels cover 128 / subgroupSize subgroups).

---

## 9. Change log

* 2026-11-02: Research done; roadmap written.
* 2026-11-02: gpt-oss-20b chosen as the first model. MXFP4 GEMV/MoE kernels and
  the MoE expert address table landed (115 GPU checks pass; MXFP4 GEMV 88.5% of
  copy at 2880 x 2880). C++ host (`src/ml_*`, `src/gpt_oss_*`) runs the model
  end to end; matches the official PyTorch model on a random tiny checkpoint
  (see `gpt_oss.md`, Testing).
* 2026-11-02: P0 + P1 kernels landed with GPU tests (104 checks) and
  benchmarks; family docs added; aligned-load fix applied to all LLM hot paths.
  All 104 checks pass on the RTX 3050 and on the AMD iGPU (coopmat skipped there).

---

## 10. How to run

```bash
cd shaders/ML/tests
python llm_kernel_tests.py            # all GPU correctness checks (or a name filter)
python llm_bench.py                   # all benchmarks (or: gemv, gemm, bench_attention, linear, small)
powershell -File compile_llm.ps1      # compile + spirv-val every LLM entry point
```

Set `SLANGML_DEVICE=<name substring>` to pick a GPU (default: first discrete GPU).

gpt-oss (after building the engine in Release):

```bash
python gpt_oss_e2e_test.py            # C++ host vs the official PyTorch model (tiny random checkpoint)
python llm_bench.py gptoss            # MXFP4 / MoE kernels at gpt-oss-20b shapes
```
