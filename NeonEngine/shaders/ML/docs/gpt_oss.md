# gpt-oss on SlangML

Running OpenAI's open-weight **gpt-oss** models (target: `gpt-oss-20b`) on the
SlangML LLM kernels, with a C++ Vulkan host in the engine. Training smaller gpt-oss
models from scratch, whose exports this host runs, is in `gpt_oss_training.md`. Status and to-dos
live in `LLM_ROADMAP.md`; kernel contracts in `quant.md`, `llm_ops.md` and
`llm_attention.md`.

## Source of truth

The translation follows the official reference implementation,
[openai/gpt-oss](https://github.com/openai/gpt-oss) (Apache-2.0):

* `gpt_oss/torch/model.py`: the model math (RMSNorm, YaRN RoPE, attention
  with sinks and alternating sliding window, top-k MoE, clamped SwiGLU).
* `gpt_oss/torch/weights.py`: the checkpoint layout and MXFP4 decode.
* `gpt_oss/tokenizer.py`: o200k_harmony (tiktoken o200k_base + harmony tokens).

`tests/gpt_oss_reference.py` is `model.py` vendored at commit `2770201`
with only the checkpoint loader removed; the end-to-end test runs it as the
reference.

## gpt-oss-20b architecture

| Field | Value |
| --- | --- |
| Layers | 24; even layers use a 128-token sliding window, odd layers full attention |
| Hidden | 2880; RMSNorm eps 1e-5 with an fp32 scale |
| Attention | 64 query heads, 8 KV heads (GQA group 8), head_dim 64, QKV and output bias, per-head learned sink logit in the softmax denominator, scale 1/8 |
| RoPE | NeoX halves, theta 150000, YaRN factor 32 on 4096 (NTK-by-parts alpha 1, beta 32), cos/sin times the concentration `0.1 ln 32 + 1` |
| MoE | 32 experts, top-4, softmax over the 4 selected router logits, router bias |
| Expert MLP | `mlp1` 2880 -> 2 x 2880 with rows interleaved (even = gate, odd = linear), `swiglu(g, u) = (clamp(u, ±7) + 1) * min(g, 7) * sigmoid(1.702 min(g, 7))`, `mlp2` 2880 -> 2880; biases on both |
| Weights | experts MXFP4 (e2m1 + e8m0 per 32), everything else bf16 |
| Vocab | 201088, untied embedding / unembedding |

## Code map

| File | Role |
| --- | --- |
| `src/ml_compute.{hpp,cpp}` | Headless Vulkan 1.3 compute context (no window, so it runs on cloud GPUs): enables BDA, scalar layout, 8/16-bit storage, int8 dot product, subgroup control, coopmat; buffers (VRAM / host / BAR), staged uploads, runtime Slang compile with an on-disk SPIR-V cache (`%TEMP%/neonengine_spv`), dispatch recorder, timestamps, `ParamArena` for per-dispatch parameter blocks. |
| `src/ml_safetensors.{hpp,cpp}` | Streaming `.safetensors` reader (header only in RAM; tensors read on demand) plus bf16/fp16 helpers. |
| `src/ml_json.{hpp,cpp}` | Minimal JSON parser (config + safetensors headers). |
| `src/gpt_oss_model.{hpp,cpp}` | Config, weight repacking and upload, KV cache, the per-token forward pass. |
| `src/gpt_oss_app.{hpp,cpp}` | `LveEngine.exe --gpt-oss` (or headless `NeonML --gpt-oss`) command line: prompt scoring, greedy or sampled generation (`--temperature`, `--top-k`, `--seed`), logit dumps. |
| `slangml/quant/gemv_*.slang` | MXFP4 GEMV and MoE kernels (`_mxfp4`), expert address table. |
| `tests/gpt_oss_e2e_test.py` | End-to-end test against the official model. |
| `tests/gpt_oss_tokens.py` | Text <-> token ids (tiktoken) until the host has a tokenizer; `--model <dir>` uses the tokenizer a from-scratch export names in its `config.json` (GPT-2 or its own `tokenizer.tiktoken` BPE). |

## Running

1. Download the official checkpoint's `original/` folder (13 GB):
   `huggingface-cli download openai/gpt-oss-20b --include "original/*" --local-dir gpt-oss-20b`
2. Build the engine (Release), then from `NeonEngine/NeonEngine`:

```text
python shaders/ML/tests/gpt_oss_tokens.py chat "What is the capital of France?" > prompt.txt
build/Release/LveEngine.exe --gpt-oss gpt-oss-20b/original --tokens-file prompt.txt --max-new 128
python shaders/ML/tests/gpt_oss_tokens.py decode <ids printed by the engine>
```

Options: `--ctx N` (default 4096), `--dense bf16|q8`, `--experts auto|device|host`,
`--score --dump-logits f.bin` (teacher-forced logits for every prompt position),
`--device <name>`, `--shaders <shaders/ML dir>`. The first run compiles ~15
kernels (about a minute); later runs load them from the SPIR-V cache.

## Memory plan (gpt-oss-20b)

| Part | Size | Where |
| --- | ---: | --- |
| Experts, MXFP4 (24 layers x 32 experts) | 10.15 GB (423 MB per layer) | VRAM or host memory, per layer |
| Attention + router weights | 1.28 GB bf16 / 0.68 GB q8 | VRAM |
| Unembedding (LM head) | 1.16 GB bf16 / 0.61 GB q8 | VRAM |
| Embedding table | 1.16 GB bf16 | host memory (one 5.6 KB row read per token) |
| KV cache | 48 KiB per token (200 MB at 4K context) | VRAM |

* **16 GB GPU** (the cloud target): everything fits in VRAM, about 12.8 GB with
  bf16 dense weights or 11.6 GB with `--dense q8`.
* **8 GB GPU**: `--experts auto` keeps as many layers' experts in VRAM as fit
  (about 11 of 24 with bf16, about 14 with `--dense q8`, at 4K context) and leaves the rest in
  host-visible system memory. The MoE kernels read those over PCIe through the
  expert address table, without stalls or uploads per layer. Each host layer
  costs 4 x 13.2 MB of PCIe reads per token, and the machine needs enough RAM
  for the host layers plus the embedding (about 6 to 7 GB).
* `--experts host` puts every expert in system memory (test path).

## Forward pass (one token, one command buffer)

```text
x = embedding[token]                                   sml_quant_embedding_fwd_bf16 (table in host memory)
per layer:
  t = rmsnorm(x)                                       sml_llm_rmsnorm_fwd_f32_row (fp32 + int8 outputs)
  qkv = t Wqkv^T + b                                   sml_quant_gemv_fwd_{bf16,q8}_m1
  q, k = rope_yarn(q, k); K/V cache append (fp16)      sml_llm_rope_qknorm_fwd_f32 (FREQ_TABLE | WRITE_V)
  a = attention(q, cache, sinks, window)               sml_llm_attention_fwd_f16kv_d64 (+ combine when split)
  x += a Wo^T + bo                                     sml_quant_gemv_fwd_*_m1 (bias + in-place residual)
  t = rmsnorm(x)
  logits_e = t Wg^T + bg                               sml_quant_gemv_fwd_bf16_m1 (32 rows)
  ids, w = top4 + softmax                              sml_llm_moe_router_fwd_f32 (NORM_TOPK)
  h = swiglu_oai(t W1^T + b1) for 4 experts            sml_quant_gemv_moe_swiglu_fwd_mxfp4 (expert table)
  x += sum_e w_e (h_e W2_e^T + b2_e)                   quantize + sml_quant_gemv_moe_down_fwd_mxfp4
logits = rmsnorm(x) Wu^T; next = argmax                sml_quant_gemv_fwd_*_m1 + argmax partial/final
```

Load-time repacking: `mlp1` rows are split into separate gate (even rows) and up
(odd rows) matrices so the fused SwiGLU kernel can pair them, and MXFP4 nibbles
are reordered into split-nibble order. Sliding-window layers shift the KV-cache
view (base address + `start * D`) so attention only walks the last 128
positions. Prompt tokens skip the LM head.

## Testing

`python tests/gpt_oss_e2e_test.py` (and `--seq 300` for many KV chunks) builds a
small random gpt-oss checkpoint in the official format (4 layers, 8 experts,
hidden 256, sliding window 16, the same RoPE settings and GQA group as 20b) and
checks three things:

1. A NumPy float64 restatement of the GPU pipeline equals the official PyTorch
   model (2e-7; the official RoPE runs in fp32).
2. The GPU equals that restatement with the GPU's int8 activations, fp16 KV
   cache and Q8 weights emulated, to within the noise floor of rounding (an
   untrained random network amplifies single int8 rounding flips). The tolerance
   is calibrated by perturbing the emulation's quantizer inputs at fp32 scale.
3. Accuracy against the official model: median logit error 1.3% (bf16 dense) /
   2.9 to 4.2% (q8 dense), top-1 agreement 97 to 98% / 92 to 93%. A trained model
   is far less sensitive than this random one.

Both expert placements are tested (VRAM and host memory give identical results).
The real 20b checkpoint has not been run yet: the dev machine (RTX 3050 Laptop,
4 GB, 7.3 GB RAM) cannot hold it.

## Known limits and next steps

* Prefill runs token by token (decode path). Batch it: GEMM coopmat for the
  dense projections, multi-token attention (supported), and MoE prefill (token
  permutation + grouped GEMM over experts, roadmap item 10).
* MoE gate/up kernel reaches 68% of copy bandwidth at H = I = 2880 (short K
  loops); interleave gate and up loads in one loop or use more rows per subgroup.
* Greedy decoding only on the CLI; the top-k / top-p sampling kernels exist and
  need wiring.
* No built-in tokenizer or harmony chat formatting in C++ yet.
* Expert cache: with the address table, hot experts of host layers can be
  promoted into a VRAM pool between tokens (LRU by router counts).
