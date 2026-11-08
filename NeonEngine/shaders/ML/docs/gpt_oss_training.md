# Training gpt-oss from scratch on SlangML

NeonEngine can train a gpt-oss-architecture model from scratch on its own
Vulkan/Slang kernels and then run it with the existing inference host
(`gpt_oss.md`). The architecture is the official one (`gpt_oss/torch/model.py`):
RMSNorm, QKV with bias, YaRN RoPE, attention with learned sinks and alternating
sliding-window / full layers, GQA, top-k MoE with softmax over the selected
logits and clamped SwiGLU, untied LM head. Only the size changes.

## Quick start

```text
python train/prepare_data.py --out data/tinystories --vocab-size 8192    # trains a BPE tokenizer, writes uint16 tokens
build/Release/NeonML.exe --gpt-oss-train train --config train/configs/t4.json --data data/tinystories --out run
build/Release/NeonML.exe --gpt-oss-train export --ckpt run --out run/export
python shaders/ML/tests/gpt_oss_tokens.py --model run/export encode "Once upon a time" > p.txt
python shaders/ML/tests/gpt_oss_tokens.py --model run/export eot                       # stop token id
build/Release/NeonML.exe --gpt-oss run/export --tokens-file p.txt --max-new 200 --temperature 0.8 --top-k 50 --stop <eot>
python shaders/ML/tests/gpt_oss_tokens.py --model run/export decode <ids>
```

On Google Colab (T4) use `train/colab/NeonEngine_T4_train.ipynb`: it installs the
Vulkan SDK and the NVIDIA Vulkan driver pieces, builds NeonML, prepares data on
Drive and trains with checkpoints mirrored to Drive.

`NeonML` is a headless build target (no GLFW) with the same `--gpt-oss` and
`--gpt-oss-train` commands as `LveEngine.exe`; on Linux configure with
`-DNEON_BUILD_ENGINE=OFF`.

## Presets (`train/configs/`)

| Preset | Layers | Hidden | Heads (q/kv) | Experts (top-k) | Expert FFN | Params with 8k vocab (non-embedding) | Use |
| --- | ---: | ---: | --- | --- | ---: | ---: | --- |
| `tiny` | 2 | 128 | 4 / 2 | 4 (2) | 128 | 2.7 M (0.6 M) | local smoke tests |
| `t4` | 8 | 512 | 8 / 2 | 16 (4) | 512 | ~115 M (~106 M; ~31 M active per token) | Colab T4, 16 GB |

`"vocab_size": 0` takes the vocabulary from the data's `meta.json`, padded to a
multiple of 64 (8192 for the 8k BPE tokenizer, 50304 for GPT-2 tokens). head_dim
is 64 (required by the attention kernels). The full gpt-oss-20b does not fit any
single 16 GB GPU for training (fp32 weights + Adam alone need 335 GB).

## Code map

| File | Role |
| --- | --- |
| `slangml/slangml-train.slang` | `slangml_train` module: tensor pointers, MXFP4 rounding rule |
| `slangml/train/gemm_train_coopmat.slang` | tensor-core GEMM (fp16 inputs, fp32 accumulate) for every linear-layer product, split-K, M-/K-grouped MoE modes; one entry point per operand layout |
| `slangml/train/gemm_train_f32.slang` | the same products in fp32 SIMT (`--precision fp32`, and shapes the tensor-core kernel cannot take) |
| `slangml/train/attention_train_coopmat.slang` | causal GQA attention with sinks and window on tensor cores (flash-attention style): forward, dq, dk/dv |
| `slangml/train/attention_train_f32.slang` | the same three kernels in fp32 SIMT (`--precision fp32`) |
| `slangml/train/moe_train_f32.slang` | routing sort, gather, SwiGLU fwd/bwd, combine fwd/bwd, router bwd + aux loss |
| `slangml/train/train_ops_f32.slang` | embedding, RMSNorm, RoPE, column sums, cross-entropy, grad norm, AdamW (skips non-finite steps), MXFP4 fake quant |
| `src/gpt_oss_train.{hpp,cpp}` | trainer: layout, init, step recording, loss scaling, LR schedule, data, checkpoints, export, CLI |
| `train/prepare_data.py` | tokenizer -> `train.bin` / `val.bin` / `meta.json` (+ `tokenizer.tiktoken`) |
| `train/bpe_tokenizer.py` | byte-level BPE training, tiktoken-format save / load |
| `tests/gpt_oss_train_test.py` | gradients vs the official model (float64 autograd), `--precision fp32/fp16` |
| `tests/train_gemm_test.py` | both training GEMMs vs NumPy for every layout, mode and epilogue; `--bench` times them |

The forward router is the inference kernel (`llm/moe_router_f32.slang`, NORM_TOPK).

## One training step

Per micro-batch (one command buffer, `N = batch * seq` tokens):

```text
x0 = E[tokens]
per layer l:  t1 = rmsnorm(x); qkv = t1 Wqkv^T + b; rope(q, k); O, L = attn(qkv, sinks, window if l even)
              x = x + O Wo^T + bo
              t2 = rmsnorm(x); z = t2 Wg^T + bg; ids, w = top-k softmax(z)
              sort (token, slot) entries by expert into 128-row padded segments
              H1 = Xe W1_e^T + b1_e (grouped GEMM); h = swiglu(H1); Y2 = h W2_e^T + b2_e
              x = x + sum_s w_s Y2[slot s]
logits = rmsnorm(x) Wu^T in chunks of --lm-chunk rows, fused softmax cross-entropy -> dlogits in place
backward in reverse; weight gradients accumulate into one flat fp32 buffer
```

Then gradient norm (two-pass sum of squares) and AdamW with clipping, fused
with zeroing the gradients. Saved per layer: t1, qkv (rotated), O, L, t2, rstd,
router logits/ids/weights, routing tables, H1, Y2. h and Xe are recomputed in
the backward. Residual-stream snapshots are kept for every block.

Losses: mean next-token cross-entropy plus a Switch-style load-balancing loss
per MoE layer, `aux * E * sum_e f_e mean_n P_ne` (default `--aux 0.01`, not part
of the official model, which was not trained from this code). The log's
`balance` column is that loss divided by `aux`, so 1.0 means perfectly even routing.

## Mixed precision (tensor cores)

`--precision auto` (default) runs every GEMM and the attention products on the
tensor cores when the device has cooperative matrices (T4, any RTX): A and B are
rounded to fp16 while they are staged in shared memory, products accumulate in
fp32, and everything else (activations, gradients, weights, optimizer, norms,
softmax statistics) stays fp32. `--precision fp32` is the SIMT path; `fp16`
insists on tensor cores.

Attention (`attention_train_coopmat.slang`) computes Q K^T, P V, dO V^T, dS K,
P^T dO and dS^T Q on the tensor cores with q, k, v, dO, P and dS in fp16. A
workgroup of 4 subgroups owns 64 query rows (forward, dq) or 64 key rows (dk/dv),
16 per subgroup, kept as fp16 fragments in registers; the other side streams
through shared memory 64 rows at a time. Accumulator layouts are opaque in
`VK_KHR_cooperative_matrix`, so each S / dP block goes through a small fp32
scratch where lanes apply the causal / window mask and the softmax, and P / dS
return as fp16 fragments; the forward keeps an online softmax (the sink starts
the denominator) and rescales O with a stride-0 broadcast load of the per-row
factor. dO carries the loss scale like every other gradient.

fp16 gradients would underflow, so the loss is scaled (dynamic loss scaling, as
in PyTorch AMP): the cross-entropy gradient and the router's aux-loss gradient
carry a factor S (initially `--loss-scale 65536`), AdamW divides it out (it also
enters the clipping norm). If the gradient norm is inf / NaN, the AdamW kernel
skips the update, the trainer halves S and prints `fp16 gradient overflow`;
after 1000 clean steps S doubles (up to 2^24). S is saved in checkpoints and
logged in `train_log.csv`.

The tensor-core kernel takes any M, N, K (edges are zero-filled) when the
contiguous strides are multiples of 4 floats; other shapes fall back to the fp32
kernel automatically. Full tiles load C / residual / bias straight into the
accumulators (the bias as a stride-0 load) and store them directly.

## Tokenizer

`prepare_data.py --vocab-size 8192` trains a byte-level BPE tokenizer
(`train/bpe_tokenizer.py`, GPT-2 pre-tokenization, merges learned on the first
`--tokenizer-sample-mb` 200 MB of training text) and encodes the corpus with
it through tiktoken. `tokenizer.tiktoken` travels with the data, the run
directory and the export (`config.json` gets `"neon_tokenizer":
"tokenizer.tiktoken"`), so `gpt_oss_tokens.py --model <export>` finds it.
`--tokenizer <file>` reuses a tokenizer; without either option the corpus is
encoded with a tiktoken encoding (`--encoding gpt2`, the old default).

Why: for small models the LM head is the biggest single cost of a step (40% of
GPU time for a 4-layer t4-sized model with GPT-2's 50k vocabulary), and the
logits, cross-entropy and embedding tables scale with the vocabulary too. A
corpus-specific vocabulary is just as compact: on TinyStories validation text,
GPT-2 gives 4.08 bytes per token, an 8k BPE 4.14 and a 4k BPE 3.98.

`meta.json` records the bytes behind each split, and the trainer prints
validation loss in **bits per byte** (`val 3.37 (1.17 bpb)`, column `val_bpb`),
which, unlike per-token loss, is comparable across tokenizers.

## Learning-rate schedule

`--schedule wsd` (default) is warmup-stable-decay: linear warmup, constant
`--lr`, then a linear cooldown to `--min-lr` over the last `--decay-frac` (0.2)
of `--steps`. Until the cooldown starts, a run can be extended by raising
`--steps` on resume. `cosine` (the old default) and `linear` are available. The
QAT start (80% of the steps by default) coincides with the cooldown.

## Quantization-aware training and export

The exported checkpoint has the official layout (`config.json` +
`model.safetensors`): dense weights bf16, norms fp32, experts MXFP4
(`.blocks` / `.scales`). From `--qat-start` (default 80% of `--steps`; `-2` turns
it off) the forward and dX products use MXFP4-rounded expert weights while
AdamW updates the fp32 masters (straight-through gradient), so the 4-bit export
matches what the network was trained with. The rounding rule is identical in the
GPU fake quantizer, the C++ exporter and `tests/llm_formats.py`. MXFP4 values
are exact in fp16, so QAT and tensor cores combine without extra rounding.

## Checkpoints and resume

`<out>/latest.nckpt` (and the previous one as `prev.nckpt`): magic, JSON header
(model config, step, best val loss, loss scale), fp32 params, Adam m, Adam v, end
magic. It is about 12 bytes per parameter (1.4 GB for `t4` with 8k tokens).
Files are written to a temp name and renamed, and are only accepted when size
and end magic check out. A checkpoint is written every `--ckpt-minutes` (30), on
Ctrl-C / SIGTERM, and at the end. `--mirror <dir>` copies each one in a
background thread (temp + rename). On start the trainer resumes from the newest
valid checkpoint in `--out` or `--mirror` (the model config then comes from the
checkpoint; `--fresh` ignores them). Batches are a pure function of (seed,
step), so a resumed run reads the same data it would have read. The LR schedule
comes from the command line, so resume with the same `--schedule` and `--steps`.

## Profiling

`--profile 5` runs two warm-up steps, then times every dispatch of the next 5
steps with GPU timestamps, prints a table (ms per step, share, calls, kernel or
GEMM shape) and exits without writing a checkpoint. Use it to check a new GPU
or config.

## Verification

* `python tests/gpt_oss_train_test.py [--batch 3 --seq 256]` runs one micro-batch
  on a tiny random model through `NeonML --gpt-oss-train gradcheck` and compares
  the loss and all 29 parameter gradients with the official PyTorch model in
  float64 (aux loss off). fp32 (default for gradcheck): loss difference 4e-7,
  worst relative gradient error 1.3e-6 (seq 64 and 3 x 256).
* `--precision fp16` (GEMMs and attention on tensor cores): worst 7.7e-4
  (seq 64) and 2.3e-3 (3 x 256: the router bias gradient, a small sum of
  cancelling terms; sinks 1.5e-3, all others <= 8e-4). With the test config
  patched to 8 / 2 heads and a 100- or 128-token window at 2 x 512: worst 7.2e-4. This mode
  routes every token to all experts: with top-k, a random-init router has
  near-tied logits, fp16 rounding flips a few choices, and a token sent to
  another expert changes every downstream gradient (a few % at 3 x 256),
  which is not a GEMM error.
* `python tests/train_gemm_test.py` checks both GEMMs against NumPy for all four
  operand layouts, split-K, both grouped modes (including empty groups and
  unused tiles), bias / residual / accumulate, alpha, ragged edges and the direct
  store path (tensor cores within 3e-6 of an fp16-input reference).
* Tiny-model A/B on 3 M TinyStories tokens, RTX 3050 Laptop (600 steps of
  8 x 256, `--eval-batches 40`):

  | Run | val bits/byte | tokens/s |
  | --- | ---: | ---: |
  | GPT-2 vocab, fp32, cosine (the old defaults) | 1.207 | 25.5 k |
  | GPT-2 vocab, fp16 tensor cores, cosine | 1.207 | 43 k |
  | GPT-2 vocab, fp16, wsd | 1.151 | 43 k |
  | 8k BPE, fp16, cosine | 1.190 | 123 k |
  | 4k BPE, fp16, cosine | 1.265 | 140 k |
  | **8k BPE, fp16, wsd** (new defaults) | **1.140** | 123 k |

  fp16 learns like fp32; seed-to-seed noise is about 0.001 bpb. WSD with decay
  fractions 0.1 to 0.3 and min-lr 0 or 1e-4 all land at 1.140 to 1.144. The
  tiny model also prefers a higher peak (`--lr 2e-3`: 1.117); that does not
  transfer to the 512-wide t4 preset, whose default stays 1e-3.
* End to end: the exported 8k-vocab MXFP4 checkpoint runs through `--gpt-oss`
  and generates TinyStories-style text; resume from the mirror alone keeps the
  step and loss scale; a forced `--loss-scale 1e12` backs off through 19
  skipped steps and then trains normally.

## Performance notes and limits

* Measured on an RTX 3050 Laptop, a 4-layer t4-sized model (hidden 512, 16
  experts top-4), 4 x 512 tokens per step:

  | Setup | ms per step |
  | --- | ---: |
  | fp32 GEMMs, GPT-2 vocab (old) | 384 |
  | tensor cores, GPT-2 vocab | 207 |
  | tensor cores, 8k BPE | 115 |
  | + tensor-core attention | 116 (vs 137 for the line above, re-measured the same day) |

  GEMM alone: 2.1-2.3x over fp32 on this GPU (GeForce Ampere runs fp16 with
  fp32 accumulate at half rate). A T4 runs it at full rate (65 vs 8.1 TFLOPS
  fp32), so the gain there should be larger; not yet measured.

  The laptop GPU's clocks drift between sessions, so compare rows measured
  together: on the same day, the same profile (`--profile 5`, 4 layers, 4 x 512,
  8k BPE) went from 137 to 116 ms per step when attention moved to the tensor
  cores (the all-fp32 path, `--precision fp32`, took 221 ms):

  | Kernel (ms per step, 4 layers) | fp32 SIMT | tensor cores |
  | --- | ---: | ---: |
  | dk / dv | 11.55 | 2.70 |
  | dq | 8.40 | 2.03 |
  | forward | 6.03 | 1.77 |
  | total attention | 26.0 (19% of the step) | 6.5 (5.6%) |

* What is left after that (8k vocab profile): MoE expert GEMMs about 45%,
  AdamW 11% (memory-bound over all parameters once per optimizer step, so
  `--accum` amortizes it), attention 6%.
* Weight-gradient GEMMs with few output tiles (qkv, out, router) use split-K into
  a partial buffer plus a reduce pass.
* Attention: head_dim 64 only; seq must be a multiple of 64. The tensor-core
  forward runs at roughly 2.3 TFLOPS on the 3050 Laptop (rough FLOP count), about
  a fifth of its fp16 / fp32-accumulate tensor rate, so there is headroom
  (larger key tiles, fewer shared-memory round trips); the fp32 kernels
  use one query or key row per 4 lanes and 32-row shared tiles.
* Column sums (bias, norm and sink gradients) use 4 row lanes per 64 columns.
  They are cheap but not tuned.
