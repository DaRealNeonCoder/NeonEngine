# Memorilla on NeonEngine: implementation plan

Status (2026-11-08): **P0, P1, P3, P4 and P6 built and tested; P2 built (self
encoder); P5 run locally and on the Colab T4** (trained memory 4.5% lower answer
loss than closed book with the `t4` decoder). P7 not started. See
[Section 11](#11-implementation-status).
Paper: *Memorilla: Latent Semantic Memory for LLMs*, Fallahpour, Idehpour, Kothapalli
et al. (Stanford), arXiv 2610.21375v3, 3 Nov 2026. Local copy:
`NeonEngine/memorilla stanford..pdf`.

> **Off by default.** Memorilla must be a setting that is **disabled unless asked
> for**. With it off, the gpt-oss inference host and the trainer behave exactly as
> they do today: no extra kernels compiled, no extra buffers, no change in output
> (bit-identical logits). Every phase below keeps this true, and Phase 0 adds a
> test for it. See [Section 5](#5-settings-off-by-default).

---

## 1. What Memorilla is (one paragraph)

A frozen embedding model turns each document chunk into one vector, once, and
keeps the vectors in a store `Z_D` (`N x h_e`). For each query, a small trainable
**memory module** reads the whole store and writes **K = 16 soft tokens** (vectors
of the LLM's input-embedding width). Those vectors go in front of the prompt as if
they were token embeddings, and a **frozen** LLM answers. Only the memory module
is trained (next-token loss on answer tokens only). Documents never enter the
context as text, so the context stays at K tokens however large the store gets.

## 2. Paper digest (what we need to build it)

### 2.1 Architecture (Section 2, Table 3)

Inputs: `Z_D = [f_enc(d_1) ... f_enc(d_N)]` (`N x h_e`), `z_q = f_enc(q)` (`h_e`).

| Stage | Math | Notes |
| --- | --- | --- |
| 1. Slots | `Q0` (`K x h_e`), learned, Xavier-uniform | shared across all stores and queries |
| 2. Query conditioning | `Qc = LN(Q0 + CrossAttn(Q=Q0, K=z_q, V=z_q))` | see 2.2: reduces to a broadcast add |
| 3. Retrieval, `L_ref = 2` blocks | `Ẑ = LN(Z_D)` (once); `R = R + CrossAttn(LN(R), Ẑ, Ẑ)`; `R = R + FFN(LN(R))` | Perceiver-style latent array, 8 heads, FFN = 4x width with GELU, dropout 0.1, **no self-attention** |
| 4. Projection | `M = LN(R_Lref) W_proj` (`h_e -> h_L`) | linear, with bias |
| Injection | `X_LLM = [m_1 ... m_K ; embed(query tokens)]` | RoPE over the combined sequence as usual; no LLM change |
| Loss | `-sum log P(y_i | y_<i, X_LLM)` over answer tokens only | memory and prompt positions carry no loss |

Paper setup: encoder Qwen3-Embedding-4B (`h_e = 2560`, mean-pooled per document),
decoder Qwen3-8B (`h_L = 4096`), K = 16.

### 2.2 Two facts the paper does not spell out

1. **Parameter count reproduces exactly.** With 4x FFN, biases on every linear,
   `W_q/W_k/W_v/W_o` in all three cross-attentions, one LayerNorm on `Z_D` shared by
   both blocks, 7 LayerNorms in total and a projection bias, the count is
   **194,123,776** for `h_e = 2560, h_L = 4096`, which is exactly the paper's number
   (and 190,190,080 for `h_L = 2560`, also exact). So this reading of the
   architecture is the right one. Formula:
   `4h² + 2(4h² + 8h²) + h·h_L + K·h + 14h + 12h + 2(5h) + h_L`, `h = h_e`.
2. **Stage 2 is not real attention.** The query is one vector, so the softmax is
   over a single key and is always 1. `CrossAttn(Q0, z_q, z_q) = W_o(W_v z_q + b_v) + b_o`
   for every slot, whatever `Q0` is. `W_q` and `W_k` get zero gradient. We implement
   Stage 2 as two linears and a broadcast add (`Qc = LN(Q0 + u)`, `u` one row). Keep
   `W_q/W_k` out of the parameter file; note the difference in the param counter.

### 2.3 Training recipe (Section 2.6, Tables 4-5, decoded from the garbled PDF table)

| | Stage 1 | Stage 2 | Stage 3 |
| --- | --- | --- | --- |
| Data | 1.98M Wikipedia paraphrase reconstruction | 1.34M mixed QA / reading / summarization | 8 benchmark train splits |
| Init | random | from stage 1 | from stage 2 |
| Optimizer | AdamW, cosine with warmup | same | same |
| Peak LR | 3e-4 | 3e-4 | 1e-4 |
| Weight decay | 1e-2 | 1e-2 | 1e-2 |
| Warmup | 3% | 3% | 10% |
| Max decoder length | 1024 | 1024 | 2048 |
| Effective batch | 128 | 128 | 64 |
| Epochs | 1 | 1 | 5 |

bf16, document embeddings precomputed and cached, chunks of about 128 tokens
(Table 2: 13 to 280 tokens per chunk depending on the benchmark).

### 2.4 Results that shape the design

* **K = 16 is enough.** Training at K = 128 did not help (Fig. 9). At inference
  the first `k < K` slots can be used without retraining, with graceful loss
  (Fig. 10). -> support `--mem-k` at inference.
* **Query conditioning before reading is what matters** (Table 13): query-free
  pooling 29-35, cosine top-16 34, Q-Former 43, Memorilla 44 on PMv2. Replacing
  the documents with noise costs Memorilla 5.9 points but the documents-only
  variant nothing. -> the noise test is our "is it really reading the store" check.
* **Not LoRA in disguise** (Fig. 11): a LoRA of similar size without documents loses.
* **Encoder size barely matters** (Fig. 12, 4B vs 8B: within 1.3 points). -> a
  small encoder is acceptable.
* **Weak spots:** exact surface details (who / which questions, short sources like
  PubMedQA). RAG and Memorilla answer different questions (Fig. 5), so a hybrid is
  the paper's suggested next step.
* **Cost:** the module forward is about 2 ms on an H100 regardless of N; end to end
  is dominated by decoding (Table 14). The store is 5 KB per document.
* **Agent / continual mode** (Section 6, App. G): one document per turn, memory
  recomputed every turn and placed before the system prompt, K = 8, one extra
  self-attention layer over documents, trained with GRPO from environment reward.
  Out of scope for the first version (Section 8, P7).

## 3. Where NeonEngine stands

Repo: `Downloads/NeonEngine/NeonEngine`. The LLM stack is our own: Slang kernels
(`shaders/ML/slangml`), a Vulkan compute host (`src/ml_compute.*`), the gpt-oss
inference host (`src/gpt_oss_model.*`, `gpt_oss_app.cpp`, doc `gpt_oss.md`) and a
from-scratch trainer (`src/gpt_oss_train.*`, doc `gpt_oss_training.md`). No PyTorch
at runtime; Python is used for data prep and float64 reference tests.

| Need | What exists | Gap |
| --- | --- | --- |
| Frozen decoder | gpt-oss host (per-token decode, int8 activations, MXFP4 experts); trained `t4` preset (~115 M) | an input path that takes a vector instead of a token id |
| Backprop through a frozen decoder | trainer computes `dX` all the way to `resid[0]` (`dR` before `sml_train_embed_bwd_f32`) | a frozen mode that skips weight gradients and AdamW for the LLM, and reads `dR` at memory rows |
| Answer-only loss | `sml_train_xent_f32` gives loss 0 and gradient 0 when the target is `SML_INVALID` | loss mean over valid targets, not over all `N` rows |
| Encoder | none (no embedding model ported) | see decision D1 |
| LayerNorm fwd/bwd | `norm/layernorm_f32.slang` | check for tests; wire into the training-style host |
| GELU fwd/bwd | `elementwise_f32.slang` (`gelu`, `gelu_exact`) | none |
| Linear layers fwd/bwd | trainer GEMMs (`gemm_train_coopmat.slang`, `gemm_train_f32.slang`) | none |
| Cross-attention (16 queries over N keys, head dim `h_e/8`) | `pair_bias_attention` is self-attention, D <= 64, untested; LLM attention is causal | new (D4) |
| Softmax rows | `softmax_loss/softmax_f32.slang` | masked variant for padded documents |
| Dropout | `pairformer/dropout_residual_f32.slang` | optional |
| Mean pooling | `reduce/` kernels | segment mean over document tokens |
| AdamW | `sml_train_adamw_f32` (flat buffer, decay split) | reuse on a separate memory-module buffer |
| Checkpoints | `.nckpt` writer/reader, atomic rename, mirror | a module checkpoint with the same rules |

## 4. Design decisions (with the recommended choice)

**D1. Encoder.** Options:
(a) *Self-encoder*: the frozen gpt-oss model itself, mean-pooled hidden state of
each chunk (`h_e = h_L`). No new model to port, works today for `t4` and 20b.
(b) *External encoder*: a real embedding model (Qwen3-Embedding-0.6B, `h_e = 1024`)
run in Python offline, written in our store format.
(c) Port Qwen3-Embedding into the engine (QK-norm RoPE and d128 attention already
exist; dense SwiGLU, last-token pooling and its tokenizer do not).
**Recommend (a) first, (b) as an import path** (the store format is just vectors, so
the module does not care who made them), (c) only if the user wants live encoding
with an external model. Make the pooled layer configurable (`encoder_layer`,
default last, after the final RMSNorm), since middle layers of a decoder often embed
better. The paper's store is LLM-independent; with (a) it is tied to the decoder,
which is fine for our use but should be written down in the store header.

**D2. Which frozen LLM we train against.** Training needs a backward pass through
the decoder. The trainer keeps fp32 master weights, so the 20b model (21 B
params, about 84 GB in fp32) cannot be the training backbone on a 16 GB GPU.
**Recommend: train and evaluate against the from-scratch `t4` model** (and `tiny`
for tests). 20b support for *running* a module is cheap (Phase 6), but *training*
one for 20b needs a dX-only backward through the quantized inference weights,
which is a separate large project (P7). A module trained for one decoder does not
transfer to another (different `h_L`, different embedding space).

**D3. Where memory tokens sit.** Positions `0..K-1`, before everything else
(paper Eq. 6 and Fig. 13), RoPE as usual. Two gpt-oss-specific effects:
* *Sliding window*: even layers see only the last 128 positions, so once the prompt
  passes ~112 tokens, half the layers stop seeing the memory. Full-attention
  layers still do. Option for later: let window layers always see positions
  `0..K-1` (a "pinned prefix", a change to the attention mask in both the
  inference and training kernels). Measure before deciding; off by default.
* *Attention sinks*: unchanged; the learned sink stays in the softmax denominator.

**D4. Cross-attention kernel.** The read is tiny on the query side (16 slots) and
long on the key side (N up to a few thousand). **Recommend composing it from
pieces that are already tested**: per-head `S = Q K^T` and `O = P V` as batched
GEMMs, row softmax with a key-padding mask in between; backward `dP = dO V^T`,
softmax backward, `dQ = dS K`, `dK = dS^T Q`, `dV = P^T dO`. Store `P`
(`B x 8 x 16 x N` floats, small). A fused kernel is an optimization for later.
The heavy part is the `K/V` projections of the store (`N x h_e x h_e` per block),
not the attention.

**D5. Inference cache.** `K = LN(Z_D) W_k + b_k` and `V = LN(Z_D) W_v + b_v` do not
depend on the query. Cache them per (store, module) on the GPU, so a query costs
only the slot side (`16 x N x h_e`), and invalidate rows when documents are
added or removed.

**D6. Multi-turn chat.** Memory recomputed per query sits at position 0, so it
invalidates the whole KV cache behind it. For one-shot QA that does not matter.
For chat: default to computing the memory **once per conversation** from the
first query; `--mem-refresh per-turn` re-prefills (the paper's agent setting,
expensive with token-by-token prefill). Leave other placements as an experiment.

**D7. Numerics.** Training forward uses fp32 / fp16 tensor-core GEMMs; inference
uses bf16 or q8 weights and int8 activations. The module will see slightly
different decoder numerics at inference. Measure the gap (Phase 6 test) rather
than build a matching training path.

## 5. Settings (off by default)

One rule: **absent means off.** Nothing is enabled by a config file the user did
not pass.

| Where | Setting | Default | Meaning |
| --- | --- | --- | --- |
| `GptOssOptions` (C++) | `MemorillaOptions memorilla` with `bool enabled = false` | off | when false the model never touches any Memorilla code path |
| `--gpt-oss` CLI | `--memorilla <module dir>` | not set = off | loads the module; requires `--mem-store` |
| | `--mem-store <file.nmem>` | - | document store |
| | `--mem-k N` | module's K | use the first N slots (paper Fig. 10) |
| | `--mem-query-file / --mem-query` | the prompt tokens | text whose embedding is `z_q` |
| | `--mem-refresh once\|per-turn` | `once` | D6 |
| `--gpt-oss-train` CLI | new mode `memorilla` (alongside `train / export / gradcheck`) | - | plain `train` is unchanged and never builds a module |
| Module dir | `memorilla.json` | - | `slots, encoder_dim, llm_dim, heads, refine_layers, ffn_mult, dropout, encoder ("self" or a name), encoder_layer, pooling, decoder fingerprint` |

Checks at load: `encoder_dim` equals the store's width, `llm_dim` equals the
decoder's hidden size, and the decoder fingerprint (config hash) matches, with a
clear error otherwise. A later GUI/app toggle maps onto the same `enabled` flag.

## 6. File formats

* **Store `.nmem`**: magic, JSON header (`encoder`, `encoder_layer`, `pooling`,
  `dim`, `dtype` f32|bf16, `count`, decoder fingerprint if self-encoded), then
  `count x dim` vectors, then a per-document table (id, token count, byte offset of
  the source text or token ids in an optional sidecar), end magic. Append-only
  writes plus a tombstone list for deletes (the paper's "delete on request"); a
  `compact` command rewrites it. 5 KB per document at bf16 and `dim = 2560`;
  1 KB at `dim = 512`.
* **Module**: `memorilla.json` + `memorilla.safetensors` (fp32 or bf16), names like
  `slots`, `cond.v.weight`, `cond.o.weight`, `cond.norm.scale`, `docs.norm.scale`,
  `block.{l}.attn.{q,k,v,o}.weight/bias`, `block.{l}.ffn.{up,down}.weight/bias`,
  `block.{l}.norm{1,2}.scale/bias`, `out.norm.*`, `proj.weight/bias`.
* **Training checkpoint**: module params + Adam m/v + step, same atomic-write and
  mirror rules as `.nckpt` (reuse that code).
* **Training examples `mem_{train,val}.bin`** (written by Python): per example the
  document range in a store, the query embedding row, query token ids and answer
  token ids. Batches are a pure function of (seed, step), as in the trainer.

## 7. Phases

Each phase ends with something testable and leaves the default path untouched.

### P0. Settings and plumbing (no behaviour)

* `MemorillaOptions` in `gpt_oss_model.hpp`, CLI flag parsing in
  `gpt_oss_app.cpp`, `memorilla` mode stub in `gpt_oss_train.cpp`, CMake entries for
  new files.
* **Test**: `gpt_oss_e2e_test.py` passes unchanged, and logits with the flag absent
  are bit-identical to a build without the feature (dump with `--dump-logits`, compare).

### P1. Soft-token input and hidden-state output in the inference host

* `GptOssModel::forwardEmbedding(uint64_t vecAddr, ...)`: same as `forward` but the
  first dispatch copies a GPU vector into `x` instead of the embedding gather.
* Optional hidden output: `forward(..., hiddenOut)` returns `x` after the final
  RMSNorm (or after layer `encoder_layer`) for the self-encoder.
* **Test (exactness)**: feed the bf16 embedding rows of a prompt as soft tokens;
  logits must equal the normal token path bit for bit.

### P2. Store and encoder

* `src/memorilla_store.{hpp,cpp}`: read/append/tombstone/compact `.nmem`.
* Encoding: `NeonML --memorilla encode --model <export|ckpt> --docs <tokens> --out store.nmem`.
  For data prep, run the **trainer's batched forward** (fast) and segment-mean the
  chosen layer's residual over each chunk's tokens. For live inference appends,
  the per-token host path from P1 (slow but fine for a few documents).
* `train/encode_external.py` (D1-b): Qwen3-Embedding via `sentence-transformers`,
  writes the same `.nmem`.
* **Test**: batched and per-token encodings of the same chunk agree (within fp16 /
  int8 tolerance); round-trip file test including tombstones.

### P3. Memory module forward

* `slangml/memorilla/` (new folder, `slangml-memorilla.slang` module): masked row
  softmax (padding docs), broadcast-add + LayerNorm for Stage 2, any glue the
  existing kernels lack. Reuse LayerNorm, GELU and the trainer GEMMs.
* `src/memorilla.{hpp,cpp}`: `MemorillaModule` owns a flat fp32 param buffer
  (same `ParamLayout` style as the trainer), records the forward into a
  `Recorder`, writes `M` (`B x K x h_L`) to a buffer the decoder reads.
* `tests/memorilla_reference.py`: NumPy float64 reference (and the parameter
  counter, which must print 194,123,776 for the paper config with `W_q/W_k` included).
* **Test**: GPU forward vs reference, fp32 and fp16 GEMM paths, with padded and
  unpadded stores, N from 1 to 4096.

### P4. Training (frozen decoder)

* Trainer frozen mode (`GptOssTrainer`, flag set only by the `memorilla` mode):
  skip every `linearDW` / weight `colsum` / embedding backward / LLM AdamW; skip the
  aux loss; keep all `dX` work including the router path. Allocate no `G / Mo / Ve`
  for LLM params (saves 12 bytes per param). Load the decoder from `.nckpt`
  (`readCheckpointParams` exists); loading an MXFP4 export (dequantize experts) comes later.
* Sequence layout per example: `[K memory rows][query tokens][answer tokens][pad]`
  to a multiple of 64 (attention needs it). Targets `SML_INVALID` except answer
  positions. After `sml_train_embed_fwd_f32`, a copy writes `M` into the memory
  rows of `resid[0]`. Loss = sum over valid rows / valid count (fix the mean).
* Module backward: read `dR` at memory rows -> projection, LN, FFN, cross-attn,
  Stage 2 backward -> module grads -> grad-norm clip + AdamW on the module buffer
  (reuse kernels). Dropout 0.1 behind a flag (deterministic seed per step).
* `NeonML --gpt-oss-train memorilla --decoder run/latest.nckpt --store ... --data ... --out memrun`
  with the paper's schedule shape (cosine + warmup, LR 3e-4, WD 1e-2) and the
  trainer's `--precision`, loss scaling, checkpoint, mirror and resume logic.
* `memorilla gradcheck` + `tests/memorilla_train_test.py`: all module gradients vs
  PyTorch float64 autograd through the tiny decoder (like `gpt_oss_train_test.py`).
  Also check that the decoder's parameters are bit-identical before and after a step.

### P5. Data and first experiment (TinyStories, `t4` decoder)

The decoder is a base TinyStories model, not an instruction model, so the tasks
must be continuations, not QA. Built by `train/prepare_memorilla_data.py` from
the existing TinyStories data:

| Task | Documents | Query (in context) | Answer (loss) | Paper analogue |
| --- | --- | --- | --- | --- |
| Reconstruct | chunks of one story | BOS only | the story | Stage 1 reconstruction |
| Continue | earlier chunks of a story | last 1-2 sentences before the target | next sentence(s) | long-document QA |
| Distractor recall | story chunks + chunks of 8-64 other stories | as above | as above | needle in a growing store |

Report validation **bits per byte on answer tokens** (comparable across setups) for:
closed book (query only), Memorilla, untrained module (random init), noise store
(Gaussian vectors with matching mean/variance), full context as text (when it
fits), cosine RAG top-k as text, and Memorilla at `--mem-k 1/2/4/8/16`.
**Success**: Memorilla clearly below closed book and below the untrained and noise
variants; it degrades less than RAG as distractors grow. If it fails, check D3
(window) and the output scale of `M` relative to token embeddings first (init
`W_proj` so `||m_k||` starts near the mean embedding-row norm).

Budget: `t4` module ~7 M params; each step is a forward plus dX backward of the
115 M decoder on short sequences, so a run should take on the order of an hour
on the free Colab T4. Prove it on the local RTX 3050 with `tiny`/small `t4`
first. Any Modal run gets a step/time cap under the remaining credit and is
dispatched only after asking.

### P6. Inference integration

* `--memorilla` path in `gpt_oss_app.cpp`: load module + store, compute `z_q` from
  the query, run the module (with the D5 K/V cache), feed `K` soft tokens through
  `forwardEmbedding`, then the prompt, then generate. Print module ms and store size.
* `--mem-k`, `--mem-refresh`; harmony chat format for 20b puts the memory before
  `<|start|>system`.
* **Tests**: engine output equals the trainer's forward for the same example (D7
  gap measured and documented); `--mem-k` truncation; memory off = bit-identical.

### P7. Later / optional

* Training a module for **gpt-oss-20b**: dX-only backward through the bf16/q8
  dense and MXFP4 expert inference weights (transposed GEMV/GEMM kernels, saved
  activations or recomputation), and batched prefill. Large; decide after P5.
* Pinned memory prefix in sliding-window layers (D3), behind its own flag.
* Hybrid Memorilla + RAG (text top-k after the memory tokens), the paper's main
  future-work item.
* Agent mode: one document per turn, extra self-attention layer, GRPO on reward.
* Fused cross-attention kernel; bf16 store; qualified Qwen3-Embedding port (D1-c).

## 8. New and changed files

| File | Phase | Change |
| --- | --- | --- |
| `src/gpt_oss_model.{hpp,cpp}` | P0, P1, P6 | `MemorillaOptions` (off), `forwardEmbedding`, hidden output |
| `src/gpt_oss_app.cpp` | P0, P6 | CLI flags, query flow |
| `src/gpt_oss_train.{hpp,cpp}` | P0, P4 | `memorilla` mode, frozen-backbone flag, memory-row injection, masked loss mean |
| `src/memorilla.{hpp,cpp}` | P3, P4 | module: params, forward, backward, optimizer, checkpoints |
| `src/memorilla_store.{hpp,cpp}` | P2 | `.nmem` store |
| `slangml/memorilla/*.slang`, `slangml/slangml-memorilla.slang` | P3, P4 | masked softmax, Stage 2, glue |
| `CMakeLists.txt` | P0 | add sources to `LveEngine` and `NeonML` |
| `train/prepare_memorilla_data.py`, `train/encode_external.py` | P2, P5 | data and external embeddings |
| `train/configs/memorilla_t4.json` | P4 | module preset |
| `tests/memorilla_reference.py`, `tests/memorilla_train_test.py` | P3, P4 | reference and gradcheck |
| `docs/memorilla.md` | P6 | user doc once it works (this file stays the plan) |
| `docs/LLM_ROADMAP.md` | P0 | one roadmap line pointing here |

## 9. Parameter sizes for our configs

`W_q/W_k` of Stage 2 dropped (D2.2), 8 heads, 2 blocks, 4x FFN.

| Encoder -> decoder | `h_e` | `h_L` | Head dim | Module params |
| --- | ---: | ---: | ---: | ---: |
| self (`t4`) | 512 | 512 | 64 | ~7.1 M |
| Qwen3-Embedding-0.6B -> `t4` | 1024 | 512 | 128 | ~27.8 M |
| self (gpt-oss-20b) | 2880 | 2880 | 360 | ~224 M |
| paper (Qwen3-Emb-4B -> Qwen3-8B, with `W_q/W_k`) | 2560 | 4096 | 320 | 194.1 M |

Head dims of 320-360 rule out reusing the d64-only attention kernels and are one
more reason for the GEMM-composed cross-attention (D4).

## 10. Risks and open questions

1. **A base 115 M TinyStories decoder may use soft tokens poorly.** The paper's
   decoders are 4-8 B instruction models. P5 is the real test; the fallback is a
   bigger from-scratch decoder or P7.
2. **Self-encoder quality**: mean-pooled causal-decoder states are a weak embedding.
   The external encoder (D1-b) is the control experiment.
3. **Sliding window hides the memory** in half the layers for longer prompts (D3).
4. **Soft-token scale**: memory vectors far from the embedding-row distribution can
   blow up the first layers; watch the initial loss and the norms of `M`.
5. **Training vs inference numerics** (D7).
6. **Per-turn memory forces a full re-prefill** (D6) until batched prefill exists.
7. Privacy: a store holds embeddings of user text; keep stores local and
   support delete (tombstones + compact), as the paper's ethics section asks.

Decisions to confirm with the user before P2/P4: encoder choice (D1), training
backbone (`t4`, D2), and whether the 20b training path (P7) is wanted at all.

## 11. Implementation status

### How it differs from the plan above

* The command line is its own mode, `NeonML --memorilla encode|train|eval|gradcheck|init|params`
  (also `LveEngine.exe --memorilla`), not a `--gpt-oss-train` sub-mode.
* `MemorillaOptions` lives in `src/memorilla.hpp` and is used by `gpt_oss_app.cpp`. The
  model class only gained `forwardEmbedding` (a vector instead of a token id) and an
  optional hidden-state output; `GptOssOptions` is unchanged.
* Stage 2 drops `W_q / W_k` (see 2.2). `NeonML --memorilla params` prints both counts
  (paper config: 181,011,456 used, 194,123,776 with the unused pair).
* The self encoder's mean pooling runs on the CPU after a batched trainer forward
  (fast enough: the hidden state of one batch is a few MB).
* Only `encoder_layer = -1` is supported for live query encoding at inference; other
  layers or external encoders pass `z_q` with `--mem-query-vec`.

### Off by default (checked)

* No `--memorilla` flag: no Memorilla code runs, no kernel compiles, no buffer is
  allocated. `gpt_oss_e2e_test.py` gives the same numbers as before the change.
* `--mem-*` options without `--memorilla` are refused with an error.
* Normal training (`--gpt-oss-train train`) is unchanged: every new trainer path is
  gated on `TrainOptions::frozen` / `memRows`, both off by default.
  `gpt_oss_train_test.py` gives the same errors as before (fp32 1.29e-6, fp16 7.65e-4).

### What exists

| File | What |
| --- | --- |
| `slangml/memorilla/memorilla_f32.slang` | LayerNorm fwd/bwd, exact GELU fwd/bwd, slot broadcast add, grouped row sums, cross-attention pieces (rowdot with padding mask, row softmax fwd/bwd, weighted sums incl. transposed), strided row copy |
| `src/memorilla.{hpp,cpp}` | config JSON, parameter layout, `.nmem` store (tombstones, compact), `.seq` reader, `MemorillaModule` (forward, backward, AdamW, safetensors save/load, training state) |
| `src/memorilla_train.cpp` | `--memorilla` CLI: self encoder, training loop (cosine + warmup, loss scaling, checkpoints + mirror, resume, best-val module), eval with ablations `closed / noise / untrained`, gradcheck dump, random `init` |
| `src/gpt_oss_train.{hpp,cpp}` | frozen mode (no weight gradients, no LLM optimizer state), memory-row injection after the embedding, answer-only loss mean, `setBatch / recordStep / useMxfp4Experts` |
| `src/gpt_oss_model.{hpp,cpp}`, `src/gpt_oss_app.{hpp,cpp}` | `forwardEmbedding`, hidden output, `--memorilla / --mem-store / --mem-k / --mem-query(-file|-vec) / --mem-dump / --soft-prefix` |
| `train/prepare_memorilla_data.py` | TinyStories-style examples (`continue`, `reconstruct`, `mix`, distractors), runs the encoder with `--decoder` |
| `tests/memorilla_formats.py` | Python readers/writers for `.nmem`, `.seq`, examples; the sequence layout |
| `tests/memorilla_train_test.py` | all module gradients vs float64 PyTorch through the official decoder |
| `tests/memorilla_infer_test.py` | soft tokens exact, z_q and memory tokens vs references, `--mem-k`, off-by-default |

### Verification (RTX 3050 Laptop, 4 GB)

* `memorilla_train_test.py`: loss and all 45 parameter gradients vs PyTorch float64
  through the official reference decoder. fp32: worst relative error 6.5e-7
  (3 examples x 64), 8.8e-7 (4 x 128), 8.4e-7 (5 x 192); fp16 tensor cores 1.1e-3.
  This test found one bug while it was being written: the block LayerNorm gains
  (`norm1/norm2.scale`) started at 0, which silently zeroed the query, key and FFN
  gradients. The test now also rejects zero gains, all-zero weights and any tensor
  without a gradient.
* `memorilla_infer_test.py`: soft-token prefix bit-identical to the token path;
  `z_q` within 7.6e-3 (rel L2) of the official model's mean hidden state (int8
  activations); memory tokens within 5e-7 of a float64 module; `--memorilla` logits
  identical to `--soft-prefix` with the same tokens; `--mem-k 2` exact.

### First learning run (local, 2026-11-07)

A small frozen decoder (4 layers, hidden 256, 8 experts top-2, GPT-2 vocabulary;
2000 steps on the 3 M-token `data/tinystories_small`, val loss 2.41) and the
`continue` task (`prepare_memorilla_data.py` defaults: 32-token documents, 16-token
query, 48-token answer; 12,884 training / 640 validation examples, ~4 documents
each). Module from `train/configs/memorilla.json` (K = 16, 8 heads, 2 blocks):
1.78 M parameters, 1500 steps of 16 examples, LR 3e-4 cosine, fp16 tensor cores,
about 195 examples/s on the RTX 3050 (2 minutes). Full validation set:

| Setup | answer loss (nats/token) | bits/byte |
| --- | ---: | ---: |
| closed book (no memory tokens) | 2.624 | 0.915 |
| untrained module | 2.668 | 0.930 |
| trained module, documents replaced by noise | 2.652 | 0.925 |
| **trained module** | **2.582** | **0.900** |

Same pattern as the paper (Table 13 noise test, Fig. 8 untrained memory): trained
memory beats closed book, while untrained memory and noise documents are worse
than no memory at all, so the gain comes from reading the store. The gain is small
(1.6%), as expected from a 7 M-parameter decoder trained for 3 minutes; the real
test is the `t4` decoder on Colab. Commands:

```text
python train/prepare_memorilla_data.py --data data/tinystories_small --out mem_data --decoder dec_run --no-mxfp4
NeonML --memorilla train --decoder dec_run --data mem_data --out mem_run --mem-config train/configs/memorilla.json \
       --batch 16 --seq 128 --steps 1500 --lr 3e-4 --eval-every 250 --eval-batches 40 --no-mxfp4
NeonML --memorilla eval --decoder dec_run --data mem_data --module mem_run/module --eval-batches 40 --no-mxfp4 \
       --ablation closed|untrained|noise|none
```

(`--no-mxfp4` because that decoder was trained without QAT; a QAT-trained `t4`
decoder should drop it, so training sees the experts the MXFP4 export has.)

### Colab T4 (2026-11-07)

`train/colab/memorilla_colab.sh` runs everything on a Colab VM: `setup_colab.sh`,
8k-BPE TinyStories (`prepare_data.py --streaming`), the `t4` decoder (MXFP4 QAT from
80%), Memorilla data + self encoder, module training, and the four evaluations
(`MODE=smoke` for a few-minute check, `MODE=full` for the experiment; stages
resume). Driven from Windows with the Colab CLI (`colab new --gpu T4`, `upload`,
`exec`), see `docs/colab-t4-gpu.md` in the Desktop folder.

* **Smoke run passed** (first time the Colab path ran at all): Vulkan through
  `libEGL_nvidia.so.0`, GCC build of NeonML, data, 30 decoder steps, encoding,
  40 module steps with the decoder unchanged, all four evaluations. 4 min in total
  after setup (~2 min).
* Measured on the T4: `t4` decoder about 10k tokens/s at 8 x 256 (114.6 M
  parameters, 2.9 GiB); memory module about 190 examples/s at 32 x 128.

* **Full run** (`MODE=full DEC_STEPS=5000 TRAIN_TOKENS=45000000`, about 1 h on the
  free tier):
  * Decoder: `t4` preset, 5000 steps of 16 x 512 (41 M tokens), WSD, MXFP4 QAT from
    step 4000. About 22-24k tokens/s; val 1.807 nats/token (**0.631 bpb**); 31 min.
  * Data: 192,083 training / 2,000 validation `continue` examples (4.1 documents
    each); self-encoding 983k sequences took 17 min.
  * Module: 7.10 M parameters, 4000 steps of 32 examples, about 245 examples/s,
    9 min; decoder parameters unchanged. Validation (bpb): 0.740 (untrained),
    0.703 (step 500), 0.694 (1000), 0.688 (2000), 0.684 (3000), 0.683 (4000); still
    improving slowly at the end.

  | Setup (2,000 val examples) | answer loss (nats/token) | bits/byte |
  | --- | ---: | ---: |
  | closed book | 2.084 | 0.715 |
  | untrained module | 2.158 | 0.740 |
  | trained module, noise documents | 2.164 | 0.742 |
  | **trained module** | **1.991** | **0.683 (4.5% lower)** |

  The gain is about 3x the local run's (1.6%) with a stronger decoder, and the
  ablations keep the paper's pattern. Through the inference host
  (`--gpt-oss --memorilla`, int8 activations, MXFP4 experts) on 60 validation
  examples, the answer loss drops by 4.6%, the same as in training, so D7 (training
  vs inference numerics) costs nothing measurable. Single greedy samples do not yet
  show visible use of the memory: the gain is statistical.
* Artifacts are kept in `data/memorilla_t4_run/` (gitignored): `decoder_export/`
  (the inference checkpoint, MXFP4), `mem_run/module/`, validation stores and
  examples, the tokenizer, and the logs. The decoder's training checkpoint (`.nckpt`,
  needed to train more modules against it) stayed on the VM and is gone.

### Still to do

* Bigger effect sizes need a task where the documents carry more of the answer
  (distractor and reconstruct tasks, longer stories, a QA-tuned decoder) and a longer module
  run (the curve had not flattened).
* D5 K/V cache of the store projections; dropout (0.1 in the paper; not implemented, 0 here).
* Loading the decoder from an MXFP4 export (today: `.nckpt` only); `encode_external.py`.
* Chat use (`--mem-refresh`), harmony formatting for 20b, pinned memory prefix (D3).
* P7 items unchanged.
