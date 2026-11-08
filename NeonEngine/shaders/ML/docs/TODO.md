# TODO

Open work for the gpt-oss stack: the inference host, the from-scratch trainer and Memorilla. What is built and tested lives in [LLM_ROADMAP.md](LLM_ROADMAP.md) (section 5) and [memorilla_plan.md](memorilla_plan.md) (section 11). This file holds the to-do list, the future-improvements list and their sources.

Last updated 2026-11-08.

## How to read this file

- **Citations.** External sources are numbered [1], [2], ... in order of first use, and listed under [References](#references). Repo files are cited as [R1]-[R4].
- **Source quality.** Most numbers come from paper abstracts or secondary summaries (blogs, aggregators, vendor posts). "Author claim" means the paper's own result. I have not read the full papers, so check any number before relying on it.
- **Cost tags** are rough planning estimates, not quotes:
  - **free**: engineering time only
  - **$**: under about $50 of GPU time
  - **$$**: hundreds of dollars
  - **$$$**: thousands
  - **$$$$**: tens of thousands or more
- **Cost method.** Forward-only work costs about 2 x N x tokens FLOPs. Backward to inputs only (no weight gradients) costs about 4 x N x tokens. Full training costs about 6 x N x tokens. N is the 3.6B active parameters of gpt-oss-20b [R3, section 4]. Priced at about $4/h per H100 (the Modal H100 rate in the project notes) at 10-30% utilization. Utilization is the largest unknown.

## Blocking work (do first)

- [ ] **Run the real gpt-oss-20b checkpoint end to end.** Still unchecked in the roadmap [R3, section 5]. Dense bf16 weights are about 12.8 GB at 4K context [R2, memory plan]. Cost: free to $, depending on GPU hours.
- [ ] **Batched prefill.** Prompt tokens run one at a time today [R2, known limits]. Needed before generating training data from the 20b. Cost: free (engineering).
- [ ] **Tokenizer and harmony chat format in C++.** Only the Python helper exists [R3, section 5]. Cost: free.
- [ ] **Measure the expert union for a k-token block.** This decides whether speculative decoding helps (see Decode speed). Cost: free (a forward pass with logging).

## Future improvements

### Memory

- [ ] **Ring-buffer KV for the sliding-window layers.** Every layer reserves `maxContext` tokens of K and V [R4, lines 502-504], but the 12 sliding-window layers read only the last 128 positions [R4, lines 649-668]. At 131,072 tokens the full KV is 6.0 GiB across all 24 layers, and only the 12 full-attention layers need 3.0 GiB (our arithmetic). Cost: free.
- [ ] **KV quantization at 3-3.5 bits (TurboQuant-style rotation).** A secondary overview reports near-neutral quality at 3.5 bits and small loss at 2.5 bits [1]. KIVI is the 2-bit baseline [2]. An independent single-GPU blog test measured cosine similarity 0.9943 at 3 bits and 8K context [3]. At 128K the full-attention KV would drop from 3.0 GiB (fp16) to about 0.7 GiB (our arithmetic). Cost: free to $ (kernel work).
- [ ] **Attention Matching: training-free KV compaction.** Closed-form, no gradients. The authors report about 50x compaction with little loss (author claim) [4]. Open question: how it handles RoPE positions for compacted entries. I have not checked this in the paper. Cost: $ per corpus.
- [ ] **Cartridges: a trained KV prefix per corpus.** Gradient-trained, so it needs backprop through the model. Author claim: 38.6x less memory and 26.4x higher throughput than in-context learning, at matched quality [5]. Cost: $ per corpus on the 20b, with backprop through the frozen model.
- [ ] **Memory Decoder: a logit-space memory.** A small decoder is trained to match kNN-retrieval distributions over a corpus, using only forward passes of the base model. It is interpolated with the base model at inference [6]. The reported gains are perplexity, not QA. The 20b vocabulary is 201,088 [R2]. A plain kNN datastore costs about 5.6 KB per token (2880 dims in fp16), so 1M tokens is about 5.6 GB (our arithmetic). Cost: $.
- [ ] **Memorilla on gpt-oss-20b (P7).** Needs a dX-only backward through the MXFP4 experts [R1, section 7, P7]. Compute is small: about 4N FLOPs per token, so 100M tokens is a few H100-hours (our estimate, roughly $5-20). The cost is engineering. Base method: Memorilla [7].
- [ ] **Engram: hashed n-gram lookup in host memory.** Lookup addresses depend only on input tokens, so the table can be prefetched from host RAM. Secondary reports claim under 3% overhead for a 100B-parameter table [8]. Needs pretraining with the module. Cost: $$ to $$$$, depending on tokens. Test on the t4 trainer first.
- [ ] **Product-key memory layers.** Sparse key-value memory in place of some FFN layers. Meta FAIR reports large factual-QA gains and results at matched training FLOPs (author claim) [9]. Test on the t4 trainer first. Cost: $$ at t4 scale.
- [ ] **TTT-E2E: context as training data at test time.** Late MLP blocks update during inference. Meta-training needs gradients of gradients [10]. NVIDIA reports about 2.7x faster than full attention at 128K (vendor blog) [11]. Cost: $$$ for meta-training; test at t4 scale first.

### Long-context attention

- [ ] **Trainable sparse attention (NSA, DeepSeek Sparse Attention).** NSA trains sparse attention natively, from pretraining [12]. DeepSeek-V3.2 uses a lightning indexer to pick top-k keys, with k = 2048 per a secondary summary [13][14]. Sources disagree on the continued-training budget: about 944B tokens [15][16] against about 100B [14]. A 20b retrofit on about 2B tokens would cost roughly $150-500 (our estimate). Cost: $$.
- [ ] **Hybrid linear-attention retrofit (Jet-Nemotron, PostNAS).** Keeps a few full-attention layers and replaces the rest. The MLP weights stay frozen during the search [17][18]. Author claim: up to 53.6x generation throughput at 256K on an H100, for the 2B model [17]. The repo already has Gated DeltaNet kernels [R3, section 3, D6]. Retrofitting the 20b still needs gradients through frozen MLPs. Cost: $$$.

### Decode speed

- [ ] **Trained draft head on frozen 20b features (EAGLE-3 / MTP-style).** Forward-only data collection, with only the draft head trained [19]. About $5 of compute for roughly 100M tokens (our estimate). Caveat: verifying k draft tokens loads the union of their routed experts, which can cancel the gain. A paper on the limits of speculation in MoE models bounds this effect [20]. Measure the union first. Cost: $.
- [ ] **Training-free speculation from n-grams or the prompt.** SSSD reports up to 2.9x lower latency (abstract, via a secondary listing) [21]. Lookahead decoding is about 1.5-2.3x in secondary write-ups [22]. Training-free n-gram drafting is described in [23]. Works best when output copies input, such as code edits or RAG. Cost: free.
- [ ] **Megakernel decode.** A secondary summary of megakernel work puts each kernel launch at about 5-10 us [24]. Published results: Cohere reports 1.25-1.41x over vLLM on one H100 at BF16 (its 1.58x headline does not match this range) [25]; Event Tensor reports 1.48x over vLLM on Qwen3-30B-A3B at batch 1 [24]; Ada-MK reports up to 23.6% over TensorRT-LLM on an L20 [26]. Our estimate: about 1-2.5 ms of launch time per token, assuming about 10 dispatches per layer and 24 layers, out of the roughly 25 ms per token projected for the 20b [R3, section 6]. Cost: free (engineering).
- [ ] **Expert prefetch from the next layer's router.** Apply the next layer's gate to the current hidden state to predict which experts it will need [27]. Helps the host-offloaded layers in the 8 GB plan [R2, memory plan]. Cost: free to $.
- [ ] **REAP: one-shot expert pruning.** Router-weighted saliency, no training. The authors test pruning down to 50% on models of 20B to 1T parameters (ICLR 2026) [28]. For the 20b, 32 to 16 experts would cut expert weights from about 10 GB to about 5 GB [R2, memory plan]. Quality on gpt-oss-20b is unknown. Cost: free to $ (one calibration pass).
- [ ] **Adaptive top-k: drop the 4th expert when its weight is small.** This is an original idea, not from the sources listed here. Renormalize the remaining weights and test with `--score`, since greedy generation is the only CLI mode [R2]. Cost: free.
- [ ] **Diffusion conversion for parallel decoding.** REPR-ALIGN adapts Qwen3 models of 0.6B-4B on up to 50B tokens and reports up to 4x faster training than pure denoising conversion [29]. OPDLM reports 15x to 7,000x fewer tokens than earlier conversions [30]. Both are self-reported and mostly on code. Cost for the 20b: $$$$.

### Model quality and training

- [ ] **Muon optimizer for hidden matrices.** Moonlight reports about 2x compute efficiency over AdamW (author claim) [31]. A separate paper finds Muon keeps its data efficiency at large batch sizes [32]. The trainer uses AdamW only [R5: `src/gpt_oss_train.cpp`]. Newton-Schulz iterations are matrix products that the existing tensor-core GEMM covers. Test on the tiny config first. Cost: free to implement, and it cuts the compute needed for a given quality if the claim holds.
- [ ] **Expert upcycling.** Add experts during continued pretraining while keeping the number of routed experts per token fixed [33]. DivMoE reports that fine-grained upcycling can leave the router ineffective [34]. Cost: $$ to $$$.
- [ ] **Continued pretraining of the 20b.** About $150-500 for 2B tokens and $4k-12k for 50B tokens (our estimate). Cost: $$ to $$$.
- [ ] **On-policy distillation from the 20b into a smaller student.** The student generates rollouts and the teacher scores each token under reverse KL [35]. Secondary sources report large savings over RL on math [36]. Reverse KL is mode-seeking and can be brittle [37]. Training FLOPs for 1B tokens are about $100-300 (our estimate), but rollout generation is decode-bound and dominates wall-clock. Cost: $$ to $$$.
- [ ] **RL with verifiable rewards (GRPO) on the 20b.** A vendor case study compares SFT and GRPO on multilingual reasoning. The base model scored 22% on a 50-case eval [38]. Tokyo Tech's Swallow variant used continued pretraining, SFT and RL [39]. I found no 2026 result specific to gpt-oss-20b. Cost: $$ to $$$, depending on rollout count.
- [ ] **Manifold-constrained hyper-connections (mHC).** Constrains the residual-stream mixing to stabilize training at scale [40]. It is an architecture change, so test it on the t4 trainer first. Cost: $$$$ at 20b scale.

### Serving

- [ ] **Batching and prefix caching.** Continuous batching for several users, and prefix caches for documents (combine with KV compaction). Cost: free (engineering).
- [ ] **Test-time compute.** Best-of-N with a verifier, or self-consistency. Inference cost only. Cost: $ per eval.

## Notes

- Items marked "original idea" are mine, not from the sources.
- When an item lands, move its status to [LLM_ROADMAP.md](LLM_ROADMAP.md) section 5 and remove it here.
- [LLM_ROADMAP.md](LLM_ROADMAP.md) section 7 is the kernel and host to-do list. This file covers research ideas and gaps.

## References

### Repo files

- [R1] [memorilla_plan.md](memorilla_plan.md): section 7 (P7), section 11 (status).
- [R2] [gpt_oss.md](gpt_oss.md): memory plan (expert sizes, 12.8 GB dense), known limits (token-by-token prefill, greedy decoding only).
- [R3] [LLM_ROADMAP.md](LLM_ROADMAP.md): section 3 (D6, Gated DeltaNet), section 4 (3.6B active parameters), section 5 (status), section 6 (projected 20b decode).
- [R4] `src/gpt_oss_model.cpp`, lines 502-504 (KV allocation per layer) and 649-668 (sliding-window cache view). Path: `C:\Users\ZyBros\Downloads\NeonEngine\NeonEngine\src\gpt_oss_model.cpp`.
- [R5] `src/gpt_oss_train.cpp`: no occurrences of Muon (checked by grep); AdamW only.

### External

1. Google TurboQuant overview (secondary). https://www.morphllm.com/turboquant (the ICLR 2026 paper was not located in my searches)
2. Liu et al., "KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache," arXiv:2402.02750 (ICML 2024). https://arxiv.org/html/2402.02750v2
3. Independent KV-cache benchmark on one GPU (blog, not peer reviewed). https://ai-muninn.com/en/blog/turboquant-kv-cache-benchmark-gx10
4. Zweiger et al., "Fast KV Compaction via Attention Matching," arXiv:2602.16284 (2026). https://arxiv.org/html/2602.16284v1
5. "Cartridges: Lightweight and general-purpose long context representations via self-study," arXiv:2506.06266 (ICLR 2026). https://arxiv.org/abs/2506.06266
6. Cao et al. (SJTU LUMIA Lab), "Memory Decoder: A Pretrained, Plug-and-Play Memory for Large Language Models," arXiv:2508.09874 (NeurIPS 2025). https://arxiv.org/html/2508.09874v1
7. Fallahpour, Idehpour, Kothapalli et al., "Memorilla: Latent Semantic Memory for LLMs," arXiv:2610.21375 (the paper implemented in [R1]). Local copy: `memorilla stanford..pdf` in the repo root.
8. DeepSeek-AI, "Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models," arXiv:2601.07372 (January 2026). https://arxiv.org/pdf/2601.07372
9. Meta FAIR, "Memory Layers at Scale," arXiv:2412.09764. https://arxiv.org/abs/2412.09764
10. "End-to-End Test-Time Training for Long Context" (TTT-E2E), arXiv:2512.23675 (Stanford and NVIDIA). https://arxiv.org/pdf/2512.23675v1
11. NVIDIA Developer Blog, "Reimagining LLM memory using context as training data" (vendor blog). https://developer.nvidia.com/blog/reimagining-llm-memory-using-context-as-training-data-unlocks-models-that-learn-at-test-time
12. DeepSeek-AI, "Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention," arXiv:2502.11089. https://arxiv.org/abs/2502.11089
13. DeepSeek-V3.2 technical report, arXiv:2512.02556 (title not verified; cited via an EmergentMind summary page). https://www.emergentmind.com/papers/2512.02556
14. DeepLearning.AI, "DeepSeek-V3.2-Exp streamlines processing using a lightning indexer" (secondary). https://www.deeplearning.ai/the-batch/deepseek-v3-2-exp-streamlines-processing-using-a-lightning-indexer-boosting-efficiency
15. DAIR.AI paper summary, "DeepSeek-V3.2-Exp" (secondary). https://academy.dair.ai/papers/deepseek-v3-2-exp
16. Analytics Vidhya, "DeepSeek-V3.2-Exp: 50% cheaper, 3x faster" (secondary). https://www.analyticsvidhya.com/?p=243729
17. "Jet-Nemotron: Efficient Language Model with Post Neural Architecture Search," arXiv:2508.15884 (NeurIPS 2025, NVIDIA with MIT Han Lab). https://arxiv.org/abs/2508.15884
18. Jet-Nemotron project page (MIT Han Lab). https://hanlab.mit.edu/projects/jet-nemotron
19. "EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test," arXiv:2503.01840. https://arxiv.org/html/2503.01840v3
20. "The Limits of Speculation: Bounding Speculative Decoding in Mixture-of-Experts," arXiv:2609.22156. https://arxiv.org/pdf/2609.22156
21. "SSSD: Simply-Scalable Speculative Decoding," ACL 2026 (summary page). https://en.papernotes.org/ACL2026/model_compression/sssd_simply-scalable_speculative_decoding/
22. General Compute blog, lookahead decoding (secondary write-up; one of several that give the 1.5-2.3x range, not checked against the original paper). https://www.generalcompute.com/blog/lookahead-decoding-parallel-token-generation-without-draft-models
23. Stewart et al., "The N-Grammys: Accelerating Autoregressive Inference with Learning-Free Batched Speculation," ICML 2024 (PMLR 262). https://proceedings.mlr.press/v262/stewart24a.html
24. "Event Tensor: A Unified Abstraction for Compiling Dynamic Megakernel," arXiv:2604.13327. https://arxiv.org/pdf/2604.13327
25. Cohere, "Megakernels" blog. https://cohere.com/blog/megakernels
26. "Ada-MK: Adaptive MegaKernel Optimization via Automated DAG-based Search for LLM Inference," arXiv:2605.11581. https://arxiv.org/pdf/2605.11581
27. "Accurate Expert Predictions in MoE Inference via Cross-Layer Gate," arXiv:2502.12224. https://arxiv.org/abs/2502.12224
28. "REAP the Experts: Why Pruning Prevails for One-Shot MoE Compression," arXiv:2510.13999 (ICLR 2026, Cerebras). https://arxiv.org/html/2510.13999v2
29. "Don't Retrain, Align: Adapting Autoregressive LMs to Diffusion LMs via Representation Alignment," arXiv:2605.06885. https://arxiv.org/abs/2605.06885
30. "Data-Efficient Autoregressive-to-Diffusion Language Models via On-Policy Distillation" (OPDLM), arXiv:2606.06712. https://huggingface.co/papers/2606.06712
31. Liu et al., "Muon is Scalable for LLM Training" (Moonlight), arXiv:2502.16982. https://arxiv.org/abs/2502.16982
32. "Practical Efficiency of Muon for Pretraining," arXiv:2505.02222. https://arxiv.org/abs/2505.02222
33. "Expert Upcycling" (title from a secondary summary; not verified), arXiv:2604.19835. https://www.emergentmind.com/papers/2604.19835
34. "DivMoE," arXiv:2610.11317. https://www.emergentmind.com/papers/2610.11317
35. "On-policy Distillation of Language Models: Learning from Self-Generated Mistakes" (GKD), arXiv:2306.13649 (ICLR 2024). https://arxiv.org/pdf/2306.13649
36. Prime Intellect, "On-policy distillation" RL glossary (secondary; summarizes the Thinking Machines post). https://primeintellect.ai/rl-glossary/on-policy-distillation
37. "Escaping the KL Agreement Trap in On-Policy Distillation," arXiv:2606.09471. https://arxiv.org/pdf/2606.09471
38. AWS Machine Learning Blog, "Fine-tuning gpt-oss-20b for multilingual reasoning on Amazon SageMaker HyperPod" (vendor case study). https://aws.amazon.com/blogs/machine-learning/fine-tuning-gpt-oss-20b-for-multilingual-reasoning-on-amazon-sagemaker-hyperpod/
39. Tokyo Tech, GPT-OSS-Swallow-20B-RL-v0.1 (model card). https://huggingface.co/tokyotech-llm/GPT-OSS-Swallow-20B-RL-v0.1
40. DeepSeek-AI, "Manifold-Constrained Hyper-Connections," arXiv:2512.24880. https://arxiv.org/pdf/2512.24880v2
