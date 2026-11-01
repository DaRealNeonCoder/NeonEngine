# SlangML: AlphaFold 3 Operator Family

Family doc for the AlphaFold 3 kernels added on top of the existing SlangML set.
Governed by [CONTRACT.md](CONTRACT.md) (BDA descriptors, `KernelHeader`, push-constant
ABI, entry naming, scalar layout). Math is taken from the AF3 main paper and
Supplementary Information (SI); section references are given per entry.

Compile and validate every entry with
[compile_af3.ps1](../tests/compile_af3.ps1). Status: all 32 entries compile with
`slangc` and pass `spirv-val`. **No numerical test harness exists yet**, so
correctness is verified by reading only.

## 1. Entry-point table

| Entry | Source | Dir. | Threads | Math reference |
|---|---|---|---|---|
| `sml_pairformer_triangle_attention_start_fwd_f32` | `pairformer/triangle_attention_f32.slang` | fwd | 128 | Alg 14, starting-node triangle attention |
| `sml_pairformer_triangle_attention_end_fwd_f32` | same | fwd | 128 | Alg 15, ending-node triangle attention |
| `sml_attention_pair_bias_fwd_f32` | `attention/pair_bias_attention_f32.slang` | fwd | 64 | Alg 24, pair-biased single attention |
| `sml_attention_pair_bias_bwd_f32_stats` | `attention/pair_bias_attention_bwd_f32.slang` | bwd (pass 1) | 64 | recomputes lse, oAcc; writes dOAcc, delta, dGate |
| `sml_attention_pair_bias_bwd_f32_dq_dbias` | same | bwd (pass 2) | 64 | dQ, dBias (thread per (i,h)) |
| `sml_attention_pair_bias_bwd_f32_dkdv` | same | bwd (pass 3) | 64 | dK, dV (thread per (j,h)) |
| `sml_diffusion_adaln_fwd_f32` | `diffusion/adaln_f32.slang` | fwd | 256 | Alg 26, AdaLN (adaptive LayerNorm) |
| `sml_diffusion_adaln_bwd_f32` | same | bwd | 256 | AdaLN backward |
| `sml_diffusion_adaln_zero_residual_fwd_f32` | same | fwd | 256 | adaLN-Zero gate on residual |
| `sml_diffusion_adaln_zero_residual_bwd_f32` | same | bwd | 256 | adaLN-Zero backward |
| `sml_diffusion_fourier_fwd_f32` | `diffusion/conditioning_f32.slang` | fwd | 256 | Alg 22, Fourier noise embedding (Alg 21 step 8 argument) |
| `sml_diffusion_input_scale_fwd_f32` | same | fwd | 256 | EDM input preconditioning (VERIFY) |
| `sml_diffusion_output_combine_fwd_f32` | same | fwd | 256 | c_skip, c_out combine |
| `sml_diffusion_sampler_init_fwd_f32` | `diffusion/sampler_f32.slang` | fwd | 256 | Alg 18, sampler init |
| `sml_diffusion_sampler_step_fwd_f32` | same | fwd | 256 | Alg 18, sampler step |
| `sml_diffusion_centre_augment_fwd_f32` | same | fwd | 256 | Alg 19, centre augmentation (random rotation and translation) |
| `sml_atom_sequence_local_attention_fwd_f32` | `atom/sequence_local_attention_f32.slang` | fwd | 64 | Alg 5 and 7, sequence-local atom attention |
| `sml_atom_band_geometry_fwd_f32` | same | fwd | 128 | band key geometry |
| `sml_msa_outer_product_mean_fwd_f32` | `msa/outer_product_mean_f32.slang` | fwd | 256 | Alg 9, outer product mean |
| `sml_msa_pair_weighted_avg_fwd_f32` | `msa/msa_pair_weighted_avg_f32.slang` | fwd | 64 | MSA pair-weighted averaging |
| `sml_pairformer_dropout_residual_fwd_f32` | `pairformer/dropout_residual_f32.slang` | fwd | 256 | Alg 17, dropout + residual |
| `sml_pairformer_dropout_bwd_f32` | same | bwd | 256 | dropout backward |
| `sml_features_relative_position_encoding_fwd_f32` | `features/pair_embedding_f32.slang` | fwd | 256 | Alg 3, relative position encoding |
| `sml_features_distance_bin_embed_fwd_f32` | same | fwd | 256 | distance-bin embedding |
| `sml_confidence_ptm_row_fwd_f32` | `confidence/ptm_f32.slang` | fwd | 64 | SI Eq. 17, 18 (per-row pTM / ipTM score) |
| `sml_confidence_ptm_max_fwd_f32` | same | fwd | 256 | max over rows (single workgroup) |
| `sml_loss_smooth_lddt_fwd_f32` | `loss/smooth_lddt_f32.slang` | fwd | 64 | SI Alg 27 (smooth LDDT, forward split) |
| `sml_loss_smooth_lddt_bwd_f32` | same | bwd | 64 | smooth LDDT analytic gradient |
| `sml_loss_weighted_rigid_align_fwd_f32` | `loss/diffusion_loss_f32.slang` | fwd | see source | weighted Kabsch alignment |
| `sml_loss_diffusion_mse_fwd_f32` | same | fwd | see source | weighted diffusion MSE (Alg 20 loss path; equation numbers unverified) |
| `sml_loss_diffusion_mse_bwd_f32` | same | bwd | see source | MSE backward |
| `sml_loss_bond_fwd_f32` | same | fwd | see source | bond loss |
| `sml_loss_bond_bwd_f32` | same | bwd | see source | bond loss backward |
| `sml_loss_binned_cross_entropy_fwd_f32` | `loss/binned_cross_entropy_f32.slang` | fwd | 64 | SI Eq. 9, 10, 12, 14 |
| `sml_loss_binned_cross_entropy_bwd_f32` | same | bwd | 64 | binned CE backward |

"see source" means the dispatch mapping is set in the file header and is not
repeated here, to avoid drift.

## 2. Parameter ABI

Each entry takes `ConstantBuffer<KernelPushConstants>`. `headerAddress` points to a
params struct whose first member is `KernelHeader` (48 bytes), followed by the
fields below. Tensor slots are indexed by the `slangml_tensor_desc(header, slot)`
argument in source order. Scalars are 4 bytes; scalar layout is enabled.

| Struct | Fields after header (in order) | Used by |
|---|---|---|
| `TriangleAttentionParams` | `N, H, C, float scale` | triangle attention start/end |
| `PairBiasAttentionParams` | `N, H, D, float scale` | pair-bias attention |
| `AdaLNParams` | `rows, cols, float epsilon, reserved` | adaLN fwd/bwd, zero-residual |
| `AdaLNGateParams` | `numel, reserved0..2` | adaLN-Zero gate |
| `FourierParams` | `rows, dims, float sigmaData, reserved` | Fourier embedding |
| `PrecondParams` | `atoms, reserved0, float sigmaData, reserved1` | input scale, output combine |
| `SamplerParams` | `elements, stepIndex, seed, stream, float cPrev, c, gamma0, gammaMin, noiseScale, stepScale, reserved0, reserved1` | sampler init/step |
| `AugmentParams` | `structures, atomsPerStructure, seed, stream, float sTrans, reserved0..2` | centre augment |
| `AtomAttentionParams` | `N, H, D, float scale` | sequence-local attention |
| `AtomBandParams` | `N, reserved0..2` | band geometry |
| `OuterProductMeanParams` | `S, N, reserved0, reserved1` | OPM |
| `MsaPairWeightedAvgParams` | `S, N, H, C` | MSA pair-weighted avg |
| `DropoutParams` | `A, B, C, sharedAxis, float dropProbability, seed, stream, reserved0` | dropout fwd/bwd |
| `RelPosParams` | `N, CZ, hasBonds, reserved` | relative position encoding |
| `DistanceBinParams` | `N, CZ, bins, reserved` | distance-bin embed |
| `PTMRowParams` | `N, NB, interfaceOnly, reserved, float d0, reserved1..3` | pTM row |
| `PTMMaxParams` | `N, reserved0..2` | pTM max |
| `SmoothLDDTParams` | `N, reserved0..2, float gradScale, reserved3..5` | smooth LDDT fwd/bwd |
| `BinnedCEParams` | `rows, bins, reserved0, reserved1` | binned CE fwd/bwd |

Byte offsets are not hand-maintained here. Derive them from `slangc -reflection-json`
or the `-fspv-reflect` output, and check them when the header or any field changes.

Tensor slot order (from source, `slangml_tensor_desc(p.header, slot)`):

- `sml_pairformer_triangle_attention_{start,end}_fwd_f32`: 0 = q, 1 = k, 2 = v, 3 = bias, 4 = gate, 5 = output.
- `sml_attention_pair_bias_fwd_f32`: 0 = q, 1 = k, 2 = v, 3 = bias, 4 = gate, 5 = output.
- `sml_atom_sequence_local_attention_fwd_f32`: 0 = q, 1 = k, 2 = v, 3 = bias, 4 = gate, 5 = output.
- `sml_atom_band_geometry_fwd_f32`: 0 = refPos, 1 = refUid, 2 = output.
- `sml_diffusion_adaln_fwd_f32`: 0 = a, 1 = scalePre, 2 = shift, 3 = output.
- `sml_diffusion_adaln_bwd_f32`: 0 = a, 1 = scalePre, 2 = dy, 3 = da, 4 = dScalePre.
- `sml_diffusion_adaln_zero_residual_fwd_f32`: 0 = x, 1 = gatePre, 2 = resid, 3 = output.
- `sml_diffusion_adaln_zero_residual_bwd_f32`: 0 = x, 1 = gatePre, 2 = dy, 3 = dx, 4 = dGatePre.
- `sml_diffusion_fourier_fwd_f32`: 0 = t, 1 = w, 2 = b, 3 = output.
- `sml_diffusion_input_scale_fwd_f32`: 0 = x, 1 = t, 2 = output.
- `sml_diffusion_output_combine_fwd_f32`: 0 = xNoisy, 1 = rUpdate, 2 = t, 3 = output.
- `sml_diffusion_sampler_init_fwd_f32`: 0 = output.
- `sml_diffusion_sampler_step_fwd_f32`: 0 = x, 1 = xDenoised, 2 = output.
- `sml_diffusion_centre_augment_fwd_f32`: 0 = x, 1 = output.
- `sml_msa_outer_product_mean_fwd_f32`: 0 = a, 1 = b, 2 = w, 3 = bias, 4 = zIn, 5 = zOut.
- `sml_msa_pair_weighted_avg_fwd_f32`: 0 = v, 1 = gate, 2 = bias, 3 = output.
- `sml_pairformer_dropout_residual_fwd_f32`: 0 = y, 1 = resid, 2 = output.
- `sml_pairformer_dropout_bwd_f32`: 0 = dOut, 1 = dy.
- `sml_features_relative_position_encoding_fwd_f32`: 0 = residueIndex, 1 = tokenIndex, 2 = asymId, 3 = symId, 4 = entityId, 5 = wRelPos, 6 = bonds, 7 = wBond, 8 = output.
- `sml_features_distance_bin_embed_fwd_f32`: 0 = d, 1 = vBins, 2 = w, 3 = output.
- `sml_loss_weighted_rigid_align_fwd_f32`: 0 = xPred, 1 = xGt, 2 = w, 3 = xAligned.
- `sml_loss_diffusion_mse_fwd_f32`: 0 = xPred, 1 = aligned, 2 = w, 3 = perStructure.
- `sml_loss_diffusion_mse_bwd_f32`: 0 = xPred, 1 = aligned, 2 = w, 3 = gradScale, 4 = gradX.
- `sml_loss_bond_fwd_f32`: 0 = xPred, 1 = xGt, 2 = bondPairs, 3 = perBond.
- `sml_loss_bond_bwd_f32`: 0 = xPred, 1 = xGt, 2 = bondPairs, 3 = scale, 4 = contrib.
- `sml_confidence_ptm_row_fwd_f32`: 0 = pae [N,N,NB], 1 = binCenters, 2 = inD,
  3 = chainId (uint), 4 = validFrame, 5 = rowScore.
- `sml_confidence_ptm_max_fwd_f32`: 0 = rowScore, 1 = output.
- `sml_loss_smooth_lddt_fwd_f32`: 0 = xPred, 1 = xGt, 2 = nucFlag, 3 = rowNum, 4 = rowDen.
- `sml_loss_smooth_lddt_bwd_f32`: 0 = xPred, 1 = xGt, 2 = nucFlag, 3 = gradX.
- `sml_loss_binned_cross_entropy_fwd_f32`: 0 = logits [R,B], 1 = targets [R,B], 2 = loss [R].
- `sml_loss_binned_cross_entropy_bwd_f32`: 0 = logits, 1 = targets, 2 = upstream [R], 3 = dLogits [R,B].

## 3. Dispatch

- Entries with `[numthreads(T)]` and a per-element mapping:
  `groups = ceil(elements / T)`.
- `sml_diffusion_adaln_*`, `sml_msa_outer_product_mean_fwd_f32`,
  `sml_diffusion_centre_augment_fwd_f32`: group-per-row or group-per-pair
  (`SV_GroupID` drives the index). Dispatch the group count from the header comment.
- `sml_confidence_ptm_max_fwd_f32`: exactly **1** group of 256.
- `sml_loss_smooth_lddt_*`: `groups = ceil(N / 64)`. The forward is O(N²) work.
- Every entry honors `KernelHeader.dispatchBase` (contract section 13). A flat
  element index is `dispatchBase.x + dtid.x`, and each numel guard compares that
  global index. A split dispatch sets `dispatchBase` to the first global index of its
  slice; the harness exposes this as `run_entry(..., dispatch_base=...)`.

## 4. Workspace and initialization

- `sml_confidence_ptm_row` writes `-1` for rows with `validFrame == 0`. Those rows
  never win the max.
- `sml_confidence_ptm_row` writes `0` for rows with no valid j. This is a choice;
  it must match the host-side reference.
- `sml_loss_smooth_lddt_fwd` writes `rowNum` and `rowDen` for every row, including
  rows with `den == 0`. The host divides only the sums, so `den == 0` rows are safe.
- `sml_loss_smooth_lddt_bwd` takes `gradScale = -upstream / B`, where
  `B = sum(rowDen)`. The host must pass the value in `SmoothLDDTParams.gradScale`.
- The sampler `init` entry seeds the state. `step` reads `cPrev` and `c` from the
  push constants, so the host must update them for each step.
- Dropout uses a Philox-style counter keyed on `(seed, stream, a, b, c)`. `stream`
  must differ between call sites that share a seed.

## 5. Read and write sets

Each entry reads its declared input slots and writes its declared output slots. The
`*_bwd` entries that produce gradients write them with `=`, not `+=`, so the host
must zero the destination or accumulate across calls itself. Not checked
exhaustively for every entry; treat as a guideline until the harness exists.

`sml_loss_smooth_lddt_bwd_f32` computes the full gradient for every atom k in one
pass over m, so it does not need a separate accumulation step.

## 6. Performance notes

- All kernels use scalar layout and BDA loads (`slangml_bda_load_f32`), with no
  descriptor sets. Keep tensors contiguous, row-major, and 16-byte aligned.
- `sml_loss_smooth_lddt_*` is compute-bound on four sigmoids per pair. A tiled
  shared-memory variant is the main optimization for N above about 4096. Not done.
- `sml_msa_outer_product_mean_fwd_f32` stages `o[1024]` in groupshared and runs the
  projection as a 128×1024 multiply per pair. This is the heaviest per-pair kernel.
  Tile over S if S is large.
- `sml_atom_sequence_local_attention_fwd_f32` uses the band key window
  `m = 32·(l/32) − 48 + t`, `t ∈ [0,128)`. Only 128 keys are read per query, so
  bandwidth is O(N·128·H·D).
- Reduction helpers: `slangml_workgroup_reduce_add_f32_256` (`common/subgroup.slang`)
  ends with a barrier, then returns `scratch[0]`. The caller reads `scratch[0]` after
  that barrier, but nothing separates that read from the next call's first write
  `scratch[lane] = value`. A fast lane can overwrite `scratch[0]` while a slow lane
  is still reading it. The subgroup branch of `sml_layernorm_reduce` has the same
  shape (line 27 writes `scratch[subgroup]` again with no barrier after the read at
  line 38). The AF3 kernels avoid this with wrappers (`sml_adaln_reduce`,
  `sml_sampler_reduce`, `sml_loss_reduce`) that add a barrier after each reduction.
  **Fixed.** Each `slangml_workgroup_reduce_add_*` helper now starts with a
  `GroupMemoryBarrierWithGroupSync()`, and so does every local reduce function with
  its own subgroup branch (layernorm, rmsnorm, l2normalize, softmax max and sum,
  reduce_extra sum, product, any and all, reduce_f32 sum, min and max, reduce_stats
  sum and max). The entry barrier orders every lane's read of the previous result
  before any lane's next write. The AF3 wrappers keep their trailing barrier; it is
  now redundant but harmless. The regression test
  (`test_reduce_stats_back_to_back`) is timing-dependent, so it does not prove the
  race is gone; it only confirms the back-to-back reductions still give correct
  results.
- Fp16 and bf16 variants are not implemented.

## 7. Profiling checklist

For each entry, before and after any change:

1. Run `compile_af3.ps1`. Confirm 32 PASS.
2. Record `spirv-val` output and the `-O3` SPIR-V size.
3. Time the dispatch on the target GPU at the AF3 shape:
   - N = 384 tokens for pairformer kernels
   - N = 768 tokens for the final stage
   - N_atom ≈ 4 × N_token for atom kernels
4. Check achieved bandwidth against peak. Pairwise and triangle kernels should be
   bandwidth-bound. Smooth LDDT should be ALU-bound.
5. Check occupancy: confirm the groupshared footprint (`smlPTMScratch[256]`,
   `o[1024]`) does not limit residency.
6. Confirm the reduction barrier in each `*_reduce` helper is present.
7. Compare against a CPU reference once one exists. Until then, record which
   entries have been compared. Currently **none**.

## 8. VERIFY items

These are mapping or normalization choices that need checking against the paper or
a reference implementation before results are trusted.

- Dropout axis: SI Alg 17 uses `DropoutRowwise0.25` for pair updates and
  `DropoutColumnwise0.25` for the ending-node update. Rates are 0.25 (pairformer) and
  0.15 (MSA pair-weighted averaging, Alg 8). The SI text does not define the mask
  shape. `sharedAxis = 0` (shared across the first index, keyed on `(b, c)`) follows
  the AF2 convention. Confirm the mask shape against the reference implementation.
  The kernel does not yet implement the Columnwise variant as a separate mode.
- Relative-position chain branch: confirm the same-chain and different-chain paths
  match SI Alg 3.
- Resolved against SI Alg 20 and Alg 22 (rendered page 21 and 22):
  `input_scale` is `x / sqrt(t̂² + σ²)`, `output_combine` is
  `c_skip = σ²/(σ²+t̂²)` and `c_out = σ·t̂/sqrt(σ²+t̂²)`, and the Fourier
  embedding is `cos(2π(t̂·w + b))` with `t̂` replaced by `(1/4)·log(t̂/σ_data)`.
  Kernels match these forms.
- Table 3 normalization: the loss-term normalization by N_atom or N_token² is applied
  on the host. Confirm the constants.
- Sampler step: `t̂ = c_prev(γ+1)`, `x̃ = x + λ√(t̂² − c_prev²)ε`,
  `x = x̃ + η(c − t̂)d`. Confirm λ and η against SI Alg 18.

## 9. Not included

- Host orchestration (recycling loop, LayerNorm and GEMM sequencing, stage
  scheduling, learning-rate schedule, EMA, Adam). Deferred by request.
- Backward kernels are written for every attention and MSA op in the table above:
  - pair-bias attention: 3 entries (stats, dq_dbias, dkdv)
  - triangle attention: 8 entries (start and end, each with stats, dq, dbias, dkdv)
  - sequence-local attention: 4 entries (stats, dq, dbias, dkdv)
  - outer product mean: 5 entries (pairs, dbias, dw, da, db)
  - MSA pair-weighted average: 4 entries (lse, rows, dv, dbias)
  Known costs: the outer product mean backward stores N²×1024 workspace (see its
  header), and the triangle and sequence-local backward gather passes are O(N³) and
  O(N·128·N/32) respectively. Both are for correctness first.

## 10. Test harness

Built and running on the GPU (NVIDIA RTX 3050 Laptop, Vulkan, through the Python
`vulkan` binding).

- `tests/af3_harness.py`: compiles one entry point with `slangc`, builds the BDA
  tensor table and parameter block, dispatches, and reads outputs back.
- `tests/af3_kernel_tests.py`: one test per kernel, each compared with a NumPy
  reference in float64. Backward kernels are also checked against central finite
  differences of the reference forward.

Run: `python shaders/ML/tests/af3_kernel_tests.py` (optional name filter as argument).

Coverage at last run: 79 of 79 checks pass (including the reduce regression test and
two split-dispatch checks).
Largest error was about 2.5e-5 on
binned cross-entropy backward. Tolerance is 1e-4 relative to the largest reference
value for forward checks, and 1e-3 for backward checks against finite differences.
The outer product mean dW check is against an analytic reference, not finite
differences, because dW has 131k entries.

Covered by checks: binned cross-entropy, Fourier, input scale, output combine,
smooth LDDT, pair-bias attention, triangle attention (start and end), sequence-local
attention, outer product mean, MSA pair-weighted average, AdaLN and adaLN-Zero
(forward and backward), sampler step (with noiseScale = 0), relative position
encoding, distance-bin embedding, pTM row and max, weighted rigid alignment,
diffusion MSE (forward and backward), and bond loss (forward and backward).

Checked only through properties, because the kernel uses Philox noise that NumPy
does not reproduce: sampler init (mean and standard deviation), centre augmentation
(zero centroid and preserved pairwise distances), and dropout (scale is 0 or 1/(1-p),
mask shared along the declared axis, keep fraction near 1-p, backward mask matches
forward).

Not covered yet: `sml_atom_band_geometry_fwd_f32`; the sampler noise path
(noiseScale > 0) against the Philox stream; and the chunked outer product mean
backward, which does not exist yet.
