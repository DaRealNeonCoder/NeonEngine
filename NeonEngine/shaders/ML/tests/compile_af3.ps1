param(
    [string]$Slangc = "C:\VulkanSDK\1.4.328.1\Bin\slangc.exe",
    [string]$SpirvVal = "C:\VulkanSDK\1.4.328.1\Bin\spirv-val.exe"
)

# Compiles and validates every AlphaFold 3 entry point added to SlangML
# (see shaders/ML/docs/af3_ops.md).  Output SPIR-V goes to a temp directory
# that is removed on exit.

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\.."))
$shaderRoot = Join-Path $repoRoot "shaders\ML"
$includeRoot = Join-Path $shaderRoot "slangml"
$outputRoot = Join-Path $env:TEMP ("neonengine-af3-" + [guid]::NewGuid().ToString("N"))

if (-not (Test-Path -LiteralPath $Slangc)) {
    throw "slangc was not found at '$Slangc'. Pass -Slangc with the local Vulkan SDK path."
}
New-Item -ItemType Directory -Path $outputRoot | Out-Null

$jobs = @(
    @{ Source = "slangml\pairformer\triangle_attention_f32.slang"; Entry = "sml_pairformer_triangle_attention_start_fwd_f32" },
    @{ Source = "slangml\pairformer\triangle_attention_f32.slang"; Entry = "sml_pairformer_triangle_attention_end_fwd_f32" },
    @{ Source = "slangml\pairformer\triangle_attention_bwd_f32.slang"; Entry = "sml_pairformer_triangle_attention_start_bwd_f32_stats" },
    @{ Source = "slangml\pairformer\triangle_attention_bwd_f32.slang"; Entry = "sml_pairformer_triangle_attention_start_bwd_f32_dq" },
    @{ Source = "slangml\pairformer\triangle_attention_bwd_f32.slang"; Entry = "sml_pairformer_triangle_attention_start_bwd_f32_dbias" },
    @{ Source = "slangml\pairformer\triangle_attention_bwd_f32.slang"; Entry = "sml_pairformer_triangle_attention_start_bwd_f32_dkdv" },
    @{ Source = "slangml\pairformer\triangle_attention_bwd_f32.slang"; Entry = "sml_pairformer_triangle_attention_end_bwd_f32_stats" },
    @{ Source = "slangml\pairformer\triangle_attention_bwd_f32.slang"; Entry = "sml_pairformer_triangle_attention_end_bwd_f32_dq" },
    @{ Source = "slangml\pairformer\triangle_attention_bwd_f32.slang"; Entry = "sml_pairformer_triangle_attention_end_bwd_f32_dbias" },
    @{ Source = "slangml\pairformer\triangle_attention_bwd_f32.slang"; Entry = "sml_pairformer_triangle_attention_end_bwd_f32_dkdv" },
    @{ Source = "slangml\attention\pair_bias_attention_f32.slang"; Entry = "sml_attention_pair_bias_fwd_f32" },
    @{ Source = "slangml\attention\pair_bias_attention_bwd_f32.slang"; Entry = "sml_attention_pair_bias_bwd_f32_stats" },
    @{ Source = "slangml\attention\pair_bias_attention_bwd_f32.slang"; Entry = "sml_attention_pair_bias_bwd_f32_dq_dbias" },
    @{ Source = "slangml\attention\pair_bias_attention_bwd_f32.slang"; Entry = "sml_attention_pair_bias_bwd_f32_dkdv" },
    @{ Source = "slangml\diffusion\adaln_f32.slang"; Entry = "sml_diffusion_adaln_fwd_f32" },
    @{ Source = "slangml\diffusion\adaln_f32.slang"; Entry = "sml_diffusion_adaln_bwd_f32" },
    @{ Source = "slangml\diffusion\adaln_f32.slang"; Entry = "sml_diffusion_adaln_zero_residual_fwd_f32" },
    @{ Source = "slangml\diffusion\adaln_f32.slang"; Entry = "sml_diffusion_adaln_zero_residual_bwd_f32" },
    @{ Source = "slangml\diffusion\conditioning_f32.slang"; Entry = "sml_diffusion_fourier_fwd_f32" },
    @{ Source = "slangml\diffusion\conditioning_f32.slang"; Entry = "sml_diffusion_input_scale_fwd_f32" },
    @{ Source = "slangml\diffusion\conditioning_f32.slang"; Entry = "sml_diffusion_output_combine_fwd_f32" },
    @{ Source = "slangml\diffusion\sampler_f32.slang"; Entry = "sml_diffusion_sampler_init_fwd_f32" },
    @{ Source = "slangml\diffusion\sampler_f32.slang"; Entry = "sml_diffusion_sampler_step_fwd_f32" },
    @{ Source = "slangml\diffusion\sampler_f32.slang"; Entry = "sml_diffusion_centre_augment_fwd_f32" },
    @{ Source = "slangml\atom\sequence_local_attention_f32.slang"; Entry = "sml_atom_sequence_local_attention_fwd_f32" },
    @{ Source = "slangml\atom\sequence_local_attention_f32.slang"; Entry = "sml_atom_band_geometry_fwd_f32" },
    @{ Source = "slangml\msa\outer_product_mean_f32.slang"; Entry = "sml_msa_outer_product_mean_fwd_f32" },
    @{ Source = "slangml\msa\msa_pair_weighted_avg_f32.slang"; Entry = "sml_msa_pair_weighted_avg_fwd_f32" },
    @{ Source = "slangml\pairformer\dropout_residual_f32.slang"; Entry = "sml_pairformer_dropout_residual_fwd_f32" },
    @{ Source = "slangml\pairformer\dropout_residual_f32.slang"; Entry = "sml_pairformer_dropout_bwd_f32" },
    @{ Source = "slangml\features\pair_embedding_f32.slang"; Entry = "sml_features_relative_position_encoding_fwd_f32" },
    @{ Source = "slangml\features\pair_embedding_f32.slang"; Entry = "sml_features_distance_bin_embed_fwd_f32" },
    @{ Source = "slangml\confidence\ptm_f32.slang"; Entry = "sml_confidence_ptm_row_fwd_f32" },
    @{ Source = "slangml\confidence\ptm_f32.slang"; Entry = "sml_confidence_ptm_max_fwd_f32" },
    @{ Source = "slangml\loss\smooth_lddt_f32.slang"; Entry = "sml_loss_smooth_lddt_fwd_f32" },
    @{ Source = "slangml\loss\smooth_lddt_f32.slang"; Entry = "sml_loss_smooth_lddt_bwd_f32" },
    @{ Source = "slangml\loss\diffusion_loss_f32.slang"; Entry = "sml_loss_weighted_rigid_align_fwd_f32" },
    @{ Source = "slangml\loss\diffusion_loss_f32.slang"; Entry = "sml_loss_diffusion_mse_fwd_f32" },
    @{ Source = "slangml\loss\diffusion_loss_f32.slang"; Entry = "sml_loss_diffusion_mse_bwd_f32" },
    @{ Source = "slangml\loss\diffusion_loss_f32.slang"; Entry = "sml_loss_bond_fwd_f32" },
    @{ Source = "slangml\loss\diffusion_loss_f32.slang"; Entry = "sml_loss_bond_bwd_f32" },
    @{ Source = "slangml\loss\binned_cross_entropy_f32.slang"; Entry = "sml_loss_binned_cross_entropy_fwd_f32" },
    @{ Source = "slangml\loss\binned_cross_entropy_f32.slang"; Entry = "sml_loss_binned_cross_entropy_bwd_f32" },
    @{ Source = "slangml\atom\sequence_local_attention_bwd_f32.slang"; Entry = "sml_atom_sequence_local_attention_bwd_f32_stats" },
    @{ Source = "slangml\atom\sequence_local_attention_bwd_f32.slang"; Entry = "sml_atom_sequence_local_attention_bwd_f32_dq" },
    @{ Source = "slangml\atom\sequence_local_attention_bwd_f32.slang"; Entry = "sml_atom_sequence_local_attention_bwd_f32_dbias" },
    @{ Source = "slangml\atom\sequence_local_attention_bwd_f32.slang"; Entry = "sml_atom_sequence_local_attention_bwd_f32_dkdv" },
    @{ Source = "slangml\msa\outer_product_mean_bwd_f32.slang"; Entry = "sml_msa_outer_product_mean_bwd_f32_pairs" },
    @{ Source = "slangml\msa\outer_product_mean_bwd_f32.slang"; Entry = "sml_msa_outer_product_mean_bwd_f32_dbias" },
    @{ Source = "slangml\msa\outer_product_mean_bwd_f32.slang"; Entry = "sml_msa_outer_product_mean_bwd_f32_dw" },
    @{ Source = "slangml\msa\outer_product_mean_bwd_f32.slang"; Entry = "sml_msa_outer_product_mean_bwd_f32_da" },
    @{ Source = "slangml\msa\outer_product_mean_bwd_f32.slang"; Entry = "sml_msa_outer_product_mean_bwd_f32_db" },
    @{ Source = "slangml\msa\msa_pair_weighted_avg_bwd_f32.slang"; Entry = "sml_msa_pair_weighted_avg_bwd_f32_lse" },
    @{ Source = "slangml\msa\msa_pair_weighted_avg_bwd_f32.slang"; Entry = "sml_msa_pair_weighted_avg_bwd_f32_rows" },
    @{ Source = "slangml\msa\msa_pair_weighted_avg_bwd_f32.slang"; Entry = "sml_msa_pair_weighted_avg_bwd_f32_dv" },
    @{ Source = "slangml\msa\msa_pair_weighted_avg_bwd_f32.slang"; Entry = "sml_msa_pair_weighted_avg_bwd_f32_dbias" }
)

$failed = @()
try {
    foreach ($job in $jobs) {
        $source = Join-Path $shaderRoot $job.Source
        $output = Join-Path $outputRoot ($job.Entry + ".spv")
        Write-Host "Compiling $($job.Entry)..."
        & $Slangc $source `
            -I $includeRoot `
            -target spirv `
            -profile spirv_1_5 `
            -stage compute `
            -entry $job.Entry `
            -fvk-use-entrypoint-name `
            -fvk-use-scalar-layout `
            -fspv-reflect `
            -O3 `
            -o $output
        if ($LASTEXITCODE -ne 0) {
            $failed += $job.Entry
            continue
        }

        if (Test-Path -LiteralPath $SpirvVal) {
            & $SpirvVal $output
            if ($LASTEXITCODE -ne 0) {
                $failed += $job.Entry
            }
        }
    }
    if ($failed.Count -gt 0) {
        throw ("AF3 shader compilation/validation failed for: " + ($failed -join ", "))
    }
    Write-Host "AF3 shader compilation and validation passed ($($jobs.Count) entry points)."
}
finally {
    if (Test-Path -LiteralPath $outputRoot) {
        Remove-Item -LiteralPath $outputRoot -Recurse -Force
    }
}
