param(
    [string]$Slangc = "C:\VulkanSDK\1.4.328.1\Bin\slangc.exe",
    [string]$SpirvVal = "C:\VulkanSDK\1.4.328.1\Bin\spirv-val.exe"
)

# Compiles and validates every SlangML LLM entry point (see shaders/ML/docs/LLM_ROADMAP.md).
# Output SPIR-V goes to a temp directory that is removed on exit.
# GPU correctness: python llm_kernel_tests.py ; speed: python llm_bench.py

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\.."))
$shaderRoot = Join-Path $repoRoot "shaders\ML"
$includeRoot = Join-Path $shaderRoot "slangml"
$outputRoot = Join-Path $env:TEMP ("neonengine-llm-" + [guid]::NewGuid().ToString("N"))

if (-not (Test-Path -LiteralPath $Slangc)) {
    throw "slangc was not found at '$Slangc'. Pass -Slangc with the local Vulkan SDK path."
}
New-Item -ItemType Directory -Path $outputRoot | Out-Null

$gemv = @()
foreach ($fmt in @("q4", "q8")) { foreach ($m in @("m1", "m2", "m4", "m8")) { $gemv += "sml_quant_gemv_fwd_${fmt}_$m" } }
foreach ($fmt in @("mxfp4", "f16", "bf16")) { foreach ($m in @("m1", "m4")) { $gemv += "sml_quant_gemv_fwd_${fmt}_$m" } }
$gemv += @("sml_quant_gemv_swiglu_fwd_q4_m1", "sml_quant_gemv_swiglu_fwd_q4_m4", "sml_quant_gemv_swiglu_fwd_q8_m1",
           "sml_quant_gemv_swiglu_fwd_q8_m4", "sml_quant_gemv_swiglu_fwd_f16_m1", "sml_quant_gemv_swiglu_fwd_bf16_m1")

$jobs = @()
$jobs += @{ Source = "slangml\quant\quantize_f32.slang"; Entries = @("sml_quant_quantize_q8act_fwd_f32") }
$jobs += @{ Source = "slangml\quant\gemv_quant.slang"; Entries = $gemv }
$jobs += @{ Source = "slangml\quant\gemv_moe.slang"; Entries = @(
    "sml_quant_gemv_moe_swiglu_fwd_q4", "sml_quant_gemv_moe_swiglu_fwd_q8", "sml_quant_gemv_moe_swiglu_fwd_bf16",
    "sml_quant_gemv_moe_swiglu_fwd_mxfp4",
    "sml_quant_gemv_moe_down_fwd_q4", "sml_quant_gemv_moe_down_fwd_q8", "sml_quant_gemv_moe_down_fwd_bf16",
    "sml_quant_gemv_moe_down_fwd_mxfp4") }
$jobs += @{ Source = "slangml\quant\gemm_quant_coopmat.slang"; Entries = @(
    "sml_quant_gemm_fwd_q4_coopmat", "sml_quant_gemm_fwd_q8_coopmat", "sml_quant_gemm_fwd_f16_coopmat",
    "sml_quant_gemm_fwd_bf16_coopmat", "sml_quant_gemm_fwd_q4_coopmat_m64", "sml_quant_gemm_fwd_q8_coopmat_m64",
    "sml_quant_gemm_fwd_bf16_coopmat_m64") }
$jobs += @{ Source = "slangml\quant\embedding.slang"; Entries = @(
    "sml_quant_embedding_fwd_q4", "sml_quant_embedding_fwd_q8", "sml_quant_embedding_fwd_f16", "sml_quant_embedding_fwd_bf16") }
$jobs += @{ Source = "slangml\llm\rmsnorm_fused_f32.slang"; Entries = @("sml_llm_rmsnorm_fwd_f32_row", "sml_llm_rmsnorm_fwd_f32_small") }
$jobs += @{ Source = "slangml\llm\rope_qknorm_f32.slang"; Entries = @("sml_llm_rope_qknorm_fwd_f32") }
$jobs += @{ Source = "slangml\llm\attention_flash_f32.slang"; Entries = @(
    "sml_llm_attention_fwd_f16kv_d64", "sml_llm_attention_fwd_f16kv_d128", "sml_llm_attention_fwd_f16kv_d256",
    "sml_llm_attention_combine_fwd_f32") }
$jobs += @{ Source = "slangml\llm\moe_router_f32.slang"; Entries = @("sml_llm_moe_router_fwd_f32") }
$jobs += @{ Source = "slangml\llm\sampling_f32.slang"; Entries = @(
    "sml_llm_argmax_partial_fwd_f32", "sml_llm_argmax_final_fwd_f32", "sml_llm_topk_partial_fwd_f32",
    "sml_llm_topk_merge_fwd_f32", "sml_llm_sample_topk_fwd_f32", "sml_llm_logits_penalty_fwd_f32") }
$jobs += @{ Source = "slangml\linear_attention\conv1d_f32.slang"; Entries = @("sml_linattn_conv1d_fwd_f32") }
$jobs += @{ Source = "slangml\linear_attention\gated_delta_rule_f32.slang"; Entries = @("sml_linattn_gated_delta_rule_fwd_f32") }

$failed = @()
$count = 0
try {
    foreach ($job in $jobs) {
        $source = Join-Path $shaderRoot $job.Source
        foreach ($entry in $job.Entries) {
            $count++
            $output = Join-Path $outputRoot ($entry + ".spv")
            Write-Host "Compiling $entry..."
            & $Slangc $source `
                -I $includeRoot `
                -target spirv `
                -profile spirv_1_5 `
                -stage compute `
                -entry $entry `
                -fvk-use-entrypoint-name `
                -fvk-use-scalar-layout `
                -O3 `
                -warnings-disable 41012 `
                -o $output
            if ($LASTEXITCODE -ne 0) {
                $failed += $entry
                continue
            }
            if (Test-Path -LiteralPath $SpirvVal) {
                & $SpirvVal --target-env vulkan1.3 $output
                if ($LASTEXITCODE -ne 0) {
                    $failed += $entry
                }
            }
        }
    }
    if ($failed.Count -gt 0) {
        throw ("LLM shader compilation/validation failed for: " + ($failed -join ", "))
    }
    Write-Host "LLM shader compilation and validation passed ($count entry points)."
}
finally {
    if (Test-Path -LiteralPath $outputRoot) {
        Remove-Item -LiteralPath $outputRoot -Recurse -Force
    }
}
