param(
    [string]$Slangc = "C:\VulkanSDK\1.4.328.1\Bin\slangc.exe",
    [string]$SpirvVal = "C:\VulkanSDK\1.4.328.1\Bin\spirv-val.exe"
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\.."))
$shaderRoot = Join-Path $repoRoot "shaders\ML"
$includeRoot = Join-Path $shaderRoot "slangml"
$outputRoot = Join-Path $env:TEMP ("neonengine-phase3-" + [guid]::NewGuid().ToString("N"))

if (-not (Test-Path -LiteralPath $Slangc)) {
    throw "slangc was not found at '$Slangc'. Pass -Slangc with the local Vulkan SDK path."
}
New-Item -ItemType Directory -Path $outputRoot | Out-Null

$jobs = @(
    @{ Source = "slangml\gemm\gemm_portable_f32.slang"; Entry = "sml_gemm_gemm_fwd_f32_portable" },
    @{ Source = "slangml\gemm\gemm_lowp.slang"; Entry = "sml_gemm_gemm_fwd_f16_accum_f32" },
    @{ Source = "slangml\gemm\gemm_lowp.slang"; Entry = "sml_gemm_gemm_fwd_bf16_accum_f32" },
    @{ Source = "slangml\einsum\contraction_f32.slang"; Entry = "sml_einsum_contract_ik_kj_fwd_f32" },
    @{ Source = "slangml\einsum\contraction_f32.slang"; Entry = "sml_einsum_triangle_mul_outgoing_fwd_f32" },
    @{ Source = "slangml\einsum\contraction_f32.slang"; Entry = "sml_einsum_triangle_mul_incoming_fwd_f32" }
)

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
            throw "slangc failed for $($job.Entry) with exit code $LASTEXITCODE."
        }

        if (Test-Path -LiteralPath $SpirvVal) {
            & $SpirvVal $output
            if ($LASTEXITCODE -ne 0) {
                throw "spirv-val failed for $($job.Entry) with exit code $LASTEXITCODE."
            }
        }
    }
    Write-Host "Phase 3 shader compilation and validation passed."
}
finally {
    if (Test-Path -LiteralPath $outputRoot) {
        Remove-Item -LiteralPath $outputRoot -Recurse -Force
    }
}
