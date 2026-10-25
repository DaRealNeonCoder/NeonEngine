param(
    [string]$Slangc = "C:\VulkanSDK\1.4.328.1\Bin\slangc.exe",
    [string]$SpirvVal = "C:\VulkanSDK\1.4.328.1\Bin\spirv-val.exe"
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\.."))
$shaderRoot = Join-Path $repoRoot "shaders\ML"
$includeRoot = Join-Path $shaderRoot "slangml"
$outputRoot = Join-Path $env:TEMP ("neonengine-digit-recognizer-" + [guid]::NewGuid().ToString("N"))

if (-not (Test-Path -LiteralPath $Slangc)) {
    throw "slangc was not found at '$Slangc'. Pass -Slangc with the local Vulkan SDK path."
}
New-Item -ItemType Directory -Path $outputRoot | Out-Null

$source = Join-Path $shaderRoot "slangml\models\digit_recognizer.slang"
$entries = @(
    "sml_model_digit_recognizer_fwd_f16_f32",
    "sml_model_digit_recognizer_fwd_bf16_f32"
)

try {
    foreach ($entry in $entries) {
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
            -fspv-reflect `
            -O3 `
            -o $output
        if ($LASTEXITCODE -ne 0) {
            throw "slangc failed for $entry with exit code $LASTEXITCODE."
        }

        if (Test-Path -LiteralPath $SpirvVal) {
            & $SpirvVal $output
            if ($LASTEXITCODE -ne 0) {
                throw "spirv-val failed for $entry with exit code $LASTEXITCODE."
            }
        }
    }
    Write-Host "Digit recognizer shader compilation and validation passed."
}
finally {
    if (Test-Path -LiteralPath $outputRoot) {
        Remove-Item -LiteralPath $outputRoot -Recurse -Force
    }
}
