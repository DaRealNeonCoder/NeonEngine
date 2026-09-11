$SLANGC = "C:\VulkanSDK\1.4.328.1\Bin\slangc.exe"
$SHADER_DIR = "C:\Users\ZyBros\Downloads\NeonEngine\NeonEngine\shaders\restir_pt"

Write-Host "Compiling updated Slang shaders..."

# Grab all Slang shader files
Get-ChildItem -Path $SHADER_DIR -File | Where-Object {
    $_.Extension -eq ".slang"
} | ForEach-Object {
    $source = $_.FullName
    $spv = "$source.spv"

    # Check if .spv is missing, OR if the source is newer than the .spv
    if (-not (Test-Path $spv) -or ($_.LastWriteTime -gt (Get-Item $spv).LastWriteTime)) {
        Write-Host "Compiling $($_.Name)..."

        & $SLANGC `
            $source `
            -target spirv `
            -o $spv
    }
}

Write-Host "Done."