$GLSLC = "C:\VulkanSDK\1.4.328.1\Bin\glslc.exe"
$SHADER_DIR = "C:\Users\ZyBros\Downloads\NeonEngine\NeonEngine\shaders"

Write-Host "Compiling updated shaders..."

# Grab all files in the directory that match your shader extensions
Get-ChildItem -Path $SHADER_DIR -File | Where-Object {
    $_.Extension -match "\.(vert|frag|rgen|rmiss|rchit|comp)$"
} | ForEach-Object {
    $source = $_.FullName
    $spv = "$source.spv"
    
    # Check if .spv is missing, OR if the source file is newer than the .spv
    if (-not (Test-Path $spv) -or ($_.LastWriteTime -gt (Get-Item $spv).LastWriteTime)) {
        Write-Host "Compiling $($_.Name)..."
        & $GLSLC --target-env=vulkan1.2 $source -o $spv
    }
}

Write-Host "Done."