#!/usr/bin/env bash
# Sets up a Google Colab GPU runtime (T4) for NeonEngine's gpt-oss trainer:
#   1. the LunarG Vulkan SDK for Linux (Vulkan loader + headers + Slang)
#   2. the NVIDIA Vulkan driver pieces matching the runtime's kernel driver
#      (Colab ships CUDA userspace libraries but no Vulkan ICD)
#   3. a Release build of the headless NeonML tool
# Usage:  bash setup_colab.sh <NeonEngine/NeonEngine source dir> [build dir]
# Safe to rerun: finished steps are skipped. Writes /content/neon_env.sh.
set -euo pipefail

SRC="$(cd "${1:?usage: setup_colab.sh <NeonEngine/NeonEngine dir> [build dir]}" && pwd)"
BUILD="${2:-/content/neon_build}"
SDK_VER="${NEON_VULKAN_SDK_VERSION:-1.4.328.1}"
SDK_ROOT="/content/vulkan_sdk"
SDK="$SDK_ROOT/$SDK_VER/x86_64"
LIBDIR=/usr/lib/x86_64-linux-gnu

say() { printf '\n==> %s\n' "$*"; }
die() { printf '\nERROR: %s\n' "$*" >&2; exit 1; }

command -v nvidia-smi >/dev/null || die "no nvidia-smi: switch the runtime to a GPU (Runtime > Change runtime type > T4 GPU)"
GPU="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
DRV="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | tr -d ' ')"
say "GPU: $GPU, driver $DRV"

# ---- 1. Vulkan SDK ---------------------------------------------------------
if [ ! -f "$SDK/lib/libslang.so" ]; then
  say "downloading Vulkan SDK $SDK_VER (~330 MB)"
  mkdir -p "$SDK_ROOT"
  curl -fL --retry 3 "https://sdk.lunarg.com/sdk/download/$SDK_VER/linux/vulkan_sdk.tar.xz" | tar -xJ -C "$SDK_ROOT"
  [ -f "$SDK/lib/libslang.so" ] || die "SDK unpacked but $SDK/lib/libslang.so is missing"
fi
export VULKAN_SDK="$SDK"
export PATH="$SDK/bin:$PATH"
export LD_LIBRARY_PATH="$SDK/lib:${LD_LIBRARY_PATH:-}"

# ---- 2. NVIDIA Vulkan ICD --------------------------------------------------
# Libraries the NVIDIA Vulkan driver needs, beyond the CUDA ones Colab has.
NV_LIBS="libGLX_nvidia libEGL_nvidia libnvidia-glcore libnvidia-eglcore libnvidia-glsi libnvidia-glvkspirv libnvidia-tls libnvidia-gpucomp"
have_nv_lib() { ls "$LIBDIR/$1.so.$DRV" /usr/lib64-nvidia/"$1.so.$DRV" >/dev/null 2>&1; }

vk_ok() { vulkaninfo --summary 2>/dev/null | grep -qE "driverName *= *NVIDIA"; }

write_icd() {  # $1 = library soname
  mkdir -p /usr/share/vulkan/icd.d /etc/vulkan/icd.d
  cat > /usr/share/vulkan/icd.d/nvidia_icd.json <<EOF
{ "file_format_version": "1.0.1", "ICD": { "library_path": "$1", "api_version": "1.4.0" } }
EOF
  cp /usr/share/vulkan/icd.d/nvidia_icd.json /etc/vulkan/icd.d/nvidia_icd.json
}

if ! vk_ok; then
  say "installing the NVIDIA Vulkan driver userspace for $DRV"
  apt-get -qq update >/dev/null && apt-get -qq install -y libx11-6 libxext6 libglvnd0 >/dev/null || true
  missing=0
  for l in libGLX_nvidia libnvidia-glcore libnvidia-glvkspirv libnvidia-tls; do have_nv_lib "$l" || missing=1; done
  if [ "$missing" = 1 ]; then
    RUN="/content/NVIDIA-Linux-x86_64-$DRV.run"
    if [ ! -s "$RUN" ]; then
      curl -fL --retry 3 -o "$RUN" "https://us.download.nvidia.com/tesla/$DRV/NVIDIA-Linux-x86_64-$DRV.run" ||
      curl -fL --retry 3 -o "$RUN" "https://download.nvidia.com/XFree86/Linux-x86_64/$DRV/NVIDIA-Linux-x86_64-$DRV.run" ||
        die "could not download the NVIDIA $DRV driver package (needed only for its Vulkan libraries)"
    fi
    rm -rf /content/nvidia_extract
    sh "$RUN" -x --target /content/nvidia_extract >/dev/null
    for l in $NV_LIBS; do
      f="/content/nvidia_extract/$l.so.$DRV"
      if [ -f "$f" ] && ! have_nv_lib "$l"; then cp "$f" "$LIBDIR/"; fi
    done
  else
    for l in $NV_LIBS; do
      [ -f "/usr/lib64-nvidia/$l.so.$DRV" ] && [ ! -e "$LIBDIR/$l.so.$DRV" ] && ln -s "/usr/lib64-nvidia/$l.so.$DRV" "$LIBDIR/"
    done
  fi
  # sonames the ICD loader and the libraries look up
  for l in libGLX_nvidia libEGL_nvidia; do [ -e "$LIBDIR/$l.so.$DRV" ] && ln -sf "$l.so.$DRV" "$LIBDIR/$l.so.0"; done
  for l in libnvidia-glcore libnvidia-eglcore libnvidia-glsi libnvidia-glvkspirv libnvidia-tls libnvidia-gpucomp; do
    [ -e "$LIBDIR/$l.so.$DRV" ] && ln -sf "$l.so.$DRV" "$LIBDIR/$l.so"
  done
  ldconfig
  for icd in libGLX_nvidia.so.0 libEGL_nvidia.so.0; do
    write_icd "$icd"
    if vk_ok; then say "Vulkan works through $icd"; break; fi
  done
  vk_ok || { vulkaninfo --summary || true; die "Vulkan does not see the GPU (see vulkaninfo output above)"; }
fi
vulkaninfo --summary 2>/dev/null | grep -E "apiVersion|deviceName|driverVersion" | head -3

# ---- 3. build NeonML -------------------------------------------------------
say "building NeonML"
cmake -S "$SRC" -B "$BUILD" -DNEON_BUILD_ENGINE=OFF -DCMAKE_BUILD_TYPE=Release \
  -DVULKAN_SDK_PATH="$SDK" -DCMAKE_BUILD_RPATH="$SDK/lib" >/dev/null
cmake --build "$BUILD" -j"$(nproc)" --target NeonML
[ -x "$BUILD/NeonML" ] || die "build finished without $BUILD/NeonML"

cat > /content/neon_env.sh <<EOF
export VULKAN_SDK="$SDK"
export PATH="$SDK/bin:\$PATH"
export LD_LIBRARY_PATH="$SDK/lib:\${LD_LIBRARY_PATH:-}"
export NEON_SRC="$SRC"
export NEON_ML="$BUILD/NeonML"
EOF
say "done: $BUILD/NeonML (environment in /content/neon_env.sh)"
