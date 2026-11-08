"""Zips the parts of NeonEngine that NeonML needs (sources, CMake, SlangML shaders,
train/ scripts) so you can upload one small file to Google Drive and unzip it on Colab.

    python train/colab/pack_for_colab.py            # writes NeonEngine_colab.zip next to the repo folder
    python train/colab/pack_for_colab.py out.zip

Upload the zip to Drive at MyDrive/neon/NeonEngine_colab.zip (the notebook's default).
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

ENGINE = Path(__file__).resolve().parents[2]          # NeonEngine/NeonEngine
INCLUDE = ["CMakeLists.txt", "src", "shaders/ML", "train"]
SKIP_DIRS = {"__pycache__", "build", ".git"}
SKIP_SUFFIXES = {".obj", ".spv", ".pdf", ".bin", ".nckpt", ".zip"}


def main() -> int:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else ENGINE.parent / "NeonEngine_colab.zip"
    count = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for item in INCLUDE:
            root = ENGINE / item
            files = [root] if root.is_file() else sorted(p for p in root.rglob("*") if p.is_file())
            for f in files:
                rel = f.relative_to(ENGINE)
                if SKIP_DIRS & set(rel.parts) or f.suffix.lower() in SKIP_SUFFIXES:
                    continue
                z.write(f, Path("NeonEngine") / rel)
                count += 1
    print(f"wrote {out} ({count} files, {out.stat().st_size / 1e6:.1f} MB)")
    print("upload it to Google Drive as MyDrive/neon/NeonEngine_colab.zip")
    return 0


if __name__ == "__main__":
    sys.exit(main())
