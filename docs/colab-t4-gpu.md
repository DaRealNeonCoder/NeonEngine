# Running code on a Colab T4 GPU

Verified 2026-11-04 with a GPU check script (no model run).

## Setup (already done)

- The Colab CLI (`google-colab-cli` 0.7.5, from https://github.com/googlecolab/google-colab-cli) does not run on native Windows. It runs in WSL.
- WSL distro: `Ubuntu-24.04`
- CLI venv: `/opt/colab` (inside the distro)
- Google sign-in is stored in the distro. If it expires, re-run `colab sessions` and complete the browser sign-in.

## Run a script on a T4

From a Windows shell:

```bash
wsl -d Ubuntu-24.04 -- /opt/colab/bin/colab run --gpu T4 /mnt/c/path/to/script.py
```

`colab run` provisions a fresh VM, runs the script, and tears the VM down afterward. Use `colab new --gpu T4` to keep a session open, then `colab exec` and `colab stop`.

Windows paths are under `/mnt/c/...` inside WSL.

## Verified results

- GPU: Tesla T4, 15360 MiB VRAM, driver 580.82.07
- Runtime Python: 3.13, torch 2.11.0+cu130, CUDA available
- Test: a 1024x1024 matmul on the GPU passed

## Notes

- Script used for the check: `scratchpad/colab_gpu_check.py` (`nvidia-smi` plus a small torch matmul).
- T4 may require a Colab subscription. The first attempt succeeded on the account used here.
- Keep the OAuth codes out of notes and chat. They are one-time use and expire quickly.
