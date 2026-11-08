"""Gradient test of the gpt-oss trainer (src/gpt_oss_train.cpp + slangml/train/*)
against the official PyTorch reference (gpt_oss_reference.py) in float64.

It runs `NeonML --gpt-oss-train gradcheck` on a tiny random model (every gpt-oss
feature: sliding-window and full layers, sinks, YaRN RoPE, GQA, top-k MoE with
clamped SwiGLU and biases), loads the same initial parameters into the
reference, computes the mean next-token cross-entropy over the batch with
autograd, and compares the loss and the gradient of every parameter.

Usage (from shaders/ML/tests, after building NeonML in Release):
    python gpt_oss_train_test.py [--precision fp16]
Needs numpy and torch (CPU is fine). Exit code 0 = all checks passed.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

import gpt_oss_reference as ref

TESTS_DIR = Path(__file__).resolve().parent
ENGINE_ROOT = TESTS_DIR.parents[2]
EXE_CANDIDATES = [ENGINE_ROOT / "build" / "Release" / "NeonML.exe", ENGINE_ROOT / "build" / "NeonML"]

CONFIG = dict(
    num_hidden_layers=2, num_experts=4, experts_per_token=2, vocab_size=512,
    hidden_size=128, intermediate_size=64, swiglu_limit=7.0, head_dim=64,
    num_attention_heads=4, num_key_value_heads=2, sliding_window=16,
    initial_context_length=4096, rope_theta=150000.0, rope_scaling_factor=32.0,
    rope_ntk_alpha=1.0, rope_ntk_beta=32.0,
)


def run_gradcheck(work: Path, tokens: np.ndarray, batch: int, seq: int,
                  precision: str = "fp32") -> tuple[float, dict, np.ndarray, np.ndarray]:
    exe = next((p for p in EXE_CANDIDATES if p.exists()), None)
    if exe is None:
        raise FileNotFoundError("build NeonML first (cmake --build build --config Release --target NeonML)")
    (work / "config.json").write_text(json.dumps(CONFIG))
    (work / "tokens.txt").write_text(",".join(map(str, tokens.reshape(-1).tolist())))
    cmd = [str(exe), "--gpt-oss-train", "gradcheck", "--config", str(work / "config.json"),
           "--tokens-file", str(work / "tokens.txt"), "--batch", str(batch), "--seq", str(seq),
           "--out", str(work / "out"), "--aux", "0", "--precision", precision,
           "--shaders", str(ENGINE_ROOT / "shaders" / "ML")]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=ENGINE_ROOT)
    print(proc.stdout.strip())
    if proc.returncode != 0:
        print(proc.stderr)
        raise RuntimeError(f"NeonML failed ({proc.returncode})")
    layout = json.loads((work / "out" / "layout.json").read_text())
    params = np.fromfile(work / "out" / "params.bin", dtype=np.float32)
    grads = np.fromfile(work / "out" / "grads.bin", dtype=np.float32)
    return layout["loss"], layout, params, grads


def tensor(flat: np.ndarray, t: dict) -> np.ndarray:
    n = int(np.prod(t["shape"]))
    return flat[t["offset"]:t["offset"] + n].reshape(t["shape"])


def reference(layout: dict, params: np.ndarray, tokens: np.ndarray) -> tuple[float, dict]:
    torch.manual_seed(0)
    model = ref.Transformer(ref.ModelConfig(**CONFIG), device=torch.device("cpu")).to(torch.float64)
    by_name = {t["name"]: t for t in layout["tensors"]}
    named = dict(model.named_parameters())
    if set(named) != set(by_name):
        raise KeyError(f"parameter sets differ: {sorted(set(named) ^ set(by_name))}")
    with torch.no_grad():
        for name, p in named.items():
            p.copy_(torch.from_numpy(tensor(params, by_name[name]).astype(np.float64)))
    total = 0.0
    count = 0
    for seq in tokens:
        x = torch.tensor(seq[:-1], dtype=torch.long)
        y = torch.tensor(seq[1:], dtype=torch.long)
        logits = model(x)
        total = total + torch.nn.functional.cross_entropy(logits, y, reduction="sum")
        count += len(y)
    loss = total / count
    loss.backward()
    return float(loss), {n: p.grad.numpy() for n, p in named.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--seq", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--precision", choices=["fp32", "fp16"], default="fp32",
                    help="fp16 = tensor-core GEMMs with loss scaling (looser tolerance)")
    args = ap.parse_args()
    # fp16 rounds every GEMM input to an 11-bit mantissa: ~1e-3 relative noise per product.
    tol = 2e-3 if args.precision == "fp32" else 1e-2
    if args.precision == "fp16":
        # A random-init router has near-tied logits, so fp16 rounding flips some
        # top-k choices, and a token sent to another expert changes every gradient
        # downstream (a few % at 3 x 256 tokens). Routing to all experts removes
        # the discrete choice and leaves only the GEMM rounding to check.
        CONFIG["experts_per_token"] = CONFIG["num_experts"]
        print("fp16: experts_per_token = num_experts (no top-k flips), tolerance", tol)
    rng = np.random.default_rng(args.seed)
    tokens = rng.integers(0, CONFIG["vocab_size"], size=(args.batch, args.seq + 1))

    with tempfile.TemporaryDirectory() as tmp:
        loss, layout, params, grads = run_gradcheck(Path(tmp), tokens, args.batch, args.seq, args.precision)
    ref_loss, ref_grads = reference(layout, params, tokens)

    ok = True
    print(f"loss: engine {loss:.7f}  reference {ref_loss:.7f}  diff {abs(loss - ref_loss):.2e}")
    if abs(loss - ref_loss) > (1e-4 if args.precision == "fp32" else 2e-3) * max(1.0, abs(ref_loss)):
        ok = False
    worst = 0.0
    for t in layout["tensors"]:
        got = tensor(grads, t).astype(np.float64)
        want = ref_grads[t["name"]]
        denom = max(np.linalg.norm(want), 1e-12)
        rel = np.linalg.norm(got - want) / denom
        worst = max(worst, rel)
        flag = "" if rel < tol else "   <-- FAIL"
        if flag:
            ok = False
        print(f"  {t['name']:32s} |g| {np.linalg.norm(want):10.3e}  rel err {rel:.2e}{flag}")
    print(f"worst relative gradient error {worst:.2e}")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
