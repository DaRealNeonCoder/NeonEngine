"""End-to-end test of the C++ gpt-oss host (src/gpt_oss_model.cpp) against the
official PyTorch reference (gpt_oss_reference.py, vendored from openai/gpt-oss).

It builds a tiny gpt-oss checkpoint with random weights in the *official*
format (config.json + model.safetensors with MXFP4 expert blocks/scales), runs
the reference in float64 on the CPU, runs LveEngine.exe --gpt-oss --score on the
same tokens, and compares the logits at every position.

The tiny model keeps every structural feature of gpt-oss-20b: alternating
sliding-window / full attention with sinks, YaRN RoPE (same theta/factor),
GQA with 8 query heads per KV head, head_dim 64, top-4 MoE with clamped
SwiGLU and biases, MXFP4 experts. The sequence is longer than the sliding
window and long enough to use split-KV attention.

Usage (from shaders/ML/tests, after building the engine in Release):
    python gpt_oss_e2e_test.py              # bf16 dense + VRAM experts, q8 dense + host experts
    python gpt_oss_e2e_test.py --keep DIR   # also keep the checkpoint in DIR

Needs numpy and torch (CPU is fine). Exit code 0 = all checks passed.
"""

from __future__ import annotations

import argparse
import json
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

import llm_formats as F
import gpt_oss_reference as ref

TESTS_DIR = Path(__file__).resolve().parent
ENGINE_ROOT = TESTS_DIR.parents[2]          # NeonEngine/NeonEngine
EXE = next((p for p in (ENGINE_ROOT / "build" / "Release" / "NeonML.exe", ENGINE_ROOT / "build" / "Release" / "LveEngine.exe", ENGINE_ROOT / "build" / "NeonML") if p.exists()), ENGINE_ROOT / "build" / "Release" / "NeonML.exe")

CONFIG = dict(
    num_hidden_layers=4, num_experts=8, experts_per_token=4, vocab_size=2048,
    hidden_size=256, intermediate_size=192, swiglu_limit=7.0, head_dim=64,
    num_attention_heads=16, num_key_value_heads=2, sliding_window=16,
    initial_context_length=4096, rope_theta=150000.0, rope_scaling_factor=32.0,
    rope_ntk_alpha=1.0, rope_ntk_beta=32.0,
)
SEQ_LEN = 40


def write_safetensors(path: Path, tensors: dict[str, tuple[str, np.ndarray]]) -> None:
    header, blobs, offset = {}, [], 0
    for name, (dtype, arr) in tensors.items():
        raw = np.ascontiguousarray(arr).tobytes()
        header[name] = {"dtype": dtype, "shape": list(arr.shape), "data_offsets": [offset, offset + len(raw)]}
        blobs.append(raw)
        offset += len(raw)
    text = json.dumps(header).encode()
    text += b" " * ((8 - len(text) % 8) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(text)))
        f.write(text)
        for b in blobs:
            f.write(b)


def make_checkpoint(out_dir: Path, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Writes the checkpoint; returns the dequantized fp32 parameters by reference name."""
    c = CONFIG
    H, I, E, V = c["hidden_size"], c["intermediate_size"], c["num_experts"], c["vocab_size"]
    Hq, Hkv, D = c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
    qkv_dim = D * (Hq + 2 * Hkv)
    files: dict[str, tuple[str, np.ndarray]] = {}
    params: dict[str, np.ndarray] = {}

    def bf16(name, arr):
        bits = F.bf16_bits(arr.astype(np.float32))
        files[name] = ("BF16", bits)
        params[name] = F.bf16_to_f32(bits)

    def f32(name, arr):
        files[name] = ("F32", arr.astype(np.float32))
        params[name] = arr.astype(np.float32)

    def mxfp4(name, w):  # w [E][N][K]
        blocks, scales = zip(*(F.quantize_mxfp4_checkpoint(w[e]) for e in range(w.shape[0])))
        blocks, scales = np.stack(blocks), np.stack(scales)  # [E][N][K/32][16], [E][N][K/32]
        files[name + ".blocks"] = ("U8", blocks)
        files[name + ".scales"] = ("U8", scales)
        params[name] = np.stack([F.dequant_mxfp4_checkpoint(blocks[e], scales[e]) for e in range(w.shape[0])])

    n = rng.normal
    bf16("embedding.weight", n(size=(V, H)))
    for l in range(c["num_hidden_layers"]):
        a, m = f"block.{l}.attn.", f"block.{l}.mlp."
        f32(a + "norm.scale", 1.0 + 0.1 * n(size=H))
        bf16(a + "qkv.weight", n(size=(qkv_dim, H)) / np.sqrt(H))
        bf16(a + "qkv.bias", 0.1 * n(size=qkv_dim))
        bf16(a + "sinks", n(size=Hq))
        bf16(a + "out.weight", n(size=(H, Hq * D)) / np.sqrt(Hq * D))
        bf16(a + "out.bias", 0.1 * n(size=H))
        f32(m + "norm.scale", 1.0 + 0.1 * n(size=H))
        bf16(m + "gate.weight", 2.0 * n(size=(E, H)) / np.sqrt(H))  # well-separated routing
        bf16(m + "gate.bias", 0.1 * n(size=E))
        mxfp4(m + "mlp1_weight", n(size=(E, 2 * I, H)) / np.sqrt(H))
        bf16(m + "mlp1_bias", 0.1 * n(size=(E, 2 * I)))
        mxfp4(m + "mlp2_weight", n(size=(E, H, I)) / np.sqrt(I))
        bf16(m + "mlp2_bias", 0.1 * n(size=(E, H)))
    f32("norm.scale", 1.0 + 0.1 * n(size=H))
    bf16("unembedding.weight", n(size=(V, H)) / np.sqrt(H))

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(CONFIG, indent=2))
    write_safetensors(out_dir / "model.safetensors", files)
    return params


def reference_logits(params: dict[str, np.ndarray], tokens: list[int]) -> np.ndarray:
    torch.manual_seed(0)
    model = ref.Transformer(ref.ModelConfig(**CONFIG), device=torch.device("cpu"))
    model = model.to(torch.float64).eval()
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name not in params:
                raise KeyError(f"reference parameter {name} missing from the checkpoint")
            src = torch.from_numpy(params[name].astype(np.float64))
            if tuple(src.shape) != tuple(p.shape):
                raise ValueError(f"{name}: checkpoint {tuple(src.shape)} vs model {tuple(p.shape)}")
            p.copy_(src)
        return model(torch.tensor(tokens, dtype=torch.long)).numpy()


def _rmsnorm(x, w, eps=1e-5):
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps) * w


def _act_q8(x: np.ndarray) -> np.ndarray:
    """What the GPU feeds an int8 GEMV: per-32-block int8 activations, dequantized."""
    x2 = np.atleast_2d(x).astype(np.float32)
    q, d, _ = F.quantize_act_ref(x2)
    deq = q.reshape(x2.shape[0], -1, 32).astype(np.float64) * d[..., None].astype(np.float64)
    return deq.reshape(x2.shape).reshape(x.shape)


def _yarn(c):
    D = c["head_dim"]
    half = D // 2
    base, factor = c["rope_theta"], c["rope_scaling_factor"]
    freq = base ** (np.arange(0, D, 2, dtype=np.float64) / D)
    conc = 0.1 * np.log(factor) + 1.0
    low = half * np.log(c["initial_context_length"] / (c["rope_ntk_beta"] * 2 * np.pi)) / np.log(base)
    high = half * np.log(c["initial_context_length"] / (c["rope_ntk_alpha"] * 2 * np.pi)) / np.log(base)
    ramp = (np.arange(half) - low) / (high - low)
    mask = 1 - np.clip(ramp, 0, 1)
    inv = (1.0 / (factor * freq)) * (1 - mask) + (1.0 / freq) * mask
    return inv.astype(np.float32).astype(np.float64), conc


def mirror_logits(params, tokens, quant_experts: bool, quant_dense: bool, kv_fp16: bool) -> np.ndarray:
    """NumPy float64 re-statement of the GPU pipeline. With every option off it must equal
    the official model; with them on it emulates the GPU's int8 activations / fp16 KV cache."""
    c = CONFIG
    H, I, E, k = c["hidden_size"], c["intermediate_size"], c["num_experts"], c["experts_per_token"]
    Hq, Hkv, D, W = c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"], c["sliding_window"]
    G = Hq // Hkv
    P = {n: v.astype(np.float64) for n, v in params.items()}
    if quant_dense:  # the loader stores these weights as Q8 (symmetric int8 per 32-block)
        for n in list(P):
            if n.endswith(("attn.qkv.weight", "attn.out.weight")) or n == "unembedding.weight":
                P[n] = F.pack_q8(params[n].astype(np.float32), 32)[2].astype(np.float64)
    inv, conc = _yarn(c)
    T = len(tokens)
    pos = np.arange(T, dtype=np.float32).astype(np.float64)
    ang = pos[:, None] * inv[None, :]
    cos, sin = np.cos(ang) * conc, np.sin(ang) * conc

    def rope(v):  # [T][heads][D], NeoX halves
        a, b = v[..., : D // 2], v[..., D // 2:]
        return np.concatenate([a * cos[:, None] - b * sin[:, None], b * cos[:, None] + a * sin[:, None]], axis=-1)

    dq = _act_q8 if quant_dense else (lambda v: v)
    eq = _act_q8 if quant_experts else (lambda v: v)
    x = P["embedding.weight"][tokens]
    for l in range(c["num_hidden_layers"]):
        a, m = f"block.{l}.attn.", f"block.{l}.mlp."
        t = _rmsnorm(x, P[a + "norm.scale"])
        qkv = dq(t) @ P[a + "qkv.weight"].T + P[a + "qkv.bias"]
        q = rope(qkv[:, : Hq * D].reshape(T, Hq, D))
        kk = rope(qkv[:, Hq * D:(Hq + Hkv) * D].reshape(T, Hkv, D))
        vv = qkv[:, (Hq + Hkv) * D:].reshape(T, Hkv, D)
        if kv_fp16:
            kk = kk.astype(np.float16).astype(np.float64)
            vv = vv.astype(np.float16).astype(np.float64)
        out = np.zeros((T, Hq, D))
        for ti in range(T):
            lo = max(0, ti + 1 - W) if l % 2 == 0 else 0
            for h in range(Hq):
                s = kk[lo:ti + 1, h // G] @ q[ti, h] / np.sqrt(D)
                mx = max(s.max(), P[a + "sinks"][h])
                e = np.exp(s - mx)
                out[ti, h] = (e @ vv[lo:ti + 1, h // G]) / (e.sum() + np.exp(P[a + "sinks"][h] - mx))
        x = x + dq(out.reshape(T, Hq * D)) @ P[a + "out.weight"].T + P[a + "out.bias"]

        t = _rmsnorm(x, P[m + "norm.scale"])
        g = t @ P[m + "gate.weight"].T + P[m + "gate.bias"]
        te = eq(t)
        mlp1, b1 = P[m + "mlp1_weight"], P[m + "mlp1_bias"]
        mlp2, b2 = P[m + "mlp2_weight"], P[m + "mlp2_bias"]
        for ti in range(T):
            ids = np.argsort(-g[ti], kind="stable")[:k]
            wts = np.exp(g[ti, ids] - g[ti, ids].max())
            wts /= wts.sum()
            hs = []
            for e in ids:
                hh = mlp1[e] @ te[ti] + b1[e]
                glu, lin = np.minimum(hh[0::2], 7.0), np.clip(hh[1::2], -7.0, 7.0)
                hs.append(glu / (1 + np.exp(-1.702 * glu)) * (lin + 1))
            hs = eq(np.array(hs))
            for j, e in enumerate(ids):
                x[ti] = x[ti] + wts[j] * (mlp2[e] @ hs[j] + b2[e])
    t = _rmsnorm(x, P["norm.scale"])
    return dq(t) @ P["unembedding.weight"].T


def rounding_sensitivity(params, tokens, quant_dense: bool, base: np.ndarray) -> float:
    """Median rel. L2 change of the emulated logits when every quantizer input is perturbed
    at 1e-6 relative -- the size of fp32 accumulation error in a 256..4096-long dot product.
    Worst of two seeds."""
    global _act_q8
    orig = _act_q8
    worst = 0.0
    for seed in (7, 8):
        prng = np.random.default_rng(seed)
        _act_q8 = lambda v: orig(v * (1 + 1e-6 * prng.standard_normal(np.shape(v))))
        try:
            pert = mirror_logits(params, tokens, True, quant_dense, True)
        finally:
            _act_q8 = orig
        worst = max(worst, float(np.median(np.linalg.norm(pert - base, axis=1) / np.linalg.norm(base, axis=1))))
    return worst


def run_engine(ckpt: Path, tokens: list[int], extra: list[str], out: Path) -> np.ndarray:
    cmd = [str(EXE), "--gpt-oss", str(ckpt), "--tokens", ",".join(map(str, tokens)), "--score",
           "--dump-logits", str(out), "--shaders", str(ENGINE_ROOT / "shaders" / "ML"),
           "--ctx", str(max(64, len(tokens) + 8))] + extra
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=ENGINE_ROOT)
    print(proc.stdout.strip())
    if proc.returncode != 0:
        print(proc.stderr)
        raise RuntimeError(f"engine failed ({proc.returncode})")
    return np.fromfile(out, dtype=np.float32).reshape(len(tokens), CONFIG["vocab_size"])


def log_softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    return x - np.log(np.exp(x).sum(axis=-1, keepdims=True))


def compare(name: str, got: np.ndarray, want: np.ndarray, rel_tol: float, top1_min: float) -> bool:
    rel = np.linalg.norm(got - want, axis=1) / np.linalg.norm(want, axis=1)
    lp_err = np.abs(log_softmax(got.astype(np.float64)) - log_softmax(want)).max(axis=1)
    top1 = float(np.mean(got.argmax(axis=1) == want.argmax(axis=1)))
    ok = bool(np.all(np.isfinite(got))) and float(np.median(rel)) <= rel_tol and top1 >= top1_min
    print(f"{'PASS' if ok else 'FAIL'}  {name:34s} rel L2 median {np.median(rel):.2e} max {rel.max():.2e} | "
          f"max |dlogp| {lp_err.max():.3f} | top-1 agree {top1 * 100:.0f}%")
    if not ok:
        worst = int(rel.argmax())
        print(f"      worst position {worst}: rel {rel[worst]:.3e}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", type=Path, help="write the checkpoint here and keep it")
    ap.add_argument("--seq", type=int, default=SEQ_LEN, help="tokens to score (default 40; try 300 for many KV chunks)")
    args = ap.parse_args()
    if not EXE.exists():
        print(f"missing {EXE}; build the engine in Release first")
        return 2

    rng = np.random.default_rng(1234)
    work = Path(tempfile.mkdtemp(prefix="gptoss_e2e_"))
    ckpt = args.keep or (work / "ckpt")
    try:
        params = make_checkpoint(ckpt, rng)
        tokens = rng.integers(0, CONFIG["vocab_size"], size=args.seq).tolist()
        want = reference_logits(params, tokens)

        ok = True
        # 1. The NumPy mirror (no quantization) restates the official model exactly.
        exact = mirror_logits(params, tokens, False, False, False)
        ok &= compare("mirror == official reference", exact, want, 1e-6, 1.0)  # official RoPE is fp32
        # 2. GPU vs the mirror emulating the GPU's int8 activations + fp16 KV cache:
        #    isolates kernel/wiring bugs from expected quantization error. An untrained random
        #    network amplifies single int8 rounding flips (GPU fp32 vs mirror fp64 decide a few
        #    roundings differently), so the tolerance is calibrated from how far the mirror moves
        #    when its own quantizer inputs are perturbed at fp32-accumulation scale (1e-6).
        emu_bf16 = mirror_logits(params, tokens, True, False, True)
        emu_q8 = mirror_logits(params, tokens, True, True, True)
        sens_bf16 = rounding_sensitivity(params, tokens, False, emu_bf16)
        sens_q8 = rounding_sensitivity(params, tokens, True, emu_q8)
        print(f"rounding sensitivity (median rel L2): bf16 {sens_bf16:.2e}, q8 {sens_q8:.2e}")
        got_a = run_engine(ckpt, tokens, ["--dense", "bf16", "--experts", "device"], work / "a.bin")
        got_b = run_engine(ckpt, tokens, ["--dense", "q8", "--experts", "host"], work / "b.bin")
        ok &= compare("GPU bf16 vs emulated quantization", got_a, emu_bf16, 2 * sens_bf16 + 1e-3, 0.95)
        ok &= compare("GPU q8+host vs emulated quantization", got_b, emu_q8, 2 * sens_q8 + 1e-3, 0.95)
        # 3. Accuracy vs the official model (int8 activation error; informational bounds).
        ok &= compare("GPU bf16 vs official", got_a, want, 3e-2, 0.9)
        ok &= compare("GPU q8+host vs official", got_b, want, 6e-2, 0.85)
        compare("(emulation bf16 vs official)", emu_bf16, want, 1.0, 0.0)
        compare("(emulation q8 vs official)", emu_q8, want, 1.0, 0.0)
        print("all passed" if ok else "FAILED")
        return 0 if ok else 1
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
