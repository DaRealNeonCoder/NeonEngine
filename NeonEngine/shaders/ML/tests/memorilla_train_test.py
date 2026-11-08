"""Gradient check of Memorilla training (src/memorilla.cpp, src/memorilla_train.cpp).

Builds a tiny random dataset (document / query stores, examples with different
document counts so padding and masking are exercised), runs
``NeonML --memorilla gradcheck`` (one fp32 micro-batch through a tiny random
frozen gpt-oss decoder) and recomputes the same loss in float64 PyTorch: the
memory module written out from the paper's equations, the decoder from the
official reference model (gpt_oss_reference.py) fed with input embeddings.
Every module gradient and the loss must match.

Usage (from shaders/ML/tests, after building NeonML in Release):
    python memorilla_train_test.py [--batch 3] [--seq 64] [--precision fp16]
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

import gpt_oss_reference as ref
import memorilla_formats as MF

TESTS_DIR = Path(__file__).resolve().parent
ENGINE_ROOT = TESTS_DIR.parents[2]
EXE_CANDIDATES = [ENGINE_ROOT / "build" / "Release" / "NeonML.exe", ENGINE_ROOT / "build" / "NeonML"]

DECODER = dict(
    num_hidden_layers=2, num_experts=4, experts_per_token=2, vocab_size=512,
    hidden_size=128, intermediate_size=64, swiglu_limit=7.0, head_dim=64,
    num_attention_heads=4, num_key_value_heads=2, sliding_window=16,
    initial_context_length=4096, rope_theta=150000.0, rope_scaling_factor=32.0,
    rope_ntk_alpha=1.0, rope_ntk_beta=32.0,
)
MEMORY = dict(slots=4, heads=4, refine_layers=2, ffn_mult=4)
ENC_DIM = 64


def make_data(work: Path, rng: np.random.Generator, n_examples: int) -> None:
    doc_counts = [3, 7, 1, 5, 2, 6][:n_examples] + [4] * max(0, n_examples - 6)
    docs = rng.standard_normal((sum(doc_counts), ENC_DIM)).astype(np.float32)
    queries = rng.standard_normal((n_examples, ENC_DIM)).astype(np.float32)
    MF.write_store(work / "train_docs.nmem", docs)
    MF.write_store(work / "train_queries.nmem", queries)
    examples, d0 = [], 0
    for i, n in enumerate(doc_counts):
        q = rng.integers(0, DECODER["vocab_size"], size=int(rng.integers(3, 9))).tolist()
        a = rng.integers(0, DECODER["vocab_size"], size=int(rng.integers(5, 20))).tolist()
        examples.append(MF.Example(d0, d0 + n, i, q, a, len(a) * 4))
        d0 += n
    MF.write_examples(work / "train_examples.bin", examples)


def run_gradcheck(work: Path, batch: int, seq: int, precision: str) -> None:
    exe = next((p for p in EXE_CANDIDATES if p.exists()), None)
    if exe is None:
        raise FileNotFoundError("build NeonML first (cmake --build build --config Release --target NeonML)")
    (work / "decoder.json").write_text(json.dumps(DECODER))
    (work / "mem.json").write_text(json.dumps(MEMORY))
    cmd = [str(exe), "--memorilla", "gradcheck", "--config", str(work / "decoder.json"), "--data", str(work),
           "--mem-config", str(work / "mem.json"), "--out", str(work / "out"), "--batch", str(batch),
           "--seq", str(seq), "--precision", precision, "--max-docs", "64",
           "--shaders", str(ENGINE_ROOT / "shaders" / "ML")]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=ENGINE_ROOT)
    print(proc.stdout.strip())
    if proc.returncode != 0:
        print(proc.stderr)
        raise RuntimeError(f"NeonML failed ({proc.returncode})")


def tensors(flat: np.ndarray, layout_path: Path) -> dict[str, np.ndarray]:
    out = {}
    for t in json.loads(layout_path.read_text())["tensors"]:
        n = int(np.prod(t["shape"]))
        out[t["name"]] = flat[t["offset"]:t["offset"] + n].reshape(t["shape"])
    return out


def layer_norm(x, p, name, eps=1e-5):
    return torch.nn.functional.layer_norm(x, x.shape[-1:], p[name + ".scale"], p[name + ".bias"], eps)


def lin(x, p, name):
    return x @ p[name + ".weight"].T + p[name + ".bias"]


def memory_module(p: dict, docs, counts, query, cfg: dict):
    """Memorilla (Section 2.2, Eq. 3-5) with the paper's own definitions."""
    B, Nd, h = docs.shape
    K, nh, L = cfg["slots"], cfg["heads"], cfg["refine_layers"]
    dh = h // nh
    # Stage 2: cross-attention of the slots to the single query vector (weight 1).
    u = lin(lin(query, p, "cond.v"), p, "cond.o")
    R = layer_norm(p["slots"][None] + u[:, None], p, "cond.norm")
    Z = layer_norm(docs, p, "docs.norm")
    mask = torch.arange(Nd)[None, :] < torch.tensor(counts)[:, None]  # [B, Nd]
    for l in range(L):
        b = f"block.{l}."
        n1 = layer_norm(R, p, b + "norm1")
        q = lin(n1, p, b + "attn.q").view(B, K, nh, dh).transpose(1, 2)
        k = lin(Z, p, b + "attn.k").view(B, Nd, nh, dh).transpose(1, 2)
        v = lin(Z, p, b + "attn.v").view(B, Nd, nh, dh).transpose(1, 2)
        s = (q @ k.transpose(-1, -2)) / math.sqrt(dh)
        s = s.masked_fill(~mask[:, None, None, :], float("-inf"))
        o = (torch.softmax(s, dim=-1) @ v).transpose(1, 2).reshape(B, K, h)
        R = R + lin(o, p, b + "attn.o")
        n2 = layer_norm(R, p, b + "norm2")
        R = R + lin(torch.nn.functional.gelu(lin(n2, p, b + "ffn.up")), p, b + "ffn.down")
    return lin(layer_norm(R, p, "out.norm"), p, "proj")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=3)
    ap.add_argument("--seq", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--precision", choices=["fp32", "fp16"], default="fp32")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        make_data(work, rng, max(args.batch, 3))
        run_gradcheck(work, args.batch, args.seq, args.precision)
        out = work / "out"
        bj = json.loads((out / "batch.json").read_text())
        B, T, Nd, K = bj["batch"], bj["seq"], bj["docs"], bj["slots"]
        mem_cfg = json.loads((out / "memorilla.json").read_text())
        dec_p = tensors(np.fromfile(out / "params.bin", dtype=np.float32), out / "layout.json")
        mem_np = tensors(np.fromfile(out / "mem_params.bin", dtype=np.float32), out / "mem_layout.json")
        mem_g = tensors(np.fromfile(out / "mem_grads.bin", dtype=np.float32), out / "mem_layout.json")
        docs = torch.from_numpy(np.fromfile(out / "docs.bin", dtype=np.float32).reshape(B, Nd, ENC_DIM).astype(np.float64))
        query = torch.from_numpy(np.fromfile(out / "query.bin", dtype=np.float32).reshape(B, ENC_DIM).astype(np.float64))
        examples = MF.read_examples(work / "train_examples.bin")

        torch.manual_seed(0)
        model = ref.Transformer(ref.ModelConfig(**DECODER), device=torch.device("cpu")).to(torch.float64)
        named = dict(model.named_parameters())
        with torch.no_grad():
            for name, prm in named.items():
                prm.copy_(torch.from_numpy(dec_p[name].astype(np.float64)))
                prm.requires_grad_(False)
        bad_init = [n for n, v in mem_np.items() if n.endswith(".scale") and not np.all(v == 1.0)]
        bad_init += [n for n, v in mem_np.items() if n.endswith(".weight") and not np.any(v)]
        if bad_init:
            print(f"FAIL: bad initialization of {bad_init}")
            return 1
        mem = {n: torch.tensor(v.astype(np.float64), requires_grad=True) for n, v in mem_np.items()}

        M = memory_module(mem, docs, bj["counts"], query, mem_cfg)
        total, count = 0.0, 0
        for b in range(B):
            inputs, targets = MF.sequence_layout(examples[bj["examples"][b]], K, T)
            x = model.embedding(torch.tensor(inputs, dtype=torch.long))
            x = torch.cat([M[b], x[K:]], dim=0)
            for block in model.block:
                x = block(x)
            logits = model.unembedding(model.norm(x))
            rows = [r for r, t in enumerate(targets) if t >= 0]
            total = total + torch.nn.functional.cross_entropy(
                logits[rows], torch.tensor([targets[r] for r in rows]), reduction="sum")
            count += len(rows)
        loss = total / count
        loss.backward()

        print(f"loss: GPU {bj['loss']:.7f}  reference {float(loss):.7f}")
        tol = 3e-3 if args.precision == "fp16" else 2e-4
        ok = abs(bj["loss"] - float(loss)) < tol * max(1.0, abs(float(loss)))
        worst = 0.0
        for name in mem_np:
            g_ref = mem[name].grad.numpy()
            g_gpu = mem_g[name].astype(np.float64)
            denom = max(np.linalg.norm(g_ref), 1e-12)
            err = np.linalg.norm(g_gpu - g_ref) / denom
            if name.endswith("attn.k.bias"):
                # Exactly zero in exact arithmetic (softmax ignores a constant shift): roundoff only.
                err = np.linalg.norm(g_gpu) * 1e2
            elif denom <= 1e-9:
                err = float("inf")  # every other tensor must receive a gradient
            worst = max(worst, err)
            flag = "" if err < tol else "   <-- FAIL"
            print(f"  {name:28s} |g| {denom:10.3e}  rel err {err:.2e}{flag}")
            ok = ok and err < tol
        print(f"worst relative gradient error {worst:.2e} (tolerance {tol:g})")
        print("PASS" if ok else "FAIL")
        return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
