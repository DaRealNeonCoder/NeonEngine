"""Memorilla in the gpt-oss inference host (src/gpt_oss_app.cpp, GptOssModel::forwardEmbedding).

On the tiny random gpt-oss checkpoint of gpt_oss_e2e_test.py:
1. Off by default: the --mem-* options are refused without --memorilla, and a run
   without them is the normal path.
2. Soft tokens are exact: feeding the bf16 embedding rows of the first prompt
   tokens through --soft-prefix gives bit-identical logits to feeding the tokens.
3. --memorilla: the query embedding z_q (self encoder) matches the official model's
   mean final hidden state; the K memory tokens match a float64 PyTorch module
   computed from the same store and z_q; the logits equal a --soft-prefix run with
   those memory tokens; --mem-k uses only the first slots.

Usage (from shaders/ML/tests, after building NeonML in Release):
    python memorilla_infer_test.py
"""

from __future__ import annotations

import json
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

import gpt_oss_e2e_test as E
import gpt_oss_reference as ref
import memorilla_formats as MF
import memorilla_train_test as MT

H = E.CONFIG["hidden_size"]
MEM = dict(slots=4, heads=4, refine_layers=2, ffn_mult=4, encoder_dim=H, llm_dim=H)


def read_safetensors(path: Path) -> dict[str, np.ndarray]:
    data = path.read_bytes()
    (n,) = struct.unpack_from("<Q", data, 0)
    header = json.loads(data[8:8 + n])
    out = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        a, b = info["data_offsets"]
        raw = data[8 + n + a:8 + n + b]
        if info["dtype"] == "F32":
            arr = np.frombuffer(raw, dtype=np.float32)
        elif info["dtype"] == "BF16":
            arr = (np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)
        else:
            continue
        out[name] = arr.reshape(info["shape"]).copy()
    return out


def engine(args: list[str], check: bool = True) -> subprocess.CompletedProcess:
    cmd = [str(E.EXE)] + args + ["--shaders", str(E.ENGINE_ROOT / "shaders" / "ML")]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=E.ENGINE_ROOT)
    if check and proc.returncode != 0:
        print(proc.stdout, proc.stderr)
        raise RuntimeError(f"engine failed: {' '.join(args[:3])}")
    return proc


def score(ckpt: Path, tokens: list[int], out: Path, extra: list[str]) -> np.ndarray:
    engine(["--gpt-oss", str(ckpt), "--tokens", ",".join(map(str, tokens)), "--score", "--dump-logits", str(out),
            "--ctx", "128"] + extra)
    return np.fromfile(out, dtype=np.float32).reshape(len(tokens), E.CONFIG["vocab_size"])


def reference_hidden_mean(params: dict, tokens: list[int]) -> np.ndarray:
    torch.manual_seed(0)
    model = ref.Transformer(ref.ModelConfig(**E.CONFIG), device=torch.device("cpu")).to(torch.float64).eval()
    with torch.no_grad():
        for name, p in model.named_parameters():
            p.copy_(torch.from_numpy(params[name].astype(np.float64)))
        x = model.embedding(torch.tensor(tokens, dtype=torch.long))
        for block in model.block:
            x = block(x)
        return model.norm(x).mean(dim=0).numpy()


def report(name: str, ok: bool, detail: str) -> bool:
    print(f"{'PASS' if ok else 'FAIL'}  {name:46s} {detail}")
    return ok


def main() -> int:
    if not E.EXE.exists():
        print(f"missing {E.EXE}; build NeonML in Release first")
        return 2
    rng = np.random.default_rng(7)
    work = Path(tempfile.mkdtemp(prefix="memorilla_infer_"))
    try:
        ckpt = work / "ckpt"
        params = E.make_checkpoint(ckpt, rng)
        tokens = rng.integers(0, E.CONFIG["vocab_size"], size=24).tolist()
        ok = True

        # 1. Off by default.
        base = score(ckpt, tokens, work / "base.bin", [])
        proc = engine(["--gpt-oss", str(ckpt), "--tokens", "1,2", "--score", "--mem-store", "x.nmem"], check=False)
        ok &= report("--mem-* refused without --memorilla", proc.returncode != 0 and "--memorilla" in proc.stderr,
                     proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else "")

        # 2. Soft tokens equal token embeddings exactly.
        emb = read_safetensors(next(ckpt.glob("*.safetensors")))["embedding.weight"]
        n0 = 5
        emb[tokens[:n0]].astype(np.float32).tofile(work / "soft.bin")
        soft = score(ckpt, tokens[n0:], work / "soft_logits.bin", ["--soft-prefix", str(work / "soft.bin")])
        same = np.array_equal(soft, base[n0:])
        ok &= report("soft prefix == token path (bit-exact)", same,
                     f"max |diff| {np.abs(soft - base[n0:]).max():.3e}")

        # 3. Memorilla.
        docs = rng.standard_normal((11, H)).astype(np.float32)
        MF.write_store(work / "docs.nmem", docs)
        (work / "mem.json").write_text(json.dumps(MEM))
        engine(["--memorilla", "init", "--mem-config", str(work / "mem.json"), "--out", str(work / "module"), "--seed", "5"])
        mem_logits = score(ckpt, tokens, work / "mem_logits.bin",
                           ["--memorilla", str(work / "module"), "--mem-store", str(work / "docs.nmem"),
                            "--mem-dump", str(work / "dump.bin")])
        dump = np.fromfile(work / "dump.bin", dtype=np.float32)
        zq, M = dump[:H], dump[H:].reshape(MEM["slots"], H)

        want_zq = reference_hidden_mean(params, tokens)
        rel = np.linalg.norm(zq - want_zq) / np.linalg.norm(want_zq)
        ok &= report("z_q vs official mean hidden state", rel < 3e-2, f"rel L2 {rel:.2e} (int8 activations)")

        mp = {k: torch.from_numpy(v.astype(np.float64)) for k, v in read_safetensors(work / "module" / "memorilla.safetensors").items()}
        want_M = MT.memory_module(mp, torch.from_numpy(docs.astype(np.float64))[None], [len(docs)],
                                  torch.from_numpy(zq.astype(np.float64))[None], MEM)[0].numpy()
        rel = np.linalg.norm(M - want_M) / np.linalg.norm(want_M)
        ok &= report("memory tokens vs float64 module", rel < 1e-5, f"rel L2 {rel:.2e}")

        M.astype(np.float32).tofile(work / "m.bin")
        via_soft = score(ckpt, tokens, work / "via_soft.bin", ["--soft-prefix", str(work / "m.bin")])
        ok &= report("--memorilla == --soft-prefix with its tokens", np.array_equal(via_soft, mem_logits),
                     f"max |diff| {np.abs(via_soft - mem_logits).max():.3e}")
        ok &= report("memory tokens change the logits", not np.array_equal(mem_logits, base),
                     f"mean |diff| {np.abs(mem_logits - base).mean():.3e}")

        k2 = score(ckpt, tokens, work / "k2.bin", ["--memorilla", str(work / "module"), "--mem-store",
                                                   str(work / "docs.nmem"), "--mem-k", "2"])
        M[:2].astype(np.float32).tofile(work / "m2.bin")
        k2_soft = score(ckpt, tokens, work / "k2_soft.bin", ["--soft-prefix", str(work / "m2.bin")])
        ok &= report("--mem-k 2 == first two memory tokens", np.array_equal(k2, k2_soft),
                     f"max |diff| {np.abs(k2 - k2_soft).max():.3e}")

        print("all passed" if ok else "FAILED")
        return 0 if ok else 1
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
