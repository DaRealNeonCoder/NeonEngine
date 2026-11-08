"""Build Memorilla training data from a tokenized corpus (the output of prepare_data.py).

Stories are split on <|endoftext|>. Each example takes one story and cuts it at a
point: the text before the cut becomes the documents (chunks of --chunk tokens),
the next --query tokens are the query (in the decoder's context, and the input
to z_q), and the following --answer tokens are the answer (the only tokens with a
loss). So the memory can only help by carrying what the earlier story said
(names, objects, events) into the continuation. Tasks:

    continue      documents = the story before the cut                       (default)
    reconstruct   documents = the whole story; query = <|endoftext|>; answer = its start
                  (the paper's stage-1 reconstruction, without paraphrasing)
    mix           half and half

--distractors N adds the chunks of N other random stories to every example's
documents, shuffled together, so the module has to find the relevant ones.

Writes into --out, per split (train, val):
    {split}_doc_tokens.seq, {split}_query_tokens.seq    encoder inputs
    {split}_examples.bin                                examples (memorilla_formats.py)
    memorilla_data.json                                 settings
With --decoder (and a built NeonML) it also runs the self encoder:
    {split}_docs.nmem, {split}_queries.nmem

Example:
    python train/prepare_memorilla_data.py --data data/tinystories --out data/mem_ts --decoder run
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "shaders" / "ML" / "tests"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import memorilla_formats as MF  # noqa: E402


def load_decoder_bytes(data_dir: Path, meta: dict):
    """Returns a function tokens -> UTF-8 byte count (for bits per byte), or None."""
    try:
        if meta.get("tokenizer_file"):
            import bpe_tokenizer as bpe
            enc = bpe.load_encoding(data_dir / meta["tokenizer_file"])  # a tiktoken Encoding
        else:
            import tiktoken
            enc = tiktoken.get_encoding(meta.get("tokenizer", "gpt2"))
        return lambda toks: len(enc.decode_bytes(list(toks)))
    except Exception as e:  # pragma: no cover - fall back to meta's bytes per token
        print(f"note: no tokenizer for byte counts ({e}); using bytes_per_token from meta.json")
        return None


def stories(path: Path, eot: int, min_len: int) -> list[np.ndarray]:
    toks = np.fromfile(path, dtype=np.uint16)
    cuts = np.flatnonzero(toks == eot)
    out, start = [], 0
    for c in list(cuts) + [len(toks)]:
        s = toks[start:c]
        if len(s) >= min_len:
            out.append(s.astype(np.uint32))
        start = c + 1
    return out


def build_split(name: str, src: list[np.ndarray], args, rng: np.random.Generator, eot: int, nbytes, bpt: float,
                out: Path, max_examples: int) -> int:
    chunk, ql, al = args.chunk, args.query, args.answer
    doc_seqs: list[list[int]] = []
    query_seqs: list[list[int]] = []
    examples: list[MF.Example] = []
    order = rng.permutation(len(src))
    for si in order:
        if len(examples) >= max_examples:
            break
        s = src[si]
        task = args.task if args.task != "mix" else ("continue" if rng.random() < 0.5 else "reconstruct")
        if task == "continue":
            lo = args.min_docs * chunk
            hi = len(s) - ql - al
            if hi <= lo:
                continue
            cut = int(rng.integers(lo, hi + 1))
            body = s[:cut]
            q = s[cut:cut + ql].tolist()
            a = s[cut + ql:cut + ql + al].tolist()
        else:
            body = s
            q = [eot]
            a = s[:al].tolist()
        own = [body[i:i + chunk].tolist() for i in range(0, len(body), chunk)]
        own = [c for c in own if c]
        docs = list(own)
        for _ in range(args.distractors):
            o = src[int(rng.integers(len(src)))]
            docs += [o[i:i + chunk].tolist() for i in range(0, len(o), chunk)]
        if args.distractors:
            rng.shuffle(docs)
        docs = docs[:args.max_docs]
        d0 = len(doc_seqs)
        doc_seqs += docs
        query_seqs.append(q)
        ab = nbytes(a) if nbytes else int(round(bpt * len(a)))
        examples.append(MF.Example(d0, d0 + len(docs), len(query_seqs) - 1, q, a, ab))
    MF.write_sequences(out / f"{name}_doc_tokens.seq", doc_seqs)
    MF.write_sequences(out / f"{name}_query_tokens.seq", query_seqs)
    MF.write_examples(out / f"{name}_examples.bin", examples)
    print(f"{name}: {len(examples)} examples, {len(doc_seqs)} documents "
          f"({len(doc_seqs) / max(1, len(examples)):.1f} per example)")
    return len(examples)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True, help="prepare_data.py output (train.bin, val.bin, meta.json)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--task", choices=["continue", "reconstruct", "mix"], default="continue")
    ap.add_argument("--chunk", type=int, default=32, help="tokens per document")
    ap.add_argument("--query", type=int, default=16, help="query tokens")
    ap.add_argument("--answer", type=int, default=48, help="answer tokens")
    ap.add_argument("--min-docs", type=int, default=2, help="continue: at least this many documents before the cut")
    ap.add_argument("--max-docs", type=int, default=64)
    ap.add_argument("--distractors", type=int, default=0, help="other stories mixed into every example's documents")
    ap.add_argument("--train-examples", type=int, default=200000)
    ap.add_argument("--val-examples", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--decoder", help="decoder checkpoint (dir or .nckpt): also run the self encoder")
    ap.add_argument("--exe", type=Path, default=None, help="NeonML executable (default: build/Release/NeonML.exe or build/NeonML)")
    ap.add_argument("--encode-seq", type=int, default=64, help="encoder sequence length (multiple of 64, >= --chunk)")
    ap.add_argument("--layer", type=int, default=-1, help="self encoder layer (-1 = final RMSNorm output)")
    ap.add_argument("--no-mxfp4", action="store_true",
                    help="encode with the fp32 expert weights (decoders trained without QAT)")
    args = ap.parse_args()

    meta = json.loads((args.data / "meta.json").read_text())
    eot = int(meta["eot"])
    nbytes = load_decoder_bytes(args.data, meta)
    bpt = float(meta.get("bytes_per_token", 4.0))
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    min_len = args.min_docs * args.chunk + args.query + args.answer if args.task != "reconstruct" else args.answer
    counts = {}
    for split, n in (("train", args.train_examples), ("val", args.val_examples)):
        src = stories(args.data / f"{split}.bin", eot, min(min_len, args.answer + 1))
        counts[split] = build_split(split, src, args, rng, eot, nbytes, bpt, args.out, n)
    (args.out / "memorilla_data.json").write_text(json.dumps({
        "source": str(args.data), "task": args.task, "chunk": args.chunk, "query": args.query,
        "answer": args.answer, "distractors": args.distractors, "examples": counts,
        "tokenizer": meta.get("tokenizer"), "eot": eot}, indent=2))

    if args.decoder:
        exe = args.exe or next((p for p in (ROOT / "build" / "Release" / "NeonML.exe", ROOT / "build" / "NeonML")
                                if p.exists()), None)
        if exe is None:
            raise SystemExit("build NeonML first (or pass --exe)")
        seq = max(args.encode_seq, ((max(args.chunk, args.query) + 63) // 64) * 64)
        for split in ("train", "val"):
            for kind in ("doc", "query"):
                store = "docs" if kind == "doc" else "queries"
                cmd = [str(exe), "--memorilla", "encode", "--decoder", str(args.decoder),
                       "--sequences", str(args.out / f"{split}_{kind}_tokens.seq"),
                       "--out", str(args.out / f"{split}_{store}.nmem"), "--seq", str(seq),
                       "--layer", str(args.layer), "--shaders", str(ROOT / "shaders" / "ML")]
                if args.no_mxfp4:
                    cmd.append("--no-mxfp4")
                print(" ".join(cmd))
                subprocess.run(cmd, check=True, cwd=ROOT)
    else:
        print("next: encode each split with  NeonML --memorilla encode --decoder <ckpt> "
              "--sequences <split>_{doc,query}_tokens.seq --out <split>_{docs,queries}.nmem")
    return 0


if __name__ == "__main__":
    sys.exit(main())
