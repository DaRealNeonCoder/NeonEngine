"""Tokenize a text corpus into the binary format the NeonEngine gpt-oss trainer reads.

Output (in --out):
    train.bin, val.bin   uint16 little-endian token ids, documents separated by <|endoftext|>
    meta.json            {"tokenizer": ..., "vocab_size": ..., "eot": ..., "val_bytes": ..., ...}
    tokenizer.tiktoken   (with --vocab-size / --tokenizer) the BPE vocabulary; the trainer
                         copies it next to its checkpoints and into the export

Tokenizers:
    --vocab-size 8192          train a byte-level BPE tokenizer on (a sample of) the training
                               text (bpe_tokenizer.py). Recommended for models trained from
                               scratch: a corpus-specific 4k-16k vocabulary encodes the text in
                               about as many tokens as GPT-2's 50k one, and makes the LM head,
                               logits and embeddings several times cheaper.
    --tokenizer FILE           reuse a tokenizer.tiktoken from an earlier run
    --encoding gpt2            (default without the two above) a tiktoken encoding

Sources:
    python prepare_data.py --out data/tinystories --vocab-size 8192   # TinyStories from Hugging Face (default)
    python prepare_data.py --out data/mine --text a.txt b.txt          # your own files (one document per file,
                                                                       #   or per blank-line-separated block with --split-blank)
    python prepare_data.py --out data/x --hf-dataset NAME --hf-field text

meta.json records the UTF-8 bytes behind each split, so the trainer can report
validation loss in bits per byte, which is comparable across tokenizers.

Needs: pip install tiktoken numpy (and `datasets` for Hugging Face sources).
"""

from __future__ import annotations

import argparse
import itertools
import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bpe_tokenizer as bpe  # noqa: E402


def batched(it, n):
    batch = []
    for x in it:
        batch.append(x)
        if len(batch) == n:
            yield batch
            batch = []
    if batch:
        yield batch


def write_split(path: Path, docs, enc, eot: int, max_tokens: int | None) -> tuple[int, int]:
    """Returns (tokens, UTF-8 bytes of the text those tokens cover)."""
    total = 0
    nbytes = 0
    with open(path, "wb") as f:
        for batch in batched(docs, 2048):
            ids = enc.encode_ordinary_batch(batch, num_threads=8)
            flat = []
            for d in ids:
                flat.extend(d)
                flat.append(eot)
            arr = np.asarray(flat, dtype=np.uint16)
            bb = sum(len(d.encode("utf-8")) for d in batch)
            if max_tokens is not None and total + len(arr) > max_tokens:
                keep = max_tokens - total
                bb = int(bb * keep / len(arr))
                arr = arr[:keep]
            arr.tofile(f)
            total += len(arr)
            nbytes += bb
            print(f"\r  {path.name}: {total / 1e6:8.1f} M tokens", end="", flush=True)
            if max_tokens is not None and total >= max_tokens:
                break
    print()
    return total, nbytes


def build(out: Path, train_docs, val_docs, args, source: str) -> dict:
    """Tokenizes both splits into `out` and returns the meta dict."""
    out.mkdir(parents=True, exist_ok=True)
    tok_file = None
    if args.vocab_size:
        # Train on the first --tokenizer-sample-mb of training text, then encode everything.
        sample = []
        nbytes = 0
        train_docs = iter(train_docs)
        for d in train_docs:
            sample.append(d)
            nbytes += len(d.encode("utf-8"))
            if nbytes >= args.tokenizer_sample_mb * 1e6:
                break
        print(f"training a {args.vocab_size}-token BPE tokenizer on {nbytes / 1e6:.1f} MB of text")
        counts, _ = bpe.count_words(sample)
        ranks, special = bpe.train_bpe(counts, args.vocab_size)
        tok_file = out / "tokenizer.tiktoken"
        bpe.save_tiktoken(tok_file, ranks, special)
        train_docs = itertools.chain(sample, train_docs)
    elif args.tokenizer:
        tok_file = out / "tokenizer.tiktoken"
        if Path(args.tokenizer).resolve() != tok_file.resolve():
            shutil.copyfile(args.tokenizer, tok_file)
    if tok_file is not None:
        enc = bpe.load_encoding(tok_file)
        name = "neon_bpe"
    else:
        import tiktoken

        enc = tiktoken.get_encoding(args.encoding)
        name = args.encoding
    eot = enc.eot_token
    if enc.n_vocab > 65535:
        raise SystemExit(f"tokenizer has {enc.n_vocab} tokens; uint16 storage needs <= 65535")
    print(f"tokenizing {source} with {name} ({enc.n_vocab} tokens)")
    n_train, b_train = write_split(out / "train.bin", train_docs, enc, eot, args.max_train_tokens)
    n_val, b_val = write_split(out / "val.bin", val_docs, enc, eot, args.max_val_tokens)
    meta = {"tokenizer": name, "vocab_size": enc.n_vocab, "eot": eot,
            "train_tokens": n_train, "val_tokens": n_val, "train_bytes": b_train, "val_bytes": b_val,
            "bytes_per_token": round(b_val / max(1, n_val), 4), "source": source}
    if tok_file is not None:
        meta["tokenizer_file"] = tok_file.name
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    return meta


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--encoding", default="gpt2", help="tiktoken encoding (vocab must fit uint16)")
    ap.add_argument("--vocab-size", type=int, default=0, help="train a BPE tokenizer with this many tokens")
    ap.add_argument("--tokenizer", default=None, help="existing tokenizer.tiktoken to reuse")
    ap.add_argument("--tokenizer-sample-mb", type=float, default=200.0, help="training text for the BPE tokenizer")
    ap.add_argument("--hf-dataset", default="roneneldan/TinyStories")
    ap.add_argument("--hf-field", default="text")
    ap.add_argument("--hf-val-split", default="validation")
    ap.add_argument("--text", nargs="*", help="local text files instead of a Hugging Face dataset")
    ap.add_argument("--split-blank", action="store_true", help="treat blank-line-separated blocks as documents")
    ap.add_argument("--val-fraction", type=float, default=0.005, help="for local text: share of documents for val")
    ap.add_argument("--streaming", action="store_true", help="stream the Hugging Face dataset (no full download)")
    ap.add_argument("--max-train-tokens", type=int, default=None)
    ap.add_argument("--max-val-tokens", type=int, default=2_000_000)
    args = ap.parse_args()
    if args.vocab_size and args.tokenizer:
        ap.error("use either --vocab-size or --tokenizer")

    if args.text:
        docs = []
        for p in args.text:
            text = Path(p).read_text(encoding="utf-8", errors="replace")
            docs.extend([d for d in text.split("\n\n") if d.strip()] if args.split_blank else [text])
        rng = np.random.default_rng(0)
        rng.shuffle(docs)
        n_val = max(1, int(len(docs) * args.val_fraction))
        train_docs, val_docs = docs[n_val:], docs[:n_val]
        source = ",".join(args.text)
    else:
        from datasets import load_dataset

        ds_train = load_dataset(args.hf_dataset, split="train", streaming=args.streaming)
        ds_val = load_dataset(args.hf_dataset, split=args.hf_val_split, streaming=args.streaming)
        train_docs = (r[args.hf_field] for r in ds_train)
        val_docs = (r[args.hf_field] for r in ds_val)
        source = args.hf_dataset

    build(Path(args.out), train_docs, val_docs, args, source)
    return 0


if __name__ == "__main__":
    sys.exit(main())
