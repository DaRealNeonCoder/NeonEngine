"""Byte-level BPE tokenizer training for the NeonEngine gpt-oss trainer.

A model trained from scratch does not need GPT-2's 50k-token vocabulary: on a
narrow corpus (TinyStories) a vocabulary of 4k-16k tokens trained on that corpus
encodes the text in about as many tokens, while the LM head, the logits and the
embedding tables shrink 3-12x. For small models the LM head is the largest
single cost of a training step (about 40% for the t4 preset with GPT-2 tokens).

Algorithm: the GPT-2 pre-tokenization regex splits text into words; BPE merges
are learned on word counts (the most frequent adjacent pair is merged, ties go
to the smallest token ids). The result is written in tiktoken's format (one
"base64(bytes) rank" line per token), so encoding runs in tiktoken's Rust code:

    enc = load_encoding("tokenizer.tiktoken")
    ids = enc.encode_ordinary("Once upon a time")

Token ids: 0-255 are the raw bytes, then the merges in order, then the special
tokens (<|endoftext|> is the last id). Every byte string is encodable and
decoding is exact.

Command line (normally driven by prepare_data.py --vocab-size):
    python bpe_tokenizer.py train --vocab-size 8192 --out tokenizer.tiktoken a.txt b.txt
    python bpe_tokenizer.py encode tokenizer.tiktoken "Once upon a time"
    python bpe_tokenizer.py decode tokenizer.tiktoken 1,2,3
"""

from __future__ import annotations

import base64
import heapq
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

# tiktoken's GPT-2 pattern (contractions, letters, numbers, punctuation, whitespace).
GPT2_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}++| ?\p{N}++| ?[^\s\p{L}\p{N}]++|\s++$|\s+(?!\S)|\s"""
EOT = "<|endoftext|>"


def count_words(texts: Iterable[str], pat: str = GPT2_PAT, max_bytes: int | None = None) -> tuple[Counter, int]:
    """Counts pre-tokenized words (as bytes). Returns (counts, bytes read)."""
    import regex

    rx = regex.compile(pat)
    counts: Counter = Counter()
    total = 0
    for text in texts:
        counts.update(m.encode("utf-8") for m in rx.findall(text))
        total += len(text.encode("utf-8"))
        if max_bytes is not None and total >= max_bytes:
            break
    return counts, total


def train_bpe(word_counts: dict[bytes, int], vocab_size: int, specials: list[str] = (EOT,),
              verbose: bool = True) -> tuple[dict[bytes, int], dict[str, int]]:
    """Learns merges until there are vocab_size - len(specials) byte tokens.

    Returns (mergeable_ranks, special_tokens) for tiktoken.Encoding.
    """
    n_tokens = vocab_size - len(specials)
    if n_tokens < 256:
        raise ValueError("vocab_size must leave room for the 256 byte tokens")
    vocab: list[bytes] = [bytes([i]) for i in range(256)]
    index: dict[bytes, int] = {b: i for i, b in enumerate(vocab)}
    words = [list(w) for w in word_counts]
    counts = list(word_counts.values())
    pairs: dict[tuple[int, int], int] = defaultdict(int)
    where: dict[tuple[int, int], set[int]] = defaultdict(set)
    for wi, w in enumerate(words):
        c = counts[wi]
        for a, b in zip(w, w[1:]):
            pairs[(a, b)] += c
            where[(a, b)].add(wi)
    heap = [(-c, a, b) for (a, b), c in pairs.items()]
    heapq.heapify(heap)
    t0 = time.time()
    while len(vocab) < n_tokens and heap:
        negc, a, b = heapq.heappop(heap)
        if pairs.get((a, b), 0) != -negc or negc == 0:
            continue  # stale heap entry
        merged = vocab[a] + vocab[b]
        new = index.get(merged)
        if new is None:  # the same bytes can come from two different pairs
            new = len(vocab)
            vocab.append(merged)
            index[merged] = new
        touched: dict[tuple[int, int], int] = {}
        for wi in list(where.pop((a, b), ())):
            w = words[wi]
            if len(w) < 2:
                continue
            c = counts[wi]
            out = []
            i = 0
            changed = False
            while i < len(w):
                if i + 1 < len(w) and w[i] == a and w[i + 1] == b:
                    out.append(new)
                    i += 2
                    changed = True
                else:
                    out.append(w[i])
                    i += 1
            if not changed:
                continue
            for p in zip(w, w[1:]):
                pairs[p] -= c
                touched[p] = pairs[p]
            for p in zip(out, out[1:]):
                pairs[p] += c
                touched[p] = pairs[p]
                where[p].add(wi)
            words[wi] = out
        pairs.pop((a, b), None)
        for p, c in touched.items():
            if p == (a, b):
                continue
            if c > 0:
                heapq.heappush(heap, (-c, p[0], p[1]))
            else:
                pairs.pop(p, None)
        if verbose and len(vocab) % 1000 == 0:
            print(f"\r  bpe: {len(vocab)} / {n_tokens} tokens ({time.time() - t0:.0f} s)", end="", flush=True)
    if verbose:
        print(f"\r  bpe: {len(vocab)} tokens in {time.time() - t0:.0f} s" + " " * 20)
    ranks = {tok: i for i, tok in enumerate(vocab)}
    special = {s: len(vocab) + i for i, s in enumerate(specials)}
    return ranks, special


def save_tiktoken(path: str | Path, ranks: dict[bytes, int], special: dict[str, int]) -> None:
    """tiktoken format; special tokens follow as '#special <name> <id>' comment lines."""
    with open(path, "w", encoding="ascii") as f:
        for tok, r in sorted(ranks.items(), key=lambda kv: kv[1]):
            f.write(f"{base64.b64encode(tok).decode('ascii')} {r}\n")
        for name, i in special.items():
            f.write(f"#special {base64.b64encode(name.encode()).decode('ascii')} {i}\n")


def load_tiktoken(path: str | Path) -> tuple[dict[bytes, int], dict[str, int]]:
    ranks: dict[bytes, int] = {}
    special: dict[str, int] = {}
    for line in Path(path).read_text(encoding="ascii").splitlines():
        if not line.strip():
            continue
        if line.startswith("#special "):
            _, name, i = line.split()
            special[base64.b64decode(name).decode()] = int(i)
            continue
        tok, r = line.split()
        ranks[base64.b64decode(tok)] = int(r)
    if not special:
        special = {EOT: len(ranks)}
    return ranks, special


def load_encoding(path: str | Path):
    import tiktoken

    ranks, special = load_tiktoken(path)
    return tiktoken.Encoding(name=Path(path).stem, pat_str=GPT2_PAT, mergeable_ranks=ranks, special_tokens=special)


def main(argv: list[str]) -> int:
    if len(argv) >= 2 and argv[1] == "train":
        import argparse

        ap = argparse.ArgumentParser()
        ap.add_argument("cmd")
        ap.add_argument("--vocab-size", type=int, required=True)
        ap.add_argument("--out", required=True)
        ap.add_argument("files", nargs="+")
        a = ap.parse_args(argv[1:])
        texts = (Path(p).read_text(encoding="utf-8", errors="replace") for p in a.files)
        counts, nbytes = count_words(texts)
        print(f"{len(counts)} distinct words in {nbytes / 1e6:.1f} MB")
        ranks, special = train_bpe(counts, a.vocab_size)
        save_tiktoken(a.out, ranks, special)
        return 0
    if len(argv) == 4 and argv[1] in ("encode", "decode"):
        enc = load_encoding(argv[2])
        if argv[1] == "encode":
            print(",".join(map(str, enc.encode(argv[3], allowed_special="all"))))
        else:
            print(enc.decode([int(t) for t in argv[3].replace(",", " ").split()]))
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
