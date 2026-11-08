"""File formats shared by the Memorilla tools (src/memorilla.cpp, src/memorilla_train.cpp).

* ``.nmem`` document store: b"NEONMEM1", u32 version (1), u32 json length, json
  header (encoder, encoder_layer, pooling, dim, dtype "f32", decoder), u64 count,
  count x dim float32 vectors, per document (u64 id, u32 tokens, u32 flags), b"NEONMEND".
* ``.seq`` token sequences (encoder input): b"NEONSEQ1", u64 count, then per
  sequence u32 length + that many u32 token ids.
* ``examples.bin``: b"NEONMEX1", u64 count, then per example six u32
  (doc_begin, doc_end, query, nq, na, answer_bytes) followed by nq query and
  na answer token ids (u32).
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


def write_store(path: Path, vectors: np.ndarray, *, encoder: str = "self", encoder_layer: int = -1,
                decoder: str = "", tokens: list[int] | None = None) -> None:
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    count, dim = vectors.shape
    header = json.dumps({"encoder": encoder, "encoder_layer": encoder_layer, "pooling": "mean", "dim": dim,
                         "dtype": "f32", "decoder": decoder}).encode()
    with open(path, "wb") as f:
        f.write(b"NEONMEM1")
        f.write(struct.pack("<II", 1, len(header)))
        f.write(header)
        f.write(struct.pack("<Q", count))
        f.write(vectors.tobytes())
        for i in range(count):
            f.write(struct.pack("<QII", i, tokens[i] if tokens else 0, 0))
        f.write(b"NEONMEND")


def read_store(path: Path) -> tuple[dict, np.ndarray, np.ndarray]:
    """Returns (header, vectors [count, dim], flags [count])."""
    data = Path(path).read_bytes()
    assert data[:8] == b"NEONMEM1", f"{path} is not a .nmem store"
    _, n = struct.unpack_from("<II", data, 8)
    header = json.loads(data[16:16 + n])
    off = 16 + n
    (count,) = struct.unpack_from("<Q", data, off)
    off += 8
    dim = header["dim"]
    vectors = np.frombuffer(data, dtype=np.float32, count=count * dim, offset=off).reshape(count, dim)
    off += count * dim * 4
    table = np.frombuffer(data, dtype=np.uint32, count=count * 4, offset=off).reshape(count, 4)
    assert data[off + count * 16:off + count * 16 + 8] == b"NEONMEND"
    return header, vectors.copy(), table[:, 3].copy()


def write_sequences(path: Path, seqs: list[list[int]]) -> None:
    with open(path, "wb") as f:
        f.write(b"NEONSEQ1")
        f.write(struct.pack("<Q", len(seqs)))
        for s in seqs:
            f.write(struct.pack("<I", len(s)))
            f.write(np.asarray(s, dtype=np.uint32).tobytes())


@dataclass
class Example:
    doc_begin: int
    doc_end: int
    query: int
    q: list[int]
    a: list[int]
    answer_bytes: int = 0


def write_examples(path: Path, examples: list[Example]) -> None:
    with open(path, "wb") as f:
        f.write(b"NEONMEX1")
        f.write(struct.pack("<Q", len(examples)))
        for e in examples:
            f.write(struct.pack("<6I", e.doc_begin, e.doc_end, e.query, len(e.q), len(e.a), e.answer_bytes))
            f.write(np.asarray(list(e.q) + list(e.a), dtype=np.uint32).tobytes())


def read_examples(path: Path) -> list[Example]:
    data = Path(path).read_bytes()
    assert data[:8] == b"NEONMEX1"
    (count,) = struct.unpack_from("<Q", data, 8)
    off, out = 16, []
    for _ in range(count):
        db, de, q, nq, na, ab = struct.unpack_from("<6I", data, off)
        off += 24
        toks = np.frombuffer(data, dtype=np.uint32, count=nq + na, offset=off).tolist()
        off += 4 * (nq + na)
        out.append(Example(db, de, q, toks[:nq], toks[nq:], ab))
    return out


def sequence_layout(e: Example, slots: int, seq: int) -> tuple[list[int], list[int]]:
    """Decoder inputs and targets for one example, as memorilla_train.cpp builds them:
    rows [slots memory][query][answer][pad]; -1 = no loss. Memory rows hold token 0."""
    room = seq - slots - len(e.q)
    na = min(len(e.a), room)
    assert na > 0
    inputs = [0] * seq
    targets = [-1] * seq
    r = slots
    for t in e.q:
        inputs[r] = t
        r += 1
    a0 = r
    for i in range(na):
        inputs[r] = e.a[i]
        r += 1
    for i in range(na):
        if a0 + i >= 1:
            targets[a0 + i - 1] = e.a[i]
    return inputs, targets
