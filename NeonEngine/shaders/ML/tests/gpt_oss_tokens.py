"""Text <-> token ids for gpt-oss (o200k_harmony), until the C++ host has a tokenizer.

The encoding matches the official gpt_oss/tokenizer.py: tiktoken's o200k_base
plus the harmony special tokens. Needs `pip install tiktoken` (it downloads the
o200k_base vocabulary on first use).

    python gpt_oss_tokens.py encode "Hello world"
    python gpt_oss_tokens.py chat "What is 2 + 2?"      # minimal harmony chat prompt
    python gpt_oss_tokens.py decode 200006,1428,200008

Models trained from scratch with the NeonEngine trainer record their tokenizer
in config.json ("neon_tokenizer": a tiktoken name such as gpt2, or the
tokenizer.tiktoken BPE file next to it). --model <export dir> picks it up:
    python gpt_oss_tokens.py --model run/export encode "Once upon a time"
    python gpt_oss_tokens.py --model run/export decode 7454,2402,257,640
    python gpt_oss_tokens.py --model run/export eot          # stop token id for --stop
--encoding also takes a tiktoken name (gpt2) or a path to a .tiktoken file.

Example end to end (from NeonEngine/NeonEngine):
    python shaders/ML/tests/gpt_oss_tokens.py chat "Hi" > prompt.txt
    build/Release/LveEngine.exe --gpt-oss <gpt-oss-20b/original> --tokens-file prompt.txt --max-new 64
    python shaders/ML/tests/gpt_oss_tokens.py decode <printed ids>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def get_tokenizer():
    import tiktoken

    o200k_base = tiktoken.get_encoding("o200k_base")
    return tiktoken.Encoding(
        name="o200k_harmony",
        pat_str=o200k_base._pat_str,
        mergeable_ranks=o200k_base._mergeable_ranks,
        special_tokens={
            **o200k_base._special_tokens,
            "<|startoftext|>": 199998,
            "<|endoftext|>": 199999,
            "<|reserved_200000|>": 200000,
            "<|reserved_200001|>": 200001,
            "<|return|>": 200002,
            "<|constrain|>": 200003,
            "<|reserved_200004|>": 200004,
            "<|channel|>": 200005,
            "<|start|>": 200006,
            "<|end|>": 200007,
            "<|message|>": 200008,
            "<|reserved_200009|>": 200009,
            "<|reserved_200010|>": 200010,
            "<|reserved_200011|>": 200011,
            "<|call|>": 200012,
        } | {f"<|reserved_{i}|>": i for i in range(200013, 201088)},
    )


def chat_prompt(user: str) -> str:
    # Minimal harmony conversation; see https://github.com/openai/harmony for the full format
    # (system/developer messages, reasoning effort, channels, tools).
    return (
        "<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.\n"
        "Reasoning: low\n\n# Valid channels: analysis, commentary, final. Channel must be included for every message."
        "<|end|>"
        f"<|start|>user<|message|>{user}<|end|><|start|>assistant"
    )


def load_bpe_file(path: Path):
    """A tokenizer.tiktoken written by train/bpe_tokenizer.py."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "train"))
    import bpe_tokenizer

    return bpe_tokenizer.load_encoding(path)


def encoding_for_model(model_dir: str) -> str:
    cfg = Path(model_dir) / "config.json"
    name = json.loads(cfg.read_text()).get("neon_tokenizer", "o200k_harmony") if cfg.exists() else "o200k_harmony"
    if name.endswith(".tiktoken"):
        return str(Path(model_dir) / name)
    return name


def load_encoding(name: str):
    if name == "o200k_harmony":
        return get_tokenizer()
    if name.endswith(".tiktoken") or Path(name).is_file():
        return load_bpe_file(Path(name))
    import tiktoken

    return tiktoken.get_encoding(name)


def main(argv: list[str]) -> int:
    encoding = "o200k_harmony"
    while len(argv) > 2 and argv[1] in ("--encoding", "--model"):
        encoding = argv[2] if argv[1] == "--encoding" else encoding_for_model(argv[2])
        argv = argv[:1] + argv[3:]
    if len(argv) < 2 or argv[1] not in ("encode", "chat", "decode", "eot") or (argv[1] != "eot" and len(argv) < 3):
        print(__doc__)
        return 2
    enc = load_encoding(encoding)
    if argv[1] == "eot":
        print(enc.eot_token)
        return 0
    if argv[1] == "chat" and encoding != "o200k_harmony":
        print("chat prompts need the o200k_harmony encoding", file=sys.stderr)
        return 2
    if argv[1] == "decode":
        ids = [int(t) for t in argv[2].replace(",", " ").split()]
        print(enc.decode(ids))
    else:
        text = chat_prompt(argv[2]) if argv[1] == "chat" else argv[2]
        print(",".join(str(t) for t in enc.encode(text, allowed_special="all")))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
