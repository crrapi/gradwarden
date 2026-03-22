"""char level data. batches are a pure fn of (seed, split, step).

matters for rewinds. roll back to step N and you get the EXACT same batches again,
so whatever changes after a patch is the patch and not data luck. test for this
in test_rewind_is_exact.
"""

from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)


def cache_dir() -> Path:
    root = os.environ.get("GRADWARDEN_CACHE", Path.home() / ".cache" / "gradwarden")
    path = Path(root)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_tiny_shakespeare() -> str:
    path = cache_dir() / "tinyshakespeare.txt"
    if not path.exists():
        # dl to .part first and rename on success. had a half downloaded file
        # overwrite the real one once and it was annoying to debug
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(TINY_SHAKESPEARE_URL, tmp)
        if tmp.stat().st_size < 1_000_000:
            tmp.unlink()
            raise RuntimeError("tinyshakespeare download looks truncated")
        tmp.rename(path)
    return path.read_text(encoding="utf-8")


@dataclass
class CharDataset:
    train: np.ndarray
    val: np.ndarray
    vocab: list[str]

    @classmethod
    def from_text(cls, text: str, val_fraction: float = 0.1) -> CharDataset:
        vocab = sorted(set(text))
        index = {ch: i for i, ch in enumerate(vocab)}
        ids = np.fromiter((index[ch] for ch in text), dtype=np.int32, count=len(text))
        split = int(len(ids) * (1 - val_fraction))
        return cls(train=ids[:split], val=ids[split:], vocab=vocab)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def batch(self, step: int, batch_size: int, seq_len: int, seed: int = 0, split: str = "train"):
        """(inputs, targets), both int32 [batch, seq]"""
        data = self.train if split == "train" else self.val
        stream = 0 if split == "train" else 1
        rng = np.random.default_rng([seed, stream, step])
        starts = rng.integers(0, len(data) - seq_len - 1, size=batch_size)
        offsets = np.arange(seq_len + 1)
        window = data[starts[:, None] + offsets[None, :]]
        return window[:, :-1], window[:, 1:]

    def decode(self, ids) -> str:
        return "".join(self.vocab[i] for i in ids)


def synthetic_dataset(n_chars: int = 50_000, vocab_size: int = 32, seed: int = 0) -> CharDataset:
    """repeating pattern w/ some noise. just enough signal that a 2-layer net can
    learn it in ~30 steps. used by tests so they dont hit the network"""
    rng = np.random.default_rng(seed)
    pattern = rng.integers(0, vocab_size, size=97)
    ids = np.resize(pattern, n_chars)
    noise = rng.random(n_chars) < 0.05
    ids = np.where(noise, rng.integers(0, vocab_size, size=n_chars), ids).astype(np.int32)
    split = int(n_chars * 0.9)
    vocab = [chr(ord("a") + i % 26) if i < 26 else str(i) for i in range(vocab_size)]
    return CharDataset(train=ids[:split], val=ids[split:], vocab=vocab)
