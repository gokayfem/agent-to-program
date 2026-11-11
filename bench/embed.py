"""Frozen text encoders with an on-disk cache.

Encoders span three orders of magnitude in cost, which is the point: the
decision-model tier can sit anywhere on this line.

  potion     minishlab/potion-base-8M      static token embeddings, no transformer (8M params)
  minilm     all-MiniLM-L6-v2              6-layer transformer (22M)
  bge-small  BAAI/bge-small-en-v1.5        12-layer transformer (33M)
  bge-base   BAAI/bge-base-en-v1.5         12-layer transformer (109M)
  me5-small  intfloat/multilingual-e5-small  multilingual 12-layer (118M), for the locale-shift test

All vectors are L2-normalised float32.
"""
from __future__ import annotations

import os
import time
from typing import Sequence

import numpy as np

from .common import CACHE, sha_text

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
EMB_DIR = CACHE / "emb"
EMB_DIR.mkdir(parents=True, exist_ok=True)

ENCODERS = {
    "potion": "minishlab/potion-base-8M",
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "bge-small": "BAAI/bge-small-en-v1.5",
    "bge-base": "BAAI/bge-base-en-v1.5",
    "me5-small": "intfloat/multilingual-e5-small",
}
_loaded = {}


def get_encoder(name: str, device: str = "cpu"):
    key = (name, device)
    if key in _loaded:
        return _loaded[key]
    if name == "potion":
        from model2vec import StaticModel
        m = StaticModel.from_pretrained(ENCODERS[name])
    else:
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer(ENCODERS[name], device=device)
    _loaded[key] = m
    return m


def _prep(name: str, texts: Sequence[str]) -> list:
    if name == "me5-small":
        return ["query: " + t for t in texts]
    return list(texts)


def encode(name: str, texts: Sequence[str], device: str = "cpu", batch_size: int = 128) -> np.ndarray:
    m = get_encoder(name, device)
    texts = _prep(name, texts)
    if name == "potion":
        v = np.asarray(m.encode(texts), dtype=np.float32)
        v /= np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)
        return v
    return np.asarray(m.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                               show_progress_bar=False), dtype=np.float32)


def cached(name: str, tag: str, texts: Sequence[str], device: str = "cpu") -> np.ndarray:
    """Embeddings for a fixed, ordered list of texts; recomputed only if the list changes."""
    texts = list(texts)
    fp = sha_text("\x1f".join(texts))
    path = EMB_DIR / f"{name}__{tag}__{fp}.npy"
    if path.exists():
        return np.load(path)
    t0 = time.perf_counter()
    v = encode(name, texts, device=device)
    np.save(path, v)
    print(f"  embedded {tag} with {name}: {len(texts)} texts in {time.perf_counter() - t0:.1f}s", flush=True)
    return v


def single_latency_ms(name: str, texts: Sequence[str], device: str = "cpu", warmup: int = 5) -> np.ndarray:
    """Per-request encode time, one text at a time (what a production request pays)."""
    m = get_encoder(name, device)
    texts = _prep(name, texts)
    enc = (lambda t: m.encode([t])) if name == "potion" else \
          (lambda t: m.encode([t], normalize_embeddings=True, show_progress_bar=False))
    for t in texts[:warmup]:
        enc(t)
    out = []
    for t in texts:
        t0 = time.perf_counter()
        enc(t)
        out.append((time.perf_counter() - t0) * 1e3)
    return np.asarray(out)
