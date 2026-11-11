"""Embed every dataset once with every encoder (CPU), so experiments only do cheap head training.

Usage: python -m bench.precompute [encoder ...]
"""
from __future__ import annotations

import sys

from . import data, embed
from .perturb import typo_noise

SETS = {}


def corpora() -> dict:
    """tag -> ordered list of texts. Row order equals the order of the loader's frame."""
    if SETS:
        return SETS
    c = data.load_clinc()
    SETS["clinc"] = list(c.text)
    SETS["banking"] = list(data.load_banking().text)
    for loc in data.MASSIVE_LOCALES:
        SETS[f"massive_{loc}"] = list(data.load_massive(loc).text)
    # perturbed copies of the CLINC test partition (in-scope), fixed seeds
    test = c[(c.part == "test") & (~c.is_oos)]
    for rate in (0.05, 0.15):
        SETS[f"clinc_test_typo{int(rate * 100):02d}"] = [typo_noise(t, rate, seed=i) for i, t in enumerate(test.text)]
    return SETS


PLAN = {
    "potion": "all", "minilm": "all", "bge-small": "all",
    "bge-base": ["clinc", "banking", "massive_en-US", "clinc_test_typo05", "clinc_test_typo15"],
    "me5-small": [f"massive_{loc}" for loc in data.MASSIVE_LOCALES],
}


def get(encoder: str, tag: str):
    return embed.cached(encoder, tag, corpora()[tag])


def news_emb(encoder: str):
    return embed.cached(encoder, "news", list(data.load_news().text))


if __name__ == "__main__":
    encs = sys.argv[1:] or list(PLAN)
    for enc in encs:
        tags = list(corpora()) if PLAN[enc] == "all" else PLAN[enc]
        for tag in tags:
            get(enc, tag)
        if enc in ("potion", "minilm", "bge-small"):
            news_emb(enc)
    print("done")
