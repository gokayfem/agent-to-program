"""Shared paths, seeds, I/O and statistics helpers for the experiments."""
from __future__ import annotations

import hashlib
import json
import pathlib
import platform
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
CACHE = ROOT / "cache"
RESULTS = ROOT / "results"
for _d in (RAW, PROC, CACHE, RESULTS):
    _d.mkdir(parents=True, exist_ok=True)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MASTER_SEED = 20261111


def sha_ids(ids) -> str:
    """Order-independent fingerprint of a set of example ids."""
    h = hashlib.sha256()
    for i in sorted(map(str, ids)):
        h.update(i.encode())
        h.update(b"\n")
    return h.hexdigest()[:16]


def sha_text(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:16]


class NpEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


def save_json(name: str, obj) -> pathlib.Path:
    path = RESULTS / name
    obj = dict(obj)
    obj.setdefault("_meta", {})
    obj["_meta"].update({"written": time.strftime("%Y-%m-%d %H:%M:%S"), "host": platform.platform(),
                         "python": platform.python_version()})
    path.write_text(json.dumps(obj, indent=1, cls=NpEncoder))
    return path


def load_json(name: str):
    return json.loads((RESULTS / name).read_text())


def mean_sd(xs) -> dict:
    a = np.asarray(list(xs), float)
    return {"mean": float(a.mean()), "sd": float(a.std(ddof=1)) if len(a) > 1 else 0.0, "n": int(len(a)),
            "min": float(a.min()), "max": float(a.max())}


class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *a):
        self.s = time.perf_counter() - self.t0
        self.ms = self.s * 1e3
