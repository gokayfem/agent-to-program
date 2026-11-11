"""Per-request latency of every local tier, one request at a time on this machine's CPU.

Run it with nothing else using the CPU. LLM latency is measured by `label_llm latency`.
"""
from __future__ import annotations

import time

import numpy as np

from . import embed
from . import models as M
from .common import save_json
from .frozen import Frozen


def timeit(fn, items, warm: int = 5) -> dict:
    for x in items[:warm]:
        fn(x)
    ms = []
    for x in items:
        t0 = time.perf_counter()
        fn(x)
        ms.append((time.perf_counter() - t0) * 1e3)
    ms = np.asarray(ms)
    return {"p50": float(np.median(ms)), "p90": float(np.quantile(ms, 0.9)), "mean": float(ms.mean()), "n": int(len(ms))}


def run() -> dict:
    F = Frozen(with_ensemble=True)
    texts = list(F.pools["test_seen"].sort_values("uid").text[:300])
    out = {}
    out["exact cache"] = timeit(lambda t: F.cache_table.get(F.norm(t)), texts)
    out["mined rules"] = timeit(lambda t: F.rules.answer([t]), texts)
    tf_lr = F.models["tfidf-char+LR"].head
    out["tfidf-char+LR end-to-end"] = timeit(lambda t: tf_lr.proba(F.vec.transform([t])), texts)
    for enc in ("potion", "minilm", "bge-small", "bge-base"):
        ms = embed.single_latency_ms(enc, texts[:200])
        out[f"{enc} encode"] = {"p50": float(np.median(ms)), "p90": float(np.quantile(ms, 0.9)), "mean": float(ms.mean()), "n": len(ms)}
    Xq = F.views["bge-small"][F.pools["test_seen"].index.values[:300]]
    rows = [Xq[i:i + 1] for i in range(len(Xq))]
    lr = F.models["bge-small+LR"].head
    out["LR head"] = timeit(lambda x: lr.proba(x), rows)
    mem = F.views["bge-small"][F.tr]
    out["knn1 over train"] = timeit(lambda x: M.topk(x, mem, 1), rows)
    out["kNN head (k=10)"] = timeit(lambda x: F.models["bge-small+kNN"].head.proba(x), rows)
    maha = F.nov["bge-small"].maha
    out["Mahalanobis"] = timeit(lambda x: maha.dist(x), rows)
    out["MLP ensemble x5"] = timeit(lambda x: F.ens.proba(x), rows)
    res = {"tier_ms_p50": {k: v["p50"] for k, v in out.items()}, "detail": out,
           "note": "single request, CPU, Apple M4; encoders via sentence-transformers / model2vec on CPU",
           "memory_rows": int(len(F.tr))}
    for k, v in out.items():
        print(f"  {k:28s} p50 {v['p50']:.3f} ms   p90 {v['p90']:.3f} ms", flush=True)
    return res


if __name__ == "__main__":
    save_json("exp_latency.json", run())
