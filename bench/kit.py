"""Glue shared by the experiments: build views, fit a named family, score it."""
from __future__ import annotations

import time

import numpy as np

from . import models as M
from . import precompute
from wfc import selective as sel

DENSE = ("potion", "minilm", "bge-small", "bge-base", "me5-small")


def dense_view(encoder: str, tag: str) -> np.ndarray:
    return precompute.get(encoder, tag)


FAMILIES = {
    # name: (view, head factory)
    "tfidf-char+LR": ("tfidf", lambda: M.LRHead(C=20.0)),
    "tfidf-word+LR": ("tfidf-word", lambda: M.LRHead(C=20.0)),
    "potion+LR": ("potion", lambda: M.LRHead(C=20.0)),
    "minilm+LR": ("minilm", lambda: M.LRHead(C=20.0)),
    "bge-small+LR": ("bge-small", lambda: M.LRHead(C=20.0)),
    "bge-base+LR": ("bge-base", lambda: M.LRHead(C=20.0)),
    "minilm+kNN": ("minilm", lambda: M.KNNHead(k=10)),
    "bge-small+kNN": ("bge-small", lambda: M.KNNHead(k=10)),
    "bge-small+1NN": ("bge-small", lambda: M.KNNHead(k=1)),
    "bge-small+Maha": ("bge-small", lambda: M.MahaHead()),
    "bge-small+MLPx5": ("bge-small", lambda: M.MLPEnsemble()),
}


class Fitted:
    """One decision model fitted on a training set, with cached outputs per row subset."""

    def __init__(self, name: str, view_name: str, head, X_all, idx_train, y_train):
        self.name, self.view_name, self.head, self.X = name, view_name, head, X_all
        t0 = time.perf_counter()
        head.fit(X_all[idx_train], y_train)
        self.fit_s = time.perf_counter() - t0
        self.classes_ = head.classes_

    def run(self, idx) -> dict:
        X = self.X[idx]
        logits = self.head.logits(X) if hasattr(self.head, "logits") else None
        P = self.head.proba(X) if logits is None else M.softmax(logits, axis=1)
        out = {"P": P, "pred": self.classes_[P.argmax(1)]}
        out.update(M.confidence_signals(P, logits))
        if isinstance(self.head, M.MLPEnsemble):
            out["ens_agree"] = self.head.agreement(X)
        if isinstance(self.head, M.KNNHead):
            out["knn_sim"] = M.topk(X, self.head.X, 1)[0][:, 0]
        return out


def build_views(tag: str, texts_all, idx_train, names) -> dict:
    views = {}
    for n in names:
        if n == "tfidf":
            views[n], _ = M.tfidf_view(texts_all, idx_train, "char_wb")
        elif n == "tfidf-word":
            views[n], _ = M.tfidf_view(texts_all, idx_train, "word")
        else:
            views[n] = dense_view(n, tag)
    return views


def fit_family(name: str, views: dict, idx_train, y_train) -> Fitted:
    view_name, make = FAMILIES[name]
    return Fitted(name, view_name, make(), views[view_name], idx_train, y_train)


def selective_summary(score, correct, cal_score, cal_correct, alphas=(0.01, 0.02, 0.05), delta=0.05) -> dict:
    """Descriptive curve stats on test + operational numbers at thresholds picked on calibration."""
    out = {"aurc": sel.aurc(score, correct)}
    for a in alphas:
        out[f"oracle_cov@{a}"] = sel.coverage_at_risk(score, correct, a)
        thr = sel.pick_fixed_sequence(cal_score, cal_correct, a, delta)
        ev = sel.evaluate(score, correct, thr)
        out[f"fs@{a}"] = {"thr": thr, "coverage": ev["coverage"], "silent_error_rate": ev["silent_error_rate"],
                          "silent_errors": ev["silent_errors"], "answered": ev["answered"],
                          "upper95": ev["silent_error_upper95"]}
    return out


FLOOR_Q = (0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)


def fit_gate(conf_seen, fam_seen, ok_seen, conf_shift, fam_shift, ok_shift, alpha: float = 0.02,
             delta: float = 0.05, method: str = "fixed_sequence") -> dict:
    """Two-threshold gate for a fast tier: answer iff familiarity >= floor and confidence >= thr.

    Chosen on calibration data only, under two separate tests (the 'G-worst' protocol of exp3):
      * seen-population rows:   selective risk  <= alpha   (errors among answered)
      * shift-population rows:  false-accept rate <= alpha (wrong answers / all shifted inputs)
    The shift population is whatever the tier should mostly refuse: held-out groups, real
    out-of-scope inputs. Among floors, the pair that keeps the most calibration rows wins;
    delta is split across floors and across the two tests.
    """
    conf_seen, fam_seen, ok_seen = map(np.asarray, (conf_seen, fam_seen, ok_seen))
    conf_shift, fam_shift, ok_shift = map(np.asarray, (conf_shift, fam_shift, ok_shift))
    pool = np.r_[fam_seen, fam_shift]
    d = delta / (2 * len(FLOOR_Q))
    best = {"kept": -1, "floor": sel.NEVER, "thr": sel.NEVER}
    for q in FLOOR_Q:
        floor = float(np.quantile(pool, q))
        k1, k2 = fam_seen >= floor, fam_shift >= floor
        if method == "fixed_sequence":
            t1 = sel.pick_fixed_sequence(conf_seen[k1], ok_seen[k1], alpha, d)
        else:
            t1 = sel.PICKERS[method](conf_seen[k1], ok_seen[k1], alpha, d)
        t2 = sel.pick_far(np.where(k2, conf_shift, -np.inf), ok_shift, alpha, d, method)
        thr = max(t1, t2)
        kept = int((k1 & (conf_seen >= thr)).sum())
        if kept > best["kept"]:
            best = {"kept": kept, "floor": floor, "thr": thr, "q": q}
    return best
