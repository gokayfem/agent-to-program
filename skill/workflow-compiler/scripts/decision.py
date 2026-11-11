"""Decision-model tier, version 2: a fast classifier that abstains on two separate grounds.

  confidence   the classifier's own probability for its answer
  familiarity  cosine similarity of the input to its nearest training example, measured on
               a view of the input (embedding or TF-IDF) independently of the classifier head

It answers only if both clear thresholds chosen on calibration data under two tests:

  seen rows    selective risk  <= alpha   (wrong answers / answered)
  shift rows   false-accept rate <= alpha (wrong answers / ALL shifted inputs)

"Shift rows" are inputs the tier should mostly refuse: rows from whole groups held out of
training (merchants, templates, customers, intents) and real out-of-scope inputs, for which
any answer counts as wrong. Calibrating on a random split alone certified a 2% tier that was
23% wrong on unseen groups in 20 of 20 splits (see REPORT.md, experiment 3).

The thresholds come from fixed-sequence testing. That is a guarantee only for inputs
exchangeable with the calibration data, and rows of one group are not independent: treat the
result as a careful estimate, count groups rather than rows as your evidence, and keep shadow
sampling on in production.

Needs numpy, scipy, scikit-learn. `embed` is any callable texts -> L2-normalised 2-D array;
without it a character TF-IDF view is used.
"""
from __future__ import annotations

import random
from typing import Any, Callable, Hashable, Optional, Sequence

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

import selective as sel
from nsp import ABSTAIN

FLOOR_Q = (0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)


def holdout_groups(groups: Sequence[Hashable], frac: float = 0.2, seed: int = 0) -> set:
    """Pick whole groups to keep out of training and use for calibration instead."""
    ids = sorted(set(groups), key=str)
    random.Random(seed).shuffle(ids)
    return set(ids[: max(1, round(len(ids) * frac))])


def fit_gate(conf_seen, fam_seen, ok_seen, conf_shift, fam_shift, ok_shift, alpha: float = 0.02,
             delta: float = 0.05) -> dict:
    """Choose (familiarity floor, confidence threshold) under the two tests described above."""
    conf_seen, fam_seen, ok_seen = map(np.asarray, (conf_seen, fam_seen, ok_seen))
    conf_shift, fam_shift, ok_shift = map(np.asarray, (conf_shift, fam_shift, ok_shift))
    pool = np.r_[fam_seen, fam_shift]
    d = delta / (2 * len(FLOOR_Q))
    best = {"kept": -1, "floor": sel.NEVER, "thr": sel.NEVER}
    for q in FLOOR_Q:
        floor = float(np.quantile(pool, q))
        k1, k2 = fam_seen >= floor, fam_shift >= floor
        thr = max(sel.pick_fixed_sequence(conf_seen[k1], ok_seen[k1], alpha, d),
                  sel.pick_far(np.where(k2, conf_shift, -np.inf), ok_shift, alpha, d))
        kept = int((k1 & (conf_seen >= thr)).sum())
        if kept > best["kept"]:
            best = {"kept": kept, "floor": floor, "thr": thr, "q": q}
    return best


class DecisionModel:
    def __init__(self, alpha: float = 0.02, delta: float = 0.05, text: Callable[[Any], str] = lambda x: x,
                 embed: Optional[Callable[[Sequence[str]], np.ndarray]] = None):
        self.alpha, self.delta, self.text, self.embed = alpha, delta, text, embed
        self.gate = {"floor": sel.NEVER, "thr": sel.NEVER}
        self.calibration: dict = {}

    def _view(self, xs, fit: bool = False):
        texts = [self.text(x) for x in xs]
        if self.embed is not None:
            return np.asarray(self.embed(texts), dtype=np.float32)
        if fit:
            self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True)
            return self.vec.fit_transform(texts)
        return self.vec.transform(texts)

    def _prep(self, X):
        return X if sparse.issparse(X) else (np.asarray(X, dtype=np.float64) - self.mu) / self.sd

    def _score(self, xs):
        X = self._view(xs)
        P = self.clf.predict_proba(self._prep(X))
        sims = X @ self.memory.T
        fam = np.asarray(sims.max(axis=1).todense()).ravel() if sparse.issparse(sims) else sims.max(axis=1)
        return self.clf.classes_[P.argmax(1)], P.max(1), fam

    def fit(self, xs, ys, seen_xs, seen_ys, shift_xs, shift_ys) -> "DecisionModel":
        """Train on (xs, ys). Calibrate on held-out rows of known groups (seen_*) and on rows the
        tier should mostly refuse (shift_*): held-out groups with their true labels, and
        out-of-scope inputs with a label that is not in `ys` (any answer to them is wrong)."""
        X = self._view(xs, fit=True)
        if not sparse.issparse(X):
            self.mu, self.sd = np.asarray(X, dtype=np.float64).mean(0), float(np.asarray(X, dtype=np.float64).std()) or 1.0
        self.clf = LogisticRegression(C=10.0, max_iter=500).fit(self._prep(X), list(ys))
        self.memory = X
        p1, c1, f1 = self._score(seen_xs)
        p2, c2, f2 = self._score(shift_xs)
        ok1, ok2 = p1 == np.asarray(list(seen_ys), dtype=object), p2 == np.asarray(list(shift_ys), dtype=object)
        self.gate = fit_gate(c1, f1, ok1, c2, f2, ok2, self.alpha, self.delta)
        a1 = (c1 >= self.gate["thr"]) & (f1 >= self.gate["floor"])
        a2 = (c2 >= self.gate["thr"]) & (f2 >= self.gate["floor"])
        self.calibration = {"seen_rows": int(len(ok1)), "shift_rows": int(len(ok2)),
                            "seen_coverage": float(a1.mean()), "seen_silent_rate": float((a1 & ~ok1).sum() / max(1, a1.sum())),
                            "shift_coverage": float(a2.mean()), "shift_false_accept_rate": float((a2 & ~ok2).mean()),
                            "answers_at_all": bool(self.gate["thr"] != sel.NEVER)}
        return self

    def answer(self, x: Any) -> Any:
        pred, conf, fam = self._score([x])
        if conf[0] < self.gate["thr"] or fam[0] < self.gate["floor"]:
            return ABSTAIN
        return pred[0].item() if hasattr(pred[0], "item") else pred[0]
