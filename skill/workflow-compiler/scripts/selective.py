"""Selective answering: choose the abstention threshold of a fast tier.

A fast tier emits a trust score per input (higher = safer to answer) and is
allowed to answer only above a threshold. The threshold is chosen on a
calibration set so that the error rate among answered inputs (the *selective
risk*, i.e. the silent-error rate of that tier) stays under a budget `alpha`.

Three procedures, from least to most defensible:

  empirical       most permissive cut whose calibration error rate <= alpha.
                  No allowance for sampling noise.
  wilson_scan     most permissive cut whose Wilson lower bound on precision is
                  >= 1 - alpha. This is what the original skill's decision.py
                  does. It scans every cut, so the bound is optimistic.
  fixed_sequence  walks a label-independent grid of cuts from strict to
                  permissive, testing H0: risk > alpha with an exact binomial
                  tail at level delta, and stops at the first cut it cannot
                  certify. Controls P(risk > alpha) <= delta when calibration
                  and deployment inputs are exchangeable (Geifman & El-Yaniv's
                  SGR / Learn-then-Test fixed-sequence testing).

None of them protects against inputs that are not exchangeable with the
calibration set; bench/exp3 measures exactly that gap.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import beta, binom

NEVER = float("inf")          # threshold meaning "this tier never answers"


def clopper_pearson_upper(k: int, n: int, delta: float = 0.05) -> float:
    """Exact one-sided upper confidence bound for a binomial proportion."""
    if n == 0:
        return 1.0
    if k >= n:
        return 1.0
    return float(beta.ppf(1 - delta, k + 1, n - k))


def clopper_pearson(k: int, n: int, conf: float = 0.95) -> tuple[float, float]:
    """Exact two-sided interval."""
    if n == 0:
        return 0.0, 1.0
    a = (1 - conf) / 2
    lo = 0.0 if k == 0 else float(beta.ppf(a, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - a, k + 1, n - k))
    return lo, hi


def _sorted(score: np.ndarray, correct: np.ndarray):
    order = np.argsort(-score, kind="stable")
    s, c = score[order], correct[order].astype(bool)
    errs = np.cumsum(~c)
    # A cut is only allowed between distinct scores.
    last_of_tie = np.r_[s[1:] != s[:-1], True]
    return s, errs, last_of_tie


def pick_empirical(score: np.ndarray, correct: np.ndarray, alpha: float, min_n: int = 1) -> float:
    if len(score) == 0:
        return NEVER
    s, errs, ok_cut = _sorted(np.asarray(score, float), np.asarray(correct))
    n = np.arange(1, len(s) + 1)
    good = ok_cut & (errs / n <= alpha) & (n >= min_n)
    idx = np.flatnonzero(good)
    return float(s[idx[-1]]) if len(idx) else NEVER


def pick_wilson_scan(score: np.ndarray, correct: np.ndarray, alpha: float, z: float = 1.96) -> float:
    if len(score) == 0:
        return NEVER
    s, errs, ok_cut = _sorted(np.asarray(score, float), np.asarray(correct))
    n = np.arange(1, len(s) + 1).astype(float)
    p = (n - errs) / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    lo = centre - half
    idx = np.flatnonzero(ok_cut & (lo >= 1 - alpha))
    return float(s[idx[-1]]) if len(idx) else NEVER


def pick_fixed_sequence(score: np.ndarray, correct: np.ndarray, alpha: float,
                        delta: float = 0.05, grid: int = 60, start_frac: float = 0.15) -> float:
    """Fixed-sequence testing over a grid of accepted-counts chosen without looking at labels.

    The walk starts at max(n_min, start_frac * N) accepted cases rather than at n_min: starting
    at the smallest certifiable count means a single error among the very top scores vetoes the
    whole tier. Any label-independent start is valid; a later start trades a little coverage in
    the best case for not collapsing in the common case."""
    if len(score) == 0:
        return NEVER
    s, errs, ok_cut = _sorted(np.asarray(score, float), np.asarray(correct))
    N = len(s)
    # Smallest n for which zero errors could be certified at all.
    n_min = int(np.ceil(np.log(delta) / np.log(1 - alpha))) if 0 < alpha < 1 else 1
    if N < n_min:
        return NEVER
    counts = np.unique(np.linspace(min(N, max(n_min, int(start_frac * N))), N, grid).astype(int))
    chosen = NEVER
    for n in counts:
        i = n - 1
        while i < N - 1 and not ok_cut[i]:      # move to the end of a tie block
            i += 1
        k, m = int(errs[i]), i + 1
        pval = float(binom.cdf(k, m, alpha))    # P(Bin(m, alpha) <= k): small => risk < alpha
        if pval <= delta:
            chosen = float(s[i])
        else:
            break
    return chosen


PICKERS = {
    "empirical": lambda s, c, a, d=0.05: pick_empirical(s, c, a),
    "wilson_scan": lambda s, c, a, d=0.05: pick_wilson_scan(s, c, a),
    "fixed_sequence": lambda s, c, a, d=0.05: pick_fixed_sequence(s, c, a, d),
}


def evaluate(score: np.ndarray, correct: np.ndarray, thr: float) -> dict:
    """Coverage and silent errors of a tier that answers when score >= thr."""
    score, correct = np.asarray(score, float), np.asarray(correct).astype(bool)
    acc = score >= thr
    n, k = int(acc.sum()), int((acc & ~correct).sum())
    return {"n": int(len(score)), "answered": n, "coverage": n / max(1, len(score)),
            "silent_errors": k, "silent_error_rate": (k / n) if n else 0.0,
            "silent_error_upper95": clopper_pearson_upper(k, n, 0.05) if n else None}


def risk_coverage(score: np.ndarray, correct: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Coverage and selective risk at every cut (most confident first)."""
    order = np.argsort(-np.asarray(score, float), kind="stable")
    c = np.asarray(correct).astype(bool)[order]
    n = np.arange(1, len(c) + 1)
    return n / len(c), np.cumsum(~c) / n


def aurc(score: np.ndarray, correct: np.ndarray) -> float:
    """Area under the risk-coverage curve (lower is better)."""
    _, risk = risk_coverage(score, correct)
    return float(risk.mean())


def coverage_at_risk(score: np.ndarray, correct: np.ndarray, alpha: float) -> float:
    """Best achievable coverage with selective risk <= alpha (oracle threshold; descriptive only)."""
    cov, risk = risk_coverage(score, correct)
    ok = np.flatnonzero(risk <= alpha)
    return float(cov[ok[-1]]) if len(ok) else 0.0


def pick_cluster_bootstrap(score: np.ndarray, correct: np.ndarray, groups: np.ndarray, alpha: float,
                           delta: float = 0.05, grid: int = 40, B: int = 400, start_frac: float = 0.15,
                           seed: int = 0) -> float:
    """Group-aware variant: the unit of evidence is a whole group, not a row.

    Rows from one group (one merchant, one intent, one template) fail together, so a binomial
    bound over rows overstates the evidence. Here the selective risk at each cut is bootstrapped
    over groups, and the walk from strict to permissive stops at the first cut whose (1 - delta)
    bootstrap quantile exceeds alpha. With few groups this is still approximate, but it is
    approximate in the right direction: its width reflects the number of groups.
    """
    score, correct = np.asarray(score, float), np.asarray(correct).astype(bool)
    if len(score) == 0:
        return NEVER
    gids, gi = np.unique(np.asarray(groups), return_inverse=True)
    G = len(gids)
    if G < 5:
        return NEVER
    N = len(score)
    order = np.sort(score)[::-1]
    counts = np.unique(np.linspace(max(1, int(start_frac * N)), N, grid).astype(int))
    thrs = order[counts - 1]
    rng = np.random.default_rng(seed)
    boot = rng.integers(0, G, size=(B, G))
    chosen = NEVER
    for thr in thrs:
        acc = score >= thr
        a = np.bincount(gi[acc], minlength=G).astype(float)
        e = np.bincount(gi[acc & ~correct], minlength=G).astype(float)
        A, E = a[boot].sum(1), e[boot].sum(1)
        risk = np.where(A > 0, E / np.maximum(A, 1), 0.0)
        if np.quantile(risk, 1 - delta) <= alpha:
            chosen = float(thr)
        else:
            break
    return chosen


def pick_far(score: np.ndarray, correct: np.ndarray, alpha: float, delta: float = 0.05,
             method: str = "fixed_sequence", groups: np.ndarray | None = None, grid: int = 60,
             B: int = 400, seed: int = 0) -> float:
    """Most permissive cut whose *false-accept rate* (wrong answers / all inputs) is <= alpha.

    Selective risk conditions on being answered, which is the wrong question for a population
    the tier should mostly refuse (unseen groups, out-of-scope inputs): if a good gate answers
    almost none of them there is nothing to condition on. The false-accept rate is defined for
    every cut, including "answer nothing", so it can always be tested.
    Rows the gate already refuses can be passed with score = -inf.
    """
    score, correct = np.asarray(score, float), np.asarray(correct).astype(bool)
    N = len(score)
    if N == 0:
        return NEVER
    finite = np.isfinite(score)
    cand = np.unique(np.quantile(score[finite], np.linspace(1, 0, grid))) [::-1] if finite.any() else np.array([])
    if method == "cluster_bootstrap":
        gids, gi = np.unique(np.asarray(groups), return_inverse=True)
        G = len(gids)
        boot = np.random.default_rng(seed).integers(0, G, size=(B, G))
        size_g = np.bincount(gi, minlength=G).astype(float)
    chosen = NEVER
    for thr in cand:
        wrong = (score >= thr) & ~correct
        k = int(wrong.sum())
        if method == "empirical":
            ok = k / N <= alpha
        elif method == "wilson_scan":
            z, p = 1.96, k / N
            d = 1 + z * z / N
            hi = (p + z * z / (2 * N)) / d + z * np.sqrt(p * (1 - p) / N + z * z / (4 * N * N)) / d
            ok = hi <= alpha
        elif method == "fixed_sequence":
            ok = float(binom.cdf(k, N, alpha)) <= delta
        elif method == "cluster_bootstrap":
            e = np.bincount(gi[wrong], minlength=G).astype(float)
            rate = e[boot].sum(1) / size_g[boot].sum(1)
            ok = np.quantile(rate, 1 - delta) <= alpha
        else:
            raise KeyError(method)
        if ok:
            chosen = float(thr)
        elif method in ("fixed_sequence", "cluster_bootstrap"):
            break
    return chosen
