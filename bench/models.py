"""Decision-model families and familiarity (novelty) signals.

A *view* is a feature matrix for the whole corpus (TF-IDF fitted on training
rows, or a cached sentence embedding). A *head* turns a view into class
probabilities. A decision model is a (view, head) pair; a familiarity detector
is a function of a view alone. Keeping the two apart is what lets the
experiments ask whether a fast answer and its check share a failure mode.

Heads
  LRHead        multinomial logistic regression               (parametric, linear)
  KNNHead       cosine k-nearest-neighbour vote               (non-parametric; k=1 is a semantic cache)
  MahaHead      class means + shared covariance (LDA-style)   (generative; has a natural distance)
  MLPEnsemble   5 one-hidden-layer MLPs, different seeds      (deep ensemble; disagreement signal)
  MinedRules    word n-gram -> label rules mined from history (a rule committee; abstains by itself)

Signals (higher = more trustworthy), see `confidence_signals` and `Novelty`.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.special import logsumexp, softmax
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression


# --------------------------------------------------------------------------- views
def tfidf_view(texts_all, idx_train, analyzer: str = "char_wb"):
    """TF-IDF fitted on the training rows only; returns a CSR matrix for every row."""
    if analyzer == "char_wb":
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True, min_df=2)
    else:
        vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), sublinear_tf=True, min_df=2)
    texts_all = list(texts_all)
    vec.fit([texts_all[i] for i in idx_train])
    return vec.transform(texts_all).tocsr(), vec


# --------------------------------------------------------------------------- heads
class LRHead:
    """Multinomial logistic regression. Dense views are centred and scaled on the training rows
    first: sentence embeddings share a large common component, and without centring L-BFGS
    stops at its first step on some encoders (bge) and returns a constant classifier."""

    def __init__(self, C: float = 10.0, max_iter: int = 500):
        self.C, self.max_iter = C, max_iter

    def _prep(self, X):
        if sparse.issparse(X):
            return X
        return (np.asarray(X, dtype=np.float64) - self.mu) / self.sd

    def fit(self, X, y):
        if not sparse.issparse(X):
            self.mu = np.asarray(X, dtype=np.float64).mean(0)
            self.sd = float(np.asarray(X, dtype=np.float64).std()) or 1.0
        self.clf = LogisticRegression(C=self.C, max_iter=self.max_iter, tol=1e-4).fit(self._prep(X), y)
        self.classes_ = self.clf.classes_
        return self

    def logits(self, X) -> np.ndarray:
        z = self.clf.decision_function(self._prep(X))
        return np.c_[-z, z] if z.ndim == 1 else z

    def proba(self, X) -> np.ndarray:
        return softmax(self.logits(X), axis=1)


def _topk_dense(Q: np.ndarray, T: np.ndarray, k: int, chunk: int = 4096):
    sims, idxs = [], []
    for i in range(0, Q.shape[0], chunk):
        s = Q[i:i + chunk] @ T.T
        kk = min(k, s.shape[1])
        part = np.argpartition(-s, kk - 1, axis=1)[:, :kk]
        ps = np.take_along_axis(s, part, axis=1)
        order = np.argsort(-ps, axis=1)
        sims.append(np.take_along_axis(ps, order, axis=1))
        idxs.append(np.take_along_axis(part, order, axis=1))
    return np.vstack(sims), np.vstack(idxs)


def _topk_sparse(Q, T, k: int, chunk: int = 1024):
    sims, idxs = [], []
    Tt = T.T.tocsr()
    for i in range(0, Q.shape[0], chunk):
        s = np.asarray((Q[i:i + chunk] @ Tt).todense())
        kk = min(k, s.shape[1])
        part = np.argpartition(-s, kk - 1, axis=1)[:, :kk]
        ps = np.take_along_axis(s, part, axis=1)
        order = np.argsort(-ps, axis=1)
        sims.append(np.take_along_axis(ps, order, axis=1))
        idxs.append(np.take_along_axis(part, order, axis=1))
    return np.vstack(sims), np.vstack(idxs)


def topk(Q, T, k: int = 10):
    """Cosine similarities and indices of the k nearest training rows (rows are L2-normalised)."""
    return _topk_sparse(Q, T, k) if sparse.issparse(Q) else _topk_dense(Q, T, k)


class KNNHead:
    def __init__(self, k: int = 10, temp: float = 0.05):
        self.k, self.temp = k, temp

    def fit(self, X, y):
        self.X = X
        self.classes_, self.y = np.unique(y, return_inverse=True)
        return self

    def proba(self, Q) -> np.ndarray:
        sims, idx = topk(Q, self.X, self.k)
        w = np.exp((sims - sims[:, :1]) / self.temp)
        P = np.zeros((Q.shape[0], len(self.classes_)))
        lab = self.y[idx]
        for j in range(lab.shape[1]):
            np.add.at(P, (np.arange(len(P)), lab[:, j]), w[:, j])
        return P / P.sum(axis=1, keepdims=True)


class MahaHead:
    """Class-conditional Gaussians with one shared, shrunk covariance."""

    def __init__(self, shrink: float = 0.1):
        self.shrink = shrink

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        self.classes_, yi = np.unique(y, return_inverse=True)
        self.mu = np.stack([X[yi == c].mean(0) for c in range(len(self.classes_))])
        R = X - self.mu[yi]
        cov = R.T @ R / len(X)
        cov = (1 - self.shrink) * cov + self.shrink * np.eye(cov.shape[0]) * np.trace(cov) / cov.shape[0]
        self.P = np.linalg.pinv(cov)
        self.mu0 = X.mean(0)
        R0 = X - self.mu0
        cov0 = R0.T @ R0 / len(X)
        cov0 = (1 - self.shrink) * cov0 + self.shrink * np.eye(cov0.shape[0]) * np.trace(cov0) / cov0.shape[0]
        self.P0 = np.linalg.pinv(cov0)
        return self

    def dist(self, Q) -> np.ndarray:
        """Squared Mahalanobis distance to every class mean, shape (n, C)."""
        Q = np.asarray(Q, dtype=np.float64)
        QP = Q @ self.P
        return (QP * Q).sum(1)[:, None] - 2 * QP @ self.mu.T + ((self.mu @ self.P) * self.mu).sum(1)[None]

    def dist0(self, Q) -> np.ndarray:
        D = np.asarray(Q, dtype=np.float64) - self.mu0
        return ((D @ self.P0) * D).sum(1)

    def proba(self, Q) -> np.ndarray:
        return softmax(-0.5 * self.dist(Q), axis=1)


class MLPEnsemble:
    """Deep ensemble: M small MLPs trained from different seeds on the same data."""

    def __init__(self, members: int = 5, hidden: int = 512, epochs: int = 30, lr: float = 2e-3, seed: int = 0):
        self.members, self.hidden, self.epochs, self.lr, self.seed = members, hidden, epochs, lr, seed

    def fit(self, X, y):
        import torch
        self.classes_, yi = np.unique(y, return_inverse=True)
        Xt, yt = torch.tensor(np.asarray(X), dtype=torch.float32), torch.tensor(yi)
        self.nets = []
        for m in range(self.members):
            torch.manual_seed(self.seed * 1000 + m)
            net = torch.nn.Sequential(torch.nn.Linear(Xt.shape[1], self.hidden), torch.nn.GELU(),
                                      torch.nn.Dropout(0.1), torch.nn.Linear(self.hidden, len(self.classes_)))
            opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=0.01)
            sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=self.lr, total_steps=self.epochs * ((len(Xt) + 255) // 256))
            for _ in range(self.epochs):
                perm = torch.randperm(len(Xt))
                for i in range(0, len(Xt), 256):
                    b = perm[i:i + 256]
                    loss = torch.nn.functional.cross_entropy(net(Xt[b]), yt[b], label_smoothing=0.05)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    sched.step()
            net.eval()
            self.nets.append(net)
        return self

    def member_proba(self, Q) -> np.ndarray:
        import torch
        with torch.no_grad():
            Qt = torch.tensor(np.asarray(Q), dtype=torch.float32)
            return np.stack([torch.softmax(n(Qt), dim=1).numpy() for n in self.nets])

    def proba(self, Q) -> np.ndarray:
        return self.member_proba(Q).mean(0)

    def agreement(self, Q) -> np.ndarray:
        """Share of members voting for the ensemble's label (1.0 = unanimous)."""
        mp = self.member_proba(Q)
        top = mp.mean(0).argmax(1)
        return (mp.argmax(2) == top[None]).mean(0)


class MinedRules:
    """A rule committee mined from labelled history.

    Every word n-gram (1-3) seen at least `min_support` times whose smoothed precision for one
    label is at least `min_precision` becomes a rule "contains n-gram -> label". At run time
    the committee answers only if at least `min_votes` rules fire and all of them agree.
    """

    def __init__(self, min_support: int = 5, min_precision: float = 1.0, min_votes: int = 1):
        self.min_support, self.min_precision, self.min_votes = min_support, min_precision, min_votes

    def fit(self, texts, y):
        self.classes_, yi = np.unique(y, return_inverse=True)
        self.vec = CountVectorizer(ngram_range=(1, 3), binary=True, min_df=self.min_support, lowercase=True,
                                   token_pattern=r"(?u)\b\w+\b")
        X = self.vec.fit_transform(texts)
        Y = sparse.csr_matrix((np.ones(len(yi)), (np.arange(len(yi)), yi)), shape=(len(yi), len(self.classes_)))
        counts = np.asarray((X.T @ Y).todense())                 # (V, C)
        tot = counts.sum(1)
        raw = counts.max(1) / np.maximum(tot, 1)                 # observed precision of "n-gram -> label"
        prec = (counts.max(1) + 1) / (tot + 2)                   # smoothed, used only as the vote weight
        keep = (tot >= self.min_support) & (raw >= self.min_precision)
        self.rule_idx = np.flatnonzero(keep)
        self.rule_label = counts[keep].argmax(1)
        self.rule_weight = np.log(prec[keep] / (1 - prec[keep]))
        self.n_rules = int(keep.sum())
        self.L = sparse.csr_matrix((self.rule_weight, (np.arange(self.n_rules), self.rule_label)),
                                   shape=(self.n_rules, len(self.classes_)))
        self.B = sparse.csr_matrix((np.ones(self.n_rules), (np.arange(self.n_rules), self.rule_label)),
                                   shape=(self.n_rules, len(self.classes_)))
        return self

    def votes(self, texts):
        """(weighted votes (n, C), count of fired rules (n,), number of distinct labels voted (n,))."""
        Xq = self.vec.transform(texts)[:, self.rule_idx]
        w = np.asarray((Xq @ self.L).todense())
        c = np.asarray((Xq @ self.B).todense())
        return w, c.sum(1), (c > 0).sum(1)

    def answer(self, texts):
        """labels (object array, None = abstain) and a trust score (number of agreeing rules)."""
        w, fired, distinct = self.votes(texts)
        ok = (fired >= self.min_votes) & (distinct == 1)
        lab = np.where(ok, self.classes_[w.argmax(1)], None)
        return lab, np.where(ok, fired, 0).astype(float)


# --------------------------------------------------------------------------- signals
def confidence_signals(P: np.ndarray, logits: np.ndarray | None = None) -> dict:
    """Trust scores computed from a model's own output."""
    srt = np.sort(P, axis=1)
    out = {"msp": srt[:, -1], "margin": srt[:, -1] - srt[:, -2],
           "neg_entropy": (P * np.log(np.clip(P, 1e-12, 1))).sum(1)}
    if logits is not None:
        out["energy"] = logsumexp(logits, axis=1)
        out["max_logit"] = logits.max(1)
    return out


class Novelty:
    """Familiarity of an input measured on one view, independent of any classifier head."""

    def __init__(self, X_train, y_train, maha: bool = True):
        self.X = X_train
        self.sparse = sparse.issparse(X_train)
        self.maha = None
        if maha and not self.sparse:
            self.maha = MahaHead().fit(X_train, y_train)
            cls, yi = np.unique(y_train, return_inverse=True)
            C = np.stack([np.asarray(X_train[yi == c]).mean(0) for c in range(len(cls))])
            self.centroids = C / np.linalg.norm(C, axis=1, keepdims=True)

    def signals(self, Q, exclude_self: bool = False) -> dict:
        k = 6 if exclude_self else 5
        sims, _ = topk(Q, self.X, k)
        if exclude_self:
            sims = sims[:, 1:]
        out = {"knn1": sims[:, 0], "knn5": sims[:, :5].mean(1)}
        if self.maha is not None:
            d = self.maha.dist(Q)
            out["maha"] = -d.min(1)
            out["rel_maha"] = -(d.min(1) - self.maha.dist0(Q))
            out["centroid"] = (np.asarray(Q) @ self.centroids.T).max(1)
        return out


def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(score of a positive > score of a negative). Positives should score higher."""
    from scipy.stats import rankdata
    r = rankdata(np.r_[pos, neg])
    n1, n0 = len(pos), len(neg)
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def fpr_at_tpr(pos: np.ndarray, neg: np.ndarray, tpr: float = 0.95) -> float:
    """Share of negatives accepted when the threshold keeps `tpr` of positives."""
    thr = np.quantile(pos, 1 - tpr)
    return float((neg >= thr).mean())
