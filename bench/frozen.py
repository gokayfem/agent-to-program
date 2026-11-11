"""Everything about the frozen CLINC routing split in one place: fitted fast models, their
outputs on every pool, and the stored real LLM answers for the same rows.

Used by exp4 (verification), exp5 (LLM tier and end-to-end tiers) and exp6 (production loop).
"""
from __future__ import annotations

import numpy as np

from . import data, embed, holes, kit, pools
from . import models as M
from .llm import AnswerStore

NONE = holes.NONE
FAST = ("tfidf-char+LR", "minilm+LR", "bge-small+LR", "bge-small+kNN", "bge-base+LR")
VIEWS = ("tfidf", "minilm", "bge-small", "bge-base")


class Frozen:
    def __init__(self, with_ensemble: bool = True, train_idx=None, train_y=None):
        self.f = pools.frozen()
        self.pools = pools.pools()
        clinc = data.load_clinc()
        self.texts = list(clinc.text)
        self.tr = self.f.index[self.f.split == "train"].values if train_idx is None else train_idx
        self.ytr = self.f.loc[self.tr, "label"].values if train_y is None else train_y
        self.classes = sorted(self.f.label.unique())
        tf, self.vec = M.tfidf_view(self.texts, self.tr, "char_wb")
        self.views = {"tfidf": tf, **{v: kit.dense_view(v, "clinc") for v in VIEWS if v != "tfidf"}}
        self.models = {m: kit.fit_family(m, self.views, self.tr, self.ytr) for m in FAST}
        self.nov = {v: M.Novelty(self.views[v][self.tr], self.ytr, maha=(v == "bge-small")) for v in VIEWS}
        self.rules = M.MinedRules(min_support=10, min_precision=1.0).fit([self.texts[i] for i in self.tr], self.ytr)
        self.ens = M.MLPEnsemble().fit(self.views["bge-small"][self.tr], self.ytr) if with_ensemble else None
        self.cache_table = {}
        for i, lab in zip(self.tr, self.ytr):
            self.cache_table.setdefault(self.norm(self.texts[i]), set()).add(lab)
        self._out = {}

    @staticmethod
    def norm(t: str) -> str:
        return " ".join(str(t).lower().split())

    # -- features for a pool (rows of the CLINC corpus, or free text for perturbed pools) -----
    def _X(self, name: str, view: str):
        p = self.pools[name]
        if name in ("typo15",):
            if view == "tfidf":
                return self.vec.transform(list(p.text))
            return embed.cached(view, f"pool_{name}", list(p.text))
        return self.views[view][p.index.values]

    def out(self, name: str) -> dict:
        """All fast-tier outputs for a pool."""
        if name in self._out:
            return self._out[name]
        p = self.pools[name]
        o = {"y": p.label.values, "text": list(p.text), "group": p.group.values}
        for m in FAST:
            fm = self.models[m]
            X = self._X(name, fm.view_name)
            logits = fm.head.logits(X) if hasattr(fm.head, "logits") else None
            P = fm.head.proba(X) if logits is None else M.softmax(logits, axis=1)
            o[m] = {"pred": fm.classes_[P.argmax(1)], **M.confidence_signals(P, logits)}
        for v in VIEWS:
            o[f"nov/{v}"] = self.nov[v].signals(self._X(name, v))
        lab, votes = self.rules.answer(list(p.text))
        o["rules"] = {"pred": lab, "votes": votes}
        if self.ens is not None:
            X = self._X(name, "bge-small")
            Pe = self.ens.proba(X)
            o["ens"] = {"pred": self.ens.classes_[Pe.argmax(1)], "msp": Pe.max(1), "agree": self.ens.agreement(X)}
        hits = [self.cache_table.get(self.norm(t)) for t in p.text]
        o["cache"] = {"pred": np.array([next(iter(h)) if h and len(h) == 1 else None for h in hits], dtype=object)}
        self._out[name] = o
        return o

    # -- stored real LLM answers ---------------------------------------------------------------
    def llm(self, name: str, model: str = "qwen3-4b", hole=None, texts=None):
        """(labels, p, complete?) for a pool. label None = unparseable reply."""
        hole = hole or holes.clinc_routing_hole()
        store = AnswerStore(model, hole)
        texts = list(self.pools[name].text) if texts is None else list(texts)
        have = np.array([t in store.rows for t in texts])
        lab = np.array([store.rows[t]["label"] if t in store.rows else None for t in texts], dtype=object)
        p = np.array([store.rows[t]["p"] if t in store.rows else np.nan for t in texts])
        return lab, p, have

    def verify(self, name: str, proposals, model: str = "qwen3-4b"):
        """Stored yes/no verdicts on (text, proposed label). True = verifier said yes."""
        h = holes.verify_hole()
        store = AnswerStore(model, h)
        qs = [holes.verify_query(t, pr) for t, pr in zip(self.pools[name].text, proposals)]
        have = np.array([q in store.rows for q in qs])
        yes = np.array([store.rows[q]["label"] == "yes" if q in store.rows else False for q in qs])
        return yes, have
