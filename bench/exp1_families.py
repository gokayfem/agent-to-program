"""Experiment 1: which decision-model family should take over a typed judgment?

Three public intent datasets with human labels, official train / calibration / test
partitions (nothing fitted or thresholded on test). Eleven (view, head) families plus a
mined rule committee and a two-view committee. For each:

  accuracy                 answer-everything accuracy on test
  AURC                     area under the risk-coverage curve using the model's own confidence
  oracle coverage @ risk   best possible coverage at <=1/2/5% silent errors (threshold peeked on
                           test; descriptive upper bound only)
  operational @ 2%         threshold certified on the calibration split by fixed-sequence testing
                           (alpha=2%, delta=5%), then coverage and silent-error rate on test
  fit time                 seconds to train the head (embeddings are precomputed)

Latency is measured separately (exp_latency) on a quiet machine.
"""
from __future__ import annotations

import sys

import numpy as np

from . import data, kit
from . import models as M
from .common import save_json
from wfc import selective as sel

TAGS = {"clinc150": "clinc", "banking77": "banking", "massive60": "massive_en-US"}
VIEWS = ("tfidf", "tfidf-word", "potion", "minilm", "bge-small", "bge-base")


def run(name: str) -> dict:
    frame = data.fine_frame(name)
    idx = {s: frame.index[frame.split == s].values for s in ("train", "cal", "test")}
    y = {s: frame.loc[idx[s], "label"].values for s in idx}
    texts_all = data.load_clinc().text if name == "clinc150" else \
        (data.load_banking().text if name == "banking77" else data.load_massive("en-US").text)
    views = kit.build_views(TAGS[name], texts_all, idx["train"], VIEWS)
    res, keep = {}, {}
    for fam in kit.FAMILIES:
        f = kit.fit_family(fam, views, idx["train"], y["train"])
        cal, te = f.run(idx["cal"]), f.run(idx["test"])
        c_cal, c_te = cal["pred"] == y["cal"], te["pred"] == y["test"]
        score = "knn_sim" if fam.endswith("1NN") else "msp"   # a 1-NN cache is trusted by similarity
        r = {"accuracy": float(c_te.mean()), "fit_s": round(f.fit_s, 2), "view": f.view_name,
             **kit.selective_summary(te[score], c_te, cal[score], c_cal)}
        res[fam] = r
        keep[fam] = (cal, te)
        print(f"  {name:10s} {fam:18s} acc={r['accuracy']:.4f} aurc={r['aurc']:.4f} "
              f"cov@2%={r['fs@0.02']['coverage']:.3f} err={r['fs@0.02']['silent_error_rate']:.4f} fit={r['fit_s']}s",
              flush=True)

    # Mined rule committee: abstains by construction, so report it as a selective classifier.
    tr_texts = [texts_all[i] for i in idx["train"]]
    te_texts = [texts_all[i] for i in idx["test"]]
    cal_texts = [texts_all[i] for i in idx["cal"]]
    for prec, sup in ((1.0, 5), (1.0, 10), (0.95, 10)):
        rules = M.MinedRules(min_support=sup, min_precision=prec).fit(tr_texts, y["train"])
        lab, votes = rules.answer(te_texts)
        ans = lab != None  # noqa: E711
        wrong = ans & (lab != y["test"])
        lab2, votes2 = rules.answer(te_texts)
        r = {"n_rules": rules.n_rules, "coverage": float(ans.mean()),
             "silent_error_rate": float(wrong.sum() / max(1, ans.sum())), "silent_errors": int(wrong.sum())}
        # require >= 2 agreeing rules
        ans2 = votes >= 2
        wrong2 = ans2 & (lab != y["test"])
        r["coverage_2votes"] = float(ans2.mean())
        r["silent_error_rate_2votes"] = float(wrong2.sum() / max(1, ans2.sum()))
        res[f"mined-rules(p>={prec},n>={sup})"] = r
        print(f"  {name:10s} rules p>={prec}: n_rules={rules.n_rules} cov={r['coverage']:.3f} "
              f"err={r['silent_error_rate']:.4f} | 2 votes cov={r['coverage_2votes']:.3f} err={r['silent_error_rate_2votes']:.4f}",
              flush=True)

    # Two-view committee: lexical (tfidf-char) and semantic (bge-small) LR must agree; score = min confidence.
    def committee(a, b, part, yy):
        ca, cb = keep[a][part], keep[b][part]
        agree = ca["pred"] == cb["pred"]
        score = np.where(agree, np.minimum(ca["msp"], cb["msp"]), -1.0)
        return score, cb["pred"] == yy
    for a, b in (("tfidf-char+LR", "bge-small+LR"), ("minilm+LR", "bge-small+LR")):
        s_cal, c_cal = committee(a, b, 0, y["cal"])
        s_te, c_te = committee(a, b, 1, y["test"])
        r = {"accuracy": float(c_te.mean()), **kit.selective_summary(s_te, c_te, s_cal, c_cal)}
        res[f"agree({a} & {b})"] = r
        print(f"  {name:10s} agree({a},{b}) aurc={r['aurc']:.4f} cov@2%={r['fs@0.02']['coverage']:.3f} "
              f"err={r['fs@0.02']['silent_error_rate']:.4f}", flush=True)

    # Exact-match cache: how often does a test query repeat a training query verbatim?
    norm = lambda t: " ".join(str(t).lower().split())
    table = {}
    for t, lab in zip(tr_texts, y["train"]):
        table.setdefault(norm(t), set()).add(lab)
    hits = [(norm(t) in table and len(table[norm(t)]) == 1, lab) for t, lab in zip(te_texts, y["test"])]
    n_hit = sum(h for h, _ in hits)
    n_wrong = sum(1 for (h, lab), t in zip(hits, te_texts) if h and next(iter(table[norm(t)])) != lab)
    res["exact-cache"] = {"coverage": n_hit / len(hits), "silent_errors": n_wrong, "answered": n_hit}
    print(f"  {name:10s} exact cache: hit rate {n_hit / len(hits):.4f}, wrong {n_wrong}", flush=True)
    return {"n": {s: int(len(idx[s])) for s in idx}, "classes": int(frame.label.nunique()), "families": res}


if __name__ == "__main__":
    names = sys.argv[1:] or list(TAGS)
    out = {n: run(n) for n in names}
    save_json("exp1_families.json", out)
