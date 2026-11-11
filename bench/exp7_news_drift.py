"""Experiment 7: natural drift. Nothing here is scripted.

Data     HuffPost headlines with editor-assigned section and publication date, in date order,
         2012-01 to 2018-05 (the dense part of the corpus; ~190k headlines, 77 months).
Task     headline -> section. Sections come and go over the years (a section the compiled
         program has never seen is the purest form of "unfamiliar input"), topics and names
         change, and the mix of sections shifts strongly.
Compile  on the first 12 months only. Fast tier: bge-small embedding + logistic regression,
         answering when its confidence clears a threshold certified for a 10% silent-error
         budget (this is a noisy 30-way editorial labelling; 10% is the budget, not a claim).
Oracle   the editor's label, standing in for the agent. There is no LLM in this experiment.
Policies
  static-R         threshold from a random split of the first 12 months; never updated
  static-T         threshold from the last 2 of those 12 months (temporal hold-out); never updated
  T + breaker      static-T plus 10% shadow sampling; a breaker stops the fast tier when the
                   shadow estimate's lower bound exceeds the budget
  gated-quarterly  every 3 months: retrain on the last 12 months of labels the system actually
                   has (escalations + shadow samples), re-certify the threshold on the newest
                   labelled month, deploy only if something can be certified
  self-train       every 3 months: retrain on its own answers, keep the original threshold
"""
from __future__ import annotations

import hashlib
from collections import deque

import numpy as np
import pandas as pd

from . import data, precompute
from . import models as M
from .common import save_json
from wfc import selective as sel
from wfc.nsp import wilson

ALPHA, DELTA, SHADOW = 0.10, 0.05, 0.10
ENC = "bge-small"


def _u(key: str) -> float:
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big") / 2 ** 64


def fit(X, y):
    return M.LRHead(C=5.0, max_iter=200).fit(X, y)


def score(head, X):
    P = head.proba(X)
    return head.classes_[P.argmax(1)], P.max(1)


def run() -> dict:
    news = data.load_news()
    E = precompute.news_emb(ENC)
    news = news[news.date < "2018-06-01"]
    E = E[news.index.values]
    news = news.reset_index(drop=True)
    month = news.date.dt.to_period("M")
    months = sorted(month.unique())
    mi = month.map({m: i for i, m in enumerate(months)}).values
    y = news.label.values
    shadow = np.array([_u("s|" + u) < SHADOW for u in news.uid])

    first = np.flatnonzero(mi < 12)
    rng = np.random.default_rng(0)
    perm = rng.permutation(first)
    cal_R, tr_R = perm[: len(perm) // 6], perm[len(perm) // 6:]
    tr_T, cal_T = np.flatnonzero(mi < 10), np.flatnonzero((mi >= 10) & (mi < 12))
    stream_months = list(range(12, len(months)))

    def certify(head, idx):
        pred, conf = score(head, E[idx])
        return sel.pick_fixed_sequence(conf, pred == y[idx], ALPHA, DELTA)

    head_R = fit(E[tr_R], y[tr_R]); thr_R = certify(head_R, cal_R)
    head_T = fit(E[tr_T], y[tr_T]); thr_T = certify(head_T, cal_T)
    seen_labels = set(np.unique(y[first]))
    out = {"months": [str(m) for m in months], "n": int(len(news)), "compile_months": 12,
           "sections_at_compile": len(seen_labels), "sections_total": int(len(np.unique(y))),
           "alpha": ALPHA, "shadow_rate": SHADOW, "thr_R": thr_R, "thr_T": thr_T,
           "base_accuracy_first_year_holdout": float((score(head_T, E[cal_T])[0] == y[cal_T]).mean()), "policies": {}}

    def monthly(idx, fast, pred):
        n = len(idx)
        wrong = fast & (pred != y[idx])
        sh = fast & shadow[idx]
        k, m = int((sh & (pred != y[idx])).sum()), int(sh.sum())
        return {"n": n, "coverage": float(fast.mean()), "silent": float(wrong.sum() / max(1, fast.sum())),
                "wrong_per_request": float(wrong.mean()), "shadow_n": m, "shadow_est": k / m if m else None,
                "shadow_ci": list(wilson(k, m)) if m else None,
                "new_section_share": float(np.mean([l not in seen_labels for l in y[idx]])),
                "wrong_on_new_sections": float(np.sum(wrong & np.array([l not in seen_labels for l in y[idx]])) / max(1, n))}

    def summarize(rows):
        N = sum(r["n"] for r in rows)
        cov = sum(r["coverage"] * r["n"] for r in rows) / N
        wr = sum(r["wrong_per_request"] * r["n"] for r in rows) / N
        return {"coverage": cov, "wrong_per_request": wr, "silent_among_fast": wr / cov if cov else 0.0,
                "months_over_budget": int(sum(r["silent"] > ALPHA for r in rows if r["coverage"] > 0)),
                "months_with_fast_tier": int(sum(r["coverage"] > 0 for r in rows)),
                "worst_month_silent": float(max(r["silent"] for r in rows))}

    # static policies
    for name, head, thr in (("static-R", head_R, thr_R), ("static-T", head_T, thr_T)):
        rows = []
        for m in stream_months:
            idx = np.flatnonzero(mi == m)
            pred, conf = score(head, E[idx])
            rows.append(monthly(idx, conf >= thr, pred))
        out["policies"][name] = {"monthly": rows, "summary": summarize(rows)}

    # static-T with a shadow breaker
    rows, win, tripped = [], deque(maxlen=400), None
    for m in stream_months:
        idx = np.flatnonzero(mi == m)
        pred, conf = score(head_T, E[idx])
        fast = (conf >= thr_T) & (tripped is None)
        rows.append(monthly(idx, fast, pred))
        for d in (pred != y[idx])[fast & shadow[idx]]:
            win.append(bool(d))
        if tripped is None and len(win) >= 200 and wilson(int(sum(win)), len(win))[0] > ALPHA:
            tripped = str(months[m])
    out["policies"]["T + breaker"] = {"monthly": rows, "summary": summarize(rows), "breaker_tripped_after": tripped}

    # gated quarterly recompilation and ungated self-training
    for name in ("gated-quarterly", "self-train"):
        head, thr = head_T, thr_T
        have_idx, have_lab = list(np.r_[tr_T, cal_T]), list(y[np.r_[tr_T, cal_T]])     # labels the system holds
        rows, events = [], []
        for q, m in enumerate(stream_months):
            idx = np.flatnonzero(mi == m)
            pred, conf = score(head, E[idx])
            fast = conf >= thr
            rows.append(monthly(idx, fast, pred))
            if name == "gated-quarterly":
                lab_mask = ~fast | shadow[idx]                       # escalations (agent label) + shadow samples
                have_idx += list(idx[lab_mask]); have_lab += list(y[idx][lab_mask])
            else:
                have_idx += list(idx); have_lab += list(np.where(fast, pred, y[idx]))   # its own answers as labels
            if (q + 1) % 3 == 0:
                hi, hl = np.array(have_idx), np.array(have_lab, dtype=object)
                recent = mi[hi] > m - 12
                if name == "gated-quarterly":
                    hold = recent & (mi[hi] == m)
                    cand = fit(E[hi[recent & ~hold]], hl[recent & ~hold])
                    p2, c2 = score(cand, E[hi[hold]])
                    cthr = sel.pick_fixed_sequence(c2, p2 == hl[hold], ALPHA, DELTA)
                    admitted = cthr != sel.NEVER
                    events.append({"month": str(months[m]), "admitted": bool(admitted), "labels": int((recent & ~hold).sum()),
                                   "holdout": int(hold.sum())})
                    if admitted:
                        head, thr = cand, cthr
                    else:
                        thr = sel.NEVER                               # nothing certified: fast tier off until next compile
                else:
                    head = fit(E[hi[recent]], hl[recent])
                    events.append({"month": str(months[m]), "labels": int(recent.sum())})
        out["policies"][name] = {"monthly": rows, "summary": summarize(rows), "events": events}

    for k, v in out["policies"].items():
        s = v["summary"]
        print(f"  {k:16s} coverage {s['coverage']:.3f}  silent among fast {s['silent_among_fast']:.3f}  wrong/request {s['wrong_per_request']:.4f} "
              f" months over budget {s['months_over_budget']}/{s['months_with_fast_tier']}  worst month {s['worst_month_silent']:.3f}"
              + (f"  breaker tripped after {v['breaker_tripped_after']}" if "breaker_tripped_after" in v else ""), flush=True)
    yr = pd.Series([str(months[m])[:4] for m in stream_months])
    for k in ("static-R", "static-T", "gated-quarterly", "self-train"):
        rows = out["policies"][k]["monthly"]
        by = {}
        for yv in sorted(yr.unique()):
            rr = [r for r, t in zip(rows, yr) if t == yv]
            s = summarize(rr)
            by[yv] = {"coverage": s["coverage"], "silent_among_fast": s["silent_among_fast"],
                      "new_section_share": float(np.mean([r["new_section_share"] for r in rr]))}
        out["policies"][k]["by_year"] = by
        print(f"  {k:16s} by year: " + "  ".join(f"{yv}: cov {b['coverage']:.2f} err {b['silent_among_fast']:.3f}" for yv, b in by.items()), flush=True)
    print("  share of traffic from sections unseen at compile time, by year:",
          {yv: round(b["new_section_share"], 3) for yv, b in out["policies"]["static-T"]["by_year"].items()})
    return out


if __name__ == "__main__":
    save_json("exp7_news_drift.json", run())
