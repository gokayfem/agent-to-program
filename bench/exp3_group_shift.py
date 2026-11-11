"""Experiment 3: does a calibrated fast tier stay calibrated on groups it has never seen?

This is the unseen-merchant failure from the original example, re-run on human-written text
and repeated over many random group splits so that "it worked" and "it failed" are rates.

Task        route a request to its coarse class (CLINC: 10 domains; MASSIVE: 18 scenarios).
Groups      intents. In each split a fifth of the intents of every class are held out for
            testing (never seen by any model or threshold), a fifth are held out for
            calibration, the rest are training groups.
Factors
  model      tfidf-char+LR, minilm+LR, bge-small+LR, bge-small+kNN
  score      msp                 the model's own confidence
             knn1                similarity to the nearest training row (familiarity only)
             msp&knn1            both must clear a threshold (the original skill's gate)
             learned             logistic trust model over confidence, familiarity and
                                 cross-view agreement features
  protocol   R        calibrate on held-out rows of *training* groups (a random split)
             G-mix    calibrate on those rows plus an equal number from held-out groups
             G-worst  two separate tests: selective risk <= alpha on seen-group rows, and
                      false-accept rate (wrong answers / all inputs) <= alpha on held-out-group
                      rows; the stricter threshold wins
  picker     empirical | wilson_scan | fixed_sequence | cluster_bootstrap
  alpha      2% and 5% silent-error budget, delta 5%

Reported per cell, over splits: coverage and silent-error rate on test rows of seen groups and
on test rows of never-seen groups, and how often the silent-error rate exceeded the budget.
"""
from __future__ import annotations

import sys
import time

import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression

from . import data, kit
from . import models as M
from .common import mean_sd, save_json
from wfc import selective as sel

ALPHAS, DELTA = (0.02, 0.05), 0.05
MODELS = ("tfidf-char+LR", "minilm+LR", "bge-small+LR", "bge-small+kNN")
FLOOR_Q = (0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
PARTS = ("cal_seen", "cal_unseen", "test_seen", "test_unseen")
PICKERS = ("empirical", "wilson_scan", "fixed_sequence", "cluster_bootstrap")


def pick(picker: str, score, correct, groups, alpha: float, delta: float) -> float:
    if picker == "cluster_bootstrap":
        return sel.pick_cluster_bootstrap(score, correct, groups, alpha, delta)
    return sel.PICKERS[picker](score, correct, alpha, delta)


def one_split(dataset: str, seed: int, **split_kw) -> dict:
    t0 = time.perf_counter()
    if dataset == "clinc":
        base = data.load_clinc()
        frame = data.routing_frame(base[~base.is_oos], "domain", "intent", seed, **split_kw)
        tag, texts = "clinc", list(base.text)
    else:
        base = data.load_massive("en-US")
        frame = data.routing_frame(base, "scenario", "intent", seed)
        tag, texts = "massive_en-US", list(base.text)
    idx = {s: frame.index[frame.split == s].values for s in ("train",) + PARTS}
    y = {s: frame.loc[idx[s], "label"].values for s in idx}
    g = {s: frame.loc[idx[s], "group"].values for s in idx}
    views = kit.build_views(tag, texts, idx["train"], ("tfidf", "minilm", "bge-small"))
    nov = {v: M.Novelty(X[idx["train"]], y["train"], maha=False) for v, X in views.items()}
    fitted = {m: kit.fit_family(m, views, idx["train"], y["train"]) for m in MODELS}
    out = {p: {m: fitted[m].run(idx[p]) for m in MODELS} for p in PARTS}
    for p in PARTS:
        for v in views:
            s = nov[v].signals(views[v][idx[p]])
            for m in MODELS:
                if fitted[m].view_name == v:
                    out[p][m]["knn1"], out[p][m]["knn5"] = s["knn1"], s["knn5"]
            out[p][f"nov/{v}"] = s

    rng = np.random.default_rng(seed)
    n_seen = len(idx["cal_seen"])
    sub_unseen = rng.choice(len(idx["cal_unseen"]), size=min(n_seen, len(idx["cal_unseen"])), replace=False)
    res = {"seed": seed, "n": {s: int(len(idx[s])) for s in idx},
           "groups": {r: int((frame.groupby("group").role.first() == r).sum()) for r in ("train", "cal", "test")},
           "models": {}}

    for m in MODELS:
        partner = "tfidf-char+LR" if fitted[m].view_name != "tfidf" else "bge-small+LR"
        correct = {p: out[p][m]["pred"] == y[p] for p in PARTS}
        base_stats = {"acc_seen": float(correct["test_seen"].mean()), "acc_unseen": float(correct["test_unseen"].mean()),
                      "unseen_errors_conf>=0.9": float((out["test_unseen"][m]["msp"][~correct["test_unseen"]] >= 0.9).mean())
                      if (~correct["test_unseen"]).any() else 0.0,
                      "unseen_conf>=0.9_share": float((out["test_unseen"][m]["msp"] >= 0.9).mean()),
                      "unseen_err_among_conf>=0.9": float((~correct["test_unseen"])[out["test_unseen"][m]["msp"] >= 0.9].mean())
                      if (out["test_unseen"][m]["msp"] >= 0.9).any() else 0.0}

        def feats(p):
            o = out[p][m]
            other_view = "tfidf" if fitted[m].view_name != "tfidf" else "bge-small"
            return np.c_[o["msp"], o["margin"], o["neg_entropy"], o["knn1"], o["knn5"],
                         out[p][f"nov/{other_view}"]["knn1"], out[p][partner]["msp"],
                         (o["pred"] == out[p][partner]["pred"]).astype(float)]

        # learned trust scores, one per protocol family (R: seen rows only; G: seen + held-out rows)
        halves = {}
        for p in ("cal_seen", "cal_unseen"):
            perm = rng.permutation(len(idx[p]))
            halves[p] = (perm[: len(perm) // 2], perm[len(perm) // 2:])
        learned = {}
        for proto, fit_parts in (("R", ("cal_seen",)), ("G", ("cal_seen", "cal_unseen"))):
            Xf = np.vstack([feats(p)[halves[p][0]] for p in fit_parts])
            yf = np.concatenate([correct[p][halves[p][0]] for p in fit_parts])
            mu, sd = Xf.mean(0), Xf.std(0) + 1e-9
            if yf.all() or (~yf).all():
                learned[proto] = {p: out[p][m]["msp"] for p in PARTS}
            else:
                clf = LogisticRegression(C=1.0, max_iter=2000).fit((Xf - mu) / sd, yf)
                learned[proto] = {p: clf.decision_function((feats(p) - mu) / sd) for p in PARTS}

        cells = {}
        for score in ("msp", "knn1", "msp&knn1", "learned"):
            for proto in ("R", "G-mix", "G-worst"):
                # calibration material for this (score, protocol)
                def cal(p, key):
                    if score == "learned":
                        h = halves[p][1]
                        return learned["R" if proto == "R" else "G"][p][h] if key == "s" else \
                            (correct[p][h] if key == "c" else g[p][h])
                    arr = {"s": None, "c": correct[p], "g": g[p]}[key]
                    return arr
                def sc(p, name):
                    if score == "learned":
                        return learned["R" if proto == "R" else "G"][p]
                    return out[p][m][name]
                for picker in PICKERS:
                    for alpha in ALPHAS:
                        if score == "msp&knn1":
                            best = (-1, sel.NEVER, sel.NEVER)
                            d = DELTA / len(FLOOR_Q) if picker in ("fixed_sequence", "cluster_bootstrap") else DELTA
                            ms, ks = out["cal_seen"][m]["msp"], out["cal_seen"][m]["knn1"]
                            mu_, ku_ = out["cal_unseen"][m]["msp"][sub_unseen], out["cal_unseen"][m]["knn1"][sub_unseen]
                            cs, cu = correct["cal_seen"], correct["cal_unseen"][sub_unseen]
                            gs, gu = g["cal_seen"], g["cal_unseen"][sub_unseen]
                            pool_k = ks if proto == "R" else np.r_[ks, ku_]
                            for q in FLOOR_Q:
                                floor = float(np.quantile(pool_k, q))
                                if proto == "R":
                                    k = ks >= floor
                                    thr = pick(picker, ms[k], cs[k], gs[k], alpha, d)
                                    kept = int((k & (ms >= thr)).sum())
                                elif proto == "G-mix":
                                    mm, kk, cc, gg = np.r_[ms, mu_], np.r_[ks, ku_], np.r_[cs, cu], np.r_[gs, gu]
                                    k = kk >= floor
                                    thr = pick(picker, mm[k], cc[k], gg[k], alpha, d)
                                    kept = int((k & (mm >= thr)).sum())
                                else:
                                    k1, k2 = ks >= floor, ku_ >= floor
                                    thr = max(pick(picker, ms[k1], cs[k1], gs[k1], alpha, d / 2),
                                              sel.pick_far(np.where(k2, mu_, -np.inf), cu, alpha, d / 2, picker, gu))
                                    kept = int((k1 & (ms >= thr)).sum() + (k2 & (mu_ >= thr)).sum())
                                if kept > best[0]:
                                    best = (kept, floor, thr)
                            _, floor, thr = best
                            accept = {p: (out[p][m]["knn1"] >= floor) & (out[p][m]["msp"] >= thr)
                                      for p in ("test_seen", "test_unseen")}
                            never = thr == sel.NEVER
                        else:
                            if score == "learned":
                                s_seen, c_seen, g_seen = cal("cal_seen", "s"), cal("cal_seen", "c"), cal("cal_seen", "g")
                                s_un, c_un, g_un = cal("cal_unseen", "s"), cal("cal_unseen", "c"), cal("cal_unseen", "g")
                                take = rng.choice(len(s_un), size=min(len(s_seen), len(s_un)), replace=False)
                                s_un, c_un, g_un = s_un[take], c_un[take], g_un[take]
                            else:
                                s_seen, c_seen, g_seen = out["cal_seen"][m][score], correct["cal_seen"], g["cal_seen"]
                                s_un = out["cal_unseen"][m][score][sub_unseen]
                                c_un, g_un = correct["cal_unseen"][sub_unseen], g["cal_unseen"][sub_unseen]
                            if proto == "R":
                                thr = pick(picker, s_seen, c_seen, g_seen, alpha, DELTA)
                            elif proto == "G-mix":
                                thr = pick(picker, np.r_[s_seen, s_un], np.r_[c_seen, c_un], np.r_[g_seen, g_un], alpha, DELTA)
                            else:
                                thr = max(pick(picker, s_seen, c_seen, g_seen, alpha, DELTA / 2),
                                          sel.pick_far(s_un, c_un, alpha, DELTA / 2, picker, g_un))
                            accept = {p: sc(p, score) >= thr for p in ("test_seen", "test_unseen")}
                            never = thr == sel.NEVER
                        cell = {"never": bool(never)}
                        for p in ("test_seen", "test_unseen"):
                            a = accept[p]
                            n_a, n_e = int(a.sum()), int((a & ~correct[p]).sum())
                            cell[p] = {"coverage": n_a / len(a), "silent": (n_e / n_a) if n_a else 0.0,
                                       "far": n_e / len(a), "n_answered": n_a, "n_silent": n_e, "n": int(len(a))}
                        cells[f"{score}|{proto}|{picker}|{alpha}"] = cell
        res["models"][m] = {"base": base_stats, "cells": cells}
    res["seconds"] = round(time.perf_counter() - t0, 1)
    return res


def aggregate(runs: list) -> dict:
    agg = {"splits": len(runs), "n": runs[0]["n"], "groups": runs[0]["groups"], "models": {}}
    for m in MODELS:
        base = {k: mean_sd([r["models"][m]["base"][k] for r in runs]) for k in runs[0]["models"][m]["base"]}
        cells = {}
        for key in runs[0]["models"][m]["cells"]:
            alpha = float(key.split("|")[-1])
            c = [r["models"][m]["cells"][key] for r in runs]
            row = {"never_rate": float(np.mean([x["never"] for x in c]))}
            for p in ("test_seen", "test_unseen"):
                cov = [x[p]["coverage"] for x in c]
                sil = [x[p]["silent"] for x in c]
                far = [x[p]["far"] for x in c]
                tot_a, tot_e = sum(x[p]["n_answered"] for x in c), sum(x[p]["n_silent"] for x in c)
                row[p] = {"coverage_mean": float(np.mean(cov)), "coverage_sd": float(np.std(cov, ddof=1)),
                          "silent_mean": float(np.mean(sil)), "silent_max": float(np.max(sil)),
                          "silent_pooled": tot_e / max(1, tot_a),
                          "far_mean": float(np.mean(far)), "far_max": float(np.max(far)),
                          "far_violation_rate": float(np.mean([f > alpha for f in far])),
                          "violation_rate": float(np.mean([s > alpha for s in sil])),
                          "violation_2x_rate": float(np.mean([s > 2 * alpha for s in sil]))}
            cells[key] = row
        agg["models"][m] = {"base": base, "cells": cells}
    return agg


def show(name: str, agg: dict, alpha: float = 0.02) -> None:
    print(f"\n== {name}: {agg['splits']} splits, groups {agg['groups']}, rows {agg['n']}")
    for m in MODELS:
        b = agg["models"][m]["base"]
        print(f" {m}: acc seen {b['acc_seen']['mean']:.3f}  unseen {b['acc_unseen']['mean']:.3f}  | of unseen-group "
              f"errors, {b['unseen_errors_conf>=0.9']['mean']:.0%} had confidence >= 0.9")
    m = "bge-small+LR"
    print(f" cells for {m}, alpha={alpha} (coverage / silent-error mean / violation rate over splits)")
    for score in ("msp", "knn1", "msp&knn1", "learned"):
        for proto in ("R", "G-mix", "G-worst"):
            for picker in ("fixed_sequence", "cluster_bootstrap"):
                c = agg["models"][m]["cells"][f"{score}|{proto}|{picker}|{alpha}"]
                s, u = c["test_seen"], c["test_unseen"]
                print(f"  {score:9s} {proto:8s} {picker:17s} seen: cov {s['coverage_mean']:.3f} err {s['silent_mean']:.4f} "
                      f"viol {s['violation_rate']:.2f} | unseen: cov {u['coverage_mean']:.3f} err {u['silent_mean']:.4f} "
                      f"viol {u['violation_rate']:.2f} FAR {u['far_mean']:.4f} farviol {u['far_violation_rate']:.2f} "
                      f"max {u['far_max']:.3f} | never {c['never_rate']:.2f}")


if __name__ == "__main__":
    n_splits = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    out = {}
    for ds in ("clinc", "massive"):
        runs = Parallel(n_jobs=5)(delayed(one_split)(ds, s) for s in range(n_splits))
        out[ds] = aggregate(runs)
        out[ds]["per_split_seconds"] = [r["seconds"] for r in runs]
        show(ds, out[ds])
        show(ds, out[ds], 0.05)
    out["alphas"], out["delta"] = list(ALPHAS), DELTA
    save_json("exp3_group_shift.json", out)
