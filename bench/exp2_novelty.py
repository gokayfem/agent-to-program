"""Experiment 2: can a fast tier tell that an input is unfamiliar?

Part A  CLINC150 (150-way). In-scope test queries vs four kinds of input the fast tier should
        not answer: CLINC's 1000 human-written out-of-scope queries (near), 2000 news headlines
        (far), and seeded bad inputs. Every detector on every view: AUROC, FPR at 95% TPR, and
        the false-accept rate at the threshold the calibration split certifies for 2% risk.
Part B  Keyboard noise on the same test queries (5% and 15% of letters): accuracy, coverage and
        silent errors at the unchanged thresholds.
Part C  Locale shift on MASSIVE (60-way): train on en-US, then the *same test utterances*
        human-translated into de/es/fr/tr/ja. English and multilingual encoders. Thresholds here
        are certified for a 5% budget (nothing can be certified at 2% on this task, see exp1).

"Positive" always means "in-distribution, should be answerable"; scores are higher = more
trustworthy.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from . import data, embed, kit, precompute
from . import models as M
from .common import save_json
from .perturb import seeded_bad_inputs
from wfc import selective as sel

ALPHA, DELTA = 0.02, 0.05
ALPHA_C = 0.05      # MASSIVE (60-way, noisier labels) cannot be certified at 2% with these models; see exp1


def signals(head, nov, X) -> dict:
    logits = head.logits(X)
    P = M.softmax(logits, axis=1)
    s = M.confidence_signals(P, logits)
    s.update(nov.signals(X))
    s["pred"] = head.classes_[P.argmax(1)]
    return s


def part_a() -> dict:
    clinc = data.load_clinc()
    frame = data.fine_frame("clinc150")
    idx = {s: frame.index[frame.split == s].values for s in ("train", "cal", "test")}
    y = {s: frame.loc[idx[s], "label"].values for s in idx}
    oos = clinc[clinc.is_oos]
    oos_test, oos_cal = oos.index[oos.part == "test"].values, oos.index[oos.part != "test"].values
    news = data.load_news()
    news_idx = news.sample(2000, random_state=3).index.values
    bad = seeded_bad_inputs(0)
    bad_all = [t for v in bad.values() for t in v]
    bad_slices, a = {}, 0
    for k, v in bad.items():
        bad_slices[k] = slice(a, a + len(v))
        a += len(v)

    texts = list(clinc.text)
    S = {}          # view -> part -> signals
    for view in ("tfidf", "minilm", "bge-small", "bge-base"):
        if view == "tfidf":
            X, vec = M.tfidf_view(texts, idx["train"], "char_wb")
            Xn = vec.transform(list(news.text[news_idx]))
            Xb = vec.transform(bad_all)
        else:
            X = precompute.get(view, "clinc")
            Xn = (precompute.news_emb(view)[news_idx] if view != "bge-base"
                  else embed.cached(view, "news_sample2000", list(news.text[news_idx])))
            Xb = embed.cached(view, "seeded_bad", bad_all)
        head = M.LRHead(C=20.0).fit(X[idx["train"]], y["train"])
        nov = M.Novelty(X[idx["train"]], y["train"])
        S[view] = {"cal": signals(head, nov, X[idx["cal"]]), "test": signals(head, nov, X[idx["test"]]),
                   "oos_test": signals(head, nov, X[oos_test]), "oos_cal": signals(head, nov, X[oos_cal]),
                   "news": signals(head, nov, Xn), "bad": signals(head, nov, Xb)}
        if view == "bge-small":
            ens = M.MLPEnsemble().fit(X[idx["train"]], y["train"])
            for part, XX in (("cal", X[idx["cal"]]), ("test", X[idx["test"]]), ("oos_test", X[oos_test]),
                             ("oos_cal", X[oos_cal]), ("news", Xn), ("bad", Xb)):
                P = ens.proba(XX)
                S[view][part]["ens_msp"] = P.max(1)
                S[view][part]["ens_agree"] = ens.agreement(XX) + 1e-3 * P.max(1)
                S[view][part]["ens_pred"] = ens.classes_[P.argmax(1)]
        print(f"  A: view {view} done", flush=True)

    # cross-view scores, attached to the bge-small view
    parts = ("cal", "test", "oos_test", "oos_cal", "news", "bad")
    for p in parts:
        a_, b_ = S["tfidf"][p], S["bge-small"][p]
        agree = a_["pred"] == b_["pred"]
        b_["agree_tfidf"] = np.where(agree, np.minimum(a_["msp"], b_["msp"]), -1.0)
        b_["msp_x_knn1"] = b_["msp"] * np.clip(b_["knn1"], 0, 1)
        b_["min(msp,knn1)"] = np.minimum(b_["msp"], b_["knn1"])

    # learned trust model: is this safe to answer? fitted on calibration rows + real OOS (val+train)
    feats = ("msp", "margin", "energy", "knn1", "knn5", "maha", "rel_maha", "centroid")

    def F(p):
        b_, t_ = S["bge-small"][p], S["tfidf"][p]
        return np.c_[[b_[f] for f in feats] + [t_["msp"], t_["knn1"], (b_["pred"] == t_["pred"]).astype(float)]].T
    c_cal = S["bge-small"]["cal"]["pred"] == y["cal"]
    Xl = np.r_[F("cal"), F("oos_cal")]
    yl = np.r_[c_cal, np.zeros(len(oos_cal), bool)]
    mu, sd = Xl.mean(0), Xl.std(0) + 1e-9
    half = np.random.default_rng(0).permutation(len(Xl))
    fit_i, thr_i = half[: len(half) // 2], half[len(half) // 2:]
    trust = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced").fit((Xl[fit_i] - mu) / sd, yl[fit_i])
    for p in parts:
        S["bge-small"][p]["learned_trust"] = trust.decision_function((F(p) - mu) / sd)

    out = {"detectors": {}, "n": {"in_scope_test": int(len(idx["test"])), "oos_test": int(len(oos_test)),
                                  "news": int(len(news_idx)), "seeded_bad": len(bad_all)}}
    for view in S:
        c_te = S[view]["test"]["pred"] == y["test"]
        c_ca = S[view]["cal"]["pred"] == y["cal"]
        for det in [k for k in S[view]["test"] if k not in ("pred", "ens_pred", "P")]:
            pos = S[view]["test"][det]
            row = {"auroc_oos": M.auroc(pos, S[view]["oos_test"][det]),
                   "fpr95_oos": M.fpr_at_tpr(pos, S[view]["oos_test"][det]),
                   "auroc_news": M.auroc(pos, S[view]["news"][det]),
                   "fpr95_news": M.fpr_at_tpr(pos, S[view]["news"][det]),
                   "auroc_error": M.auroc(pos[c_te], pos[~c_te]) if (~c_te).any() else None}
            # Operational, calibration R: in-scope calibration rows only.
            if det == "learned_trust":
                sc, cc = S[view]["cal"][det][thr_i[thr_i < len(c_ca)]], c_ca[thr_i[thr_i < len(c_ca)]]
                oc = S[view]["oos_cal"][det][thr_i[thr_i >= len(c_ca)] - len(c_ca)]
            else:
                sc, cc, oc = S[view]["cal"][det], c_ca, S[view]["oos_cal"][det]
            thr_r = sel.pick_fixed_sequence(sc, cc, ALPHA, DELTA)
            # Calibration G: the same rows plus real OOS queries, which count as errors if answered.
            thr_g = sel.pick_fixed_sequence(np.r_[sc, oc], np.r_[cc, np.zeros(len(oc), bool)], ALPHA, DELTA)
            for tag, thr in (("R", thr_r), ("G", thr_g)):
                ev = sel.evaluate(pos, c_te, thr)
                fa_oos = float((S[view]["oos_test"][det] >= thr).mean())
                n_ans = ev["answered"] + int((S[view]["oos_test"][det] >= thr).sum())
                n_err = ev["silent_errors"] + int((S[view]["oos_test"][det] >= thr).sum())
                row[f"op_{tag}"] = {"in_scope_coverage": ev["coverage"], "in_scope_silent": ev["silent_error_rate"],
                                    "oos_false_accept": fa_oos,
                                    "news_false_accept": float((S[view]["news"][det] >= thr).mean()),
                                    "bad_false_accept": {k: float((S[view]["bad"][det][sl] >= thr).mean())
                                                         for k, sl in bad_slices.items()},
                                    "mixed_silent_rate": n_err / max(1, n_ans),
                                    "never": bool(thr == sel.NEVER)}
            out["detectors"][f"{view}/{det}"] = row
    for k, r in sorted(out["detectors"].items(), key=lambda kv: -kv[1]["auroc_oos"])[:12]:
        print(f"  A: {k:28s} auroc_oos={r['auroc_oos']:.3f} fpr95={r['fpr95_oos']:.3f} auroc_news={r['auroc_news']:.3f} "
              f"| R: cov={r['op_R']['in_scope_coverage']:.3f} oosFA={r['op_R']['oos_false_accept']:.3f} "
              f"| G: cov={r['op_G']['in_scope_coverage']:.3f} oosFA={r['op_G']['oos_false_accept']:.3f}", flush=True)

    # Part B: keyboard noise on the test partition (same rows, same thresholds)
    test_rows = clinc[(clinc.part == "test") & (~clinc.is_oos)]
    assert (test_rows.index.values == idx["test"]).all()
    noise = {}
    for view in ("tfidf", "minilm", "bge-small", "bge-base"):
        if view == "tfidf":
            X, vec = M.tfidf_view(texts, idx["train"], "char_wb")
        else:
            X = precompute.get(view, "clinc")
        head = M.LRHead(C=20.0).fit(X[idx["train"]], y["train"])
        nov = M.Novelty(X[idx["train"]], y["train"], maha=False)
        cal = signals(head, nov, X[idx["cal"]])
        c_ca = cal["pred"] == y["cal"]
        row = {}
        for level, tag in ((0, None), (5, "clinc_test_typo05"), (15, "clinc_test_typo15")):
            if tag is None:
                Xq = X[idx["test"]]
            elif view == "tfidf":
                Xq = vec.transform(precompute.corpora()[tag])
            else:
                Xq = precompute.get(view, tag)
            s = signals(head, nov, Xq)
            c = s["pred"] == y["test"]
            r = {"accuracy": float(c.mean())}
            for det in ("msp", "knn1", "min(msp,knn1)"):
                sc_cal = np.minimum(cal["msp"], cal["knn1"]) if det.startswith("min") else cal[det]
                sc = np.minimum(s["msp"], s["knn1"]) if det.startswith("min") else s[det]
                thr = sel.pick_fixed_sequence(sc_cal, c_ca, ALPHA, DELTA)
                ev = sel.evaluate(sc, c, thr)
                r[det] = {"coverage": ev["coverage"], "silent_error_rate": ev["silent_error_rate"],
                          "silent_errors": ev["silent_errors"], "answered": ev["answered"]}
            row[f"typo{level:02d}"] = r
        noise[view] = row
        print(f"  B: {view:10s} " + "  ".join(
            f"typo{lv:02d}: acc={row[f'typo{lv:02d}']['accuracy']:.3f} cov={row[f'typo{lv:02d}']['msp']['coverage']:.3f} "
            f"err={row[f'typo{lv:02d}']['msp']['silent_error_rate']:.4f}" for lv in (0, 5, 15)), flush=True)
    out["typo_noise"] = noise
    return out


def part_c() -> dict:
    en = data.fine_frame("massive60")
    idx = {s: en.index[en.split == s].values for s in ("train", "cal", "test")}
    y = {s: en.loc[idx[s], "label"].values for s in idx}
    out = {}
    for view in ("tfidf", "minilm", "bge-small", "me5-small"):
        if view == "tfidf":
            X, vec = M.tfidf_view(list(en.text), idx["train"], "char_wb")
        else:
            X = precompute.get(view, "massive_en-US")
        head = M.LRHead(C=20.0).fit(X[idx["train"]], y["train"])
        nov = M.Novelty(X[idx["train"]], y["train"])
        cal = signals(head, nov, X[idx["cal"]])
        c_ca = cal["pred"] == y["cal"]
        base = signals(head, nov, X[idx["test"]])
        thr = {d: sel.pick_fixed_sequence(cal[d], c_ca, ALPHA_C, DELTA) for d in ("msp", "knn1")}
        thr["min(msp,knn1)"] = sel.pick_fixed_sequence(np.minimum(cal["msp"], cal["knn1"]), c_ca, ALPHA_C, DELTA)
        row = {}
        for loc in data.MASSIVE_LOCALES:
            d = data.load_massive(loc)
            assert (d.mid.values == data.load_massive("en-US").mid.values).all()
            Xq = (vec.transform(list(d.text)) if view == "tfidf" else precompute.get(view, f"massive_{loc}"))[idx["test"]]
            s = signals(head, nov, Xq)
            c = s["pred"] == y["test"]
            r = {"accuracy": float(c.mean()),
                 "confident_wrong": float(((s["msp"] >= 0.9) & ~c).mean())}
            for det in ("msp", "knn1", "min(msp,knn1)"):
                sc = np.minimum(s["msp"], s["knn1"]) if det.startswith("min") else s[det]
                ev = sel.evaluate(sc, c, thr[det])
                r[det] = {"coverage": ev["coverage"], "silent_error_rate": ev["silent_error_rate"],
                          "silent_errors": ev["silent_errors"], "answered": ev["answered"]}
            if loc != "en-US":
                for det in ("msp", "energy", "knn1", "knn5") + (("maha", "rel_maha") if view != "tfidf" else ()):
                    r[f"auroc_locale/{det}"] = M.auroc(base[det], s[det])
            row[loc] = r
        out[view] = row
        print(f"  C: {view:10s} " + "  ".join(
            f"{loc}: acc={row[loc]['accuracy']:.3f} cov={row[loc]['msp']['coverage']:.2f} err={row[loc]['msp']['silent_error_rate']:.3f}"
            for loc in data.MASSIVE_LOCALES), flush=True)
    return out


if __name__ == "__main__":
    out = part_a()
    out["locale_shift"] = part_c()
    out["alpha"], out["delta"], out["alpha_locale"] = ALPHA, DELTA, ALPHA_C
    save_json("exp2_novelty.json", out)
