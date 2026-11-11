"""Experiment 4: what can verify a fast answer, and what does it cost?

Frozen CLINC routing split. The fast tier under test is bge-small+LR answering *everything*
(no abstention), so the verifier is the only guard. Three populations:

  seen     test rows of training groups            (fast model right ~97%)
  unseen   rows of groups never seen in training   (fast model right ~70%)
  oos      human-written out-of-scope queries      (every fast answer is wrong)

For each verifier: share of the fast tier's errors it rejects (catch rate), share of correct
answers it rejects (false-reject rate), the coverage and silent-error rate that result, and
the compute it adds per verified answer. LLM verifiers are real stored model answers.

Also: an error-correlation table. A verifier helps only where it does not repeat the primary's
mistake, so for every pair we report P(B wrong | A wrong) and P(B gives the same wrong label |
A wrong) next to B's marginal error rate.
"""
from __future__ import annotations

import time

import numpy as np

from . import holes
from .common import load_json, save_json
from .frozen import NONE, Frozen
from wfc import selective as sel

PRIMARY = "bge-small+LR"
POPS = ("test_seen", "test_unseen", "oos_test")


def run() -> dict:
    F = Frozen()
    O = {p: F.out(p) for p in POPS + ("cal_seen_full", "cal_unseen_full", "oos_cal")}
    prop = {p: O[p][PRIMARY]["pred"] for p in O}
    wrong = {p: prop[p] != O[p]["y"] for p in O}

    # thresholds for the score-based verifiers, chosen on calibration pools only
    c_seen = ~wrong["cal_seen_full"]
    thr_msp_R = sel.pick_fixed_sequence(O["cal_seen_full"][PRIMARY]["msp"], c_seen, 0.02)
    # familiarity floors: false-accept rate <= 2% on held-out calibration groups and on OOS-val
    def far_thr(score_fn):
        s = np.r_[score_fn(O["cal_unseen_full"]), score_fn(O["oos_cal"])]
        c = np.r_[~wrong["cal_unseen_full"], np.zeros(len(O["oos_cal"]["y"]), bool)]
        return max(sel.pick_far(s, c, 0.02), sel.pick_fixed_sequence(score_fn(O["cal_seen_full"]), c_seen, 0.02))
    thr_msp_G = far_thr(lambda o: o[PRIMARY]["msp"])
    thr_knn_G = far_thr(lambda o: o["nov/bge-small"]["knn1"])
    thr_maha_G = far_thr(lambda o: o["nov/bge-small"]["rel_maha"])
    thr_tfknn_G = far_thr(lambda o: o["nov/tfidf"]["knn1"])

    lat = {}
    try:
        L = load_json("llm_latency.json")
        lat = {"llm_classify": L["qwen3-4b/route_1shot"]["p50"], "llm_verify": L["qwen3-4b/verify_yes_no"]["p50"],
               "gemma": L["gemma3-4b/route_1shot"]["p50"]}
    except Exception:
        pass
    try:
        FL = load_json("exp_latency.json")["tier_ms_p50"]
    except Exception:
        FL = {}

    V = {}      # name -> (accept(pop) -> bool array, mask of rows where the verifier is available, cost key)
    full = lambda p: np.ones(len(prop[p]), bool)
    V["none (answer everything)"] = (lambda p: full(p), full, None)
    V["self: msp >= thr (random-split calibration)"] = (lambda p: O[p][PRIMARY]["msp"] >= thr_msp_R, full, None)
    V["self: msp >= thr (grouped calibration)"] = (lambda p: O[p][PRIMARY]["msp"] >= thr_msp_G, full, None)
    V["familiarity: knn1 (same embedding)"] = (lambda p: O[p]["nov/bge-small"]["knn1"] >= thr_knn_G, full, "knn")
    V["familiarity: rel. Mahalanobis (same embedding)"] = (lambda p: O[p]["nov/bge-small"]["rel_maha"] >= thr_maha_G, full, "maha")
    V["familiarity: knn1 (tf-idf view)"] = (lambda p: O[p]["nov/tfidf"]["knn1"] >= thr_tfknn_G, full, "tfidf")
    V["agree: kNN head, same embedding"] = (lambda p: O[p]["bge-small+kNN"]["pred"] == prop[p], full, "knn")
    V["agree: MLP ensemble unanimous, same embedding"] = (
        lambda p: (O[p]["ens"]["pred"] == prop[p]) & (O[p]["ens"]["agree"] >= 1.0), full, "mlp")
    V["agree: minilm+LR (other encoder)"] = (lambda p: O[p]["minilm+LR"]["pred"] == prop[p], full, "minilm")
    V["agree: bge-base+LR (bigger encoder)"] = (lambda p: O[p]["bge-base+LR"]["pred"] == prop[p], full, "bge-base")
    V["agree: tfidf-char+LR (lexical view)"] = (lambda p: O[p]["tfidf-char+LR"]["pred"] == prop[p], full, "tfidf")
    V["rules: mined rules do not contradict"] = (
        lambda p: (O[p]["rules"]["pred"] == None) | (O[p]["rules"]["pred"] == prop[p]), full, "rules")  # noqa: E711
    V["rules: a mined rule confirms"] = (lambda p: O[p]["rules"]["pred"] == prop[p], full, "rules")
    V["committee: tfidf + minilm both agree"] = (
        lambda p: (O[p]["tfidf-char+LR"]["pred"] == prop[p]) & (O[p]["minilm+LR"]["pred"] == prop[p]), full, "tfidf+minilm")
    V["tfidf agrees AND knn1 familiar"] = (
        lambda p: (O[p]["tfidf-char+LR"]["pred"] == prop[p]) & (O[p]["nov/bge-small"]["knn1"] >= thr_knn_G), full, "tfidf")

    llm_lab = {p: F.llm(p) for p in POPS}
    V["LLM classifies independently and agrees (qwen3-4b)"] = (
        lambda p: llm_lab[p][0] == prop[p], lambda p: llm_lab[p][2], "llm_classify")
    ver = {p: F.verify(p, prop[p]) for p in POPS}
    V["LLM yes/no check of the proposal (qwen3-4b)"] = (lambda p: ver[p][0], lambda p: ver[p][1], "llm_verify")
    gem = {p: F.llm(p, model="gemma3-4b") for p in POPS}
    V["second LLM agrees (gemma3-4b)"] = (lambda p: gem[p][0] == prop[p], lambda p: gem[p][2], "gemma")
    V["both LLMs agree with the proposal"] = (
        lambda p: (gem[p][0] == prop[p]) & (llm_lab[p][0] == prop[p]), lambda p: gem[p][2] & llm_lab[p][2], "llm_classify+gemma")

    out = {"primary": PRIMARY, "thresholds": {"msp_R": thr_msp_R, "msp_G": thr_msp_G, "knn_G": thr_knn_G},
           "n": {p: int(len(prop[p])) for p in POPS}, "verifiers": {}}
    for name, (acc_fn, avail_fn, cost) in V.items():
        row = {"cost_key": cost}
        ok_all = True
        for p in POPS:
            m = avail_fn(p)
            if m.sum() < 50:
                ok_all = False
                continue
            a, w = acc_fn(p)[m], wrong[p][m]
            n_err, n_ok = int(w.sum()), int((~w).sum())
            row[p] = {"n": int(m.sum()),
                      "catch_rate": float((~a & w).sum() / n_err) if n_err else None,
                      "false_reject": float((~a & ~w).sum() / n_ok) if n_ok else None,
                      "coverage": float(a.mean()),
                      "silent_rate": float((a & w).sum() / max(1, a.sum())),
                      "false_accept_rate": float((a & w).sum() / len(a))}
        if ok_all or any(p in row for p in POPS):
            out["verifiers"][name] = row
    for name, row in out["verifiers"].items():
        cells = []
        for p in POPS:
            if p in row:
                r = row[p]
                cells.append(f"{p.replace('test_', '').replace('_test', '')}: catch {r['catch_rate'] if r['catch_rate'] is None else round(r['catch_rate'], 2)} "
                             f"FR {r['false_reject'] if r['false_reject'] is None else round(r['false_reject'], 3)} "
                             f"cov {r['coverage']:.2f} FAR {r['false_accept_rate']:.3f}")
        print(f"  {name:52s} " + " | ".join(cells), flush=True)

    # ---- error correlation between decision makers (rows where everyone has an answer) ----
    makers = {m: {p: O[p][m]["pred"] for p in POPS} for m in ("bge-small+LR", "bge-small+kNN", "minilm+LR", "bge-base+LR", "tfidf-char+LR")}
    makers["mlp-ens(bge-small)"] = {p: O[p]["ens"]["pred"] for p in POPS}
    makers["llm:qwen3-4b"] = {p: llm_lab[p][0] for p in POPS}
    makers["llm:gemma3-4b"] = {p: gem[p][0] for p in POPS}
    corr = {}
    have = {m: {p: np.ones(len(O[p]["y"]), bool) for p in POPS} for m in makers}
    have["llm:qwen3-4b"] = {p: llm_lab[p][2] for p in POPS}
    have["llm:gemma3-4b"] = {p: gem[p][2] for p in POPS}
    for pop in ("test_seen", "test_unseen"):
        tab = {}
        for a in makers:
            for b in makers:
                if a == b:
                    continue
                m = have[a][pop] & have[b][pop]          # every row where both have an answer
                if m.sum() < 50:
                    continue
                y = O[pop]["y"][m]
                pa, pb = makers[a][pop][m], makers[b][pop][m]
                wa, wb = pa != y, pb != y
                tab[f"{a} -> {b}"] = {"n": int(m.sum()), "A_err": float(wa.mean()), "B_err": float(wb.mean()),
                                      "A_errors": int(wa.sum()),
                                      "B_wrong_given_A_wrong": float(wb[wa].mean()) if wa.any() else None,
                                      "same_wrong_label_given_A_wrong": float((pb[wa] == pa[wa]).mean()) if wa.any() else None}
        corr[pop] = tab
    out["error_correlation"] = corr
    for pop in corr:
        print(f"  error correlation on {pop} (A = {PRIMARY}):")
        for k, r in corr[pop].items():
            if k.startswith(PRIMARY + " ->"):
                print(f"    {k.split('-> ')[1]:22s} B_err {r['B_err']:.3f}  P(B wrong|A wrong) {r['B_wrong_given_A_wrong']:.2f}  "
                      f"P(same wrong label|A wrong) {r['same_wrong_label_given_A_wrong']:.2f}", flush=True)
    out["latency_ms"] = {"llm": lat, "fast": FL}
    return out


if __name__ == "__main__":
    save_json("exp4_verification.json", run())
