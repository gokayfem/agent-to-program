"""Experiment 5: the real LLM tier, and whole tier stacks end to end.

Frozen CLINC routing split; every LLM answer is a stored real output of a local model.

A  The LLM tier on its own: accuracy against human labels on seen groups, unseen groups,
   out-of-scope queries and noisy queries; how often it takes the "none of these" exit; whether
   its own token probability or a second LLM's agreement is a usable abstention signal.
B  Tier stacks. Each request goes to the cheapest tier willing to answer:
       cache -> rules -> decision model (+ independent verifier) -> LLM -> agent
   The agent is the human label: assumed correct, and counted as an escalation, not timed.
   Reported per population and for stated traffic mixes: who answered, wrong answers delivered,
   LLM calls and escalations per request, and compute time per request from measured medians.
C  Compile from traces: the same fast model trained on the LLM's own past answers instead of
   human labels.
"""
from __future__ import annotations

import numpy as np

from . import kit
from .common import load_json, save_json
from .frozen import NONE, Frozen
from wfc import selective as sel

M0 = "bge-small+LR"
POPS = ("test_seen", "test_unseen", "oos_test", "typo15")
MIXES = {"70/20/10 seen/unseen/oos": {"test_seen": 0.7, "test_unseen": 0.2, "oos_test": 0.1},
         "90/8/2": {"test_seen": 0.9, "test_unseen": 0.08, "oos_test": 0.02},
         "50/30/20": {"test_seen": 0.5, "test_unseen": 0.3, "oos_test": 0.2}}


def latencies() -> dict:
    lat = {"fast_ms": 6.0, "verify_ms": 0.6, "rules_ms": 0.1, "cache_ms": 0.01, "llm_ms": 311.0,
           "source": "preliminary defaults (probe)"}
    try:
        L, FL = load_json("llm_latency.json"), load_json("exp_latency.json")["tier_ms_p50"]
        lat = {"fast_ms": FL["bge-small encode"] + FL["LR head"] + FL["knn1 over train"],
               "verify_ms": FL["tfidf-char+LR end-to-end"], "rules_ms": FL["mined rules"],
               "cache_ms": FL["exact cache"], "llm_ms": L["qwen3-4b/route_1shot"]["p50"],
               "source": "measured: results/exp_latency.json, results/llm_latency.json"}
    except Exception:
        pass
    return lat


def llm_section(F: Frozen) -> dict:
    out = {}
    for model in ("qwen3-4b", "gemma3-4b"):
        row = {}
        for p in POPS + ("cal_seen", "cal_unseen", "oos_val"):
            lab, pr, have = F.llm(p, model=model)
            if have.sum() < 50:
                continue
            y = F.out(p)["y"][have]
            lab, pr = lab[have], pr[have]
            answered = np.array([l is not None and l != NONE for l in lab])
            row[p] = {"n": int(have.sum()), "accuracy": float((lab == y).mean()),
                      "none_exit_rate": float((lab == NONE).mean()),
                      "unparseable_rate": float(np.mean([l is None for l in lab])),
                      "answered_rate": float(answered.mean()),
                      "wrong_among_answered": float((lab[answered] != y[answered]).mean()) if answered.any() else None,
                      "p_eq_1_share": float((pr >= 0.999).mean()),
                      "acc_when_p_eq_1": float((lab == y)[pr >= 0.999].mean()) if (pr >= 0.999).any() else None,
                      "acc_when_p_lt_1": float((lab == y)[pr < 0.999].mean()) if (pr < 0.999).any() else None}
        out[model] = row
    # two-LLM agreement as an abstention signal (rows both models answered)
    agree = {}
    for p in ("test_seen", "test_unseen", "oos_test"):
        a, _, ha = F.llm(p)
        b, _, hb = F.llm(p, model="gemma3-4b")
        m = ha & hb
        if m.sum() < 50:
            continue
        y = F.out(p)["y"][m]
        a, b = a[m], b[m]
        both = (a == b) & np.array([x is not None and x != NONE for x in a])
        agree[p] = {"n": int(m.sum()), "agree_and_answer_rate": float(both.mean()),
                    "wrong_when_agree": float((a[both] != y[both]).mean()) if both.any() else None,
                    "qwen_wrong_answered": float(np.mean([(x is not None and x != NONE and x != t) for x, t in zip(a, y)]))}
    out["two_llm_agreement"] = agree
    return out


def stacks(F: Frozen, lat: dict) -> dict:
    C = {p: F.out(p) for p in ("cal_seen_full", "cal_unseen_full", "oos_cal")}
    ok = {p: C[p][M0]["pred"] == C[p]["y"] for p in C}
    conf = lambda o: o[M0]["msp"]
    fam = lambda o: o["nov/bge-small"]["knn1"]
    thr_R = sel.pick_fixed_sequence(conf(C["cal_seen_full"]), ok["cal_seen_full"], 0.02)
    shift_conf = np.r_[conf(C["cal_unseen_full"]), conf(C["oos_cal"])]
    shift_fam = np.r_[fam(C["cal_unseen_full"]), fam(C["oos_cal"])]
    shift_ok = np.r_[ok["cal_unseen_full"], np.zeros(len(C["oos_cal"]["y"]), bool)]
    gate = kit.fit_gate(conf(C["cal_seen_full"]), fam(C["cal_seen_full"]), ok["cal_seen_full"],
                        shift_conf, shift_fam, shift_ok, 0.02)
    gate5 = kit.fit_gate(conf(C["cal_seen_full"]), fam(C["cal_seen_full"]), ok["cal_seen_full"],
                         shift_conf, shift_fam, shift_ok, 0.05)
    # rules: smallest vote count that is safe on calibration (same two tests, score = votes)
    r_ok = {p: C[p]["rules"]["pred"] == C[p]["y"] for p in C}
    r_votes = lambda o: np.where(o["rules"]["pred"] == None, -np.inf, o["rules"]["votes"])  # noqa: E711
    rv = max(sel.pick_fixed_sequence(r_votes(C["cal_seen_full"]), r_ok["cal_seen_full"], 0.02, 0.025),
             sel.pick_far(np.r_[r_votes(C["cal_unseen_full"]), r_votes(C["oos_cal"])],
                          np.r_[r_ok["cal_unseen_full"], np.zeros(len(C["oos_cal"]["y"]), bool)], 0.02, 0.025))
    # LLM self-confidence threshold from LLM-labelled calibration pools (alpha 5% on all answered)
    lc, lp, ly = [], [], []
    for p in ("cal_seen", "cal_unseen", "oos_val"):
        lab, pr, have = F.llm(p)
        y = F.out(p)["y"]
        ans = have & np.array([l is not None and l != NONE for l in lab])
        lc.append((lab == y)[ans]); lp.append(pr[ans])
    llm_tau = sel.pick_fixed_sequence(np.concatenate(lp), np.concatenate(lc), 0.05) if len(np.concatenate(lp)) else sel.NEVER

    def simulate(p: str, policy: str) -> dict:
        o = F.out(p)
        y = o["y"]
        n = len(y)
        lab, pr, have = F.llm(p)
        llm_ans = np.array([l is not None and l != NONE for l in lab])
        tier = np.full(n, "agent", dtype=object)
        ans = np.full(n, None, dtype=object)
        open_ = np.ones(n, bool)
        ms = np.zeros(n)
        llm_calls = np.zeros(n)

        def take(mask, name, values):
            m = open_ & mask
            tier[m], ans[m] = name, values[m]
            open_[m] = False

        steps = policy.split(" > ")
        for s in steps:
            if s == "cache":
                ms[open_] += lat["cache_ms"]
                take(o["cache"]["pred"] != None, "cache", o["cache"]["pred"])  # noqa: E711
            elif s == "rules":
                ms[open_] += lat["rules_ms"]
                take(r_votes(o) >= rv, "rules", o["rules"]["pred"])
            elif s.startswith("fast"):
                ms[open_] += lat["fast_ms"]
                if s == "fast[all]":
                    g = np.ones(n, bool)
                elif s == "fast[R]":
                    g = conf(o) >= thr_R
                elif s.startswith("fast[G5]"):
                    g = (conf(o) >= gate5["thr"]) & (fam(o) >= gate5["floor"])
                else:
                    g = (conf(o) >= gate["thr"]) & (fam(o) >= gate["floor"])
                if s.endswith("+V"):
                    ms[open_ & g] += lat["verify_ms"]
                    g = g & (o["tfidf-char+LR"]["pred"] == o[M0]["pred"])
                take(g, "model", o[M0]["pred"])
            elif s.startswith("llm"):
                ms[open_] += lat["llm_ms"]
                llm_calls[open_] += 1
                g = llm_ans & have
                if s == "llm[conf]":
                    g = g & (pr >= llm_tau)
                take(g, "llm", lab)
        wrong = (tier != "agent") & (ans != y)
        return {"n": n, "llm_complete": bool(have.all()),
                "share": {t: float((tier == t).mean()) for t in ("cache", "rules", "model", "llm", "agent")},
                "wrong_delivered": float(wrong.mean()),
                "wrong_by_tier": {t: float((wrong & (tier == t)).mean()) for t in ("cache", "rules", "model", "llm")},
                "silent_rate_among_program_answers": float(wrong.sum() / max(1, (tier != "agent").sum())),
                "llm_calls": float(llm_calls.mean()), "escalations": float((tier == "agent").mean()),
                "compute_ms_mean": float(ms.mean()), "compute_ms_p50": float(np.median(ms))}

    policies = ["llm", "fast[all]", "fast[R] > llm", "fast[G] > llm", "fast[G]+V > llm", "fast[G5]+V > llm",
                "cache > rules > fast[G]+V > llm", "fast[G]+V > llm[conf]", "fast[G]+V", "fast[R]", "rules > llm", "agent"]
    out = {"gate_alpha2": gate, "gate_alpha5": gate5, "thr_R": thr_R, "rules_min_votes": rv, "llm_tau": llm_tau,
           "latency": lat, "policies": {}}
    for pol in policies:
        steps = "" if pol == "agent" else pol
        per = {p: simulate(p, steps) for p in POPS}
        mixes = {}
        for name, w in MIXES.items():
            agg = {}
            for k in ("wrong_delivered", "llm_calls", "escalations", "compute_ms_mean"):
                agg[k] = float(sum(w[p] * per[p][k] for p in w))
            agg["share"] = {t: float(sum(w[p] * per[p]["share"][t] for p in w)) for t in ("cache", "rules", "model", "llm", "agent")}
            prog = 1 - agg["share"]["agent"]
            agg["silent_rate_among_program_answers"] = agg["wrong_delivered"] / prog if prog > 0 else 0.0
            mixes[name] = agg
        out["policies"][pol] = {"per_population": per, "mix": mixes}
        m = mixes["70/20/10 seen/unseen/oos"]
        print(f"  {pol:34s} mix70/20/10: wrong {m['wrong_delivered']:.4f}  model {m['share']['model']:.2f} "
              f"rules {m['share']['rules']:.2f} llm {m['share']['llm']:.2f} agent {m['share']['agent']:.2f}  "
              f"llm_calls {m['llm_calls']:.2f}  ms {m['compute_ms_mean']:.0f} | wrong seen {per['test_seen']['wrong_delivered']:.4f} "
              f"unseen {per['test_unseen']['wrong_delivered']:.3f} oos {per['oos_test']['wrong_delivered']:.3f} "
              f"typo {per['typo15']['wrong_delivered']:.3f}", flush=True)
    return out


def traces_section() -> dict:
    """Train the fast model on the LLM's own answers for the 'traces' pool vs human labels."""
    base = Frozen(with_ensemble=False)
    tr = base.pools["traces"]
    lab, _, have = base.llm("traces")
    if have.mean() < 0.99:
        return {"status": f"traces pool only {have.mean():.0%} labelled"}
    keep = np.array([l is not None and l != NONE for l in lab])
    idx, y_llm, y_h = tr.index.values, lab, tr.label.values
    out = {"n_traces": int(len(tr)), "llm_label_accuracy_on_traces": float((lab == y_h).mean()),
           "dropped_none_or_invalid": int((~keep).sum()), "variants": {}}
    for name, (ii, yy) in {"human labels, all 9000 train rows": (None, None),
                           "human labels, the same trace rows": (idx, y_h),
                           "LLM labels, the same trace rows": (idx[keep], y_llm[keep])}.items():
        F = base if ii is None else Frozen(with_ensemble=False, train_idx=ii, train_y=yy)
        row = {}
        for p in ("test_seen", "test_unseen"):
            o = F.out(p)
            l2, _, h2 = base.llm(p)
            for m in ("bge-small+LR", "tfidf-char+LR"):
                row[f"{m}/{p}"] = {"accuracy_vs_human": float((o[m]["pred"] == o["y"]).mean()),
                                   "agreement_with_llm": float((o[m]["pred"][h2] == l2[h2]).mean())}
            row[f"llm/{p}"] = float((l2[h2] == o["y"][h2]).mean())
        out["variants"][name] = row
        print(f"  traces: {name:36s} bge acc seen {row['bge-small+LR/test_seen']['accuracy_vs_human']:.3f} "
              f"unseen {row['bge-small+LR/test_unseen']['accuracy_vs_human']:.3f} | LLM itself seen {row['llm/test_seen']:.3f} "
              f"unseen {row['llm/test_unseen']:.3f}", flush=True)
    return out


if __name__ == "__main__":
    F = Frozen(with_ensemble=False)
    lat = latencies()
    out = {"llm_tier": llm_section(F), "stacks": stacks(F, lat), "compile_from_traces": traces_section()}
    for model, row in out["llm_tier"].items():
        if model != "two_llm_agreement":
            print(f"  LLM {model}: " + "  ".join(f"{p}: acc {r['accuracy']:.3f} none {r['none_exit_rate']:.2f} (n={r['n']})"
                                                for p, r in row.items()), flush=True)
    save_json("exp5_tiers.json", out)
