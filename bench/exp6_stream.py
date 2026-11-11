"""Experiment 6: a production loop with drift, replayed over real model outputs.

What is real and what is simulated
  real       every request text (human-written, from held-out CLINC pools), every fast-tier
             answer (models are actually run and actually retrained inside the loop), every LLM
             answer (stored outputs of the local qwen3-4b router; after the contract change, of
             the same model with the changed contract).
  simulated  the order and mix of traffic (five scripted phases), and the agent, which is the
             human label and is assumed correct.

Phases (2,790 requests, no row used twice, clean or noised)
  1 steady          familiar groups, 3% out-of-scope background
  2 new groups      half the traffic comes from groups never seen in training
  3 oos burst       a third of the traffic is out of scope
  4 noisy input     most requests carry 15% keyboard noise
  5 policy change   the contract changes: requests about physical cards move from credit_cards
                    to banking. The LLM hole gets the new contract at once (it is text); the
                    compiled fast tier is stale until something notices.

Policies
  A always-LLM
  B static, naive      fast tier calibrated on a random split -> LLM            (never updated)
  C static, guarded    fast tier with familiarity gate + independent verifier    (never updated)
  D C + shadow         a sample of fast answers is re-asked to the LLM; a per-class circuit
                       breaker turns off any answer class whose shadow disagreement is high
  E D + gated updates  recompile from production labels; a candidate is deployed only if it
                       passes on held-out recent traffic and does not regress on the frozen suite
                       (suite re-adjudicated for classes whose breaker tripped)
  E' as E but the regression suite is left stale after the contract change
  F ungated self-training   retrain on everything the system itself answered, no checks
"""
from __future__ import annotations

import hashlib
from collections import defaultdict, deque

import numpy as np
from scipy import sparse

from . import holes, kit, pools
from . import models as M
from .common import save_json
from .exp5_tiers import latencies
from .frozen import NONE, Frozen
from wfc import selective as sel
from wfc.nsp import wilson

CHUNK, WINDOW = 50, 100
SHADOW_RATE, BREAKER_W, BUDGET = 0.10, 30, 0.05
BREAKER_MIN, BREAKER_LB, ADMIT_BUDGET = 8, 0.10, 0.08      # see the note in simulate()
RECOMPILE_EVERY, PROD_WEIGHT = 400, 10.0


def _u(key: str) -> float:
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big") / 2 ** 64


def build_stream(F: Frozen, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    P = F.pools
    pol = P["policy"]
    reserved = set(pol.uid)
    typo_uids = set(P["typo15"].uid)             # a row never appears both clean and noised
    seen = P["test_seen"][~P["test_seen"].uid.isin(reserved) & ~P["test_seen"].uid.isin(typo_uids)]
    unseen = P["test_unseen"][~P["test_unseen"].uid.isin(reserved) & (P["test_unseen"].group != "new_card")]
    oos, typo = P["oos_test"], P["typo15"][~P["typo15"].uid.isin(reserved)]
    take = {k: list(rng.permutation(len(v))) for k, v in (("seen", seen), ("unseen", unseen), ("oos", oos), ("typo", typo))}
    frames = {"seen": seen, "unseen": unseen, "oos": oos, "typo": typo}
    names = {"seen": "test_seen", "unseen": "test_unseen", "oos": "oos_test", "typo": "typo15"}
    plan = [(1, {"seen": 400, "oos": 20}), (2, {"seen": 250, "unseen": 340, "oos": 20}),
            (3, {"seen": 200, "unseen": 100, "oos": 300}), (4, {"typo": 400, "unseen": 100, "oos": 20})]
    rows = []
    for phase, mix in plan:
        block = []
        for k, n in mix.items():
            for _ in range(n):
                j = take[k].pop()
                r = frames[k].iloc[j]
                block.append({"phase": phase, "pool": names[k], "pos": int(np.flatnonzero(P[names[k]].uid.values == r.uid)[0]),
                              "uid": r.uid + ("#typo" if k == "typo" else ""), "y": r.label, "kind": k, "group": r.group})
        for i in rng.permutation(len(block)):
            rows.append(block[i])
    block = [{"phase": 5, "pool": "policy", "pos": j, "uid": r.uid + "#p5", "y": r.label, "group": r.group,
              "kind": "affected" if r.group in pools.POLICY_GROUPS else "seen"} for j, (_, r) in enumerate(pol.iterrows())]
    for i in rng.permutation(len(block)):
        rows.append(block[i])
    return rows


class World:
    """Features and stored LLM answers for every stream row."""

    def __init__(self, F: Frozen, rows: list):
        self.F, self.rows, self.n = F, rows, len(rows)
        E, T, lab_old, lab_new = [], [], [], []
        by_pool = defaultdict(list)
        for i, r in enumerate(rows):
            by_pool[r["pool"]].append(i)
        self.E = np.zeros((self.n, F.views["bge-small"].shape[1]), dtype=np.float32)
        Trows = [None] * self.n
        self.llm = np.full(self.n, None, dtype=object)
        self.llm_have = np.zeros(self.n, bool)
        pol_hole = holes.clinc_routing_hole(policy_note=holes.POLICY_NOTE, name="clinc_route_policy")
        for pool, idxs in by_pool.items():
            pos = np.array([rows[i]["pos"] for i in idxs])
            self.E[idxs] = F._X(pool, "bge-small")[pos]
            Tp = F._X(pool, "tfidf")[pos]
            for k, i in enumerate(idxs):
                Trows[i] = Tp[k]
            if pool == "policy":
                lab, _, have = F.llm(pool, hole=pol_hole)
            else:
                lab, _, have = F.llm(pool)
            self.llm[idxs], self.llm_have[idxs] = lab[pos], have[pos]
        self.T = sparse.vstack(Trows).tocsr()
        self.y = np.array([r["y"] for r in rows], dtype=object)
        self.phase = np.array([r["phase"] for r in rows])
        self.kind = np.array([r["kind"] for r in rows], dtype=object)
        self.uid = [r["uid"] for r in rows]


class Compiled:
    """One version of the fast tier: primary LR, lexical verifier LR, familiarity memory, gate."""

    def __init__(self, Xe, Xt, y, w=None, gate=None):
        self.primary = M.LRHead(C=20.0)
        self.primary.mu = np.asarray(Xe, dtype=np.float64).mean(0)
        self.primary.sd = float(np.asarray(Xe, dtype=np.float64).std()) or 1.0
        from sklearn.linear_model import LogisticRegression
        self.primary.clf = LogisticRegression(C=20.0, max_iter=500, tol=1e-4).fit(self.primary._prep(Xe), y, sample_weight=w)
        self.primary.classes_ = self.primary.clf.classes_
        self.verifier = LogisticRegression(C=20.0, max_iter=300, tol=1e-3).fit(Xt, y, sample_weight=w)
        self.memory = np.asarray(Xe, dtype=np.float32)
        self.gate = gate

    def score(self, Xe, Xt) -> dict:
        P = self.primary.proba(Xe)
        pred = self.primary.classes_[P.argmax(1)]
        return {"pred": pred, "conf": P.max(1), "fam": M.topk(np.asarray(Xe, dtype=np.float32), self.memory, 1)[0][:, 0],
                "agree": self.verifier.predict(Xt) == pred}


def fit_gate_for(c: Compiled, cal: dict, alpha: float = 0.02, extra_shift=None) -> dict:
    s_seen = c.score(cal["E_seen"], cal["T_seen"])
    s_shift = c.score(cal["E_shift"], cal["T_shift"])
    ok_seen = s_seen["pred"] == cal["y_seen"]
    ok_shift = s_shift["pred"] == cal["y_shift"]
    return kit.fit_gate(s_seen["conf"], s_seen["fam"], ok_seen, s_shift["conf"], s_shift["fam"], ok_shift, alpha)


def simulate(W: World, policy: str, base: Compiled, cal: dict, thr_R: float, lat: dict, reg: dict,
             fallback: str = "llm", shadow_rate: float = SHADOW_RATE, announce: bool = False) -> dict:
    """fallback='llm': abstentions go to the LLM, then the agent; the shadow oracle is the LLM.
    fallback='agent': abstentions go straight to the agent (human label) and the shadow oracle is
    the agent too. This isolates the compiled tier and the loop from the local LLM's own errors."""
    F = W.F
    cur = base
    disabled, windows = set(), defaultdict(lambda: deque(maxlen=BREAKER_W))
    prod = []                               # (row, label, source)
    log = {k: np.zeros(W.n) for k in ("fast", "llm", "agent", "wrong", "llm_calls", "shadow", "shadow_dis", "ms", "agent_shadow")}
    tier = np.full(W.n, "", dtype=object)
    events = []
    last_compile, reg_refresh_calls, delivered = 0, 0, np.full(W.n, None, dtype=object)
    reg_y = reg["y"].copy()
    use_shadow = policy in ("D", "E", "E'")
    change_at = int(np.flatnonzero(W.phase == 5)[0])

    def recompile(t: int, reason: str):
        nonlocal cur, disabled, last_compile, reg_y, reg_refresh_calls
        rows = [(i, l) for i, l, src in prod if l is not None]
        if len(rows) < 60:
            return
        r_idx = np.array([i for i, _ in rows])
        r_lab = np.array([l for _, l in rows], dtype=object)
        hold = np.array([_u(f"hold|{W.uid[i]}") < 0.3 for i in r_idx])
        tr_in = ~hold & (r_lab != NONE)
        Xe = np.vstack([F.views["bge-small"][F.tr], W.E[r_idx[tr_in]]])
        Xt = sparse.vstack([F.views["tfidf"][F.tr], W.T[r_idx[tr_in]]]).tocsr()
        yy = np.r_[F.ytr, r_lab[tr_in]]
        w = np.r_[np.ones(len(F.tr)), np.full(int(tr_in.sum()), PROD_WEIGHT)]
        cand = Compiled(Xe, Xt, yy, w)
        # out-of-scope requests seen in production join the population the gate must refuse
        none_rows = r_idx[(r_lab == NONE) & ~hold]
        cal2 = dict(cal)
        if len(none_rows):
            cal2["E_shift"] = np.vstack([cal["E_shift"], W.E[none_rows]])
            cal2["T_shift"] = sparse.vstack([cal["T_shift"], W.T[none_rows]]).tocsr()
            cal2["y_shift"] = np.r_[cal["y_shift"], np.full(len(none_rows), NONE, dtype=object)]
        cand.gate = fit_gate_for(cand, cal2)
        # admission 1: held-out recent labelled traffic
        h = r_idx[hold]
        s = cand.score(W.E[h], W.T[h])
        acc = (s["conf"] >= cand.gate["thr"]) & (s["fam"] >= cand.gate["floor"]) & s["agree"]
        n_a, n_e = int(acc.sum()), int((acc & (s["pred"] != r_lab[hold])).sum())
        # The reference labels here are mostly the LLM's, which is itself wrong a few percent of the
        # time, so the bar is the silent-error budget plus an allowance for oracle noise.
        ok_recent = n_a >= 20 and n_e / n_a <= ADMIT_BUDGET
        # admission 2: frozen regression suite (optionally re-adjudicated for tripped classes)
        if policy == "E" and disabled:
            cur_pred = cur.score(reg["E"], reg["T"])["pred"]
            need = np.isin(cur_pred, list(disabled)) & ~reg["refreshed"]
            reg_y[need] = reg["y_new"][need]
            reg["refreshed"] |= need
            reg_refresh_calls += int(need.sum())
        acc_cur = float((cur.score(reg["E"], reg["T"])["pred"] == reg_y).mean())
        acc_new = float((cand.score(reg["E"], reg["T"])["pred"] == reg_y).mean())
        ok_reg = acc_new >= acc_cur - 0.01
        admitted = bool(ok_recent and ok_reg and cand.gate["thr"] != sel.NEVER)
        events.append({"t": t, "event": "recompile", "reason": reason, "admitted": admitted,
                       "recent_answered": n_a, "recent_errors": n_e, "regression_acc": [round(acc_cur, 4), round(acc_new, 4)],
                       "labels_used": int(tr_in.sum()), "breakers_before": sorted(disabled)})
        last_compile = t
        if admitted:
            cur = cand
            disabled = set()
            windows.clear()

    for a in range(0, W.n, CHUNK):
        b = min(W.n, a + CHUNK)
        idx = np.arange(a, b)
        if announce and a <= change_at < b + CHUNK and pools.POLICY_FROM not in disabled and a + CHUNK > change_at:
            # The contract change is announced: every fast answer in the class it touches is
            # invalidated at once, without waiting for shadow evidence.
            disabled.add(pools.POLICY_FROM)
            events.append({"t": a, "event": "announced", "class": pools.POLICY_FROM})
        if policy == "A":
            accept = np.zeros(b - a, bool)
            s = None
        else:
            s = cur.score(W.E[idx], W.T[idx])
            if policy in ("B", "F"):
                accept = s["conf"] >= thr_R
            else:
                accept = (s["conf"] >= cur.gate["thr"]) & (s["fam"] >= cur.gate["floor"]) & s["agree"]
                accept &= ~np.isin(s["pred"], list(disabled))
            log["ms"][idx] += lat["fast_ms"] + (lat["verify_ms"] if policy not in ("B", "F") else 0.0)
        for k, i in enumerate(idx):
            llm_ok = W.llm_have[i] and W.llm[i] is not None and W.llm[i] != NONE
            if accept[k]:
                tier[i], delivered[i] = "fast", s["pred"][k]
                log["fast"][i] = 1
                if use_shadow and _u(f"shadow|{W.uid[i]}") < shadow_rate:
                    # shadow checks are asynchronous: they cost a call, not request latency
                    log["shadow"][i] = 1
                    if fallback == "llm":
                        log["llm_calls"][i] += 1
                        ref = W.llm[i]
                    else:
                        log["agent_shadow"][i] = 1
                        ref = W.y[i]
                    dis = ref != s["pred"][k]
                    log["shadow_dis"][i] = float(dis)
                    windows[s["pred"][k]].append(dis)
                    prod.append((i, ref if ref is not None else None, "shadow"))
                elif policy == "F":
                    prod.append((i, s["pred"][k], "self"))
            elif fallback == "agent":
                tier[i], delivered[i] = "agent", W.y[i]
                log["agent"][i] = 1
                prod.append((i, W.y[i], "agent"))
            else:
                log["llm_calls"][i] += 1
                log["ms"][i] += lat["llm_ms"]
                if llm_ok:
                    tier[i], delivered[i] = "llm", W.llm[i]
                    log["llm"][i] = 1
                    prod.append((i, W.llm[i], "llm"))
                else:
                    tier[i], delivered[i] = "agent", W.y[i]
                    log["agent"][i] = 1
                    prod.append((i, W.y[i], "agent"))
            log["wrong"][i] = float(tier[i] != "agent" and delivered[i] != W.y[i])
        if use_shadow:
            for c, win in list(windows.items()):
                if c not in disabled and len(win) >= BREAKER_MIN and wilson(int(sum(win)), len(win))[0] > BREAKER_LB:
                    disabled.add(c)
                    events.append({"t": b, "event": "breaker", "class": c, "window": [int(sum(win)), len(win)]})
                    if policy in ("E", "E'") and b - last_compile >= 100:
                        recompile(b, f"breaker:{c}")
        if policy in ("E", "E'") and b - last_compile >= RECOMPILE_EVERY:
            recompile(b, "periodic")
        if policy == "F" and b - last_compile >= RECOMPILE_EVERY:
            rows = [(i, l) for i, l, _ in prod if l is not None and l != NONE]
            r_idx = np.array([i for i, _ in rows])
            r_lab = np.array([l for _, l in rows], dtype=object)
            cur = Compiled(np.vstack([F.views["bge-small"][F.tr], W.E[r_idx]]),
                           sparse.vstack([F.views["tfidf"][F.tr], W.T[r_idx]]).tocsr(),
                           np.r_[F.ytr, r_lab], np.r_[np.ones(len(F.tr)), np.full(len(r_idx), PROD_WEIGHT)])
            events.append({"t": b, "event": "self-train", "labels_used": int(len(r_idx))})
            last_compile = b

    def summ(mask) -> dict:
        n = int(mask.sum())
        sh_n, sh_d = int(log["shadow"][mask].sum()), int(log["shadow_dis"][mask].sum())
        return {"n": n, "fast": float(log["fast"][mask].mean()), "llm": float(log["llm"][mask].mean()),
                "agent": float(log["agent"][mask].mean()), "wrong": float(log["wrong"][mask].mean()),
                "wrong_via_fast": float((log["wrong"][mask] * log["fast"][mask]).mean()),
                "wrong_via_llm": float((log["wrong"][mask] * log["llm"][mask]).mean()),
                "agent_shadow_calls": float(log["agent_shadow"][mask].mean()),
                "wrong_fast": float((log["wrong"][mask] * log["fast"][mask]).sum() / max(1, log["fast"][mask].sum())),
                "llm_calls": float(log["llm_calls"][mask].mean()), "ms": float(log["ms"][mask].mean()),
                "shadow_n": sh_n, "shadow_est": (sh_d / sh_n) if sh_n else None,
                "shadow_ci": list(wilson(sh_d, sh_n)) if sh_n else None}
    out = {"total": summ(np.ones(W.n, bool)), "by_phase": {int(p): summ(W.phase == p) for p in np.unique(W.phase)},
           "windows": [summ((np.arange(W.n) >= a) & (np.arange(W.n) < a + WINDOW)) for a in range(0, W.n, WINDOW)],
           "events": events, "regression_readjudication_agent_calls": reg_refresh_calls}
    aff = W.kind == "affected"
    out["phase5_affected"] = summ(aff)
    return out


def run() -> dict:
    F = Frozen(with_ensemble=False)
    rows = build_stream(F)
    W = World(F, rows)
    lat = latencies()
    C = {p: F.out(p) for p in ("cal_seen_full", "cal_unseen_full", "oos_cal")}
    X = lambda p, v: F._X(p, v)
    cal = {"E_seen": X("cal_seen_full", "bge-small"), "T_seen": X("cal_seen_full", "tfidf"), "y_seen": C["cal_seen_full"]["y"],
           "E_shift": np.vstack([X("cal_unseen_full", "bge-small"), X("oos_cal", "bge-small")]),
           "T_shift": sparse.vstack([X("cal_unseen_full", "tfidf"), X("oos_cal", "tfidf")]).tocsr(),
           "y_shift": np.r_[C["cal_unseen_full"]["y"], C["oos_cal"]["y"]]}
    base = Compiled(F.views["bge-small"][F.tr], F.views["tfidf"][F.tr], F.ytr)
    base.gate = fit_gate_for(base, cal)
    s = base.score(cal["E_seen"], cal["T_seen"])
    thr_R = sel.pick_fixed_sequence(s["conf"], s["pred"] == cal["y_seen"], 0.02)
    g = F.pools["cal_seen_full"].group.values
    y_new = cal["y_seen"].copy()
    y_new[np.isin(g, pools.POLICY_GROUPS)] = pools.POLICY_TO
    out = {"n_requests": W.n, "llm_answers_available": float(W.llm_have.mean()),
           "phases": {int(p): {"n": int((W.phase == p).sum()),
                               "kinds": {k: int(((W.phase == p) & (W.kind == k)).sum()) for k in np.unique(W.kind[W.phase == p])}}
                      for p in np.unique(W.phase)},
           "params": {"shadow_rate": SHADOW_RATE, "breaker_window": BREAKER_W, "budget": BUDGET,
                      "breaker_min_samples": BREAKER_MIN, "breaker_lower_bound": BREAKER_LB, "admit_budget": ADMIT_BUDGET,
                      "recompile_every": RECOMPILE_EVERY, "production_label_weight": PROD_WEIGHT, "gate": base.gate,
                      "thr_R": thr_R}, "latency": lat, "policies": {}}
    def fresh_reg():
        return {"E": cal["E_seen"], "T": cal["T_seen"], "y": cal["y_seen"], "y_new": y_new,
                "refreshed": np.zeros(len(y_new), bool)}
    runs = ([(p, "llm", False) for p in ("A", "B", "C", "D", "E", "E'", "F")] + [("D", "llm", True), ("E", "llm", True)] +
            [(p, "agent", False) for p in ("B", "C", "D", "E", "E'", "F")] + [("D", "agent", True), ("E", "agent", True)])
    for pol, fb, ann in runs:
        r = simulate(W, pol, base, cal, thr_R, lat, fresh_reg(), fallback=fb, announce=ann)
        pol = pol + ("+" if ann else "") + ("" if fb == "llm" else "*")
        out["policies"][pol] = r
        t = r["total"]
        print(f"  {pol:3s} total: fast {t['fast']:.2f} llm {t['llm']:.2f} agent {t['agent']:.2f} wrong {t['wrong']:.4f} (fast {t['wrong_via_fast']:.4f}) "
              f"llm_calls {t['llm_calls']:.2f} ms {t['ms']:.0f} | wrong by phase " +
              " ".join(f"{p}:{v['wrong']:.3f}" for p, v in r["by_phase"].items()) +
              f" | affected wrong {r['phase5_affected']['wrong']:.2f} | events {len(r['events'])}", flush=True)
        for e in r["events"][:14]:
            if e["event"] != "self-train":
                print("       ", {k: v for k, v in e.items() if k != "breakers_before"}, flush=True)
    # how fast does shadow sampling notice the policy change? (agent as shadow oracle)
    change_at = int(np.flatnonzero(W.phase == 5)[0])
    sweep = {}
    for rate in (0.05, 0.1, 0.2, 0.4):
        r = simulate(W, "D", base, cal, thr_R, lat, fresh_reg(), fallback="agent", shadow_rate=rate)
        trip = [e["t"] for e in r["events"] if e["event"] == "breaker" and e["class"] == pools.POLICY_FROM and e["t"] > change_at]
        other = [e for e in r["events"] if e["event"] == "breaker" and e["class"] != pools.POLICY_FROM]
        sweep[str(rate)] = {"requests_until_breaker": (trip[0] - change_at) if trip else None,
                            "wrong_on_rerouted": r["phase5_affected"]["wrong"], "wrong_total": r["total"]["wrong"],
                            "agent_shadow_calls_per_request": r["total"]["agent_shadow_calls"], "other_breakers": len(other)}
        print(f"  shadow rate {rate}: breaker after {sweep[str(rate)]['requests_until_breaker']} requests, wrong on re-routed "
              f"{sweep[str(rate)]['wrong_on_rerouted']:.2f}, other breakers {len(other)}", flush=True)
    out["shadow_sweep"] = sweep
    out["change_at"] = change_at
    return out


if __name__ == "__main__":
    save_json("exp6_stream.json", run())
