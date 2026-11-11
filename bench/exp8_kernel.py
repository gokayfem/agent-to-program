"""Experiment 8: the compiled program as a real `wfc.Program`, run one request at a time.

Everything local is executed live per request (guards, cache lookup, mined rules, sentence
embedding, classifier, familiarity search, lexical verifier). The LLM tier replays the stored
real answer of the local model for that exact text; the agent is the human label.

Then the admission harness: held-out cases, seeded bad inputs that must not be answered by
the program, and a determinism check.
"""
from __future__ import annotations

import numpy as np

from . import embed, holes, kit
from . import models as M
from .common import save_json
from .frozen import NONE, Frozen
from .llm import AnswerStore
from .perturb import seeded_bad_inputs
from wfc import selective as sel
from wfc.admit import admit
from wfc.nsp import ABSTAIN, Decision, GuardedCache, Program, check

M0 = "bge-small+LR"


def build(F: Frozen, with_verifier: bool = True, with_guards: bool = True, gate_kind: str = "G",
          with_llm: bool = True, with_rules: bool = True):
    C = {p: F.out(p) for p in ("cal_seen_full", "cal_unseen_full", "oos_cal")}
    ok = {p: C[p][M0]["pred"] == C[p]["y"] for p in C}
    conf, fam = (lambda o: o[M0]["msp"]), (lambda o: o["nov/bge-small"]["knn1"])
    if gate_kind == "G":
        gate = kit.fit_gate(conf(C["cal_seen_full"]), fam(C["cal_seen_full"]), ok["cal_seen_full"],
                            np.r_[conf(C["cal_unseen_full"]), conf(C["oos_cal"])],
                            np.r_[fam(C["cal_unseen_full"]), fam(C["oos_cal"])],
                            np.r_[ok["cal_unseen_full"], np.zeros(len(C["oos_cal"]["y"]), bool)], 0.02)
    else:
        gate = {"thr": sel.pick_fixed_sequence(conf(C["cal_seen_full"]), ok["cal_seen_full"], 0.02), "floor": -1.0}
    r_ok = {p: C[p]["rules"]["pred"] == C[p]["y"] for p in C}
    r_votes = lambda o: np.where(o["rules"]["pred"] == None, -np.inf, o["rules"]["votes"])  # noqa: E711
    rv = max(sel.pick_fixed_sequence(r_votes(C["cal_seen_full"]), r_ok["cal_seen_full"], 0.02, 0.025),
             sel.pick_far(np.r_[r_votes(C["cal_unseen_full"]), r_votes(C["oos_cal"])],
                          np.r_[r_ok["cal_unseen_full"], np.zeros(len(C["oos_cal"]["y"]), bool)], 0.02, 0.025))
    enc = embed.get_encoder("bge-small")
    primary, memory = F.models[M0].head, F.views["bge-small"][F.tr]
    tf_lr = F.models["tfidf-char+LR"].head
    store = AnswerStore("qwen3-4b", holes.clinc_routing_hole())
    classes = set(F.classes)

    class Rules:
        def answer(self, x):
            lab, votes = F.rules.answer([x])
            return lab[0] if lab[0] is not None and votes[0] >= rv else ABSTAIN

    class Model:
        def answer(self, x):
            e = np.asarray(enc.encode([x], normalize_embeddings=True, show_progress_bar=False), dtype=np.float32)
            P = primary.proba(e)[0]
            if P.max() < gate["thr"] or M.topk(e, memory, 1)[0][0, 0] < gate["floor"]:
                return ABSTAIN
            return str(primary.classes_[P.argmax()])

    class ReplayLLM:
        missing = 0

        def answer(self, x):
            r = store.rows.get(x)
            if r is None:
                ReplayLLM.missing += 1
                return ABSTAIN
            return r["label"] if r["label"] in classes else ABSTAIN       # none_of_these / unparseable -> agent

    def verifier(x, v, tier):
        return str(tf_lr.classes_[tf_lr.proba(F.vec.transform([x])).argmax()]) == v

    decide = Decision("route", F.classes,
                      cache=GuardedCache({k: next(iter(v)) for k, v in F.cache_table.items() if len(v) == 1}, key=F.norm),
                      rules=Rules() if with_rules else None, model=Model(), llm=ReplayLLM() if with_llm else None,
                      verifier=verifier if with_verifier else None, verify_tiers=("model",))

    def fn(x, t):
        if with_guards:
            check(t, "is_text", isinstance(x, str), "not a string")
            check(t, "length", 3 <= len(x.strip()) <= 300, "empty or implausibly long")
            letters = sum(c.isalpha() or c.isspace() for c in x) / max(1, len(x))
            check(t, "mostly_words", letters >= 0.8, "not natural-language text")
            check(t, "has_words", len([w for w in x.split() if w.isalpha()]) >= 2, "fewer than two words")
        return decide(x, t)

    return fn, ReplayLLM, {"gate": gate, "rules_min_votes": rv}


def run() -> dict:
    F = Frozen(with_ensemble=False)
    P = F.pools
    cases = (list(P["test_seen"].sort_values("uid").text[:300]) + list(P["test_unseen"].sort_values("uid").text[:200])
             + list(P["oos_test"].sort_values("uid").text[:100]))
    expected = (list(P["test_seen"].sort_values("uid").label[:300]) + list(P["test_unseen"].sort_values("uid").label[:200])
                + [NONE] * 100)
    truth = dict(zip(cases, expected))
    bad = seeded_bad_inputs(0)
    bad_all = [t for v in bad.values() for t in v]
    agent = lambda x, e: truth.get(x, NONE)
    out = {"cases": {"seen": 300, "unseen_groups": 200, "out_of_scope": 100}, "bad_inputs": {k: len(v) for k, v in bad.items()},
           "builds": {}}
    for name, kw in {"guards + gate(G) + verifier": {},
                     "gate(G) + verifier, no structural guards": {"with_guards": False},
                     "guards + gate(G), no verifier": {"with_verifier": False},
                     "guards + naive gate(R), no verifier": {"with_verifier": False, "gate_kind": "R"},
                     "full build without the LLM tier (abstentions go to the agent)": {"with_llm": False},
                     "the same without the rule tier": {"with_llm": False, "with_rules": False}}.items():
        fn, replay, info = build(F, **kw)
        replay.missing = 0
        prog = Program(name, fn, fallback=agent)
        rep = admit(prog, cases, expected, bad_inputs=bad_all, max_silent_error=0.02, repeats=2)
        rep["llm_answers_missing_in_replay"] = replay.missing
        rep["admitted_at_5pct"] = bool(rep["silent_error_rate"] <= 0.05 and rep["guard_false_accepts"] == 0 and rep["deterministic"])
        # which seeded bad inputs got an answer from the program, by category and by tier
        fa = {}
        for k, items in bad.items():
            hits = []
            for x in items:
                try:
                    r = prog.run(x)
                except Exception:
                    continue
                if r.path == "fast":
                    hits.append(r.trace.highest_tier())
            fa[k] = {"n": len(items), "answered": len(hits), "by_tier": {t: hits.count(t) for t in set(hits)}}
        rep["bad_input_detail"] = fa
        rep["info"] = info
        out["builds"][name] = rep
        print(f"  {name:44s} admitted={rep['admitted']} fast {rep['fast_path_rate']:.2f} no-LLM {rep['no_llm_rate']:.2f} "
              f"silent {rep['silent_errors']} ({rep['silent_error_rate']:.3f}, ub {rep['silent_error_upper95']}) "
              f"bad answered {rep['guard_false_accepts']}/{rep['bad_inputs']} p50 no-LLM run {rep['run_ms_p50_no_llm']} ms "
              f"tiers {rep['by_tier']}", flush=True)
        print("       bad inputs answered:", {k: f"{v['answered']}/{v['n']} {v['by_tier']}" for k, v in fa.items()}, flush=True)
    return out


if __name__ == "__main__":
    save_json("exp8_kernel.json", run())
