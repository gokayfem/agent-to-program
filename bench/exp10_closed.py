"""Experiment 10: a closed, hosted model as the LLM tier.

The local runs used 4-bit open models of about 4B parameters. This runs the same router
contract through a closed model on a fixed subset of the same held-out pools and scores it
with the same code.

How the closed model is called matters, so it is stated here and in the report:
  * It is run through Claude Code subagents (no API key on this machine), in blind batches:
    each subagent gets one file with the contract and ~220 requests as JSON lines, and writes
    one label per request. The files contain no labels and sit outside the project.
  * That is not one API call per request. There is no per-request latency, no token
    probability, and the model sees other requests in the same batch.
  * Effort is set on the subagent ("medium" for Opus 5.5 in the run recorded here).

Usage:
  python -m bench.exp10_closed prepare <dir>            write batch files + a private key
  python -m bench.exp10_closed ingest <dir> <model-key>  read answer files into cache/llm/
  python -m bench.exp10_closed score <model-key>         results/exp10_closed.json
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import sys

import numpy as np

from . import holes, kit, pools
from .common import CACHE, save_json
from .frozen import NONE, Frozen
from .llm import AnswerStore
from .perturb import seeded_bad_inputs
from wfc import selective as sel

KEY = CACHE / "closed" / "key.json"
M0 = "bge-small+LR"
BATCH = 225


def subset() -> dict:
    """Fixed subsets of the LLM-labelled pools (so the local model's answers exist for the same rows)."""
    P = pools.pools()
    take = lambda name, n: P[name].sort_values("uid").groupby("group").head(n)
    bad = [(k, t) for k, v in seeded_bad_inputs(0).items() for t in v if t.strip()]
    pol = P["policy"]
    aff = pol[pol.group.isin(pools.POLICY_POOL_GROUPS)]
    ctrl = pol[~pol.group.isin(pools.POLICY_POOL_GROUPS)].sort_values("uid").head(160)
    return {"test_seen": take("test_seen", 5), "test_unseen": take("test_unseen", 15),
            "oos_test": P["oos_test"].sort_values("uid").head(300),
            "typo15": P["typo15"].sort_values("uid").head(200),
            "bad": bad, "policy": __import__("pandas").concat([aff, ctrl])}


def prepare(out_dir: str):
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    S = subset()
    base, pol = holes.clinc_routing_hole(), holes.clinc_routing_hole(policy_note=holes.POLICY_NOTE, name="clinc_route_policy")
    rows = []
    for pool in ("test_seen", "test_unseen", "oos_test", "typo15"):
        for _, r in S[pool].iterrows():
            rows.append({"contract": "base", "pool": pool, "text": r.text, "truth": r.label, "group": r.group})
    for kind, t in S["bad"]:
        rows.append({"contract": "base", "pool": "bad", "text": t, "truth": NONE, "group": kind})
    for _, r in S["policy"].iterrows():
        rows.append({"contract": "policy", "pool": "policy", "text": r.text, "truth": r.label, "group": r.group})
    for i, r in enumerate(rows):                       # opaque ids, order shuffled within each contract
        r["id"] = "q" + hashlib.sha256(f"{i}|{r['text']}".encode()).hexdigest()[:8]
    rng = np.random.default_rng(10)
    key, n_files = {}, 0
    for contract, hole in (("base", base), ("policy", pol)):
        part = [r for r in rows if r["contract"] == contract]
        order = rng.permutation(len(part))
        n_batches = int(np.ceil(len(part) / BATCH))
        for b in range(n_batches):
            idx = order[b::n_batches]
            name = f"batch_{n_files:02d}"
            lines = [hole.instruction, "", "REQUESTS (one JSON object per line)"]
            for j in idx:
                lines.append(json.dumps({"id": part[j]["id"], "request": part[j]["text"]}, ensure_ascii=False))
                key[part[j]["id"]] = {**part[j], "batch": name}
            (out / f"{name}.txt").write_text("\n".join(lines) + "\n")
            n_files += 1
    KEY.parent.mkdir(parents=True, exist_ok=True)
    KEY.write_text(json.dumps(key))
    print(f"{len(rows)} requests in {n_files} batch files under {out}; key kept at {KEY}")


def ingest(in_dir: str, model_key: str):
    key = json.loads(KEY.read_text())
    base, pol = holes.clinc_routing_hole(), holes.clinc_routing_hole(policy_note=holes.POLICY_NOTE, name="clinc_route_policy")
    stores = {"base": AnswerStore(model_key, base), "policy": AnswerStore(model_key, pol)}
    got, problems = {}, []
    for f in sorted(pathlib.Path(in_dir).glob("answers_*.tsv")):
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) != 2 or parts[0].strip() not in key:
                problems.append((f.name, line[:60]))
                continue
            got[parts[0].strip()] = parts[1].strip()
    add = {"base": [], "policy": []}
    for qid, k in key.items():
        if qid not in got:
            continue
        hole = base if k["contract"] == "base" else pol
        add[k["contract"]].append({"q": k["text"], "raw": got[qid][:80], "label": hole.parse(got[qid]), "p": float("nan"),
                                   "n_gen": None, "batch": k["batch"], "via": "subagent batch"})
    for c in add:
        fresh = [r for r in add[c] if r["q"] not in stores[c].rows]
        stores[c].add(fresh)
    print(f"answers read: {len(got)} of {len(key)}; malformed lines: {len(problems)} {problems[:3]}")
    return len(got), len(key)


def lookup(model_key: str, hole, texts):
    st = AnswerStore(model_key, hole)
    have = np.array([t in st.rows for t in texts])
    lab = np.array([st.rows[t]["label"] if t in st.rows else None for t in texts], dtype=object)
    return lab, have


def score(model_key: str, local: str = "qwen3-4b") -> dict:
    key = json.loads(KEY.read_text())
    base, pol = holes.clinc_routing_hole(), holes.clinc_routing_hole(policy_note=holes.POLICY_NOTE, name="clinc_route_policy")
    F = Frozen(with_ensemble=False)
    S = subset()
    out = {"model": model_key, "compared_with": local, "protocol": "Claude Code subagents, blind batches of about 220 requests, "
           "one label per request; not one API call per request; no latency or token probabilities",
           "n_requests": len(key), "populations": {}}

    def stats(lab, y, in_scope=True):
        answered = np.array([l is not None and l != NONE for l in lab])
        return {"n": int(len(y)), "correct": float((lab == y).mean()), "none_exit": float((lab == NONE).mean()),
                "unparseable": float(np.mean([l is None for l in lab])),
                "wrong_answers_per_request": float((answered & (lab != y)).mean())}

    names = {"test_seen": "seen groups", "test_unseen": "never-seen groups", "oos_test": "out of scope", "typo15": "seen groups, 15% keyboard noise"}
    per_pool = {}
    for pool in names:
        texts, y = list(S[pool].text), S[pool].label.values
        row = {}
        for m in (model_key, local):
            lab, have = lookup(m, base, texts)
            assert have.all(), (m, pool, int((~have).sum()))
            row[m] = stats(lab, y)
            per_pool[(pool, m)] = lab
        out["populations"][names[pool]] = row

    # seeded bad inputs: any in-scope answer is a failure
    bad = S["bad"]
    kinds = sorted({k for k, _ in bad})
    row = {}
    for m in (model_key, local):
        lab, have = lookup(m, base, [t for _, t in bad])
        ans = np.array([l is not None and l != NONE for l in lab]) & have
        row[m] = {k: {"n": int(sum(kk == k for kk, _ in bad)), "answered_in_scope": int(sum(a for a, (kk, _) in zip(ans, bad) if kk == k))}
                  for k in kinds}
        row[m]["missing"] = int((~have).sum())
    out["bad_inputs"] = row

    # the changed contract
    polS = S["policy"]
    aff = polS.group.isin(pools.POLICY_GROUPS).values
    near = (polS.group == "new_card").values          # a card request the policy does not cover: must stay put
    ctrl = ~aff & ~near
    row = {}
    for m in (model_key, local):
        lab, have = lookup(m, pol, list(polS.text))
        assert have.all(), (m, "policy")
        yv = polS.label.values
        row[m] = {"rerouted_n": int(aff.sum()), "rerouted_correct": float((lab[aff] == yv[aff]).mean()),
                  "rerouted_still_old_label": float((lab[aff] == pools.POLICY_FROM).mean()),
                  "not_covered_n": int(near.sum()), "not_covered_correct": float((lab[near] == yv[near]).mean()),
                  "not_covered_moved_by_mistake": float((lab[near] == pools.POLICY_TO).mean()),
                  "control_n": int(ctrl.sum()), "control_correct": float((lab[ctrl] == yv[ctrl]).mean())}
    out["changed_contract"] = row

    # does the closed model repeat the fast model's mistakes on never-seen groups?
    u = S["test_unseen"]
    fast_pred = F.models[M0].run(u.index.values)["pred"]
    wrong = fast_pred != u.label.values
    out["same_wrong_answer_as_fast_model_on_unseen"] = {
        m: {"fast_errors": int(wrong.sum()), "also_wrong": float((per_pool[("test_unseen", m)][wrong] != u.label.values[wrong]).mean()),
            "same_wrong_label": float((per_pool[("test_unseen", m)][wrong] == fast_pred[wrong]).mean())} for m in (model_key, local)}

    # tier stacks on the same rows (gate fitted on the full calibration partitions, as in exp5)
    C = {p: F.out(p) for p in ("cal_seen_full", "cal_unseen_full", "oos_cal")}
    ok = {p: C[p][M0]["pred"] == C[p]["y"] for p in C}
    conf, fam = (lambda o: o[M0]["msp"]), (lambda o: o["nov/bge-small"]["knn1"])
    gate = kit.fit_gate(conf(C["cal_seen_full"]), fam(C["cal_seen_full"]), ok["cal_seen_full"],
                        np.r_[conf(C["cal_unseen_full"]), conf(C["oos_cal"])], np.r_[fam(C["cal_unseen_full"]), fam(C["oos_cal"])],
                        np.r_[ok["cal_unseen_full"], np.zeros(len(C["oos_cal"]["y"]), bool)], 0.02)
    thr_R = sel.pick_fixed_sequence(conf(C["cal_seen_full"]), ok["cal_seen_full"], 0.02)
    mix = {"test_seen": 0.7, "test_unseen": 0.2, "oos_test": 0.1}
    stacks = {}
    for m in (model_key, local):
        res = {}
        for pol_name in ("always-LLM", "fast[R] -> LLM", "fast[G]+V -> LLM", "fast[G]+V -> agent"):
            agg = {"wrong": 0.0, "llm_calls": 0.0, "agent": 0.0, "local": 0.0}
            for pool, w in mix.items():
                sub = S[pool]
                pos = np.array([np.flatnonzero(F.pools[pool].uid.values == uid)[0] for uid in sub.uid])
                o = F.out(pool)
                y = o["y"][pos]
                lab = per_pool[(pool, m)]
                llm_ok = np.array([l is not None and l != NONE for l in lab])
                if pol_name == "always-LLM":
                    acc = np.zeros(len(y), bool)
                elif pol_name.startswith("fast[R]"):
                    acc = conf(o)[pos] >= thr_R
                else:
                    acc = ((conf(o)[pos] >= gate["thr"]) & (fam(o)[pos] >= gate["floor"]) &
                           (o["tfidf-char+LR"]["pred"][pos] == o[M0]["pred"][pos]))
                use_llm = ~acc & (not pol_name.endswith("agent"))
                wrong = (acc & (o[M0]["pred"][pos] != y)) | (use_llm & llm_ok & (lab != y))
                agent = ~acc & (~llm_ok if use_llm.any() or not pol_name.endswith("agent") else True)
                if pol_name.endswith("agent"):
                    agent = ~acc
                agg["wrong"] += w * float(wrong.mean())
                agg["llm_calls"] += w * float(use_llm.mean())
                agg["agent"] += w * float(agent.mean())
                agg["local"] += w * float(acc.mean())
            res[pol_name] = agg
        stacks[m] = res
    out["stacks_mix_70_20_10"] = stacks

    for pool, row in out["populations"].items():
        print(f"  {pool:34s} " + " | ".join(f"{m}: correct {r['correct']:.3f} none {r['none_exit']:.2f} wrong/req {r['wrong_answers_per_request']:.3f}"
                                           for m, r in row.items()), flush=True)
    print("  bad inputs answered in scope:", {m: {k: f"{v['answered_in_scope']}/{v['n']}" for k, v in r.items() if isinstance(v, dict)}
                                           for m, r in out["bad_inputs"].items()}, flush=True)
    print("  changed contract:", out["changed_contract"], flush=True)
    print("  same wrong answer on unseen:", out["same_wrong_answer_as_fast_model_on_unseen"], flush=True)
    for m, res in stacks.items():
        for k, v in res.items():
            print(f"  stack {m:18s} {k:20s} wrong {v['wrong']:.4f} llm calls {v['llm_calls']:.2f} agent {v['agent']:.2f} local {v['local']:.2f}", flush=True)
    return out


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "prepare":
        prepare(sys.argv[2])
    elif cmd == "ingest":
        ingest(sys.argv[2], sys.argv[3])
    elif cmd == "score":
        save_json("exp10_closed.json", score(sys.argv[2]))
