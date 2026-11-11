"""Run the real local LLMs over the frozen pools and store every answer.

Usage: python -m bench.label_llm [stage ...]     (stages run in the order given; all are resumable)

Stages
  route       qwen3-4b router (1-shot contract) on every CLINC pool
  policy      qwen3-4b router with the changed contract on the policy pool
  verify      qwen3-4b yes/no verifier on the fast model's proposals
  gemma       gemma3-4b router on test/oos pool subsets (a second, differently built LLM)
  fine        qwen3-4b zero-shot 150-way / 77-way intent holes on test subsets
  massive     qwen3-4b scenario router on MASSIVE en-US and the same utterances in es-ES / de-DE
  latency     single-request latency probes
  bad         seeded bad inputs through the router hole
"""
from __future__ import annotations

import gc
import statistics
import sys
import time

import numpy as np

from . import data, embed, holes, kit, pools
from .common import save_json
from .llm import AnswerStore, LocalLLM, label_queries

_llms = {}


def llm(key: str) -> LocalLLM:
    if key not in _llms:
        for k in list(_llms):                      # one model in memory at a time
            del _llms[k]
        gc.collect()
        _llms[key] = LocalLLM(key)
        print(f"loaded {key} in {_llms[key].load_s:.1f}s", flush=True)
    return _llms[key]


def fast_proposals() -> dict:
    """Predictions of the reference fast model (bge-small + LR, fitted on the frozen train split)."""
    f = pools.frozen()
    tr = f.index[f.split == "train"].values
    X = kit.dense_view("bge-small", "clinc")
    m = kit.Fitted("bge-small+LR", "bge-small", kit.FAMILIES["bge-small+LR"][1](), X, tr, f.loc[tr, "label"].values)
    out = {}
    for name, p in pools.pools().items():
        if name in ("typo15", "policy", "traces"):
            continue
        r = m.run(p.index.values)
        out[name] = (list(p.text), list(r["pred"]))
    return out


def stage_route():
    h = holes.clinc_routing_hole()
    m = llm("qwen3-4b")
    for name in ("test_seen", "test_unseen", "oos_test", "cal_seen", "cal_unseen", "oos_val", "typo15", "traces"):
        label_queries(m, h, list(pools.pools()[name].text), batch_size=4)


def stage_policy():
    h = holes.clinc_routing_hole(policy_note=holes.POLICY_NOTE, name="clinc_route_policy")
    label_queries(llm("qwen3-4b"), h, list(pools.pools()["policy"].text), batch_size=4)


def stage_verify():
    h = holes.verify_hole()
    m = llm("qwen3-4b")
    for name, (texts, preds) in fast_proposals().items():
        if name in ("test_seen", "test_unseen", "oos_test"):
            label_queries(m, h, [holes.verify_query(t, p) for t, p in zip(texts, preds)], batch_size=8)


def gemma_subsets() -> dict:
    p = pools.pools()
    return {"test_seen": p["test_seen"].sort_values("uid").groupby("group").head(7),
            "test_unseen": p["test_unseen"].sort_values("uid").groupby("group").head(20),
            "oos_test": p["oos_test"].sort_values("uid").head(300)}


def stage_gemma():
    h = holes.clinc_routing_hole()
    m = llm("gemma3-4b")
    for name, sub in gemma_subsets().items():
        label_queries(m, h, list(sub.text), batch_size=4)


def fine_subsets() -> dict:
    c = data.fine_frame("clinc150")
    b = data.fine_frame("banking77")
    oos = data.clinc_oos()
    return {"clinc150": (c[c.split == "test"].sort_values("uid").groupby("label").head(5),
                         oos[oos.part == "test"].sort_values("uid").head(250)),
            "banking77": (b[b.split == "test"].sort_values("uid").groupby("label").head(7), None)}


def stage_fine():
    m = llm("qwen3-4b")
    for name, (sub, oos) in fine_subsets().items():
        h = holes.fine_hole(name)
        label_queries(m, h, list(sub.text) + ([] if oos is None else list(oos.text)), batch_size=8)


def massive_subsets() -> dict:
    f = data.massive_routing()
    seen = f[f.split == "test_seen"].sort_values("uid").sample(400, random_state=21)
    unseen = f[f.split == "test_unseen"].sort_values("uid").sample(400, random_state=22)
    out = {"en_seen": seen, "en_unseen": unseen}
    for loc in ("es-ES", "de-DE"):
        d = data.load_massive(loc).set_index("mid")
        s = seen.copy()
        s["text"] = d.loc[seen.mid.values, "text"].values
        out[f"{loc}_seen"] = s
    return out


def stage_massive():
    h = holes.massive_routing_hole()
    m = llm("qwen3-4b")
    for name, sub in massive_subsets().items():
        label_queries(m, h, list(sub.text), batch_size=4)


def stage_latency():
    p = pools.pools()
    qs = list(p["test_seen"].sort_values("uid").text[:60])
    out = {}
    def summ(ms):
        ms = sorted(ms)
        return {"p50": statistics.median(ms), "p90": ms[int(0.9 * len(ms))], "mean": statistics.mean(ms), "n": len(ms)}
    m = llm("qwen3-4b")
    route, zero, ver, fine = holes.clinc_routing_hole(), holes.clinc_routing_hole(shots=0), holes.verify_hole(), holes.fine_hole("clinc150")
    for name, h in (("route_1shot", route), ("route_0shot", zero), ("fine_150way", fine)):
        m.latency_probe(h, qs[:4])
        out[f"qwen3-4b/{name}"] = {**summ(m.latency_probe(h, qs)), "prompt_tokens": len(m._tokens(h, qs[0]))}
    vq = [holes.verify_query(q, "banking") for q in qs]
    m.latency_probe(ver, vq[:4])
    out["qwen3-4b/verify_yes_no"] = {**summ(m.latency_probe(ver, vq)), "prompt_tokens": len(m._tokens(ver, vq[0]))}
    out["qwen3-4b/route_1shot_no_prefix_cache"] = summ(m.latency_probe(route, qs[:12], use_prefix_cache=False))
    out["qwen3-4b/route_0shot_no_prefix_cache"] = summ(m.latency_probe(zero, qs[:12], use_prefix_cache=False))
    # batch throughput
    for bs in (4, 8):
        t0 = time.perf_counter()
        m.ask(route, qs[:48], batch_size=bs)
        out[f"qwen3-4b/route_1shot_batch{bs}_ms_per_item"] = (time.perf_counter() - t0) / 48 * 1e3
    g = llm("gemma3-4b")
    g.latency_probe(route, qs[:4])
    out["gemma3-4b/route_1shot"] = {**summ(g.latency_probe(route, qs[:40])), "prompt_tokens": len(g._tokens(route, qs[0]))}
    save_json("llm_latency.json", out)
    for k, v in out.items():
        print(k, v, flush=True)


def stage_bad():
    """Seeded bad inputs through the router hole: does the LLM tier take its 'none' exit?"""
    from .perturb import seeded_bad_inputs
    h = holes.clinc_routing_hole()
    qs = [t for v in seeded_bad_inputs(0).values() for t in v if t.strip()]
    label_queries(llm("qwen3-4b"), h, qs, batch_size=4)


STAGES = {"bad": stage_bad, "route": stage_route, "policy": stage_policy, "verify": stage_verify, "gemma": stage_gemma,
          "fine": stage_fine, "massive": stage_massive, "latency": stage_latency}

if __name__ == "__main__":
    for s in sys.argv[1:] or list(STAGES):
        t0 = time.perf_counter()
        print(f"=== stage {s}", flush=True)
        STAGES[s]()
        print(f"=== stage {s} done in {(time.perf_counter() - t0) / 60:.1f} min", flush=True)
