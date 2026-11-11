"""Experiment 9: a wrapper harness that watches an LLM do an open-ended task, then runs it faster.

The idea under test (the user's own framing): do not change the model. Wrap it. Let it do a task
such as "analyze my profile" the slow way while the harness records what it did; from those
recordings turn the task into a program in which the remaining decisions are much smaller than
the first time.

Task       "Analyze this profile": public resume profiles (opensporks/resumes, 2,484 profiles in
           24 human-labelled fields). The uncompiled agent is three free-form LLM calls:
             S1 extract   profile -> facts JSON {current_title, years_experience, education_level, skills}
             S2 assess    profile + facts -> JSON {field, seniority, strengths, gaps}
             S3 write     facts + assessment -> two-sentence summary
Observe    The harness records every call (prompt/answer tokens, time, parsed output) for
           `observed` profiles.
Compile    From observed traces only, per output field: infer its type, search a small family of
           candidate programs (code extractors, mined rules), keep the best, and admit it only if
           it agrees with the LLM on separate `admission` traces. Fields that are not admitted
           stay with the model, but as a narrow typed hole: one field, closed answer set, the
           smallest input that suffices. Free text stays a generation hole fed by compact facts.
Run        The compiled program on `test` profiles the compile step never saw, compared with the
           uncompiled agent's own output on the same profiles (fidelity), and with the human
           field label (quality).

Subcommands: collect | compile | run | report      (python -m bench.exp9_harness <cmd>)
"""
from __future__ import annotations

import json
import re
import statistics
import sys
import time
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from .common import CACHE, RAW, save_json
from .data import fetch

HDIR = CACHE / "harness"
HDIR.mkdir(parents=True, exist_ok=True)
TRACES = HDIR / "traces.jsonl"
MODEL = "qwen3-4b"
MAX_CHARS = 4000
N_OBS, N_ADMIT, N_TEST = 60, 40, 80
EDU = ["none", "high_school", "associate", "bachelor", "master", "doctorate"]
SENIORITY = ["junior", "mid", "senior", "executive"]


def profiles() -> pd.DataFrame:
    d = pd.read_csv(RAW / "resumes.csv")[["ID", "Resume_str", "Category"]]
    d["text"] = d.Resume_str.astype(str).str.replace(r"[ \t]+", " ", regex=True).str.strip()
    d = d[d.text.str.len() >= 800].sort_values("ID").reset_index(drop=True)
    pick = d.sample(N_OBS + N_ADMIT + N_TEST, random_state=20261111).reset_index(drop=True)
    pick["role"] = ["observed"] * N_OBS + ["admission"] * N_ADMIT + ["test"] * N_TEST
    pick["text"] = pick.text.str[:MAX_CHARS]
    return pick


FIELDS = sorted(pd.read_csv(fetch("resumes.csv"), usecols=["Category"]).Category.unique())   # downloads on first use


# --------------------------------------------------------------------------- the uncompiled agent
def p_extract(text: str) -> str:
    return ("You are analyzing a person's professional profile. Read it and extract the facts as a JSON object "
            "with exactly these keys:\n"
            '- "current_title": the most recent job title (string)\n'
            '- "years_experience": total years of professional experience as an integer (work it out from the dates)\n'
            f'- "education_level": one of {", ".join(EDU)}\n'
            '- "skills": a list of up to 6 short skill names taken from the profile\n\n'
            f"Profile:\n{text}\n\nReply with only the JSON object.")


def p_assess(text: str, facts: dict) -> str:
    return ("You are analyzing a person's professional profile. Using the profile and the extracted facts, assess it "
            "as a JSON object with exactly these keys:\n"
            f'- "field": the professional field, one of {", ".join(FIELDS)}\n'
            f'- "seniority": one of {", ".join(SENIORITY)}\n'
            '- "strengths": a list of 2 short phrases\n'
            '- "gaps": a list of 2 short phrases (what is missing or weak)\n\n'
            f"Profile:\n{text}\n\nExtracted facts:\n{json.dumps(facts)}\n\nReply with only the JSON object.")


def p_write(facts: dict, assess: dict) -> str:
    return ("Write a two-sentence summary of this professional profile for a recruiter. Use only the facts and "
            f"assessment below.\n\nFacts:\n{json.dumps(facts)}\n\nAssessment:\n{json.dumps(assess)}\n\n"
            "Reply with only the two sentences.")


def parse_json(raw: str):
    m = re.search(r"\{.*\}", raw, flags=re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def clean_facts(f) -> dict | None:
    if not isinstance(f, dict):
        return None
    try:
        yrs = f.get("years_experience")
        yrs = int(round(float(yrs))) if yrs is not None and str(yrs).strip() != "" else None
    except Exception:
        yrs = None
    edu = str(f.get("education_level", "")).strip().lower().replace(" ", "_")
    skills = f.get("skills") if isinstance(f.get("skills"), list) else []
    return {"current_title": str(f.get("current_title", "")).strip(), "years_experience": yrs,
            "education_level": edu if edu in EDU else None,
            "skills": [str(s).strip() for s in skills if str(s).strip()][:6]}


def clean_assess(a) -> dict | None:
    if not isinstance(a, dict):
        return None
    field = str(a.get("field", "")).strip().upper().replace(" ", "-").replace("_", "-")
    sen = str(a.get("seniority", "")).strip().lower()
    lst = lambda v: [str(s).strip() for s in v][:2] if isinstance(v, list) else []
    return {"field": field if field in FIELDS else None, "seniority": sen if sen in SENIORITY else None,
            "strengths": lst(a.get("strengths")), "gaps": lst(a.get("gaps"))}


def run_agent(llm, texts: list, batch_size: int = 4) -> list:
    """The uncompiled agent on a batch of profiles. Every call is recorded."""
    r1 = llm.chat([p_extract(t) for t in texts], max_tokens=160, batch_size=batch_size)
    facts = [clean_facts(parse_json(r["text"])) for r in r1]
    safe = [f or {"current_title": "", "years_experience": None, "education_level": None, "skills": []} for f in facts]
    r2 = llm.chat([p_assess(t, f) for t, f in zip(texts, safe)], max_tokens=140, batch_size=batch_size)
    assess = [clean_assess(parse_json(r["text"])) for r in r2]
    safe2 = [a or {"field": None, "seniority": None, "strengths": [], "gaps": []} for a in assess]
    r3 = llm.chat([p_write(f, a) for f, a in zip(safe, safe2)], max_tokens=110, batch_size=batch_size)
    out = []
    for i in range(len(texts)):
        calls = [{"step": s, "n_prompt": r[i]["n_prompt"], "n_gen": r[i]["n_gen"],
                  "batch_s_per_item": r[i]["batch_s_per_item"], "raw": r[i]["text"][:1200]}
                 for s, r in (("extract", r1), ("assess", r2), ("write", r3))]
        out.append({"facts": facts[i], "assess": assess[i], "summary": r3[i]["text"], "calls": calls,
                    "valid": facts[i] is not None and assess[i] is not None})
    return out


def collect():
    from .llm import LocalLLM
    P = profiles()
    done = set()
    if TRACES.exists():
        done = {json.loads(line)["ID"] for line in TRACES.read_text().splitlines()}
    todo = P[~P.ID.isin(done)]
    llm = LocalLLM(MODEL)
    t0 = time.perf_counter()
    for i in range(0, len(todo), 8):
        chunk = todo.iloc[i:i + 8]
        res = run_agent(llm, list(chunk.text), batch_size=4)
        with open(TRACES, "a") as f:
            for (_, row), r in zip(chunk.iterrows(), res):
                f.write(json.dumps({"ID": int(row.ID), "role": row.role, "category": row.Category, **r}) + "\n")
        print(f"  agent traces {i + len(chunk)}/{len(todo)}  {(time.perf_counter() - t0) / (i + len(chunk)):.1f} s/profile",
              flush=True)
    # single-request latency of the uncompiled agent (batch size 1), on observed profiles only
    obs = P[P.role == "observed"].head(12)
    lat = []
    for t in obs.text:
        s = time.perf_counter()
        run_agent(llm, [t], batch_size=1)
        lat.append(time.perf_counter() - s)
    (HDIR / "agent_latency.json").write_text(json.dumps({"single_request_s": lat}))
    print("agent single-request s: p50", statistics.median(lat), flush=True)


def load_traces() -> pd.DataFrame:
    rows = [json.loads(line) for line in TRACES.read_text().splitlines()]
    T = pd.DataFrame(rows).drop_duplicates("ID")
    return profiles().merge(T.drop(columns=["role"]), on="ID", how="inner")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "collect"
    if cmd == "collect":
        collect()
    else:
        from . import exp9_compile
        getattr(exp9_compile, cmd)()
