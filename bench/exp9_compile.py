"""Compile, run and score the observing harness (see exp9_harness.py for the experiment).

compile   observed traces -> per-field candidate search -> admission on separate traces
          -> results/exp9_spec.json  (which fields became code, which stay holes, and why)
run       compiled program on test profiles (real LLM for the remaining holes), the agent's
          self-agreement under a reworded instruction, and single-request latencies
"""
from __future__ import annotations

import json
import re
import statistics
import time
from collections import Counter

import numpy as np
from sklearn.tree import DecisionTreeClassifier

from .common import RESULTS, save_json
from .exp9_harness import (EDU, FIELDS, HDIR, MODEL, SENIORITY, clean_assess, clean_facts, load_traces, p_extract,
                           p_write, parse_json, run_agent)

YEAR = r"(?:19[6-9]\d|20[0-2]\d)"
ADMIT = {"years_experience": 0.80, "education_level": 0.85, "seniority": 0.80, "field": 0.90,
         "skills": 0.50, "current_title": 0.80}


def norm(s) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()


# --------------------------------------------------------------------------- candidate programs
def _explicit(text):
    return [int(x) for x in re.findall(r"(\d{1,2})\s*\+?\s*(?:years|yrs)", text, flags=re.I) if 0 < int(x) < 50]


def _years(text):
    return [int(y) for y in re.findall(rf"\b({YEAR})\b", text)]


def _ranges(text):
    """Union length (years) of 'MM/YYYY to MM/YYYY|Current' style ranges."""
    ys = _years(text)
    now = max(ys) if ys else None
    spans = []
    for a, b in re.findall(rf"({YEAR})\s*(?:to|-|–|—)\s*(?:[A-Za-z]{{3,9}}\.?\s+|\d{{1,2}}/)?({YEAR}|current|present)", text, flags=re.I):
        end = now if b.lower() in ("current", "present") else int(b)
        if end is not None and end >= int(a):
            spans.append((int(a), end))
    if not spans:
        return None
    spans.sort()
    total, (cs, ce) = 0, spans[0]
    for s, e in spans[1:]:
        if s <= ce:
            ce = max(ce, e)
        else:
            total += ce - cs
            cs, ce = s, e
    return total + ce - cs


def y_explicit_first(t):
    e = _explicit(t)
    return e[0] if e else None


def y_explicit_max(t):
    e = _explicit(t)
    return max(e) if e else None


def y_span(t):
    ys = _years(t)
    return min(45, max(ys) - min(ys)) if len(ys) >= 2 else None


def y_ranges(t):
    return _ranges(t)


def _else(*fs):
    def f(t):
        for g in fs:
            v = g(t)
            if v is not None:
                return v
        return None
    f.__name__ = " else ".join(g.__name__ for g in fs)
    return f


def y_since_first_job(now: int):
    """Years from the earliest year in the profile to an assumed 'now', when a job is marked
    Current/Present. Observed traces suggest the agent counts this way; `now` is fitted."""
    def f(t):
        ys = _years(t)
        if not ys:
            return None
        if re.search(r"\b(current|present)\b", t, flags=re.I):
            return max(0, min(45, now - min(ys)))
        return min(45, max(ys) - min(ys))
    f.__name__ = f"y_since_first_job(now={now})"
    return f


YEARS_CANDS = ([y_explicit_first, y_explicit_max, y_span, y_ranges, _else(y_explicit_first, y_ranges),
                _else(y_explicit_first, y_span), _else(y_ranges, y_span), _else(y_ranges, y_explicit_max),
                _else(y_explicit_max, y_span)]
               + [y_since_first_job(n) for n in range(2016, 2026)]
               + [_else(y_explicit_max, y_since_first_job(n)) for n in range(2016, 2026)])

DEG = [("doctorate", r"\b(ph\.?\s?d|doctorate|doctor of|j\.?d\.?|m\.?d\.?)\b"),
       ("master", r"\b(master|masters|mba|m\.?s\.?c?|m\.?a\.?|m\.?ed)\b"),
       ("bachelor", r"\b(bachelor|bachelors|b\.?s\.?c?|b\.?a\.?|b\.?b\.?a|b\.?tech|b\.?e\.?)\b"),
       ("associate", r"\b(associate|a\.?a\.?s?|a\.?s\.?)\b"),
       ("high_school", r"\b(high school|ged|diploma)\b")]
DEG_STRICT = [("doctorate", r"\b(ph\.?d|doctorate|doctor of)\b"), ("master", r"\b(master|masters|mba)\b"),
              ("bachelor", r"\b(bachelor|bachelors)\b"), ("associate", r"\bassociate of\b|\bassociate degree\b|\bassociates?\b(?= (of|in|degree))"),
              ("high_school", r"\b(high school|ged)\b")]


def _edu_section(t):
    m = re.search(r"\beducation\b", t, flags=re.I)
    return t[m.start():m.start() + 700] if m else ""


def _deg(text, table):
    low = text.lower()
    for name, rx in table:
        if re.search(rx, low):
            return name
    return None


def e_any_loose(t): return _deg(t, DEG) or "none"
def e_any_strict(t): return _deg(t, DEG_STRICT) or "none"
def e_section_loose(t): return _deg(_edu_section(t), DEG) or "none"
def e_section_strict(t): return _deg(_edu_section(t), DEG_STRICT) or "none"
def e_section_strict_else_any(t): return _deg(_edu_section(t), DEG_STRICT) or _deg(t, DEG_STRICT) or "none"


def e_keyword_else_bachelor(t):
    """NOT grounded: says 'bachelor' when the profile shows no degree at all. It is in the search
    only to show what compiling from traces alone would pick up; it can never be admitted."""
    return _deg(t, DEG_STRICT) or "bachelor"


e_keyword_else_bachelor.ungrounded = True
EDU_CANDS = [e_any_loose, e_any_strict, e_section_loose, e_section_strict, e_section_strict_else_any, e_keyword_else_bachelor]


def t_heading(t):
    return t.strip().split("\n")[0].strip()[:80]


def t_first_job(t):
    m = re.search(r"\b(?:experience|work history)\b(.{0,400})", t, flags=re.I | re.S)
    if not m:
        return None
    seg = m.group(1)
    d = re.search(rf"(\d{{1,2}}/)?{YEAR}|\b(january|february|march|april|may|june|july|august|september|october|november|december)\b", seg, flags=re.I)
    cand = seg[: d.start()] if d else seg[:60]
    cand = re.sub(r"\s+", " ", cand).strip(" :-|")
    return cand[-60:] if cand else None


TITLE_CANDS = [t_heading, t_first_job, _else(t_first_job, t_heading)]


def _items(seg):
    parts = re.split(r"\s{2,}|,|;|\n|•|\|", seg)
    out = []
    for p in parts:
        p = p.strip(" .:-")
        if 2 <= len(p) <= 40 and len(p.split()) <= 4:
            out.append(p)
    return out


def s_section(t):
    m = re.search(r"\b(skills|highlights|core qualifications|areas of expertise)\b[:\s]", t, flags=re.I)
    return _items(t[m.end(): m.end() + 500])[:6] if m else None


def make_gazetteer(observed_skills):
    vocab = Counter(norm(s) for sk in observed_skills for s in sk)
    terms = [w for w, c in vocab.most_common() if len(w) > 2]

    def s_gazetteer(t):
        low = " " + norm(t) + " "
        hits = [(low.find(" " + w + " "), w) for w in terms if (" " + w + " ") in low]
        return [w for _, w in sorted(hits)][:6] or None
    return s_gazetteer


FIELD_KEYS = {"ACCOUNTANT": ["accountant", "accounting"], "ADVOCATE": ["advocate", "attorney", "lawyer", "legal"],
              "AGRICULTURE": ["agriculture", "agricultural", "farm"], "APPAREL": ["apparel", "fashion", "merchandis"],
              "ARTS": ["arts", "artist"], "AUTOMOBILE": ["automobile", "automotive"], "AVIATION": ["aviation", "pilot", "aircraft"],
              "BANKING": ["banking", "bank", "teller"], "BPO": ["bpo", "call center"], "BUSINESS-DEVELOPMENT": ["business development"],
              "CHEF": ["chef", "cook", "culinary"], "CONSTRUCTION": ["construction"], "CONSULTANT": ["consultant"],
              "DESIGNER": ["designer"], "DIGITAL-MEDIA": ["digital media", "digital marketing", "social media"],
              "ENGINEERING": ["engineer", "engineering"], "FINANCE": ["finance", "financial"], "FITNESS": ["fitness", "trainer"],
              "HEALTHCARE": ["healthcare", "health care", "nurse", "medical"], "HR": ["hr", "human resources"],
              "INFORMATION-TECHNOLOGY": ["information technology", "it"], "PUBLIC-RELATIONS": ["public relations", "pr"],
              "SALES": ["sales"], "TEACHER": ["teacher", "teaching", "instructor"]}


def f_heading_keyword(t):
    """Rule written from the contract (the label names), not from traces: a label keyword in the heading."""
    head = " " + norm(t_heading(t)) + " "
    hits = [f for f, keys in FIELD_KEYS.items() if any((" " + k + " ") in head for k in keys)]
    return hits[0] if len(hits) == 1 else None


def f_heading_or_title_keyword(t):
    v = f_heading_keyword(t)
    if v:
        return v
    head = " " + norm(t[:300]) + " "
    hits = [f for f, keys in FIELD_KEYS.items() if any((" " + k + " ") in head for k in keys)]
    return hits[0] if len(hits) == 1 else None


FIELD_CANDS = [f_heading_keyword, f_heading_or_title_keyword]

SEN_FLAGS = [("exec", r"\b(director|vice president|vp|chief|ceo|cfo|cto|president|head of|owner|executive)\b"),
             ("senior", r"\b(senior|sr|lead|principal|manager|supervisor)\b"),
             ("junior", r"\b(junior|jr|intern|assistant|trainee|entry)\b")]


def sen_features(text, years):
    head = text[:400].lower()
    return [(-1 if years is None else years)] + [1.0 if re.search(rx, head) else 0.0 for _, rx in SEN_FLAGS]


# --------------------------------------------------------------------------- agreement measures
def agree_years(a, b, tol=2): return a is not None and b is not None and abs(a - b) <= tol
def agree_exact(a, b): return a is not None and a == b


def agree_title(a, b):
    a, b = norm(a), norm(b)
    return bool(a) and bool(b) and (a == b or a in b or b in a)


def skill_overlap(a, b):
    """Share of the LLM's skills (b) that the program's list (a) also names (token-set match)."""
    if not a or not b:
        return 0.0
    A = [set(norm(x).split()) for x in a]
    hit = 0
    for y in b:
        Y = set(norm(y).split())
        if any(Y and (Y <= X or X <= Y) for X in A if X):
            hit += 1
    return hit / len(b)


def search(cands, texts, targets, agree):
    """Best candidate by mean agreement with the LLM on observed traces."""
    scored = []
    for c in cands:
        vals = [c(t) for t in texts]
        scored.append((float(np.mean([agree(v, y) for v, y in zip(vals, targets)])), c))
    scored.sort(key=lambda x: -x[0])
    return scored


def compile():   # noqa: A001
    T = load_traces()
    T = T[T.valid]
    obs, adm = T[T.role == "observed"], T[T.role == "admission"]
    get = lambda d, part, key: [(r[part] or {}).get(key) for _, r in d.iterrows()]
    spec = {"n_observed": int(len(obs)), "n_admission": int(len(adm)), "fields": {}}
    chosen = {}

    def decide(name, cands, part, agree, selective=False):
        tx_o, tx_a = list(obs.text), list(adm.text)
        ranked_all = search(cands, tx_o, get(obs, part, name), agree)
        ranked = [(sc, c) for sc, c in ranked_all if not getattr(c, "ungrounded", False)]   # the contract outranks the traces
        best_score, best = ranked[0]
        vals = [best(t) for t in tx_a]
        tgt = get(adm, part, name)
        if selective:                              # the rule may abstain; judge it on what it answers
            ans = [v is not None for v in vals]
            cov = float(np.mean(ans))
            adm_score = float(np.mean([agree(v, y) for v, y, a in zip(vals, tgt, ans) if a])) if any(ans) else 0.0
            ok = adm_score >= ADMIT[name] and sum(ans) >= 10
        else:
            cov = 1.0
            adm_score = float(np.mean([agree(v, y) for v, y in zip(vals, tgt)]))
            ok = adm_score >= ADMIT[name]
        ung = [(sc, c) for sc, c in ranked_all if getattr(c, "ungrounded", False)]
        spec["fields"][name] = {"candidates": {c.__name__: round(s, 3) for s, c in ranked}, "chosen": best.__name__,
                                "rejected_ungrounded": {c.__name__: round(sc, 3) for sc, c in ung},
                                "observed_agreement": round(best_score, 3), "admission_agreement": round(adm_score, 3),
                                "admission_coverage": round(cov, 3), "bar": ADMIT[name], "admitted": bool(ok),
                                "compiled_to": "code" if ok else "narrow LLM hole"}
        chosen[name] = best if ok else None
        print(f"  {name:17s} best={best.__name__:38s} observed {best_score:.2f}  admission {adm_score:.2f} "
              f"(coverage {cov:.2f}, bar {ADMIT[name]}) -> {'CODE' if ok else 'stays a hole'}", flush=True)

    decide("years_experience", YEARS_CANDS, "facts", agree_years)
    decide("education_level", EDU_CANDS, "facts", agree_exact)
    decide("current_title", TITLE_CANDS, "facts", agree_title)
    gaz = make_gazetteer(get(obs, "facts", "skills"))
    decide("skills", [s_section, gaz, _else(s_section, gaz)], "facts", lambda a, b: skill_overlap(a, b))
    decide("field", FIELD_CANDS, "assess", agree_exact, selective=True)

    # seniority: a small tree over (years, title keyword flags), learned from the LLM's own answers
    ybest = chosen["years_experience"] or search(YEARS_CANDS, list(obs.text), get(obs, "facts", "years_experience"), agree_years)[0][1]
    Xo = np.array([sen_features(t, ybest(t)) for t in obs.text])
    yo = np.array([s or "mid" for s in get(obs, "assess", "seniority")])
    tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=4, random_state=0).fit(Xo, yo)
    Xa = np.array([sen_features(t, ybest(t)) for t in adm.text])
    ya = get(adm, "assess", "seniority")
    sen_adm = float(np.mean(tree.predict(Xa) == np.array(ya, dtype=object)))
    sen_obs = float(np.mean(tree.predict(Xo) == yo))
    majority = Counter(yo).most_common(1)[0]
    spec["fields"]["seniority"] = {"candidates": {"decision tree(depth 3) on years + title keywords": round(sen_obs, 3)},
                                   "chosen": "decision tree", "observed_agreement": round(sen_obs, 3),
                                   "admission_agreement": round(sen_adm, 3), "bar": ADMIT["seniority"],
                                   "majority_class_baseline": round(float(np.mean(np.array(ya, dtype=object) == majority[0])), 3),
                                   "admitted": bool(sen_adm >= ADMIT["seniority"]),
                                   "compiled_to": "code" if sen_adm >= ADMIT["seniority"] else "narrow LLM hole"}
    print(f"  {'seniority':17s} tree on years+title flags: observed {sen_obs:.2f} admission {sen_adm:.2f} "
          f"(majority baseline {spec['fields']['seniority']['majority_class_baseline']:.2f}) -> "
          f"{'CODE' if sen_adm >= ADMIT['seniority'] else 'stays a hole'}", flush=True)
    for k in ("strengths", "gaps", "summary"):
        spec["fields"][k] = {"type": "free text", "compiled_to": "generation hole fed by compact facts (or a template)"}
    # Audit of the agent: did it report a degree when the text it was given shows no sign of one?
    # "Evidence" is deliberately broad: any degree word or abbreviation, or any mention of
    # education, school, college, university, graduation or a diploma.
    ev = re.compile(r"bachelor|master|mba|ph\.?d|doctorate|degree|universit|college|diploma|graduat|school|education|"
                    r"\bB\.[AS]\.|\bM\.[AS]\.|\bBBA\b|\bBSc\b", flags=re.I)
    both = T[T.role.isin(["observed", "admission"])]
    edu_o = get(both, "facts", "education_level")
    no_ev = [ev.search(t) is None for t in both.text]
    spec["agent_audit"] = {
        "runs_audited": int(len(both)), "profiles_with_no_education_evidence_in_input": int(sum(no_ev)),
        "of_those_agent_still_reported_a_degree": int(sum(1 for e, n in zip(edu_o, no_ev) if n and e not in (None, "none"))),
        "agent_reported_bachelor_overall": float(np.mean([e == "bachelor" for e in edu_o]))}
    # The contract-derived field rule against the human label, on the rows where it answers.
    fr = [f_heading_keyword(t) for t in both.text]
    ans = [v is not None for v in fr]
    af = get(both, "assess", "field")
    spec["field_rule_vs_human"] = {
        "rows": int(len(both)), "rule_answers": int(sum(ans)),
        "rule_matches_human_label": float(np.mean([v == c for v, c, a in zip(fr, both.Category, ans) if a])),
        "agent_matches_human_label_same_rows": float(np.mean([v == c for v, c, a in zip(af, both.Category, ans) if a])),
        "rule_matches_agent": float(np.mean([v == x for v, x, a in zip(fr, af, ans) if a]))}
    spec["agent_field_vs_human_label_on_observed"] = float(np.mean(np.array(get(obs, "assess", "field"), dtype=object) == obs.Category.values))
    spec["tokens_per_agent_run"] = {"prompt": float(np.mean([sum(c["n_prompt"] for c in cs) for cs in T.calls])),
                                    "generated": float(np.mean([sum(c["n_gen"] for c in cs) for cs in T.calls]))}
    save_json("exp9_spec.json", spec)
    return spec, chosen, tree, ybest, gaz


# --------------------------------------------------------------------------- the compiled program
def hole_prompt(text: str, need: list) -> str:
    desc = {"current_title": '"current_title": the most recent job title (string)',
            "years_experience": '"years_experience": total years of professional experience as an integer',
            "education_level": f'"education_level": one of {", ".join(EDU)}',
            "skills": '"skills": a list of up to 6 short skill names taken from the profile',
            "field": f'"field": one of {", ".join(FIELDS)}',
            "seniority": f'"seniority": one of {", ".join(SENIORITY)}'}
    return ("Read this professional profile and reply with a JSON object with exactly these keys:\n" +
            "\n".join("- " + desc[k] for k in need) + f"\n\nProfile:\n{text}\n\nReply with only the JSON object.")


def prose_prompt(facts: dict) -> str:
    return ("From these facts about a professional profile, reply with a JSON object with keys "
            '"strengths" (2 short phrases), "gaps" (2 short phrases) and "summary" (two sentences for a recruiter).\n\n'
            f"Facts:\n{json.dumps(facts)}\n\nReply with only the JSON object.")


def template_summary(f: dict) -> str:
    sk = ", ".join((f.get("skills") or [])[:3])
    return (f"{str(f.get('seniority') or 'mid').capitalize()}-level {f.get('current_title') or 'professional'} with "
            f"{f.get('years_experience') if f.get('years_experience') is not None else 'several'} years of experience in "
            f"{str(f.get('field') or 'their field').replace('-', ' ').title()}. Key skills: {sk or 'not listed'}.")


def guards(text: str) -> str | None:
    if not isinstance(text, str) or len(text) < 400:
        return "profile too short"
    if sum(c.isalpha() for c in text) / len(text) < 0.6:
        return "profile is not prose"
    if not re.search(r"\b(experience|work history|employment)\b", text, flags=re.I):
        return "no experience section"
    return None


def run():
    from . import embed
    from .llm import LocalLLM
    spec, chosen, tree, ybest, gaz = compile()
    T = load_traces()
    test = T[(T.role == "test") & T.valid].reset_index(drop=True)
    llm = LocalLLM(MODEL)
    short_fields = ["current_title", "years_experience", "education_level", "skills", "field", "seniority"]
    HOLE_CHARS = 1800

    def structured(text: str):
        """Code for admitted fields, one narrow typed call for the rest. Returns (output, stats)."""
        st = {"calls": 0, "n_prompt": 0, "n_gen": 0, "code_fields": [], "hole_fields": [], "guard": guards(text)}
        if st["guard"]:
            return None, st                                   # hand the whole profile to the agent
        out = {}
        for k in ("years_experience", "education_level", "current_title", "skills", "field"):
            v = chosen[k](text) if chosen.get(k) else None
            if v is not None:
                out[k] = v
                st["code_fields"].append(k)
        if spec["fields"]["seniority"]["admitted"]:
            out["seniority"] = str(tree.predict(np.array([sen_features(text, out.get("years_experience", ybest(text)))]))[0])
            st["code_fields"].append("seniority")
        need = [k for k in short_fields if k not in out]
        if need:
            r = llm.chat([hole_prompt(text[:HOLE_CHARS], need)], max_tokens=30 + 22 * len(need), batch_size=1)[0]
            st["calls"] += 1
            st["n_prompt"] += r["n_prompt"]
            st["n_gen"] += r["n_gen"]
            got = parse_json(r["text"]) or {}
            f = clean_facts(got) or {}
            a = clean_assess(got) or {}
            for k in need:
                out[k] = f.get(k) if k in ("current_title", "years_experience", "education_level", "skills") else a.get(k)
            st["hole_fields"] = need
        return out, st

    def prose(out: dict):
        r = llm.chat([prose_prompt({k: out.get(k) for k in short_fields})], max_tokens=150, batch_size=1)[0]
        g = parse_json(r["text"]) or {}
        return {"strengths": g.get("strengths", []), "gaps": g.get("gaps", []), "summary": str(g.get("summary", ""))}, r

    res = {"n_test": int(len(test)), "modes": {}}
    agent_fields = {k: [(r["facts"] if k in ("current_title", "years_experience", "education_level", "skills") else r["assess"]).get(k)
                        for _, r in test.iterrows()] for k in short_fields}
    # One pass: the structured program is run and timed once per profile; the prose hole is timed
    # as an additional call on its output; the template adds no model call.
    outs, stats, lat_s, lat_p, prose_out, prose_tok = [], [], [], [], [], []
    for t in test.text:
        t0 = time.perf_counter()
        o, st = structured(t)
        lat_s.append(time.perf_counter() - t0)
        outs.append(o)
        stats.append(st)
        if o is not None:
            t0 = time.perf_counter()
            po, r = prose(o)
            lat_p.append(time.perf_counter() - t0)
            prose_out.append(po)
            prose_tok.append((r["n_prompt"], r["n_gen"]))
        else:
            lat_p.append(0.0)
            prose_out.append(None)
            prose_tok.append((0, 0))
    ok = [o is not None for o in outs]
    ag = {"years_experience(±2)": np.mean([agree_years(o["years_experience"], y) for o, y, k in zip(outs, agent_fields["years_experience"], ok) if k]),
          "years_experience(exact)": np.mean([o["years_experience"] == y for o, y, k in zip(outs, agent_fields["years_experience"], ok) if k]),
          "education_level": np.mean([o["education_level"] == y for o, y, k in zip(outs, agent_fields["education_level"], ok) if k]),
          "current_title": np.mean([agree_title(o["current_title"], y) for o, y, k in zip(outs, agent_fields["current_title"], ok) if k]),
          "skills(overlap)": np.mean([skill_overlap(o["skills"], y) for o, y, k in zip(outs, agent_fields["skills"], ok) if k]),
          "field": np.mean([o["field"] == y for o, y, k in zip(outs, agent_fields["field"], ok) if k]),
          "seniority": np.mean([o["seniority"] == y for o, y, k in zip(outs, agent_fields["seniority"], ok) if k])}
    a_sum = [r["summary"] for _, r in test.iterrows()]
    Ea = embed.encode("bge-small", a_sum)
    floor = float((Ea * np.roll(Ea, 1, axis=0)).sum(1).mean())
    summ = lambda xs: {"p50": statistics.median(xs), "mean": statistics.mean(xs), "p90": sorted(xs)[int(0.9 * len(xs))]}
    for mode in ("structured", "prose_hole", "template"):
        extra_lat = lat_p if mode == "prose_hole" else [0.0] * len(lat_s)
        extra_tok = prose_tok if mode == "prose_hole" else [(0, 0)] * len(lat_s)
        row = {"fallback_to_agent": float(1 - np.mean(ok)), "agreement_with_agent": {k: float(v) for k, v in ag.items()},
               "field_accuracy_vs_human": float(np.mean([o["field"] == c for o, c, k in zip(outs, test.Category, ok) if k])),
               "agent_field_accuracy_vs_human": float(np.mean(np.array(agent_fields["field"], dtype=object) == test.Category.values)),
               "llm_calls": float(np.mean([s["calls"] + (1 if mode == "prose_hole" and k else 0) for s, k in zip(stats, ok)])),
               "prompt_tokens": float(np.mean([s["n_prompt"] + e[0] for s, e in zip(stats, extra_tok)])),
               "generated_tokens": float(np.mean([s["n_gen"] + e[1] for s, e in zip(stats, extra_tok)])),
               "latency_s": summ([a + b for a, b in zip(lat_s, extra_lat)]),
               "code_fields": dict(Counter(k for s in stats for k in s["code_fields"])),
               "hole_fields": dict(Counter(k for s in stats for k in s["hole_fields"]))}
        if mode != "structured":
            c_sum = [(po["summary"] if mode == "prose_hole" else template_summary(o)) if o is not None else "" for o, po in zip(outs, prose_out)]
            Ec = embed.encode("bge-small", c_sum)
            row["summary_cosine_to_agent"] = float((Ea * Ec).sum(1).mean())
            row["summary_cosine_floor_other_profile"] = floor
        res["modes"][mode] = row
        print(f"  {mode:11s} calls {row['llm_calls']:.2f} gen {row['generated_tokens']:.0f} tok  latency p50 {row['latency_s']['p50']:.2f}s  "
              + " ".join(f"{k}={v:.2f}" for k, v in row["agreement_with_agent"].items()), flush=True)

    # the agent on the same machine, same conditions: single-request latency and token counts
    lat, toks = [], []
    for t in list(test.text)[:25]:
        t0 = time.perf_counter()
        r = run_agent(llm, [t], batch_size=1)[0]
        lat.append(time.perf_counter() - t0)
        toks.append((sum(c["n_prompt"] for c in r["calls"]), sum(c["n_gen"] for c in r["calls"])))
    res["agent"] = {"llm_calls": 3, "latency_s": {"p50": statistics.median(lat), "mean": statistics.mean(lat), "p90": sorted(lat)[int(0.9 * len(lat))]},
                    "prompt_tokens": float(np.mean([a for a, _ in toks])), "generated_tokens": float(np.mean([b for _, b in toks]))}
    # how stable is the agent itself? same profiles, extraction instruction reworded
    reworded = lambda text: p_extract(text).replace("Read it and extract the facts", "Go through it carefully and pull out the key facts") \
        .replace("work it out from the dates", "estimate it from the employment dates")
    r2 = llm.chat([reworded(t) for t in list(test.text)[:60]], max_tokens=160, batch_size=4)
    f2 = [clean_facts(parse_json(r["text"])) or {} for r in r2]
    sub = test.head(60)
    res["agent_self_agreement_reworded_prompt"] = {
        "n": 60,
        "years_experience(±2)": float(np.mean([agree_years(a.get("years_experience"), r["facts"]["years_experience"]) for a, (_, r) in zip(f2, sub.iterrows())])),
        "years_experience(exact)": float(np.mean([a.get("years_experience") == r["facts"]["years_experience"] for a, (_, r) in zip(f2, sub.iterrows())])),
        "education_level": float(np.mean([a.get("education_level") == r["facts"]["education_level"] for a, (_, r) in zip(f2, sub.iterrows())])),
        "current_title": float(np.mean([agree_title(a.get("current_title"), r["facts"]["current_title"]) for a, (_, r) in zip(f2, sub.iterrows())])),
        "skills(overlap)": float(np.mean([skill_overlap(a.get("skills"), r["facts"]["skills"]) for a, (_, r) in zip(f2, sub.iterrows())]))}
    # guard test: inputs that are not profiles must be handed back, not analysed
    bad = ["", "hello", "lorem ipsum dolor sit amet " * 30, "1234567890 " * 60,
           "Preheat the oven to 200C. Mix flour, sugar and butter until crumbly, then press into a tin and bake. " * 6,
           "def main():\n    print('hello world')\n" * 20, "The quick brown fox jumps over the lazy dog. " * 12]
    res["guard_test"] = {"bad_inputs": len(bad), "rejected": int(sum(guards(b) is not None for b in bad))}
    res["spec"] = spec
    save_json("exp9_harness.json", res)
    print("  agent:", res["agent"], flush=True)
    print("  agent self-agreement (reworded):", res["agent_self_agreement_reworded_prompt"], flush=True)
    print("  guard test:", res["guard_test"], flush=True)
    return res
