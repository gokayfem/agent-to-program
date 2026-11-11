"""admit: decide whether a compiled program may enter service.

Ported from the `workflow-compiler.skill` artifact (scripts/admit.py), with the
silent-error upper bound switched to an exact (Clopper-Pearson) bound and the
verifier cost reported separately.

A program is admitted on evidence, before it runs on real traffic:

  - held-out cases the compiler never saw, compared with reviewed outputs;
  - silent errors: fast-path answers that are wrong (the number that matters);
  - guard test: seeded bad inputs must all be handed to the agent;
  - determinism: repeated runs give identical outputs and paths;
  - cost and speed: LLM calls per run, share of runs needing no LLM, per-tier latency.

Nothing here relaxes a threshold. A failed check is reported as failed.
"""
from __future__ import annotations

import json
import statistics
import time
from collections import defaultdict
from typing import Any, Callable, Sequence

from .nsp import TIER_ORDER, Escalate, Program
from .selective import clopper_pearson_upper


def _run(program: Program, x: Any):
    try:
        return program.run(x)
    except Escalate:                       # no fallback configured: still a rejection, not an answer
        return None


def admit(program: Program, cases: Sequence[Any], expected: Sequence[Any],
          bad_inputs: Sequence[Any] = (), equal: Callable[[Any, Any], bool] = lambda a, b: a == b,
          max_silent_error: float = 0.01, repeats: int = 2, bound_must_pass: bool = False) -> dict:
    """`bound_must_pass=True` admits only if the 95% upper bound (not the point
    estimate) of the silent-error rate is within budget."""
    n = len(cases)
    by_tier: dict = defaultdict(lambda: {"n": 0, "wrong": 0})
    tier_ms: dict = defaultdict(list)
    wall_local: list = []
    verify_ms: list = []
    fast = local = silent = correct = llm_calls = 0
    first: list = []

    for x, want in zip(cases, expected):
        t0 = time.perf_counter()
        res = _run(program, x)
        wall = (time.perf_counter() - t0) * 1e3
        if res is None:
            first.append(("agent", None))
            by_tier["agent"]["n"] += 1
            continue
        ok = bool(equal(res.output, want))
        first.append((res.path, json.dumps(res.output, sort_keys=True, default=str)))
        tier = res.trace.highest_tier()
        by_tier[tier]["n"] += 1
        by_tier[tier]["wrong"] += (not ok)
        correct += ok
        llm_calls += res.trace.llm_calls()
        verify_ms.append(res.trace.verify_ms())
        for e in res.trace.events:
            if e.tier in TIER_ORDER and e.answered:
                tier_ms[e.tier].append(e.ms)
        if res.path == "fast":
            fast += 1
            silent += (not ok)
            if res.trace.llm_calls() == 0:
                local += 1
                wall_local.append(wall)

    deterministic = True
    for _ in range(max(0, repeats - 1)):
        for x, (path, out) in zip(cases, first):
            res = _run(program, x)
            got = ("agent", None) if res is None else (res.path, json.dumps(res.output, sort_keys=True, default=str))
            if got != (path, out):
                deterministic = False

    false_accepts = []
    for x in bad_inputs:
        res = _run(program, x)
        if res is not None and res.path == "fast":
            false_accepts.append(x)

    silent_rate = silent / fast if fast else 0.0
    upper = clopper_pearson_upper(silent, fast) if fast else None
    report = {
        "program": program.name,
        "cases": n,
        "accuracy_overall": round(correct / n, 4) if n else None,
        "fast_path_rate": round(fast / n, 4) if n else None,
        "no_llm_rate": round(local / n, 4) if n else None,
        "agent_rate": round((n - fast) / n, 4) if n else None,
        "silent_errors": silent,
        "silent_error_rate": round(silent_rate, 4),
        "silent_error_upper95": round(upper, 4) if upper is not None else None,
        "llm_calls_per_run": round(llm_calls / n, 3) if n else None,
        "by_tier": {t: by_tier[t] for t in TIER_ORDER if t in by_tier},
        "latency_ms_p50": {t: round(statistics.median(tier_ms[t]), 4) for t in TIER_ORDER if tier_ms[t]},
        "run_ms_p50_no_llm": round(statistics.median(wall_local), 4) if wall_local else None,
        "verify_ms_mean": round(sum(verify_ms) / len(verify_ms), 4) if verify_ms else 0.0,
        "deterministic": deterministic,
        "bad_inputs": len(bad_inputs),
        "guard_false_accepts": len(false_accepts),
    }
    reasons = []
    if silent_rate > max_silent_error:
        reasons.append(f"silent error rate {silent_rate:.2%} exceeds budget {max_silent_error:.2%}")
    elif bound_must_pass and upper is not None and upper > max_silent_error:
        reasons.append(f"silent error upper bound {upper:.2%} exceeds budget {max_silent_error:.2%} "
                       f"(not enough evidence: {fast} fast-path cases)")
    if false_accepts:
        reasons.append(f"{len(false_accepts)} bad input(s) were answered on the fast path")
    if not deterministic:
        reasons.append("outputs or paths changed between repeated runs")
    report["admitted"] = not reasons
    report["reasons"] = reasons
    return report
