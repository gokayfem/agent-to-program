"""nsp: runtime for compiled LLM workflows (neuro-symbolic programs).

Ported from the `workflow-compiler.skill` artifact produced in the shared chat
(scripts/nsp.py, read in full and re-authored here), with two extensions used by
the experiments in bench/:

  * `Decision(verifier=...)`: an independent check applied to every answer from
    a tier cheaper than the LLM. A rejected answer is discarded and the next
    tier is tried, so a fast answer is never its own sole verifier.
  * `Trace.tier_of_answer()` and per-tier timing kept on the trace.

A compiled workflow is ordinary Python. Judgment that cannot become code is a
`Decision`: a typed question answered by the cheapest tier that is sure:

    cache -> rules -> decision model -> LLM hole -> Escalate (the agent takes over)

Standard library only.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

__all__ = [
    "ABSTAIN", "Escalate", "Event", "Trace", "check", "GuardedCache",
    "RuleCommittee", "Hole", "Decision", "Program", "Result", "wilson", "TIER_ORDER",
]

# Cheapest first. "code" means no judgment tier was needed at all.
TIER_ORDER = ("code", "cache", "rules", "model", "llm", "agent")


class _Abstain:
    """Returned by a tier that will not answer. Distinct from None and False."""

    def __repr__(self) -> str:
        return "ABSTAIN"

    def __bool__(self) -> bool:
        return False


ABSTAIN = _Abstain()


class Escalate(Exception):
    """The program cannot answer safely. Control passes to the agent."""

    def __init__(self, step: str, reason: str):
        super().__init__(f"{step}: {reason}")
        self.step, self.reason = step, reason


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for k successes out of n (95% by default)."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, centre - half), min(1.0, centre + half)


@dataclass
class Event:
    step: str
    tier: str            # one of TIER_ORDER, "guard", or "verify"
    answered: bool       # tier answered / guard passed / verifier accepted
    detail: str = ""
    ms: float = 0.0


@dataclass
class Trace:
    events: list[Event] = field(default_factory=list)

    def add(self, step: str, tier: str, answered: bool, detail: Any = "", ms: float = 0.0) -> None:
        self.events.append(Event(step, tier, answered, str(detail)[:200], ms))

    def highest_tier(self) -> str:
        """The most expensive tier that produced an answer in this run."""
        used = [e.tier for e in self.events if e.answered and e.tier in TIER_ORDER]
        return max(used, key=TIER_ORDER.index) if used else "code"

    def llm_calls(self) -> int:
        return sum(1 for e in self.events if e.tier == "llm")

    def verify_ms(self) -> float:
        return sum(e.ms for e in self.events if e.tier == "verify")

    def to_json(self) -> list[dict]:
        return [e.__dict__ for e in self.events]


def check(trace: Trace, name: str, ok: bool, detail: str = "") -> None:
    """A guard. Every assumption the program relies on gets one; failing escalates."""
    trace.add(name, "guard", bool(ok), "" if ok else detail)
    if not ok:
        raise Escalate(name, detail or "guard failed")


class GuardedCache:
    """A value observed in earlier runs, reused only while its guard holds."""

    def __init__(self, table: dict, key: Callable[[Any], Any] = lambda x: x,
                 guard: Callable[[Any, Any], bool] = lambda x, v: True):
        self.table, self.key, self.guard = table, key, guard

    def answer(self, x: Any) -> Any:
        try:
            k = self.key(x)
            if k in self.table and self.guard(x, self.table[k]):
                return self.table[k]
        except TypeError:
            pass
        return ABSTAIN


class RuleCommittee:
    """Several small rules vote; the committee abstains unless they agree.

    Each rule returns a label or None. Call fit() with oracle-labelled examples
    to weight rules by measured precision. A rule that has not earned its
    precision on real examples does not vote.
    """

    def __init__(self, rules: Sequence[Callable[[Any], Any]], min_votes: int = 1, min_agree: float = 1.0):
        self.rules = list(rules)
        self.min_votes, self.min_agree = min_votes, min_agree
        self.weights = [1.0] * len(self.rules)
        self.precision: list[Optional[float]] = [None] * len(self.rules)

    def fit(self, xs: Sequence[Any], ys: Sequence[Any], min_precision: float = 0.9) -> "RuleCommittee":
        for i, rule in enumerate(self.rules):
            fired = [(rule(x), y) for x, y in zip(xs, ys)]
            fired = [(p, y) for p, y in fired if p is not None]
            n, k = len(fired), sum(p == y for p, y in fired)
            p = (k + 1) / (n + 2)                     # Laplace: an unseen rule scores 0.5
            self.precision[i] = p
            self.weights[i] = math.log(p / (1 - p)) if p >= min_precision else 0.0
        return self

    def votes(self, x: Any) -> list[tuple[Any, float]]:
        """(label, weight) from every trusted rule that fires. Useful as an independent view."""
        out = []
        for rule, w in zip(self.rules, self.weights):
            if w > 0:
                label = rule(x)
                if label is not None:
                    out.append((label, w))
        return out

    def answer(self, x: Any) -> Any:
        votes = self.votes(x)
        if len(votes) < self.min_votes or not votes:
            return ABSTAIN
        tally: Counter = Counter()
        for label, w in votes:
            tally[label] += w
        label, top = tally.most_common(1)[0]
        return label if top / sum(tally.values()) >= self.min_agree else ABSTAIN


class Hole:
    """A typed LLM call at a fixed point in the program.

    `llm` is any callable prompt -> text. The reply must parse and, if `choices`
    is given, be one of them; otherwise the hole abstains and the run escalates.
    """

    def __init__(self, llm: Callable[[str], str], prompt: Callable[[Any], str],
                 choices: Optional[Sequence[str]] = None,
                 parse: Optional[Callable[[str], Any]] = None, retries: int = 1):
        self.llm, self.prompt, self.parse, self.retries = llm, prompt, parse, retries
        self.choices = list(choices) if choices else None

    def answer(self, x: Any) -> Any:
        p = self.prompt(x)
        for _ in range(self.retries + 1):
            raw = self.llm(p)
            try:
                v = self.parse(raw) if self.parse else str(raw).strip()
            except Exception:
                continue
            if self.choices is None:
                return v
            for c in self.choices:
                if str(v).lower() == str(c).lower():
                    return c
        return ABSTAIN


class Decision:
    """One typed judgment, answered by the cheapest tier that is sure.

    `verifier(x, answer, tier) -> bool` is consulted for answers from tiers
    listed in `verify_tiers` (default: every tier cheaper than the LLM). It must
    be built from a different view than the tier it checks; that is the
    caller's responsibility and what bench/exp4 measures.
    """

    def __init__(self, name: str, choices: Optional[Sequence[Any]] = None, cache: Any = None,
                 rules: Any = None, model: Any = None, llm: Any = None,
                 verifier: Optional[Callable[[Any, Any, str], bool]] = None,
                 verify_tiers: Sequence[str] = ("cache", "rules", "model")):
        self.name = name
        self.choices = list(choices) if choices else None
        named = (("cache", cache), ("rules", rules), ("model", model), ("llm", llm))
        self.tiers = [(t, obj) for t, obj in named if obj is not None]
        self.verifier, self.verify_tiers = verifier, set(verify_tiers)

    def __call__(self, x: Any, trace: Trace) -> Any:
        for tier, obj in self.tiers:
            t0 = time.perf_counter()
            v = obj.answer(x)
            ms = (time.perf_counter() - t0) * 1e3
            valid = v is not ABSTAIN and (self.choices is None or v in self.choices)
            trace.add(self.name, tier, valid, v, ms)
            if not valid:
                continue
            if self.verifier is not None and tier in self.verify_tiers:
                t0 = time.perf_counter()
                ok = bool(self.verifier(x, v, tier))
                trace.add(self.name, "verify", ok, f"{tier}:{v}", (time.perf_counter() - t0) * 1e3)
                if not ok:
                    trace.events[-2].answered = False      # the tier's answer was not accepted
                    continue
            return v
        raise Escalate(self.name, "no tier could answer")


@dataclass
class Result:
    output: Any
    path: str                              # "fast": program finished. "agent": fallback took over.
    trace: Trace
    reason: Optional[str] = None
    shadow_agrees: Optional[bool] = None


def _sampled(x: Any, rate: float) -> bool:
    """Deterministic sampling, so the same input is always (or never) shadowed."""
    if rate <= 0:
        return False
    h = hashlib.sha256(json.dumps(x, sort_keys=True, default=str).encode()).digest()
    return int.from_bytes(h[:8], "big") / 2 ** 64 < rate


class Program:
    """A compiled workflow plus its escape hatch.

    fn(x, trace) is the compiled program. If it raises Escalate (or crashes),
    fallback(x, escalation) finishes the task and the case is queued for
    recompilation. With an oracle and shadow_rate > 0, a sample of fast-path
    runs is also sent to the oracle, which is the only way to see silent errors
    in production.
    """

    def __init__(self, name: str, fn: Callable[[Any, Trace], Any],
                 fallback: Optional[Callable[[Any, Escalate], Any]] = None,
                 oracle: Optional[Callable[[Any], Any]] = None, shadow_rate: float = 0.0,
                 equal: Callable[[Any, Any], bool] = lambda a, b: a == b,
                 queue_path: Optional[str] = None):
        self.name, self.fn, self.fallback = name, fn, fallback
        self.oracle, self.shadow_rate, self.equal = oracle, shadow_rate, equal
        self.queue_path = queue_path
        self.stats: Counter = Counter()

    def run(self, x: Any) -> Result:
        trace = Trace()
        try:
            try:
                res = Result(self.fn(x, trace), "fast", trace)
            except Escalate:
                raise
            except Exception as exc:                   # a crash is an escalation, never an answer
                raise Escalate("crash", f"{type(exc).__name__}: {exc}") from exc
        except Escalate as e:
            if self.fallback is None:
                raise
            t0 = time.perf_counter()
            out = self.fallback(x, e)
            trace.add("fallback", "agent", True, e.reason, (time.perf_counter() - t0) * 1e3)
            res = Result(out, "agent", trace, reason=str(e))
            self._enqueue(x, e.step, e.reason, out)
        self.stats[res.path] += 1
        if res.path == "fast" and self.oracle is not None and _sampled(x, self.shadow_rate):
            res.shadow_agrees = bool(self.equal(res.output, self.oracle(x)))
            self.stats["shadow_n"] += 1
            if not res.shadow_agrees:
                self.stats["shadow_disagree"] += 1
                self._enqueue(x, "shadow", "fast path disagreed with oracle", None)
        return res

    def silent_error_estimate(self) -> dict:
        """Share of sampled fast-path runs the oracle disagreed with, with a 95% upper bound."""
        n, k = self.stats["shadow_n"], self.stats["shadow_disagree"]
        return {"sampled": n, "disagreed": k, "rate": k / n if n else None,
                "upper95": wilson(k, n)[1] if n else None}

    def _enqueue(self, x: Any, step: str, reason: str, agent_output: Any) -> None:
        if not self.queue_path:
            return
        row = {"program": self.name, "input": x, "step": step, "reason": reason,
               "agent_output": agent_output}
        with open(self.queue_path, "a") as f:
            f.write(json.dumps(row, default=str) + "\n")
