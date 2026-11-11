"""Kernel tests. Run from the project root:  .venv/bin/python -m unittest discover -s tests -v

The first eleven mirror the tests shipped in the original skill (re-authored
against the ported kernel). The rest cover what this project added: the
independent verifier hook and the selective-risk threshold procedures.
"""
import json
import pathlib
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from wfc.admit import admit                                               # noqa: E402
from wfc.nsp import (ABSTAIN, Decision, Escalate, GuardedCache, Hole,     # noqa: E402
                     Program, RuleCommittee, Trace, check, wilson)
from wfc import selective as sel                                          # noqa: E402


class Fixed:
    def __init__(self, value):
        self.value, self.calls = value, 0

    def answer(self, x):
        self.calls += 1
        return self.value


class KernelTests(unittest.TestCase):
    def test_wilson(self):
        lo, hi = wilson(0, 300)
        self.assertEqual(lo, 0.0)
        self.assertTrue(0.012 < hi < 0.013)
        self.assertEqual(wilson(0, 0), (0.0, 1.0))

    def test_cache_hit_miss_and_guard(self):
        c = GuardedCache({"a": 1}, guard=lambda x, v: x != "blocked")
        self.assertEqual(c.answer("a"), 1)
        self.assertIs(c.answer("b"), ABSTAIN)
        self.assertIs(GuardedCache({"a": 1}, key=lambda x: "a", guard=lambda x, v: False).answer("z"), ABSTAIN)
        self.assertIs(c.answer(["unhashable"]), ABSTAIN)

    def test_committee_abstains_on_disagreement(self):
        yes = lambda x: "y" if "a" in x else None
        no = lambda x: "n" if "b" in x else None
        c = RuleCommittee([yes, no])
        self.assertEqual(c.answer("a"), "y")
        self.assertIs(c.answer("ab"), ABSTAIN)
        self.assertIs(c.answer("zzz"), ABSTAIN)

    def test_unproven_rule_does_not_vote(self):
        good = lambda x: "y" if x.startswith("a") else None
        bad = lambda x: "n" if x.endswith("!") else None
        xs = ["a%d" % i for i in range(30)] + ["a%d!" % i for i in range(10)]
        c = RuleCommittee([good, bad]).fit(xs, ["y"] * 40)
        self.assertGreater(c.weights[0], 0)
        self.assertEqual(c.weights[1], 0.0)
        self.assertEqual(c.answer("a1!"), "y")

    def test_hole_is_typed(self):
        calls = []
        ok = Hole(lambda p: calls.append(p) or " Travel ", lambda x: x, choices=["travel", "meals"])
        self.assertEqual(ok.answer("x"), "travel")
        calls.clear()
        bad = Hole(lambda p: calls.append(p) or "something else", lambda x: x, choices=["travel"], retries=2)
        self.assertIs(bad.answer("x"), ABSTAIN)
        self.assertEqual(len(calls), 3)

    def test_decision_uses_cheapest_tier_that_answers(self):
        rules, model, llm = Fixed(ABSTAIN), Fixed("b"), Fixed("c")
        d = Decision("d", ["a", "b", "c"], cache=Fixed(ABSTAIN), rules=rules, model=model, llm=llm)
        t = Trace()
        self.assertEqual(d("x", t), "b")
        self.assertEqual(llm.calls, 0)
        self.assertEqual(t.highest_tier(), "model")
        self.assertEqual([e.answered for e in t.events], [False, False, True])

    def test_decision_rejects_out_of_set_answers_and_escalates(self):
        d = Decision("d", ["a"], model=Fixed("zzz"), llm=Fixed(ABSTAIN))
        with self.assertRaises(Escalate):
            d("x", Trace())

    def test_program_fallback_queue_and_crash(self):
        def fn(x, t):
            check(t, "positive", x["n"] > 0, "n must be positive")
            return 10 // x["n"]
        with tempfile.TemporaryDirectory() as tmp:
            q = str(pathlib.Path(tmp) / "queue.jsonl")
            p = Program("p", fn, fallback=lambda x, e: "agent:" + e.step, queue_path=q)
            self.assertEqual(p.run({"n": 5}).path, "fast")
            r = p.run({"n": -1})
            self.assertEqual((r.path, r.output), ("agent", "agent:positive"))
            self.assertEqual(p.run({"n": "oops"}).output, "agent:crash")
            rows = [json.loads(line) for line in pathlib.Path(q).read_text().splitlines()]
            self.assertEqual([row["step"] for row in rows], ["positive", "crash"])
        with self.assertRaises(Escalate):
            Program("p", fn).run({"n": -1})

    def test_shadow_sampling_sees_silent_errors(self):
        p = Program("p", lambda x, t: x * 2, oracle=lambda x: x * 2 if x % 2 else -1, shadow_rate=1.0)
        for i in range(10):
            p.run(i)
        est = p.silent_error_estimate()
        self.assertEqual((est["sampled"], est["disagreed"]), (10, 5))
        none = Program("p", lambda x, t: x, oracle=lambda x: x, shadow_rate=0.0)
        none.run(1)
        self.assertEqual(none.silent_error_estimate()["sampled"], 0)

    def test_admission_catches_a_missing_guard(self):
        unguarded = Program("u", lambda x, t: abs(x), fallback=lambda x, e: None)
        rep = admit(unguarded, [1, 2, 3], [1, 2, 3], bad_inputs=[-5])
        self.assertFalse(rep["admitted"])
        self.assertEqual(rep["guard_false_accepts"], 1)

        def guarded(x, t):
            check(t, "non_negative", x >= 0)
            return x
        rep = admit(Program("g", guarded, fallback=lambda x, e: None), [1, 2, 3], [1, 2, 3], bad_inputs=[-5])
        self.assertTrue(rep["admitted"])
        self.assertEqual(rep["no_llm_rate"], 1.0)

    def test_admission_counts_silent_errors(self):
        rep = admit(Program("w", lambda x, t: 0), list(range(10)), [0] * 9 + [1])
        self.assertEqual(rep["silent_errors"], 1)
        self.assertFalse(rep["admitted"])

    def test_admission_can_require_the_bound_not_the_point_estimate(self):
        perfect = Program("p", lambda x, t: x)
        cases = list(range(50))
        self.assertTrue(admit(perfect, cases, cases)["admitted"])
        rep = admit(perfect, cases, cases, bound_must_pass=True)      # 0/50 cannot certify 1%
        self.assertFalse(rep["admitted"])
        self.assertIn("not enough evidence", rep["reasons"][0])


class VerifierTests(unittest.TestCase):
    def test_rejected_fast_answer_falls_through_to_the_llm(self):
        model, llm = Fixed("a"), Fixed("b")
        d = Decision("d", ["a", "b"], model=model, llm=llm, verifier=lambda x, v, tier: False)
        t = Trace()
        self.assertEqual(d("x", t), "b")
        self.assertEqual(t.highest_tier(), "llm")
        self.assertEqual([(e.tier, e.answered) for e in t.events],
                         [("model", False), ("verify", False), ("llm", True)])

    def test_llm_answers_are_not_sent_to_the_fast_verifier(self):
        seen = []
        d = Decision("d", ["a", "b"], model=Fixed(ABSTAIN), llm=Fixed("b"),
                     verifier=lambda x, v, tier: seen.append(tier) or True)
        self.assertEqual(d("x", Trace()), "b")
        self.assertEqual(seen, [])

    def test_accepting_verifier_keeps_the_fast_answer(self):
        llm = Fixed("b")
        d = Decision("d", ["a", "b"], model=Fixed("a"), llm=llm, verifier=lambda x, v, tier: v == "a")
        self.assertEqual(d("x", Trace()), "a")
        self.assertEqual(llm.calls, 0)


class SelectiveTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.score = rng.random(5000)
        # error probability falls with score: 30% at score 0, ~0% at score 1
        self.correct = rng.random(5000) > 0.3 * (1 - self.score) ** 3

    def test_empirical_is_the_most_permissive_picker(self):
        # Wilson (97.5% one-sided, scanned) and fixed-sequence (95%, sequential) are not
        # ordered relative to each other; both must be at least as strict as the point estimate.
        a = 0.02
        e = sel.pick_empirical(self.score, self.correct, a)
        w = sel.pick_wilson_scan(self.score, self.correct, a)
        f = sel.pick_fixed_sequence(self.score, self.correct, a)
        self.assertLessEqual(e, w)
        self.assertLessEqual(e, f)
        self.assertLess(max(w, f), sel.NEVER)

    def test_no_evidence_means_never_answer(self):
        s, c = np.linspace(0, 1, 40), np.ones(40, bool)
        self.assertEqual(sel.pick_fixed_sequence(s, c, 0.01), sel.NEVER)       # needs >= 299 clean cases
        self.assertEqual(sel.pick_wilson_scan(s, c, 0.01), sel.NEVER)
        self.assertEqual(sel.evaluate(s, c, sel.NEVER)["answered"], 0)

    def test_fixed_sequence_holds_its_guarantee_on_exchangeable_data(self):
        rng = np.random.default_rng(1)
        alpha, viol, trials = 0.05, 0, 300
        for _ in range(trials):
            s = rng.random(800)
            c = rng.random(800) > 0.4 * (1 - s) ** 2
            thr = sel.pick_fixed_sequence(s, c, alpha, delta=0.1)
            # true risk above thr under this generator, by numerical integration
            g = np.linspace(min(thr, 1.0), 1.0, 4001)
            true_risk = float((0.4 * (1 - g) ** 2).mean()) if thr < 1 else 0.0
            viol += true_risk > alpha
        self.assertLessEqual(viol / trials, 0.1 + 0.04)

    def test_cuts_never_split_ties(self):
        s = np.array([0.9] * 50 + [0.5] * 50)
        c = np.array([True] * 50 + [True] * 25 + [False] * 25)
        thr = sel.pick_empirical(s, c, 0.05)
        self.assertEqual(thr, 0.9)

    def test_cluster_bootstrap_is_stricter_when_errors_come_in_groups(self):
        rng = np.random.default_rng(3)
        groups = np.repeat(np.arange(30), 100)
        bad_group = np.isin(groups, [0, 1, 2])                     # 3 of 30 groups are wrong half the time
        correct = ~(bad_group & (rng.random(3000) < 0.5))
        score = rng.random(3000)                                   # confidence carries no signal
        iid = sel.pick_fixed_sequence(score, correct, 0.07)        # pooled risk is ~5%: rows say "fine"
        clu = sel.pick_cluster_bootstrap(score, correct, groups, 0.07)
        self.assertLess(iid, sel.NEVER)
        self.assertGreater(clu, iid)                               # groups say "not enough evidence"

    def test_clopper_pearson(self):
        self.assertAlmostEqual(sel.clopper_pearson_upper(0, 299, 0.05), 0.00997, places=4)
        lo, hi = sel.clopper_pearson(5, 100)
        self.assertTrue(0.016 < lo < 0.017 and 0.112 < hi < 0.114)


if __name__ == "__main__":
    unittest.main()
