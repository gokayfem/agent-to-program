---
name: workflow-compiler
description: Compile a recurring LLM or agent workflow into a fast, mostly deterministic program with typed model calls, a small decision model that abstains, independent verification, guards and an agent fallback. Use this whenever the user wants a workflow to run faster, cheaper or the same way every time; says "compile", "productionize", "stop re-running the agent", "make this deterministic", "cache this", "observe the agent and learn the process", or "turn this run, prompt or skill into a script or pipeline"; or has logs of a model doing the same input-to-output job repeatedly, even if they never use the word "compile".
---

# Workflow compiler (version 2)

Turn a workflow that a model solves from scratch on every run into a program that runs mostly without it. The result is ordinary Python: a deterministic shell, a few typed decisions answered by the cheapest tier that is sure, guards on every assumption, an independent check on fast answers, and the original agent as the fallback for anything the program has not earned the right to answer.

Compiling moves the model; it rarely removes it. The aim is that each remaining model call is small, typed, at a fixed point, and testable on its own.

Version 2 changes the calibration, verification and operating rules. Each change is backed by an experiment on public human-labelled data with real local models; the numbers quoted here are from that study (`REPORT.md` in the project this skill was built in). They describe those tasks and models, so re-measure on yours.

## What you need first

- The request template or spec: what stays fixed between runs and what varies.
- Recorded runs with inputs and reviewed outputs. Three is the minimum to tell a constant from a coincidence; a decision model needs a few hundred labelled answers; a per-field code extractor needs about 50 observed runs plus 30 to 40 it never saw for admission.
- **The grouping that will shift.** Which entity do inputs come from (merchant, customer, template, intent, author, language, scene family)? New members of that grouping are where a compiled program fails, and you need whole groups to hold out.
- Real examples of inputs the program should refuse (out of scope, malformed), if any exist.
- The tools the workflow calls, and which of their effects cannot be undone.

## Decide what to compile

| The step is... | Compile it to... |
|---|---|
| A transform, lookup, calculation or tool call | Code |
| A judgment with a closed answer set, made repeatedly | A `Decision` with tiers |
| An extraction the model does approximately (an estimate, a paraphrase) | Usually a narrow hole. Code is admitted only if it agrees with the model on runs it never saw, and it often will not: what the model does is rarely a simple function of the text |
| Open-ended writing or design | A `Hole` fed by compact facts; compile everything around it |
| An irreversible action whose trigger the runs do not pin down | Nothing. Leave it with the agent and tell the user why |

"This decision stays with the model" is a valid result. In the study one of three intent tasks could not be certified at a 2% silent-error budget by any of twelve model families, and five of six profile-analysis fields failed admission as code.

## Procedure

1. **Write the contract before any code.** In `CONTRACT.md`: inputs and types, the output, which values vary between runs, the properties any correct output has, and a **contract version**. Correctness comes from the spec; the runs make it cheaper and auditable.

2. **Tag every value by where it comes from.** Input, declared constant, derived by code, observed in past runs, judged, or generated. Only "judged" and "generated" may involve a model. Arithmetic, counting, sorting, aggregation and formatting are always code.

3. **Write the shell.** Plain Python, one function `fn(x, trace)`. Call `check()` for every assumption: type, length, character mix, field present, value in range. Structural guards are nearly free and stop most malformed input before any model sees it.

4. **Give each judgment a `Decision`, cheapest tier first.**
   - `GuardedCache`: an exact value seen before. Expect little from it unless traffic really repeats. Do **not** use a nearest-neighbour "semantic cache" as a tier: similarity to one stored example was the weakest predictor of correctness measured.
   - `RuleCommittee`: mined or hand-written rules that vote. A rule with no counter-example in history is still wrong a few percent of the time on new data; require two agreeing rules or run it through the same admission as everything else.
   - `DecisionModel` (`scripts/decision.py`): a frozen sentence encoder with a linear head is enough; fine-tune a small encoder if latency matters more than training time. See the next section before using it.
   - `Hole`: a typed model call with a closed set of `choices`, the smallest input that suffices, no arithmetic, and the contract as a fixed prompt prefix whose KV cache is reused (5 times faster per call for a 400-token contract, 24 times for a 2,000-token one).

5. **Add an independent verifier** (`Decision(verifier=...)`). See "Verification" below.

6. **Wire the fallback.** `Program(..., fallback=agent, queue_path=...)`. Every fallback and every shadow disagreement is appended to the queue, which is the input to the next compile.

7. **Admit it.** Run `admit()` on cases nothing was fitted on, including whole held-out groups, plus seeded bad inputs that must be handed back. Admit every tier, including the LLM hole: a small local LLM answered more than half of real out-of-scope queries as in-scope and its confidence could not be certified. If the program is not admitted, fix it and test again on fresh cases. Do not loosen a budget to pass.

8. **Operate it.** See "Operating a compiled program".

## The decision model: fast, but not self-verifying

A typed judgment is a classification problem with free labels, so a small model can answer it in a few milliseconds instead of a model call. Three things go wrong, and each has a rule.

**Its confidence says nothing about whether the input is familiar.** On groups never seen in training, accuracy fell from 97% to 69%, and 68% of those errors were made with confidence of at least 0.9. On text with typing noise an embedding classifier fell to 51% accuracy and still answered half the requests, a quarter of them wrong.

- Use a **separate familiarity score**: similarity to the nearest training example, or relative Mahalanobis distance, on the embedding. Gate on both confidence and familiarity.

**A random calibration split certifies the wrong thing.** Thresholds certified for 2% silent errors on a random split held on familiar groups and, in 20 of 20 group splits, produced 23% wrong answers on unseen groups.

- Hold whole groups out of training with `holdout_groups()` and calibrate with **two tests** (`DecisionModel.fit`, `fit_gate`): selective risk on held-out rows of known groups, and **false-accept rate** (wrong answers over all inputs) on held-out groups and on real out-of-scope examples. The second test is defined even when the gate refuses almost everything, which is what you want it to do there.
- Expect to pay for it. In the study this cut wrong answers per unseen-group input from 17.6% to 1.6% and familiar coverage from 95% to 63%.

**The certificate is weaker than it looks.** Rows from one group fail together, so thousands of calibration rows from 30 groups are about 30 observations. The certified rate was exceeded on never-seen groups in a third of splits.

- Count groups as your evidence (`pick_cluster_bootstrap` is stricter and group-aware). State the bound as an estimate. Rely on shadow sampling to catch what calibration misses.

## Verification

A fast answer must not be its own sole verifier, and a second model trained on the same features is close to the same model. Measured on unseen-group errors of an embedding classifier, the share of cases where the checker gave the **same wrong label**: a 5-member ensemble on the same embedding 70%, a k-NN head on the same embedding 52%, another encoder 55%, a character n-gram model 38%, a local LLM 32%.

- Default verifier: a **character n-gram (TF-IDF) classifier must agree**. It costs about 0.2 ms, it fails differently from an embedding model, and under typing noise it loses coverage instead of accuracy.
- Agreement checks cannot refuse an input both models misread the same way; out-of-scope inputs need the familiarity gate, not another vote.
- A verifier's cost is its false rejects, not its milliseconds. Report both.
- An LLM check on every fast answer gives the speed back. Use it on a sample (shadowing), and prefer asking it to answer independently over asking "is this proposal right?": the yes/no check caught fewer errors and rejected more correct answers.
- Never let a learned verifier be the only guard in front of an irreversible action.

## Operating a compiled program

- **Shadow-sample.** Set `shadow_rate` so a share of fast answers is re-checked by the oracle. It is the only view of silent errors in production. A program compiled on one year of dated data went from 11% to 65% silent errors over five years while its confidence stayed high.
- **Per-answer-class circuit breakers.** Track shadow disagreement per answer class; when its lower bound exceeds the budget, stop answering that class on the fast path. Detection takes roughly (samples needed) / (shadow rate × that class's share of fast traffic) requests, so a 10% rate is slow for rare classes.
- **A noisy checker hides drift.** If the shadow oracle is an LLM that is itself wrong a few percent of the time, its disagreements are mostly its own errors. Use the best oracle you can afford on the sample.
- **Version the contract and invalidate on change.** A change of policy looks like familiar input, so no input-side guard sees it. When the contract changes, open the breaker for every answer class it touches at once, and re-adjudicate the regression cases for those classes; a suite that encodes the old contract penalises exactly the update you need.
- **Gate every update.** Recompile from the contract and all labelled cases; deploy a candidate only if it passes on held-out recent labels and does not regress on the frozen suite. Labels collected in production come from escalations and are harder than fast-path traffic, so this gate is conservative.
- **Never retrain on the program's own unverified answers.** It was the worst policy in the drift test.

## Compiling from observation (the wrapper-harness pattern)

When there is no decision to train, only an agent doing an open-ended job ("analyze this profile"):

1. Wrap the agent and record every call for a few dozen inputs: prompt and answer tokens, time, parsed output.
2. Infer each output field's type from the recordings: number, closed set, list of terms, free text.
3. For each field, search a small family of candidate programs (regex and arithmetic extractors, keyword rules, a shallow tree) by agreement with the recorded answers.
4. **Admit** the best candidate only if it agrees with the agent on recordings it never saw. Measure the agent's agreement with itself under a reworded instruction first; that is the ceiling for fidelity.
5. **Audit the agent while you are there.** Check that each recorded answer is supported by the input the agent was given. In the study the agent reported a degree for 33 of 44 profiles that showed no education at all, and a rule that copied this habit agreed with it better than any honest rule. Exclude such candidates: the contract outranks the traces, and admission should use reviewed outputs as its reference wherever they exist.
6. Everything not admitted goes into **one** narrow hole: only those fields, closed answer sets, a prefix of the input. Free text becomes a small generation hole fed by the compact facts, or a template.

Expect the speed-up to be set by generated tokens: about 3 times for a structured analysis with one narrow call, less when prose is still generated, in the one case measured.

## What each tier costs

Measured per request on an Apple M4 (CPU for local tiers, MLX GPU for the LLM):

| Tier | Median time |
|---|---|
| Exact cache | about 0.001 ms |
| Mined rules | about 0.1 ms |
| Character TF-IDF + linear model | about 0.2 ms |
| Static-embedding encode | about 0.03 ms |
| Small transformer encode + linear head + nearest-neighbour search | about 5 to 9 ms |
| Typed call to a local 4B LLM, cached contract prefix | about 200 to 270 ms |
| The same call without the prefix cache | about 1,200 ms (400-token contract) to 6,300 ms (2,000-token contract) |
| Agent | the full loop |

## Reporting

Report what was measured and how: coverage, silent-error rate with its bound, **false-accept rate on each shifted population**, the verifier's catch rate and false-reject rate, the guard test by kind of bad input and by the tier that answered, model calls and escalations per request. Separate measured results from replayed, scripted and assumed ones. Say plainly what was not tested. A check that failed is listed as failed.

## Files

- `scripts/nsp.py` runtime: `Decision` (with `verifier`), `GuardedCache`, `RuleCommittee`, `Hole`, `check`, `Program` (standard library only)
- `scripts/selective.py` thresholds: `pick_fixed_sequence`, `pick_far`, `pick_cluster_bootstrap`, exact bounds, risk-coverage helpers (numpy, scipy)
- `scripts/decision.py` decision-model tier: `DecisionModel`, `fit_gate`, `holdout_groups` (scikit-learn)
- `scripts/admit.py` admission harness: `admit()`
- `tests/` 23 tests: `python -m unittest discover -s tests`

## Shape of a compiled program

```python
from nsp import Decision, GuardedCache, Hole, Program, check
from decision import DecisionModel, holdout_groups

held = holdout_groups(groups, frac=0.2)                  # whole groups, never trained on
model = DecisionModel(alpha=0.02, embed=encode).fit(
    train_x, train_y,
    seen_x, seen_y,                                      # held-out rows of known groups
    shift_x, shift_y)                                    # held-out groups + real out-of-scope inputs

route = Decision(
    "route", CHOICES,
    cache=GuardedCache(table, key=normalise),
    model=model,                                         # abstains on low confidence or low familiarity
    llm=Hole(call_model, make_prompt, choices=CHOICES),  # contract as a cached prefix
    verifier=lambda x, v, tier: lexical_model(x) == v,   # a differently built second opinion
)

def fn(x, trace):
    check(trace, "is_text", isinstance(x, str))
    check(trace, "length", 3 <= len(x) <= 300)
    check(trace, "contract_version", CONTRACT_VERSION == compiled_for)
    return route(x, trace)

program = Program("router", fn, fallback=agent, oracle=best_oracle,
                  shadow_rate=0.1, queue_path="recompile_queue.jsonl")
```
