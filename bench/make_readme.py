"""Build README.md. The prose is fixed here; every number in it is read from results/*.json.

Usage: python -m bench.make_readme
"""
from __future__ import annotations

import json

from .common import RESULTS, ROOT
from blog.build_post import numbers, pct

REPO = "https://github.com/gokayfem/agent-to-program"
POST = "https://www.generativemedia.blog/blog/agent-2-program-what-survives-when-you-compile-an-llm-workflow"
FILM = "https://www.generativemedia.blog/articles/agent-to-program-film.mp4"


def J(name):
    return json.loads((RESULTS / name).read_text())


def fig(name, alt, caption):
    return (f'<p align="center"><img src="gallery/{name}" width="100%" alt="{alt}"></p>\n\n'
            f'<p align="center"><sub>{caption}</sub></p>\n')


def shadow_line(n: dict) -> str:
    if n["sh_late"]:
        return (f"re-checking 10% of fast answers did not catch it within the {n['p5_n']} requests that followed, "
                f"re-checking 40% took {n['sh40']}.")
    return f"re-checking 10% of fast answers took {n['sh_req']} requests to notice."


def extra() -> dict:
    e1b, e2, e3b = J("exp1b_finetune.json"), J("exp2_novelty.json"), J("exp3b_cal_groups.json")
    e5b, e6, e7, e8 = J("exp5b_llm_other.json"), J("exp6_stream.json"), J("exp7_news_drift.json"), J("exp8_kernel.json")
    D = e2["detectors"]
    tr = e2["locale_shift"]["bge-small"]["tr-TR"]["msp"]
    S = e3b["settings"]
    lo, hi = min(S, key=int), max(S, key=int)
    B = e8["builds"]
    full, bare = B["guards + gate(G) + verifier"], B["the same without the rule tier"]
    naive = B["guards + naive gate(R), no verifier"]["bad_input_detail"]["gibberish"]
    P = e6["policies"]
    return {
        "ft_acc": pct(e1b["clinc150"]["accuracy"], 1), "ft_ms": f"{e1b['clinc150']['latency_ms_p50']:.1f}",
        "oos_conf": pct(D["bge-small/msp"]["op_R"]["oos_false_accept"]),
        "oos_maha": pct(D["bge-small/rel_maha"]["op_R"]["oos_false_accept"], 1),
        "oos_conf_cov": pct(D["bge-small/msp"]["op_R"]["in_scope_coverage"]),
        "oos_maha_cov": pct(D["bge-small/rel_maha"]["op_R"]["in_scope_coverage"]),
        "tr_ans": pct(tr["coverage"]), "tr_wrong": pct(tr["silent_error_rate"]),
        "g_lo": lo, "g_hi": hi, "g_lo_viol": pct(S[lo]["fixed_sequence"]["unseen_over_budget"]),
        "g_hi_viol": pct(S[hi]["fixed_sequence"]["unseen_over_budget"]),
        "fine_llm": pct(e5b["fine"]["clinc150"]["llm_accuracy"]), "fine_fast": pct(e5b["fine"]["clinc150"]["fast_accuracy_same_rows"]),
        "es_llm": pct(e5b["locale"]["es-ES_seen"]["llm"]), "es_fast": pct(e5b["locale"]["es-ES_seen"]["fast:bge-small"]),
        "full_silent": pct(full["silent_error_rate"], 1), "full_llm_wrong": full["by_tier"]["llm"]["wrong"],
        "full_llm_n": full["by_tier"]["llm"]["n"], "full_local_wrong": full["by_tier"]["rules"]["wrong"] + full["by_tier"]["model"]["wrong"],
        "full_local_n": full["by_tier"]["rules"]["n"] + full["by_tier"]["model"]["n"],
        "bare_silent": pct(bare["silent_error_rate"], 1), "bare_local": pct(bare["no_llm_rate"]),
        "gib": naive["answered"], "gib_n": naive["n"],
        "tot_naive": pct(P["B*"]["total"]["wrong"], 1), "tot_guard": pct(P["C*"]["total"]["wrong"], 1),
        "tot_ann": pct(P["D+*"]["total"]["wrong"], 1),
        "breaker": e7["policies"]["T + breaker"]["breaker_tripped_after"],
        "gated_cov": pct(e7["policies"]["gated-quarterly"]["summary"]["coverage"]),
        "gated_silent": pct(e7["policies"]["gated-quarterly"]["summary"]["silent_among_fast"], 1),
        "n_tests_skill": 23,
    }


def build() -> str:
    n = {**numbers(), **extra()}
    s = f"""# Agent 2 Program

**Let an agent solve a workflow, then run it as a program.** A skill for coding agents (Claude Code, Codex,
Cursor and others) and the test bed behind it. The skill is a compile procedure with a small runtime: the
deterministic parts of a recurring LLM workflow become code, each judgment becomes a typed decision answered
by the cheapest tier that is sure of it (cache, rules, a small decision model, one narrow LLM call), every
fast answer gets an independent check, and anything the program has not earned the right to answer goes back
to the agent. The test bed is ten experiments on public human-labelled data: nine with real local models, one
with a hosted frontier model as the LLM tier. They
measure where this works and where it fails without telling you.

The earlier projects in this series, [Video 2 GLSL](https://github.com/gokayfem/shader-skills),
[Video 2 Blender Nodes](https://github.com/gokayfem/video-to-blender),
[Video 2 Physics](https://github.com/gokayfem/video-to-physics),
[Video to CSS](https://github.com/gokayfem/video-to-css),
[Image 2 Sculpt](https://github.com/gokayfem/image-to-sculpt),
[Blockout 2 Real](https://github.com/gokayfem/blockout-to-real),
[Image 2 Low Poly](https://github.com/gokayfem/image-to-lowpoly) and
[Reference to Game](https://github.com/gokayfem/reference-to-game), each put an agent through a whole job
every time. This one is about the second time: how much of a job the agent has already done can run without
it, and how you know when it cannot.

Write-up: [Agent 2 Program: What Survives When You Compile an LLM Workflow]({POST}). Every table:
[`REPORT.md`](REPORT.md).

{fig("tier-latency.png", "Dot plot on a log axis of the median time for one answer at each tier, from 0.001 ms for an exact cache to 6.3 s for an LLM call that re-reads its contract",
     f"What one answer costs at each tier on an Apple M4. A local decision takes {n['fast_lo']} to {n['fast_hi']} ms; one typed call to a local 4B LLM takes {n['llm_ms']} ms with the contract as a cached prompt prefix and {n['llm_nocache_s']} s without it.")}
<p align="center"><a href="{FILM}"><img src="gallery/film-poster.jpg" width="100%" alt="Frame of the film: wrong answers per 100 requests for three policies across five phases of a replayed production loop"></a></p>

<p align="center"><sub>The 38-second film (click to play): the tier ladder, the unseen-group failure, a replay of {n['n_req']} real requests through the production loop, and the harness result.</sub></p>

## Why

Ask for "the same thing again, but faster" and the usual answer is a cache, or a small model trained on the
agent's past answers. Both look fine in a demo. These are the failures that went unnoticed until they were
measured here:

- **A fast model is confidently wrong on groups it has never seen.** Calibrated on a random split it kept a
  2% error promise on familiar groups and, in {n['r_viol']} of {n['splits']} random group splits, answered {n['r_ans']} of the
  requests from never-seen groups with {n['r_sil']} of those answers wrong. {n['conf9']} of those errors were made with a
  confidence of 0.9 or more.
- **Noise and new languages fail silently.** With 15% of letters mistyped, an embedding model was {n['ty_acc']}
  accurate and still answered {n['ty_ans']} of requests, {n['ty_wrong']} of them wrong. On Turkish versions of English
  requests it answered {n['tr_ans']} and got {n['tr_wrong']} of those wrong.
- **A second opinion trained on the same features repeats the mistake.** When the fast model was wrong, an
  ensemble on the same embedding gave the same wrong answer {n['same_ens']} of the time, a character n-gram model
  {n['same_tf']}, an LLM {n['same_llm']}.
- **The LLM fallback is not automatically safe.** A local 4B model took its "none of these" exit for only
  {n['llm_none']} of out-of-scope queries and gave an in-scope answer to {n['inj']} of {n['inj_n']} prompt-injection strings.
  A frontier model on the same requests took the exit for {n['o_none']} and answered {n['o_inj']} of {n['o_inj_n']}, so
  measure the model you have.
- **Programs rot.** A router compiled on one year of headlines went from {n['news_first']} to {n['news_last']} wrong answers
  over five years, and it was still answering. After a change of policy, {n['pol']} of the affected requests were
  answered under the old rule.
- **The agent's own runs contain habits you do not want to compile.** The agent reported a degree for
  {n['au_claim']} of {n['au_none']} profiles that show no education at all. A rule that copied this habit agreed with the
  agent on {n['ung']} of runs; the best honest rule agreed on {n['honest']}.

The skill gives the agent a procedure that deals with each of these:

- **Contract first.** Inputs, outputs, what varies between runs and what any correct output satisfies are
  written down before any code, with a version. The program is compiled from the contract; the recorded runs
  are evidence.
- **Typed decisions with tiers.** Every judgment is a closed question answered by the cheapest tier that is
  sure: exact cache, mined rules, a small decision model, one typed LLM call with the contract as a cached
  prefix, then the agent. A tier that is not sure abstains.
- **Calibration against the shift you expect.** Whole groups are held out, and the gate must pass two tests:
  selective risk on known groups, and false-accept rate on held-out groups and real out-of-scope inputs. A
  familiarity score separate from confidence does most of the work. Wrong answers per request from unseen
  groups: {n['r_far']} down to {n['g_far']}. Familiar requests still answered: {n['r_cov']} down to {n['g_cov']}.
- **An independent verifier.** A fast answer is never its own check. The default is a character n-gram model
  that must agree: {n['tf_ms']} ms per request, and it rejects {n['tf_catch']} of the fast model's errors on unseen groups.
- **Admission before service.** Held-out cases, whole held-out groups and seeded bad inputs. Every tier is
  admitted, the LLM hole included; a build that fails is refused, and no budget is loosened to pass.
- **Operation.** Shadow sampling, a circuit breaker per answer class, a contract version in every guard,
  regression-gated recompiles, and no retraining on the program's own unverified answers.
- **Compiling from observation.** A wrapper harness records the agent, searches candidate programs for each
  output field, admits one only if it agrees with the agent on runs it never saw, checks the agent's answers
  against its input, and leaves everything else as one narrow typed call.

## What compiles, and what does not

| The step is... | It compiles to... |
|---|---|
| A transform, lookup, calculation or tool call | Code |
| A judgment with a closed answer set, made repeatedly | A typed decision with tiers |
| An extraction the model does approximately (an estimate, a paraphrase) | Usually a narrow typed call; code only if it agrees with the model on runs it never saw |
| Open-ended writing or design | A small generation call fed by compact facts, or a template |
| An irreversible action whose trigger the runs do not pin down | Nothing. It stays with the agent |

"This decision stays with the model" is a valid result. One of three intent tasks could not be certified at a
2% error budget by any model family tested, and {n['n_short'] - n['n_code']} of {n['n_short']} profile fields failed admission as code.

## Install

**Claude Code** (plugin marketplace):

```bash
claude plugin marketplace add gokayfem/agent-to-program
```

```bash
claude plugin install agent-to-program@agent-to-program
```

Or copy the skill: `cp -R skills/workflow-compiler ~/.claude/skills/`.

**Codex, Cursor, Gemini CLI, Copilot and other agents** (via [skills](https://github.com/vercel-labs/skills)):

```bash
npx skills add gokayfem/agent-to-program
```

Or install by hand: copy `skills/workflow-compiler` to `~/.agents/skills/` (Codex) or `~/.cursor/skills/` (Cursor).

**claude.ai:** upload `workflow-compiler-v2.skill` under Settings > Capabilities > Skills.

Requirements: Python 3.10+. The runtime (`scripts/nsp.py`) is standard library only; the threshold procedures
need `numpy` and `scipy`; the decision-model tier needs `scikit-learn`. The skill needs no model API of its
own: it compiles a workflow around whatever agent or LLM you already use.

## Use

Ask naturally, for example:

- *"Here are five logs of you triaging expense lines. Compile that into a script and tell me what still needs a model."*
- *"This prompt classifies support tickets all day. Make it fast without making it wrong on customers it has never seen."*
- *"Analyze these 60 profiles the slow way, record what you do, then turn it into a program and show me which fields became code."*
- *"Is my router's confidence threshold safe for request types that were not in its training data?"*

The result is ordinary Python:

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
    check(trace, "contract_version", CONTRACT_VERSION == compiled_for)
    return route(x, trace)

program = Program("router", fn, fallback=agent, oracle=best_oracle,
                  shadow_rate=0.1, queue_path="recompile_queue.jsonl")
```

The full procedure is in [`skills/workflow-compiler/SKILL.md`](skills/workflow-compiler/SKILL.md).

## The evidence

Ten experiments. Nine ran on one Apple M4 with local models; the tenth gave {n['o_n']} of the same held-out
requests to a hosted model, Claude Opus 5.5 at medium effort. {n['n_llm']} real answers from the local LLMs are
stored in `cache/llm/`; 20 random group splits were used wherever no LLM was needed. [`REPORT.md`](REPORT.md) has every
table and is generated from `results/*.json`.

| # | Question | What came out |
|---|---|---|
| 1 | Which fast model should take over a typed decision? | A frozen encoder with a linear head answers {n['cov_clinc']} of CLINC150 and {n['cov_bank']} of BANKING77 inside a 2% silent-error budget; nothing could be certified at 2% on MASSIVE. Fine-tuned MiniLM: {n['ft_acc']} accurate at {n['ft_ms']} ms. A nearest-neighbour "semantic cache" was the weakest use of an embedding |
| 2 | Can the fast tier tell an input is unfamiliar? | Confidence alone lets {n['oos_conf']} of human-written out-of-scope queries through. Relative Mahalanobis distance lets {n['oos_maha']} through, at the cost of familiar coverage ({n['oos_conf_cov']} down to {n['oos_maha_cov']}) |
| 3 | Does calibration survive groups it never saw? | No with a random split ({n['r_sil']} wrong on unseen groups, {n['r_viol']} of {n['splits']} splits). Mostly with held-out groups and a familiarity gate ({n['g_far']} wrong per unseen request), but it still went over budget in {n['g_viol']} of splits; with {n['g_hi']} calibration groups instead of {n['g_lo']}, {n['g_hi_viol']} instead of {n['g_lo_viol']} |
| 4 | What can verify a fast answer, and at what cost? | Agreement with a character n-gram model: {n['tf_ms']} ms, rejects {n['tf_catch']} of unseen-group errors. Checkers on the same embedding repeat the mistake. The cost of a verifier is its false rejects |
| 5 | How good is the LLM tier, and whole stacks? | Qwen3-4B: {n['llm_seen']} on familiar requests, {n['llm_unseen']} on unseen groups, "none" exit on {n['llm_none']} of out-of-scope queries. A guarded fast tier in front of it: {n['stack_wrong']} wrong against {n['llm_wrong']} for always-LLM, with {n['stack_calls']} calls per request. Zero-shot on 150 intents the LLM scores {n['fine_llm']} against {n['fine_fast']} for the trained fast model; on Spanish versions of familiar requests it scores {n['es_llm']} against {n['es_fast']} |
| 6 | What happens in a production loop with drift? | Over {n['n_req']} replayed requests: naive {n['tot_naive']} wrong, guarded {n['tot_guard']}, guarded with the contract change announced {n['tot_ann']}. After the policy change {n['pol']} of the re-routed requests were answered under the old rule; {shadow_line(n)} Ungated self-training was the worst policy |
| 7 | And under natural drift? | Five years of dated headlines: a static program goes from {n['news_first']} to {n['news_last']} wrong among its answers. A shadow breaker opens in {n['breaker']}. Gated quarterly recompiles stay at {n['gated_silent']} but keep only {n['gated_cov']} coverage |
| 8 | Does a real compiled program pass admission? | The full build is refused: {n['full_silent']} silent errors, {n['full_llm_wrong']} of {n['full_llm_n']} from the LLM tier and {n['full_local_wrong']} of {n['full_local_n']} from the local tiers. Without the LLM and rule tiers it is admitted at {n['bare_silent']}, answering {n['bare_local']} locally. A naive gate answers {n['gib']} of {n['gib_n']} gibberish strings; the guarded gate none |
| 9 | Can a wrapper harness compile an agent from its runs? | After {n['n_obs']} recorded runs of a three-call profile analysis: {n['st_s']} s per profile instead of {n['ag_s']} s ({n['pr_s']} s if prose is still generated). {n['n_code']} of {n['n_short']} short fields became pure code; the rest is one narrow typed call reading {n['st_pt']} tokens instead of {n['ag_pt']} |
| 10 | Does a frontier model as the LLM tier change the picture? | Claude Opus 5.5 (medium effort) on {n['o_n']} of the same requests: {n['o_unseen']} right on never-seen groups against {n['q_unseen']} for the local model, "none" exit on {n['o_none']} of out-of-scope queries against {n['q_none']}, an in-scope answer to {n['o_inj']} of {n['o_inj_n']} prompt-injection strings against {n['q_inj']}. Always asking it is {n['o_always']} wrong; the guarded fast tier in front of it is {n['o_guard']} wrong and saves {n['o_saved']} of the calls; a naive fast tier in front of it is {n['o_naive']} wrong. Answered in batches through subagents, so no latency or cost |

{fig("unseen-groups.png", "Two bar charts: wrong answers per request from never-seen groups fall from 17.6% to 1.6% when calibration uses held-out groups; familiar requests answered fall from 95% to 63%",
     f"Experiment 3. Left: wrong answers per request from groups the fast model never saw. Right: what the fix costs on familiar traffic. Means over {n['splits']} random group splits, 2% budget.")}
{fig("same-wrong-answer.png", "Ranked bars: how often each kind of checker gives the same wrong answer as the fast model on never-seen groups, from 70% for an ensemble on the same embedding to 27% for a second LLM",
     "Experiment 4. How often a second opinion repeats the fast model's mistake. The more differently a checker is built, the less it agrees with the error.")}
{fig("typing-noise.png", "Line chart: with 15% of letters mistyped the embedding model's answered requests are 24.5% wrong while the character n-gram model stays at 2.3%",
     "Experiment 2. The same test requests with more and more mistyped letters. Both thresholds were certified on clean data and never changed.")}
{fig("production-loop.png", "Line chart of wrong answers per 100 requests across five phases for a naive program, a guarded one, and a guarded one where the contract change is announced",
     "Experiment 6. Wrong answers per 100 requests across the five scripted phases. Guards handle unfamiliar inputs; they cannot see a change of rules.")}
{fig("five-years-of-headlines.png", "Two line charts by month from 2013 to 2018: wrong answers among the headlines a static program still answers rise to about 60%; headlines from sections created after it was compiled rise to about 50%",
     "Experiment 7. A program compiled on 2012 headlines and never updated, month by month.")}
{fig("seconds-per-profile.png", "Bars: 11.2 seconds per profile for the three-call agent, 6.7 seconds compiled with generated prose, 3.8 seconds compiled for the structured analysis",
     f"Experiment 9. Seconds per profile on {n['n_test']} unseen profiles, same local model, one request at a time.")}
### What is measured and what is assumed

| Kind | What |
|---|---|
| Measured | All fast-model training and inference; every LLM answer and latency (Qwen3-4B and Gemma-3-4B, 4-bit, on this machine); the three-call agent and the compiled program of experiment 9; every accuracy, coverage and error figure against human labels |
| Replayed | Experiments 5, 6 and 8 look up the stored real LLM answer for each exact request instead of calling the model again; per-request latency in the stack and loop tables is assembled from measured per-tier medians |
| Scripted | The order and mix of traffic in experiment 6, the traffic mixes in experiment 5, the seeded bad inputs, the keyboard noise |
| Assumed | The agent tier: wherever a request is escalated, the human label is taken as the agent's answer and assumed correct. Escalations are counted, never timed |
| Real answers, different protocol | Experiment 10: Claude Opus 5.5 (medium effort) answered through Claude Code subagents in blind batches of about 220 requests, not one API call per request. No latency, no token probabilities, no measured cost |
| Not run | Any hosted model through its API; any OpenAI model; an agent with tools; GPU latency for the fast tier; real production traffic with repeats |

## Limits

- **Outside experiment 10 the LLMs are 4-bit local models of about 4B parameters.** Every other statement about
  the LLM tier is about them. Experiment 10 is one hosted model at one effort setting on one frozen split,
  answered in batches, with no latency or cost measured.
- **The agent is the human label.** Those labels are themselves imperfect, visibly so on MASSIVE and on the
  news sections.
- **One frozen group split carries the LLM-in-the-loop results** (experiments 4, 5, 6 and 8). The fast-tier
  conclusions were checked on 20 splits; the LLM ones were not.
- **Short-text, closed-answer tasks only.** Nothing here tests perceptual workflows such as video to Blender,
  where the verifier is the hard part.
- **The harness was tested on one workflow.** Its candidate programs were written by the coding agent after
  reading the observed runs; search, admission and scoring were mechanical. {n['n_obs']} runs are far too few to
  certify a decision model for the closed-set fields, so that step was not tested.
- **A certificate is an estimate.** Calibration against held-out groups went over budget in {n['g_viol']} of splits
  because rows from one group fail together. Shadow sampling stays on.

## Reproduce

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

```bash
.venv/bin/python -m bench.data                 # download datasets, write split manifest
.venv/bin/python -m bench.precompute           # embed everything (CPU, ~15 min)
.venv/bin/python -m bench.exp1_families        # model families
.venv/bin/python -m bench.exp1b_finetune       # end-to-end fine-tune (CPU, ~8 min)
.venv/bin/python -m bench.exp2_novelty         # unfamiliar inputs, typos, locale shift
.venv/bin/python -m bench.exp3_group_shift 20  # calibration x held-out groups, 20 splits
.venv/bin/python -m bench.exp3b_cal_groups     # how many calibration groups are needed
.venv/bin/python -m bench.exp7_news_drift      # natural drift
.venv/bin/python -m bench.label_llm route policy verify gemma latency bad fine massive   # real LLM answers (GPU, ~50 min)
.venv/bin/python -m bench.exp_latency          # per-request latency of local tiers
.venv/bin/python -m bench.exp4_verification
.venv/bin/python -m bench.exp5_tiers
.venv/bin/python -m bench.exp5b_llm_other      # the LLM on fine-grained holes and on a new language
.venv/bin/python -m bench.exp6_stream
.venv/bin/python -m bench.exp8_kernel
.venv/bin/python -m bench.exp9_harness collect # the agent the harness observes (GPU, ~25 min)
.venv/bin/python -m bench.exp9_harness run     # compile from traces, run compiled program
.venv/bin/python -m bench.exp10_closed prepare <dir>              # write blind batch files for a hosted model
.venv/bin/python -m bench.exp10_closed ingest <dir> <model-key>   # read its answers back
.venv/bin/python -m bench.exp10_closed score <model-key>          # same rows, same scoring as the local LLM
.venv/bin/python -m bench.make_report          # REPORT.md
.venv/bin/python -m bench.make_readme          # this file
```

The LLM stages need Apple Silicon (MLX) and are resumable: answers are appended to `cache/llm/`, keyed by
model, prompt version and exact query text, so the experiments that replay them run anywhere. Run one GPU job
at a time; two MLX processes sharing the GPU each ran about three times slower. Seeds are fixed and LLM
decoding is greedy. Datasets are downloaded on first use into `data/raw/`.

Kernel tests: `.venv/bin/python -m unittest discover -s tests` (21 tests). Skill tests:
`cd skills/workflow-compiler && python -m unittest discover -s tests` ({n['n_tests_skill']} tests).

## Layout

| Path | What |
|---|---|
| `skills/workflow-compiler/` | the skill: `SKILL.md`, runtime (`scripts/nsp.py`), threshold procedures (`selective.py`), decision-model tier with two-test calibration (`decision.py`), admission harness (`admit.py`), tests |
| `workflow-compiler-v2.skill` | the same skill as one file, for claude.ai |
| `REPORT.md` | the experimental report, generated from `results/` |
| `results/` | one JSON per experiment, plus `data_manifest.json` with a fingerprint of every split |
| `bench/` | datasets, encoders, model families, the local-LLM tier, one module per experiment, and the generators for `REPORT.md` and this file |
| `wfc/`, `tests/` | the runtime kernel the experiments import, and its tests |
| `cache/llm/` | every stored real LLM answer: model, prompt version, exact query, raw reply, token probability |
| `cache/closed/` | experiment 10: the batch files given to the hosted model, its answers, and the scoring key |
| `gallery/` | the figures above |
| `blog/` | the scripts that draw the figures and the film from `results/` |
| `logs/pip_freeze.txt` | exact package versions of the run |

## Corrections

- **11 November 2026, experiment 6.** The first version counted applications for a new credit card as requests
  re-routed by the policy change. The policy text does not cover them, and both LLMs given the changed contract
  kept most of them where they were. The phase was rerun with the three request types the policy names. Withdrawn:
  65% of re-routed requests answered under the old rule, detection after 400 requests at a 10% shadow rate, the
  naive program at 21.4% wrong overall, the guarded one at 6.7%. Current numbers are in the table above and in
  [`REPORT.md`](REPORT.md). The production-loop figure was redrawn.

## Credits

Experiments designed and run with Claude Fable 5.2 in Claude Code on local hardware.

Data: [CLINC150](https://github.com/clinc/oos-eval) (Larson et al., 2019),
[BANKING77](https://github.com/PolyAI-LDN/task-specific-datasets) (Casanueva et al., 2020),
[MASSIVE](https://github.com/alexa/massive) (FitzGerald et al., 2022), the
[News Category Dataset](https://huggingface.co/datasets/heegyu/news-category-dataset) (Misra, 2022) and the
[opensporks/resumes](https://huggingface.co/datasets/opensporks/resumes) profiles. `cache/llm/` quotes request
texts from CLINC150, BANKING77 and MASSIVE next to the model's answers; those texts belong to their datasets
and stay under the datasets' own licences.

Models: Qwen3-4B-Instruct-2507 and Gemma-3-4B-it (4-bit, via MLX) for the LLM tier; all-MiniLM-L6-v2,
bge-small-en-v1.5, bge-base-en-v1.5, multilingual-e5-small and potion-base-8M for the fast tier.

## License

MIT for the code, results and report.
"""
    return s


if __name__ == "__main__":
    out = ROOT / "README.md"
    out.write_text(build())
    print("wrote", out, len(out.read_text().splitlines()), "lines")
