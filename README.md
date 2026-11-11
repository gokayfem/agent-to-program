# Agent 2 Program

Tests of one idea: let an agent solve a workflow, then turn what it did into a program that
handles most later inputs fast, with a model stepping in only for small, typed decisions and
for inputs the program has not earned the right to answer.

Nine experiments on public human-labelled data with real local models (Apple M4, MLX). The
write-up is on the blog: https://www.generativemedia.blog/blog/agent-2-program-what-survives-when-you-compile-an-llm-workflow

**Read `REPORT.md` for every result.** It is generated from `results/*.json`; no number in it is typed by hand.


## Layout

| Path | What |
|---|---|
| `REPORT.md` | the experimental report |
| `results/` | one JSON per experiment, plus `data_manifest.json` (split fingerprints) |
| `wfc/` | the runtime kernel: tiers, guards, verifier hook (`nsp.py`), admission (`admit.py`), threshold procedures (`selective.py`) |
| `tests/` | kernel tests (`.venv/bin/python -m unittest discover -s tests`) |
| `bench/` | datasets, encoders, model families, the local-LLM tier, and one module per experiment |
| `skill/workflow-compiler/`, `workflow-compiler-v2.skill` | the skill revised with these results (procedure, runtime, decision tier with two-test calibration, 23 tests) |
| `cache/llm/` | every stored real LLM answer (model, prompt version, exact query, raw reply, token probability). Embeddings and agent traces are not in the repository; the commands below recreate them |
| `blog/` | scripts that draw the post's figures and film from `results/` |
| `logs/pip_freeze.txt` | exact package versions |

## Reproduce

```bash
python3.12 -m venv .venv && .venv/bin/pip install numpy scipy scikit-learn pandas pyarrow torch sentence-transformers model2vec mlx-lm joblib
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
.venv/bin/python -m bench.make_report
```

LLM stages are resumable: answers are appended to `cache/llm/` keyed by model, hole version and
exact query text. Run one GPU job at a time; two MLX processes sharing the GPU ran 3x slower each.
Seeds are fixed (`bench/common.py`, `bench/data.py`); LLM decoding is greedy.

## Data

Nothing in this repository is synthetic except the deliberately bad inputs. `bench/data.py` downloads
CLINC150, BANKING77, MASSIVE and the HuffPost News Category dataset from their public sources, and
`bench/exp9_harness.py` reads the public `opensporks/resumes` profiles. The files in `cache/llm/` quote
request texts from CLINC150, BANKING77 and MASSIVE next to the model's answers; those texts belong to
their datasets and stay under the datasets' own licences.

## License

MIT for the code, results and report.
