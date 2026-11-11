# LLM as a compiler: what holds up under harder tests

An earlier demonstration of this idea ran on synthetic expense lines with a stubbed LLM; this one runs on public, human-labelled data with real models on one machine. Run on 2026-11-11, Apple M4, local compute only.

## Summary

**What was run.** Nine experiments on four public datasets with human labels plus 2,484 public resume profiles; twelve fast-model families; two local LLMs with 18,529 stored real answers, and 180 recorded three-step agent runs; 20 random group splits wherever no LLM was needed. Every table below is generated from `results/*.json`.

**What it found.**

1. **The speed is real, and the amount you can take depends on the task.** A local decision costs 0.2–9 ms against 267 ms for one cached-prefix call to a local 4B LLM. On familiar traffic a frozen encoder with a linear head answers 94% of CLINC150 inside a 2% silent-error budget, 76% of BANKING77, and nothing on MASSIVE, where no model could be certified at 2%.
2. **The unseen-group failure from the first demo is general.** Calibrated on a random split, the fast tier keeps its 2% promise on familiar groups and, in all 20 splits, breaks it on groups it never saw: it answers 76% of them and 23% of those answers are wrong. 68% of those errors are made with confidence ≥ 0.9.
3. **The fix costs coverage and is not a guarantee.** Calibrating against held-out groups and real out-of-scope queries, with a familiarity score separate from confidence, cuts wrong answers per unseen-group input from 17.6% to 1.6% and familiar coverage from 95% to 63%. It still ended over budget in 35% of splits, because rows from one group fail together.
4. **A verifier is only as good as it is different.** When the fast model is wrong on an unseen group, a 5-model ensemble on the same embedding gives the same wrong label 70% of the time, a character n-gram model 38%, the LLM 32%. The lexical check adds 0.2 ms and catches 62% of those errors; verification is paid for in false rejects, not milliseconds.
5. **Embedding models fail silently on noise and on new languages; a lexical model fails honestly.** With 15% of letters mistyped, bge-small is 51% accurate and its confidence gate still answers 53% of requests, 25% of them wrong; character TF-IDF answers 43% with 2.3% wrong. On Turkish versions of English requests the same kind of model answers 26% and gets 84% of those wrong; a similarity floor refuses almost all of them.
6. **Here the local LLM is the weak tier.** Qwen3-4B is right on 95% of familiar requests and 78% of unseen-group requests and takes the "none of these" exit for only 45% of out-of-scope queries. A guarded fast tier in front of it delivers the same quality (11.4% vs 12.4% wrong on a 70/20/10 mix) with 0.46 calls per request instead of 1; with the agent as the only fallback it delivers 1.0% wrong and escalates 46%. The LLM is the stronger tier only where the fast model has seen nothing: 82% vs 72% on never-seen MASSIVE groups, 73% vs 36% on Spanish versions of familiar requests. A stronger LLM would change these absolute numbers; none was run.
7. **Compiled programs rot, and a contract change is the drift no guard sees.** On 6.4 years of dated headlines a program compiled on the first year goes from 11% to 65% silent errors while still answering. In the scripted loop the guards handle new groups, an out-of-scope burst and noise (1.3%, 2.7%, 1.0% wrong), but after a policy change 65% of re-routed requests are answered under the old policy until shadow sampling notices (400 requests at a 10% rate) or the change is announced (0%). Retraining on the system's own answers without a gate was the worst policy in the scripted loop (22% wrong) and stayed over budget in 57 of 65 months on the news stream.
8. **The wrapper-harness idea works modestly: the decisions shrink, few disappear.** Watching 60 runs of a three-call "analyze this profile" agent was enough to compile it into code plus one narrow typed call: 2.9× faster for the structured analysis (11.2 s → 3.8 s), 1.7× when prose is still generated. Only 1 of 6 short fields passed admission as pure code; the rest stayed with the model as one smaller typed call, because what the agent does for them is not a simple function of the text. The recordings also audited the agent: it reported a degree for 33 of 44 profiles that show no education at all.

**What is measured, replayed, or assumed.**

| Kind | What |
| --- | --- |
| Measured (real runs) | All fast-model training and inference; all LLM answers and latencies (Qwen3-4B, Gemma-3-4B on this machine); the three-call agent and the compiled program of Experiment 9; every accuracy, coverage and error figure against human labels |
| Replayed real outputs | Experiments 5, 6 and 8 look up the stored real LLM answer for each exact request instead of calling the model again; latency per request in the stack and loop tables is assembled from the measured per-tier medians |
| Scripted | The order and mix of traffic in Experiment 6 (five phases), the traffic mixes in Experiment 5, the seeded bad inputs, keyboard noise |
| Assumed | The agent: wherever a request is escalated, the human label is taken as the agent's answer and assumed correct; its cost is counted as escalations and never timed |
| Not run | Any hosted or frontier model; an agent with tools; GPU latency for the fast tier; real production traffic with repeats |

## 1. Test bed

**Machine.** Apple M4 (10 cores, 24 GB), macOS. All training, inference and benchmarks ran locally. Fast models ran on CPU; the LLMs ran on the GPU through MLX.

**Data.** Public datasets with human labels; nothing synthetic except the deliberately bad inputs.

| Dataset | What it is | train / calibration / test |
| --- | --- | --- |
| CLINC150 | 23,700 assistant queries, 150 intents in 10 domains; 1,200 human-written out-of-scope queries | 15,000 / 3,000 / 4,500 |
| BANKING77 | 13,083 bank-support messages, 77 fine-grained intents | 8,005 / 1,998 / 3,080 |
| MASSIVE (en-US) | 16,521 assistant utterances, 60 intents in 18 scenarios; same utterances human-translated into 5 other locales | 11,514 / 2,033 / 2,974 |
| HuffPost news | 206,581 dated headlines with editor-assigned section, 2012-01-28 to 2022-09-23 | by date |
| Resume profiles | 2,484 public resume profiles in 24 human-labelled fields (opensporks/resumes) | 60 observed / 40 admission / 80 test |

**Two kinds of decision.** *Fine*: predict the intent (150 / 77 / 60 classes). *Routing*: predict the coarse class (CLINC: 10 domains; MASSIVE: 18 scenarios) while treating each intent as a **group**. Whole groups are held out, which reproduces the earlier example's unseen-merchant failure on human-written text: the label space is known, the sub-population is new. On CLINC that is 90 training groups, 30 groups held out for calibration and 30 groups never seen by any model or threshold.

**Splits.** Models are fitted on train, thresholds are chosen on calibration, and every reported number is from test rows (or never-seen groups). Split membership is fingerprinted in `results/data_manifest.json`; one frozen group split (seed 7) is used wherever a real LLM is in the loop, and 20 random group splits wherever it is not.

**Models actually run.**

- Fast tier: TF-IDF (character and word n-grams), static embeddings (potion-base-8M), all-MiniLM-L6-v2, bge-small-en-v1.5, bge-base-en-v1.5, multilingual-e5-small; heads: logistic regression, k-NN, 1-NN (a semantic cache), class-conditional Gaussian (Mahalanobis), a 5-member MLP ensemble, mined n-gram rule committees, and MiniLM fine-tuned end to end.
- LLM tier: Qwen3-4B-Instruct-2507 (4-bit, MLX) as the main model, Gemma-3-4B-it (4-bit QAT) as a second, differently built model, Qwen3-8B (4-bit) in a pilot.
- Agent tier: **not a model**. Wherever the report says *agent* it means the human label, assumed correct, counted as an escalation and never timed.

**Terms.** *Coverage*: share of inputs a tier answers. *Silent error*: a wrong answer that was delivered without escalation; the silent-error rate is wrong answers over answered inputs. *False-accept rate*: wrong answers over **all** inputs of a population (used for populations a tier should mostly refuse). *Certified*: the threshold was chosen on calibration data by fixed-sequence testing so that the silent-error rate exceeds the budget with probability at most 5%, **if** deployment inputs are exchangeable with calibration inputs.

**Measured latency per request** (median, one request at a time on this machine):

| Tier | ms |
| --- | ---: |
| exact-match cache | 0.001 |
| mined rule committee | 0.09 |
| TF-IDF + logistic regression (end to end) | 0.19 |
| static embedding (potion) encode | 0.03 |
| MiniLM encode | 5.0 |
| bge-small encode | 8.5 |
| bge-base encode | 25.7 |
| logistic-regression head | 0.04 |
| nearest-neighbour search over 9,000 rows | 0.12 |
| MLP ensemble (5 members) | 0.07 |
| MiniLM fine-tuned end to end | 4.3 |
| LLM yes/no verification call (Qwen3-4B, cached prefix) | 195 |
| LLM routing call, short contract (Qwen3-4B, cached prefix) | 219 |
| LLM routing call, 1-shot contract (Qwen3-4B, cached prefix) | 267 |
| same call without the prefix cache | 6305 |
| LLM routing call, 1-shot contract (Gemma-3-4B) | 221 |


## 2. Experiment 1: which fast model should take over a typed decision?

*Real runs. Official train / calibration / test partitions, no group shift.* Each family answers only above a confidence threshold certified on the calibration split. "Certified coverage" is the share of test inputs it then answers; "silent" is the error rate among those answers on test.

**CLINC150, 150 intents** (test n = 4,500)

| Family | Accuracy | Coverage @2% | Silent @2% | Coverage @5% | Silent @5% | Fit (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| tfidf-char+LR | 92.1% | 79.6% | 1.68% | 91.7% | 4.32% | 8.2 |
| tfidf-word+LR | 90.9% | 72.1% | 1.39% | 87.5% | 4.04% | 3.9 |
| potion+LR | 90.8% | 74.7% | 1.43% | 88.8% | 4.30% | 0.5 |
| minilm+LR | 95.2% | 90.5% | 1.57% | 98.8% | 4.19% | 0.6 |
| bge-small+LR | 95.8% | 93.8% | 1.85% | 100.0% | 4.18% | 0.5 |
| bge-base+LR | 96.4% | 97.3% | 2.35% | 99.9% | 3.54% | 0.7 |
| minilm+kNN | 89.9% | 76.5% | 1.60% | 85.8% | 3.60% | 0.0 |
| bge-small+kNN | 93.4% | 84.8% | 1.55% | 97.0% | 5.11% | 0.0 |
| bge-small+1NN | 91.7% | 0.0% | – | 80.7% | 5.32% | 0.0 |
| bge-small+Maha | 93.6% | 88.6% | 2.18% | 96.6% | 4.46% | 0.1 |
| bge-small+MLPx5 | 96.4% | 95.7% | 2.04% | 100.0% | 3.58% | 17.1 |
| agree(tfidf-char+LR & bge-small+LR) | 95.8% | 91.4% | 1.46% | 100.0% | 4.18% | – |
| agree(minilm+LR & bge-small+LR) | 95.8% | 94.0% | 1.47% | 100.0% | 4.18% | – |
| MiniLM fine-tuned end to end | 96.0% | 93.8% | 1.45% | 100.0% | 4.02% | 269 |

| Rule committee (word n-gram → label) | Rules | Coverage | Silent | Coverage, ≥2 rules agree | Silent |
| --- | ---: | ---: | ---: | ---: | ---: |
| mined-rules(p>=1.0,n>=5) | 2816 | 51.4% | 4.24% | 32.9% | 1.28% |
| mined-rules(p>=1.0,n>=10) | 911 | 41.0% | 1.79% | 23.2% | 0.38% |
| mined-rules(p>=0.95,n>=10) | 1037 | 49.5% | 2.74% | 32.0% | 1.25% |

Exact-match cache on test: 0.04% of queries repeat a training query verbatim (2 hits, 2 of them with a different label).

**BANKING77, 77 intents** (test n = 3,080)

| Family | Accuracy | Coverage @2% | Silent @2% | Coverage @5% | Silent @5% | Fit (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| tfidf-char+LR | 90.3% | 72.0% | 1.40% | 87.2% | 4.28% | 3.9 |
| tfidf-word+LR | 88.1% | 60.6% | 1.29% | 77.5% | 3.27% | 2.8 |
| potion+LR | 87.8% | 52.1% | 1.37% | 77.5% | 4.19% | 0.4 |
| minilm+LR | 91.6% | 77.6% | 1.88% | 89.8% | 3.98% | 0.3 |
| bge-small+LR | 92.0% | 75.9% | 1.33% | 92.5% | 4.39% | 0.4 |
| bge-base+LR | 92.8% | 79.4% | 1.55% | 91.8% | 3.78% | 0.5 |
| minilm+kNN | 92.2% | 79.6% | 1.55% | 92.4% | 4.11% | 0.0 |
| bge-small+kNN | 93.4% | 86.4% | 2.03% | 94.3% | 3.96% | 0.0 |
| bge-small+1NN | 92.8% | 0.0% | – | 75.0% | 3.64% | 0.0 |
| bge-small+Maha | 89.8% | 64.5% | 1.61% | 83.1% | 3.63% | 0.0 |
| bge-small+MLPx5 | 92.1% | 70.0% | 1.02% | 91.1% | 4.17% | 8.0 |
| agree(tfidf-char+LR & bge-small+LR) | 92.0% | 74.7% | 0.87% | 88.7% | 2.82% | – |
| agree(minilm+LR & bge-small+LR) | 92.0% | 80.6% | 1.61% | 91.9% | 3.89% | – |
| MiniLM fine-tuned end to end | 92.5% | 77.8% | 1.21% | 91.5% | 3.73% | 255 |

| Rule committee (word n-gram → label) | Rules | Coverage | Silent | Coverage, ≥2 rules agree | Silent |
| --- | ---: | ---: | ---: | ---: | ---: |
| mined-rules(p>=1.0,n>=5) | 896 | 35.9% | 4.07% | 21.2% | 2.76% |
| mined-rules(p>=1.0,n>=10) | 201 | 21.9% | 1.93% | 11.5% | 0.28% |
| mined-rules(p>=0.95,n>=10) | 219 | 24.8% | 2.88% | 13.9% | 0.47% |

Exact-match cache on test: 0.16% of queries repeat a training query verbatim (5 hits, 0 of them with a different label).

**MASSIVE, 60 intents** (test n = 2,974)

| Family | Accuracy | Coverage @2% | Silent @2% | Coverage @5% | Silent @5% | Fit (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| tfidf-char+LR | 83.5% | 0.0% | – | 70.2% | 4.89% | 8.0 |
| tfidf-word+LR | 82.7% | 0.0% | – | 72.5% | 6.03% | 3.8 |
| potion+LR | 74.8% | 0.0% | – | 0.0% | – | 1.5 |
| minilm+LR | 81.6% | 0.0% | – | 52.3% | 4.50% | 0.6 |
| bge-small+LR | 83.9% | 0.0% | – | 63.4% | 5.25% | 0.6 |
| bge-base+LR | 85.3% | 0.0% | – | 70.3% | 4.55% | 0.6 |
| minilm+kNN | 82.7% | 0.0% | – | 56.2% | 2.69% | 0.0 |
| bge-small+kNN | 85.6% | 0.0% | – | 69.7% | 3.23% | 0.0 |
| bge-small+1NN | 83.7% | 0.0% | – | 33.5% | 3.81% | 0.0 |
| bge-small+Maha | 84.4% | 0.0% | – | 52.4% | 3.73% | 0.1 |
| bge-small+MLPx5 | 87.0% | 0.0% | – | 82.0% | 6.19% | 10.2 |
| agree(tfidf-char+LR & bge-small+LR) | 83.9% | 0.0% | – | 78.5% | 5.18% | – |
| agree(minilm+LR & bge-small+LR) | 83.9% | 0.0% | – | 72.5% | 4.55% | – |

| Rule committee (word n-gram → label) | Rules | Coverage | Silent | Coverage, ≥2 rules agree | Silent |
| --- | ---: | ---: | ---: | ---: | ---: |
| mined-rules(p>=1.0,n>=5) | 824 | 31.4% | 5.36% | 12.7% | 2.12% |
| mined-rules(p>=1.0,n>=10) | 202 | 17.9% | 3.38% | 6.3% | 2.69% |
| mined-rules(p>=0.95,n>=10) | 256 | 27.1% | 4.84% | 12.3% | 3.01% |

Exact-match cache on test: 0.71% of queries repeat a training query verbatim (21 hits, 2 of them with a different label).

**What this shows**

- On in-distribution traffic a small frozen encoder with a linear head is enough. On CLINC150, bge-small + LR is 95.8% accurate and can answer 93.8% of queries inside a 2% budget at about 8.6 ms; character TF-IDF + LR answers 79.6% at 0.19 ms.
- Fine-tuning MiniLM end to end (4.5 minutes on CPU) gives 96.0% and 93.8% coverage at 4.3 ms: the accuracy of the larger frozen encoders at about half the latency of bge-small.
- How much can be taken over depends on the task, not the model. BANKING77's fine-grained intents allow 72%–86% at 2%. On MASSIVE (accuracy 75%–87%) **no family could be certified at 2%**; at 5% they answer 63%–82%. A compile step has to be allowed to conclude "this decision stays with the model".
- A semantic cache (1-NN on embedding similarity) is the weakest use of an embedding. With the threshold peeked on test, it could cover 55% of CLINC150 at 2% against 95% for a linear head on the same vectors (30% vs 82% on BANKING77): similarity to one stored example says little about whether its label transfers.
- Rule committees mined from history are cheap (0.09 ms) and abstain by construction, but single rules are not safe: 911 rules with no counter-example in training cover 41% of CLINC150 test queries with 1.8% silent errors; requiring two agreeing rules gives 23% coverage at 0.4%.
- Verbatim repeats are rare in these crowd-sourced sets (0.04%–0.71%), so the exact-match cache does almost nothing here. Real traffic repeats far more; that was not measured.
- "Certified" is a statistical statement. 4 family/dataset pairs ended slightly over the 2% budget on test: bge-base+LR on clinc150 (2.35%); bge-small+Maha on clinc150 (2.18%); bge-small+MLPx5 on clinc150 (2.04%); bge-small+kNN on banking77 (2.03%). The calibration and test partitions of these datasets were collected separately, so even here exchangeability is only approximate.


## 3. Experiment 2: can the fast tier tell that an input is unfamiliar?

*Real runs.* CLINC150 models; 4,500 in-scope test queries against 1,000 human-written out-of-scope queries, 2,000 news headlines and 152 seeded bad inputs. AUROC: 1.0 means the score separates perfectly, 0.5 means it carries no information. "OOS answered" is the share of out-of-scope queries the fast tier would answer at the threshold certified for a 2% budget, when calibration contains only in-scope rows (R) or also real out-of-scope rows (G).

| View | Score | AUROC vs OOS | FPR@95 | AUROC vs news | AUROC vs own errors | R: in-scope coverage | R: OOS answered | G: in-scope coverage | G: OOS answered |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bge-small | confidence (max prob.) | 0.922 | 45.1% | 0.939 | 0.930 | 93.8% | 38.6% | 81.4% | 9.0% |
| bge-small | top-2 margin | 0.918 | 47.9% | 0.934 | 0.929 | 93.8% | 40.7% | 81.5% | 9.4% |
| bge-small | energy | 0.956 | 22.2% | 0.975 | 0.841 | 82.5% | 6.1% | 74.4% | 3.8% |
| bge-small | nearest-neighbour similarity | 0.963 | 19.1% | 0.996 | 0.744 | 0.0% | 0.0% | 0.0% | 0.0% |
| bge-small | mean similarity to 5 nearest | 0.963 | 19.5% | 0.997 | 0.744 | 0.0% | 0.0% | 54.1% | 0.4% |
| bge-small | nearest class centroid | 0.966 | 17.4% | 0.998 | 0.769 | 18.3% | 0.0% | 18.0% | 0.0% |
| bge-small | Mahalanobis | 0.946 | 26.5% | 0.994 | 0.722 | 0.0% | 0.0% | 0.0% | 0.0% |
| bge-small | relative Mahalanobis | 0.974 | 14.1% | 0.996 | 0.816 | 74.3% | 1.3% | 74.6% | 1.3% |
| bge-small | MLP-ensemble confidence | 0.965 | 16.3% | 0.985 | 0.932 | 92.6% | 11.6% | 90.2% | 8.6% |
| bge-small | MLP-ensemble unanimity | 0.960 | 16.4% | 0.978 | 0.932 | 92.6% | 11.6% | 90.2% | 8.6% |
| bge-small | agreement with TF-IDF model | 0.936 | 100.0% | 0.951 | 0.906 | 91.4% | 15.3% | 86.7% | 6.5% |
| bge-small | min(confidence, similarity) | 0.956 | 22.6% | 0.983 | 0.841 | 86.0% | 6.2% | 83.1% | 3.6% |
| bge-small | learned trust model (uses real OOS examples) | 0.971 | 13.1% | 0.989 | 0.924 | 92.6% | 8.2% | 91.1% | 6.2% |
| tfidf | confidence (max prob.) | 0.954 | 19.6% | 0.983 | 0.910 | 79.6% | 5.5% | 73.4% | 3.8% |
| tfidf | nearest-neighbour similarity | 0.917 | 39.9% | 0.983 | 0.730 | 0.0% | 0.0% | 0.0% | 0.0% |
| minilm | confidence (max prob.) | 0.915 | 49.4% | 0.945 | 0.931 | 90.5% | 28.1% | 75.9% | 7.2% |
| minilm | relative Mahalanobis | 0.972 | 12.5% | 0.989 | 0.843 | 65.2% | 0.8% | 65.0% | 0.8% |
| bge-base | confidence (max prob.) | 0.934 | 36.5% | 0.955 | 0.933 | 97.3% | 55.1% | 87.5% | 14.9% |
| bge-base | energy | 0.975 | 13.1% | 0.987 | 0.859 | 92.9% | 8.8% | 89.0% | 5.0% |
| bge-base | relative Mahalanobis | 0.980 | 8.6% | 0.997 | 0.833 | 85.8% | 2.1% | 83.8% | 1.8% |

Share of seeded bad inputs the score would let through (G thresholds, no structural guards in front):

| Score | structural | gibberish | keyboard mash | injection text | far-domain text |
| --- | ---: | ---: | ---: | ---: | ---: |
| bge-small/msp | 5% | 3% | 2% | 0% | 5% |
| bge-small/knn1 | 0% | 0% | 0% | 0% | 0% |
| bge-small/rel_maha | 0% | 0% | 0% | 0% | 0% |
| bge-small/min(msp,knn1) | 5% | 0% | 0% | 0% | 0% |
| bge-small/learned_trust | 5% | 0% | 0% | 0% | 0% |
| tfidf/msp | 0% | 0% | 0% | 0% | 0% |
| tfidf/knn1 | 0% | 0% | 0% | 0% | 0% |

**Keyboard noise on the same test queries** (thresholds unchanged, certified on clean calibration data for 2%):

| View (+LR) | Noise | Accuracy | Coverage (confidence) | Silent | Coverage (min of confidence, similarity) | Silent |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| tfidf | clean | 92.1% | 79.6% | 1.7% | 48.8% | 1.3% |
| tfidf | 5% of letters | 88.7% | 67.0% | 2.2% | 32.0% | 1.7% |
| tfidf | 15% of letters | 79.3% | 42.8% | 2.3% | 11.2% | 1.8% |
| minilm | clean | 95.2% | 90.5% | 1.6% | 69.1% | 1.6% |
| minilm | 5% of letters | 79.8% | 72.7% | 6.7% | 36.4% | 2.7% |
| minilm | 15% of letters | 48.5% | 46.3% | 25.5% | 9.2% | 3.8% |
| bge-small | clean | 95.8% | 93.8% | 1.8% | 86.0% | 1.9% |
| bge-small | 5% of letters | 81.0% | 79.4% | 8.2% | 61.0% | 4.3% |
| bge-small | 15% of letters | 51.4% | 53.2% | 24.5% | 26.1% | 7.8% |
| bge-base | clean | 96.4% | 97.3% | 2.4% | 95.4% | 2.4% |
| bge-base | 5% of letters | 83.8% | 88.2% | 9.5% | 82.0% | 7.9% |
| bge-base | 15% of letters | 56.4% | 68.8% | 28.1% | 54.7% | 20.1% |

**Locale shift** (MASSIVE, 60 intents; trained on en-US; the same test utterances in other languages; thresholds certified for 5% on en-US calibration data):

| View (+LR) | Locale | Accuracy | Coverage (confidence) | Silent | Coverage (min of confidence, similarity) | Silent | AUROC en vs locale (similarity) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| tfidf | en-US | 83.5% | 70.2% | 4.9% | 57.1% | 4.9% | – |
| tfidf | de-DE | 22.3% | 10.2% | 26.8% | 2.0% | 19.0% | 0.960 |
| tfidf | es-ES | 21.3% | 6.3% | 18.3% | 1.0% | 13.3% | 0.965 |
| tfidf | fr-FR | 28.3% | 9.5% | 19.5% | 1.4% | 17.1% | 0.958 |
| tfidf | tr-TR | 14.4% | 4.2% | 25.6% | 0.5% | 28.6% | 0.975 |
| tfidf | ja-JP | 7.7% | 1.4% | 14.3% | 1.8% | 27.8% | 0.957 |
| minilm | en-US | 81.6% | 52.3% | 4.5% | 35.1% | 3.9% | – |
| minilm | de-DE | 17.6% | 8.4% | 56.2% | 0.2% | 0.0% | 0.970 |
| minilm | es-ES | 13.8% | 16.6% | 82.0% | 0.1% | 0.0% | 0.971 |
| minilm | fr-FR | 19.1% | 7.8% | 52.4% | 0.1% | 0.0% | 0.972 |
| minilm | tr-TR | 10.4% | 4.4% | 75.6% | 0.0% | 0.0% | 0.982 |
| minilm | ja-JP | 6.8% | 4.9% | 92.5% | 0.0% | 0.0% | 0.996 |
| bge-small | en-US | 83.9% | 63.4% | 5.2% | 19.0% | 1.2% | – |
| bge-small | de-DE | 24.5% | 19.0% | 46.5% | 0.0% | 0.0% | 0.962 |
| bge-small | es-ES | 21.8% | 19.2% | 64.0% | 0.0% | 0.0% | 0.967 |
| bge-small | fr-FR | 29.1% | 18.9% | 41.6% | 0.0% | 0.0% | 0.957 |
| bge-small | tr-TR | 12.5% | 26.0% | 83.7% | 0.0% | 0.0% | 0.983 |
| bge-small | ja-JP | 10.7% | 19.1% | 82.1% | 0.0% | 0.0% | 0.967 |
| me5-small | en-US | 82.9% | 61.9% | 5.0% | 38.6% | 3.3% | – |
| me5-small | de-DE | 58.3% | 38.6% | 18.6% | 1.2% | 8.1% | 0.879 |
| me5-small | es-ES | 66.0% | 40.6% | 10.4% | 0.9% | 11.5% | 0.858 |
| me5-small | fr-FR | 65.2% | 41.0% | 11.7% | 3.0% | 3.3% | 0.791 |
| me5-small | tr-TR | 53.0% | 30.4% | 17.8% | 0.0% | 0.0% | 0.931 |
| me5-small | ja-JP | 60.5% | 39.1% | 17.0% | 0.1% | 0.0% | 0.929 |

**What this shows**

- A model's own confidence is a poor familiarity test. With the threshold certified on in-scope data only, bge-small + LR would answer 39% of human-written out-of-scope queries (bge-base: 55%), which turns a 2% tier into a 10% tier on CLINC's own 82/18 in-scope/out-of-scope test mix. The bigger, more accurate encoder is the worse one here.
- Distance in embedding space is much better. Relative Mahalanobis distance reaches AUROC 0.974 (bge-base 0.980) and lets through 1.3% of out-of-scope queries, at the price of in-scope coverage (74% instead of 94%; 86% with bge-base). Nearest-neighbour similarity ranks out-of-scope inputs almost as well but is useless as a correctness score, so on its own it certifies nothing (the 0% rows).
- Showing the calibration step real out-of-scope examples helps every score: confidence alone drops from 39% to 9% answered. A learned trust model over confidence, distances and cross-view agreement keeps the most coverage (91%) but still answers 6.2% of out-of-scope queries. Nothing gets near-scope inputs under 1% without giving up a quarter of the in-scope traffic.
- Fine-tuning improves this but does not fix it: the fine-tuned MiniLM's confidence has AUROC 0.967 and still answers 14% of out-of-scope queries at its certified threshold.
- Keyboard noise breaks embedding models *silently*. At 15% noisy letters bge-small falls to 51% accuracy while its confidence gate still answers 53% of requests, 25% of them wrong (bge-base: 28%; fine-tuned MiniLM: 11%). Character TF-IDF degrades honestly: coverage falls to 43% and silent errors stay at 2.3%. Adding the similarity floor cuts bge-small's silent errors to 8%, still several times the budget.
- A new language is the easy case for a familiarity gate and the worst case without one. English-only encoders are 7%–29% accurate on translated utterances, yet the confidence gate keeps answering (worst: bge-small on tr-TR, 26% answered, 84% of those wrong). Similarity to the training set separates English from every other locale with AUROC 0.96–1.00, and the combined gate answers at most 0.2% of them. A multilingual encoder (me5-small) generalises (53%–66% accurate) but its confidence gate is 10%–19% wrong on what it answers: better generalisation makes the shift harder to see, not safer.


## 4. Experiment 3: does a calibrated fast tier stay calibrated on groups it has never seen?

*Real runs, 20 random group splits per dataset.* The routing task, with a fifth of each class's intents held out entirely. Protocols: **R** calibrates on held-out rows of training groups (a random split); **G-mix** adds an equal number of rows from held-out calibration groups; **G-worst** runs two tests, selective risk on seen-group rows and false-accept rate on held-out-group rows. "Over budget" is the share of the 20 splits in which the realised rate exceeded the budget.

**CLINC routing (10 domains; 90 / 30 / 30 train / calibration / test groups)**

| Model | Accuracy, seen groups | Accuracy, unseen groups | Unseen inputs with confidence ≥ 0.9 | …of which wrong | Unseen-group errors made at confidence ≥ 0.9 |
| --- | ---: | ---: | ---: | ---: | ---: |
| tfidf-char+LR | 97.0% ± 0.4% | 62.0% ± 4.5% | 33% | 13% | 11% |
| minilm+LR | 96.6% ± 0.4% | 64.3% ± 4.1% | 83% | 30% | 69% |
| bge-small+LR | 97.2% ± 0.4% | 68.9% ± 3.7% | 84% | 25% | 68% |
| bge-small+kNN | 98.0% ± 0.3% | 68.6% ± 4.5% | 53% | 17% | 28% |

bge-small + LR, budget 2%, thresholds by fixed-sequence testing:

| Trust score | Calibration | Seen: coverage | Seen: silent | Seen: over budget | Unseen: coverage | Unseen: wrong answers / all unseen inputs | Unseen: over budget | Unseen: silent among answered | Splits with no fast tier at all |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| confidence | R | 95.5% | 1.27% | 0% | 76.4% | 17.60% | 100% | 22.9% | 0% |
| confidence | G-mix | 26.1% | 0.06% | 0% | 9.5% | 1.05% | 25% | 4.9% | 45% |
| confidence | G-worst | 39.4% | 0.09% | 0% | 15.2% | 1.48% | 30% | 6.8% | 20% |
| familiarity only | R | 74.4% | 1.35% | 0% | 20.0% | 3.89% | 60% | 17.2% | 5% |
| familiarity only | G-mix | 46.2% | 0.66% | 0% | 2.2% | 0.35% | 0% | 12.0% | 5% |
| familiarity only | G-worst | 33.7% | 0.44% | 0% | 3.5% | 0.52% | 10% | 7.1% | 45% |
| confidence + familiarity | R | 89.8% | 0.96% | 0% | 57.7% | 11.84% | 95% | 19.8% | 0% |
| confidence + familiarity | G-mix | 50.2% | 0.21% | 0% | 10.9% | 1.25% | 25% | 10.9% | 5% |
| confidence + familiarity | G-worst | 62.8% | 0.19% | 0% | 14.9% | 1.64% | 35% | 9.5% | 0% |
| learned trust model | R | 83.2% | 0.70% | 0% | 56.6% | 10.38% | 85% | 15.9% | 10% |
| learned trust model | G-mix | 58.9% | 0.12% | 0% | 11.0% | 1.19% | 15% | 9.1% | 5% |
| learned trust model | G-worst | 69.8% | 0.15% | 0% | 14.8% | 1.54% | 40% | 8.8% | 0% |

bge-small + LR, budget 5%, thresholds by fixed-sequence testing:

| Trust score | Calibration | Seen: coverage | Seen: silent | Seen: over budget | Unseen: coverage | Unseen: wrong answers / all unseen inputs | Unseen: over budget | Unseen: silent among answered | Splits with no fast tier at all |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| confidence | R | 99.9% | 2.76% | 0% | 99.4% | 30.64% | 100% | 30.8% | 0% |
| confidence | G-mix | 66.9% | 0.29% | 0% | 34.6% | 4.96% | 55% | 11.6% | 10% |
| confidence | G-worst | 71.7% | 0.23% | 0% | 34.0% | 4.14% | 40% | 11.2% | 0% |
| familiarity only | R | 99.9% | 2.80% | 0% | 99.6% | 30.81% | 100% | 30.9% | 0% |
| familiarity only | G-mix | 72.1% | 1.21% | 0% | 13.6% | 2.26% | 5% | 14.9% | 0% |
| familiarity only | G-worst | 78.2% | 1.44% | 0% | 19.5% | 3.46% | 20% | 17.0% | 0% |
| confidence + familiarity | R | 99.8% | 2.71% | 0% | 98.6% | 30.13% | 100% | 30.5% | 0% |
| confidence + familiarity | G-mix | 79.6% | 0.37% | 0% | 36.0% | 4.91% | 50% | 12.2% | 0% |
| confidence + familiarity | G-worst | 79.5% | 0.30% | 0% | 32.3% | 4.08% | 35% | 12.2% | 0% |
| learned trust model | R | 99.9% | 2.69% | 0% | 98.5% | 29.97% | 100% | 30.4% | 0% |
| learned trust model | G-mix | 87.4% | 0.30% | 0% | 38.0% | 4.84% | 50% | 12.0% | 0% |
| learned trust model | G-worst | 87.0% | 0.27% | 0% | 35.1% | 4.21% | 35% | 11.6% | 0% |

Threshold procedures compared (bge-small + LR, confidence + familiarity, budget 5%):

| Procedure | Calibration | Seen: coverage | Seen: silent | Seen: over budget | Unseen: coverage | Unseen: wrong / all | Unseen: over budget | Unseen: worst split |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| empirical | R | 99.9% | 2.75% | 0% | 99.0% | 30.36% | 100% | 37.0% |
| empirical | G-worst | 82.8% | 0.43% | 0% | 39.8% | 5.47% | 65% | 10.2% |
| wilson_scan | R | 99.9% | 2.75% | 0% | 99.0% | 30.36% | 100% | 37.0% |
| wilson_scan | G-worst | 80.3% | 0.35% | 0% | 34.4% | 4.56% | 45% | 9.6% |
| fixed_sequence | R | 99.8% | 2.71% | 0% | 98.6% | 30.13% | 100% | 37.0% |
| fixed_sequence | G-worst | 79.5% | 0.30% | 0% | 32.3% | 4.08% | 35% | 8.3% |
| cluster_bootstrap | R | 99.6% | 2.62% | 0% | 97.5% | 29.46% | 100% | 37.0% |
| cluster_bootstrap | G-worst | 70.8% | 0.23% | 0% | 24.3% | 2.88% | 10% | 7.0% |

Same comparison across model families (budget 5%):

| Model | R + confidence: wrong / all unseen | G-worst gate: seen coverage | Seen: silent | Unseen: coverage | Unseen: wrong / all | Unseen: over budget |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| tfidf-char+LR | 35.7% | 79.2% | 0.28% | 30.4% | 4.10% | 40% |
| minilm+LR | 33.6% | 77.7% | 0.43% | 28.2% | 3.99% | 20% |
| bge-small+LR | 30.6% | 79.5% | 0.30% | 32.3% | 4.08% | 35% |
| bge-small+kNN | 30.5% | 83.6% | 0.28% | 28.5% | 4.16% | 35% |

**MASSIVE routing (18 scenarios; 34 / 13 / 13 groups)**

| Model | Accuracy, seen groups | Accuracy, unseen groups | Unseen inputs with confidence ≥ 0.9 | …of which wrong | Unseen-group errors made at confidence ≥ 0.9 |
| --- | ---: | ---: | ---: | ---: | ---: |
| tfidf-char+LR | 91.6% ± 2.5% | 55.1% ± 9.2% | 30% | 14% | 8% |
| minilm+LR | 91.1% ± 3.7% | 52.9% ± 10.0% | 81% | 42% | 71% |
| bge-small+LR | 92.2% ± 3.1% | 51.4% ± 11.0% | 77% | 42% | 66% |
| bge-small+kNN | 93.7% ± 2.5% | 56.7% ± 11.8% | 40% | 21% | 18% |

At a 2% budget this task is barely compilable with these models: with random-split calibration the confidence gate still answers 27% of unseen-group inputs and gets 24% of those wrong, while the guarded gate (G-worst, confidence + familiarity) keeps 44% seen-group coverage with 1.3% wrong answers per unseen input. The table shows the 5% budget.

bge-small + LR, budget 5%, thresholds by fixed-sequence testing:

| Trust score | Calibration | Seen: coverage | Seen: silent | Seen: over budget | Unseen: coverage | Unseen: wrong answers / all unseen inputs | Unseen: over budget | Unseen: silent among answered | Splits with no fast tier at all |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| confidence | R | 90.1% | 3.91% | 5% | 66.5% | 26.42% | 100% | 39.0% | 0% |
| confidence | G-mix | 50.4% | 0.83% | 0% | 12.6% | 3.07% | 25% | 18.5% | 5% |
| confidence | G-worst | 64.3% | 1.06% | 0% | 18.9% | 4.41% | 30% | 21.8% | 0% |
| familiarity only | R | 72.7% | 3.29% | 10% | 50.2% | 20.84% | 70% | 35.9% | 10% |
| familiarity only | G-mix | 26.1% | 0.69% | 0% | 3.6% | 0.96% | 5% | 18.9% | 40% |
| familiarity only | G-worst | 44.7% | 1.19% | 0% | 9.5% | 2.57% | 10% | 24.7% | 20% |
| confidence + familiarity | R | 88.3% | 3.30% | 0% | 61.9% | 23.31% | 100% | 36.8% | 0% |
| confidence + familiarity | G-mix | 61.3% | 0.77% | 0% | 16.0% | 3.05% | 10% | 18.1% | 0% |
| confidence + familiarity | G-worst | 67.7% | 1.02% | 0% | 20.4% | 4.09% | 25% | 19.6% | 0% |
| learned trust model | R | 72.2% | 2.54% | 0% | 48.7% | 16.38% | 80% | 26.2% | 20% |
| learned trust model | G-mix | 64.0% | 0.70% | 0% | 20.4% | 2.93% | 20% | 13.4% | 5% |
| learned trust model | G-worst | 71.3% | 0.86% | 0% | 24.3% | 3.18% | 15% | 13.6% | 0% |

Threshold procedures compared (bge-small + LR, confidence + familiarity, budget 5%):

| Procedure | Calibration | Seen: coverage | Seen: silent | Seen: over budget | Unseen: coverage | Unseen: wrong / all | Unseen: over budget | Unseen: worst split |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| empirical | R | 93.7% | 5.26% | 70% | 80.7% | 35.51% | 100% | 61.9% |
| empirical | G-worst | 73.4% | 1.38% | 5% | 26.4% | 5.85% | 50% | 16.4% |
| wilson_scan | R | 90.5% | 4.07% | 0% | 69.1% | 27.83% | 100% | 53.5% |
| wilson_scan | G-worst | 69.8% | 1.20% | 0% | 22.1% | 4.45% | 30% | 14.6% |
| fixed_sequence | R | 88.3% | 3.30% | 0% | 61.9% | 23.31% | 100% | 49.9% |
| fixed_sequence | G-worst | 67.7% | 1.02% | 0% | 20.4% | 4.09% | 25% | 12.4% |
| cluster_bootstrap | R | 86.4% | 2.97% | 0% | 57.0% | 20.86% | 90% | 45.6% |
| cluster_bootstrap | G-worst | 62.7% | 0.91% | 0% | 16.4% | 3.38% | 20% | 9.5% |

Same comparison across model families (budget 5%):

| Model | R + confidence: wrong / all unseen | G-worst gate: seen coverage | Seen: silent | Unseen: coverage | Unseen: wrong / all | Unseen: over budget |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| tfidf-char+LR | 20.0% | 70.3% | 1.08% | 26.9% | 3.32% | 10% |
| minilm+LR | 19.7% | 59.6% | 1.11% | 18.2% | 3.39% | 25% |
| bge-small+LR | 26.4% | 67.7% | 1.02% | 20.4% | 4.09% | 25% |
| bge-small+kNN | 23.5% | 68.7% | 0.57% | 20.1% | 3.29% | 15% |

**How many held-out groups does the calibration need?** (CLINC, 20 splits per row, 30 never-seen test groups in every row, bge-small + LR, confidence + familiarity, G-worst, budget 5%; left block: row-level fixed-sequence test, right block: group-aware bootstrap)

| Calibration groups | Training groups | Seen coverage | Unseen: wrong / all | Over budget | Worst split | Seen coverage (group-aware) | Unseen: wrong / all | Over budget | Worst split |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | 110 | 75.0% | 4.57% | 40% | 12.0% | 66.8% | 3.50% | 15% | 14.0% |
| 20 | 100 | 78.5% | 4.43% | 40% | 9.2% | 69.7% | 3.08% | 15% | 8.0% |
| 30 | 90 | 79.5% | 4.08% | 35% | 8.3% | 70.8% | 2.88% | 10% | 7.0% |
| 40 | 80 | 80.7% | 3.74% | 35% | 7.2% | 72.8% | 2.68% | 15% | 6.6% |
| 50 | 70 | 81.1% | 3.50% | 20% | 6.1% | 75.2% | 2.47% | 5% | 5.1% |

**What this shows**

- The unseen-merchant failure is real on human text and it is a property of the models, not of the toy. On groups never seen in training, accuracy falls from 97% to 69% (CLINC) and from 92% to 51% (MASSIVE), and the embedding classifiers do not notice: 68% of bge-small's unseen-group errors are made with confidence ≥ 0.9. TF-IDF is wrong more often but far less sure of itself (11%).
- Random-split calibration certifies exactly the wrong thing. It keeps its promise on seen groups in every split (1.3% silent at a 2% budget) and breaks it on unseen groups in every split: the tier answers 76% of unseen-group inputs and 23% of those answers are wrong (at a 5% budget: 99% answered, 31% wrong).
- Calibrating against held-out groups works, and its price is coverage on familiar traffic. With the confidence + familiarity gate wrong answers per unseen-group input fall from 17.6% to 1.6% (2% budget) and from 31% to 4.1% (5%), while seen-group coverage drops from 95% to 63% and from 100% to 79%. The gate does not make unseen-group answers trustworthy (10% of the ones it lets through are still wrong); it works by refusing 85% of them.
- It is not a guarantee. The realised rate on never-seen groups exceeded the budget in 35% of splits at 2% and 35% at 5%, although the procedure promises 5%. Rows from one group fail together, so 4,500 calibration rows from 30 groups are closer to 30 observations than to 4,500. A group-aware bootstrap picker is over budget in 10% of splits at 5% and pays 9 points of seen coverage for it.
- More held-out groups help, steadily. Going from 10 to 50 calibration groups takes the row-level test from over budget in 40% of splits to 20% and the group-aware one from 15% to 5%; the worst split falls from 12% to 6% wrong answers per unseen input. With 20 splits per row these rates are themselves rough (one split is 5 points).
- The scores trade differently. Familiarity alone gives the fewest wrong answers on unseen groups (0.35% per unseen input at 2%) but lets more seen-group errors through (0.66%). A learned trust model keeps the most seen coverage (87% at 5%) with similar unseen-group leakage (4.2%).
- MASSIVE repeats the pattern with fewer groups and noisier labels: random calibration lets 26% of unseen-group inputs through wrong at a 5% budget, the guarded gate 4.1% with 68% seen coverage. All four model families behave the same way (last table of each block), so the remedy belongs in the calibration protocol, not in the choice of classifier.


## 5. Experiment 4: what can verify a fast answer, and what does it cost?

*Real runs; LLM verifiers are stored real answers.* Frozen CLINC routing split. The fast tier (bge-small + LR) answers everything, so the verifier is the only guard. Populations: 1,800 seen-group test rows, 1,500 rows of never-seen groups, 1,000 out-of-scope queries. *Catch* = share of the fast tier's errors rejected; *false reject* = share of its correct answers rejected; *wrong / all* = wrong answers that still get through, over all inputs.

| Verifier | Added ms | Seen: catch | Seen: false reject | Seen: wrong / all | Unseen: catch | Unseen: false reject | Unseen: wrong / all | OOS: catch | OOS: wrong / all |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| none (answer everything) | 0 | 0% | 0.0% | 2.5% | 0% | 0.0% | 27.2% | 0% | 100.0% |
| self: msp >= thr (random-split calibration) | 0 | 53% | 3.2% | 1.2% | 48% | 18.9% | 14.1% | 45% | 55.1% |
| self: msp >= thr (grouped calibration) | 0 | 93% | 54.4% | 0.2% | 95% | 85.3% | 1.3% | 95% | 4.8% |
| familiarity: knn1 (same embedding) | 0.12 | 78% | 27.4% | 0.6% | 87% | 89.7% | 3.7% | 99% | 1.0% |
| familiarity: rel. Mahalanobis (same embedding) | 0.02 | 76% | 25.4% | 0.6% | 86% | 70.1% | 3.9% | 96% | 3.7% |
| familiarity: knn1 (tf-idf view) | 0.19 | 91% | 62.7% | 0.2% | 94% | 97.5% | 1.7% | 100% | 0.5% |
| agree: kNN head, same embedding | 0.12 | 73% | 1.6% | 0.7% | 48% | 16.7% | 14.1% | 53% | 47.0% |
| agree: MLP ensemble unanimous, same embedding | 0.07 | 40% | 0.5% | 1.5% | 34% | 5.0% | 18.0% | 26% | 74.0% |
| agree: minilm+LR (other encoder) | 4.99 | 64% | 1.7% | 0.9% | 45% | 20.7% | 15.0% | 51% | 49.2% |
| agree: bge-base+LR (bigger encoder) | 26 | 82% | 0.9% | 0.4% | 52% | 8.2% | 12.9% | 45% | 55.3% |
| agree: tfidf-char+LR (lexical view) | 0.19 | 82% | 1.9% | 0.4% | 62% | 28.3% | 10.2% | 68% | 32.0% |
| rules: mined rules do not contradict | 0.09 | 27% | 0.6% | 1.8% | 15% | 1.4% | 23.1% | 5% | 94.8% |
| rules: a mined rule confirms | 0.09 | 98% | 30.9% | 0.1% | 86% | 70.0% | 3.8% | 95% | 5.4% |
| committee: tfidf + minilm both agree | 5.14 | 89% | 3.3% | 0.3% | 71% | 40.2% | 8.0% | 80% | 20.5% |
| tfidf agrees AND knn1 familiar | 0.19 | 100% | 28.0% | 0.0% | 89% | 90.7% | 3.1% | 99% | 0.9% |
| LLM classifies independently and agrees (qwen3-4b) | 267 | 82% | 4.5% | 0.4% | 68% | 12.6% | 8.7% | 79% | 21.0% |
| LLM yes/no check of the proposal (qwen3-4b) | 195 | 71% | 12.3% | 0.7% | 61% | 16.2% | 10.6% | 80% | 20.3% |
| second LLM agrees (gemma3-4b) | 221 | 87% | 11.7% | 0.3% | 73% | 25.1% | 7.3% | 77% | 23.3% |
| both LLMs agree with the proposal | 488 | 100% | 13.8% | 0.0% | 80% | 29.0% | 5.5% | 85% | 15.3% |

**Do verifiers repeat the fast model's mistakes?** For each second opinion B: its own error rate, how often it is wrong when the fast model is wrong, and how often it gives the *same* wrong label (the case no agreement check can catch).

On seen groups (fast model error rate 2.5%):

| Second opinion | Its error rate | Wrong when fast model is wrong | Same wrong label | Rows compared | Fast-model errors among them |
| --- | ---: | ---: | ---: | ---: | ---: |
| bge-small+kNN | 2.3% | 31% | 27% | 1800 | 45 |
| minilm+LR | 2.8% | 47% | 36% | 1800 | 45 |
| bge-base+LR | 1.7% | 33% | 18% | 1800 | 45 |
| tfidf-char+LR | 2.6% | 31% | 18% | 1800 | 45 |
| mlp-ens(bge-small) | 2.1% | 67% | 60% | 1800 | 45 |
| llm:qwen3-4b | 5.3% | 36% | 18% | 1800 | 45 |
| llm:gemma3-4b | 12.1% | 27% | 13% | 630 | 15 |

On never-seen groups (fast model error rate 27.2%):

| Second opinion | Its error rate | Wrong when fast model is wrong | Same wrong label | Rows compared | Fast-model errors among them |
| --- | ---: | ---: | ---: | ---: | ---: |
| bge-small+kNN | 33.1% | 77% | 52% | 1500 | 408 |
| minilm+LR | 36.5% | 79% | 55% | 1500 | 408 |
| bge-base+LR | 23.6% | 65% | 48% | 1500 | 408 |
| tfidf-char+LR | 40.6% | 74% | 38% | 1500 | 408 |
| mlp-ens(bge-small) | 25.5% | 82% | 70% | 1500 | 408 |
| llm:qwen3-4b | 21.7% | 46% | 32% | 1500 | 408 |
| llm:gemma3-4b | 31.8% | 50% | 27% | 600 | 165 |

**What this shows**

- A stricter threshold on the model's own confidence is not verification. Calibrated on a random split it catches 48% of unseen-group errors and lets 55% of out-of-scope queries through.
- Second opinions built on the same embedding mostly repeat the mistake. A k-NN head and a unanimous 5-MLP ensemble catch 48% and 34% of unseen-group errors. When the fast model is wrong on an unseen group, the ensemble gives the *same wrong label* 70% of the time; the TF-IDF model 38%; the LLM 32%. Independence is measurable, and it follows how differently the checker was built.
- The cheapest useful verifier is a lexical model: 0.19 ms, catches 82% of seen-group errors, 62% of unseen-group errors and 68% of out-of-scope answers, while rejecting 1.9% of correct seen-group answers. Agreement checks cannot refuse an input both models are wrong about in the same way, which is why out-of-scope queries still get through (32%).
- Familiarity checks do what agreement cannot: they refuse 99% of out-of-scope answers and 87% of unseen-group errors for 0.12 ms, but they also refuse 27% of correct seen-group answers. A verifier is paid for mostly in false rejects, not in milliseconds.
- An LLM is the most independent checker and the most expensive: asking the local LLM to classify independently costs 267 ms and catches 68% of unseen-group errors and 79% of out-of-scope answers. A yes/no check of the proposal is cheaper (195 ms) and worse on both counts: it catches 61% of unseen-group errors and rejects 12.3% of correct seen-group answers (independent classification: 4.5%). Running it on every fast answer gives the speed back; it belongs on a sample (shadowing) or behind a cheap pre-filter.


## 6. Experiment 5: the real LLM tier, and whole tier stacks

*LLM numbers are real outputs of local models on this machine; the agent is the human label (assumed).*

**The LLM tier alone.** For out-of-scope queries the correct behaviour is the "none of these" exit, so accuracy there is the exit rate.

| Model | Population | n | Correct | Took the none exit | Unparseable | Wrong among answered | Share answered with p = 1 | Correct when p = 1 | Correct when p < 1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| qwen3-4b | seen groups | 1800 | 94.7% | 1.0% | 0.11% | 4.2% | 98% | 95.7% | 34.5% |
| qwen3-4b | never-seen groups | 1500 | 78.3% | 2.1% | 0.00% | 20.0% | 96% | 79.4% | 55.2% |
| qwen3-4b | out of scope | 1000 | 44.7% | 44.7% | 0.00% | 100.0% | 87% | 45.2% | 40.9% |
| qwen3-4b | seen groups, 15% keyboard noise | 600 | 83.5% | 8.8% | 0.67% | 7.7% | 95% | 85.8% | 40.0% |
| gemma3-4b | seen groups | 630 | 87.9% | 5.2% | 0.00% | 7.2% | 36% | 84.8% | 89.7% |
| gemma3-4b | never-seen groups | 600 | 68.2% | 8.0% | 0.00% | 25.9% | 38% | 74.6% | 64.2% |
| gemma3-4b | out of scope | 300 | 50.7% | 50.7% | 0.00% | 100.0% | 32% | 43.2% | 54.1% |

Two differently built LLMs as each other's check (answer only when both give the same in-scope label):

| Population | n | Both agree and answer | Wrong when they agree | Qwen alone: wrong answers / all |
| --- | ---: | ---: | ---: | ---: |
| seen groups | 630 | 86.0% | 0.7% | 3.3% |
| never-seen groups | 600 | 71.8% | 14.4% | 20.2% |
| out of scope | 300 | 24.7% | 100.0% | 53.0% |

**Tier stacks.** `fast[R]`: confidence threshold from random-split calibration. `fast[G]`: confidence + familiarity gate calibrated against held-out groups and real out-of-scope queries (2% budget; `G5` = 5%). `+V`: an independent lexical model must agree. `llm[conf]`: the LLM answers only when its own token probability clears a calibrated threshold. Latency uses the measured medians (measured: results/exp_latency.json, results/llm_latency.json); escalations to the agent are counted, not timed.

Traffic mix 70/20/10 seen/unseen/oos (seen / never-seen groups / out of scope; an assumption, three variants shown):

| Stack | Wrong answers delivered | Answered locally | Answered by LLM | Escalated to agent | LLM calls / request | Compute ms / request |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| always-LLM | 12.35% | 0.0% | 94.3% | 5.7% | 1.00 | 267 |
| fast model, no abstention | 17.19% | 100.0% | 0.0% | 0.0% | 0.00 | 9 |
| fast[R] → llm | 13.27% | 87.0% | 10.6% | 2.3% | 0.13 | 43 |
| fast[G] → llm | 11.29% | 55.0% | 39.6% | 5.5% | 0.45 | 129 |
| fast[G]+V → llm | 11.38% | 54.2% | 40.3% | 5.5% | 0.46 | 131 |
| fast[G5]+V → llm | 10.87% | 62.7% | 32.0% | 5.3% | 0.37 | 108 |
| cache → rules → fast[G]+V → llm | 11.29% | 61.1% | 33.5% | 5.4% | 0.39 | 109 |
| fast[G]+V → llm[conf] | 1.03% | 54.2% | 0.0% | 45.8% | 0.46 | 131 |
| fast[G]+V | 1.03% | 54.2% | 0.0% | 45.8% | 0.00 | 9 |
| fast[R] | 9.15% | 87.0% | 0.0% | 13.0% | 0.00 | 9 |
| rules → llm | 12.17% | 36.6% | 57.9% | 5.5% | 0.63 | 169 |
| always-agent (reference) | 0.00% | 0.0% | 0.0% | 100.0% | 0.00 | 0 |

Traffic mix 90/8/2 (seen / never-seen groups / out of scope; an assumption, three variants shown):

| Stack | Wrong answers delivered | Answered locally | Answered by LLM | Escalated to agent | LLM calls / request | Compute ms / request |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| always-LLM | 6.42% | 0.0% | 97.9% | 2.1% | 1.00 | 267 |
| fast model, no abstention | 6.43% | 100.0% | 0.0% | 0.0% | 0.00 | 9 |
| fast[R] → llm | 4.74% | 93.0% | 6.5% | 0.6% | 0.07 | 27 |
| fast[G] → llm | 4.92% | 67.6% | 30.5% | 1.9% | 0.32 | 95 |
| fast[G]+V → llm | 5.02% | 66.9% | 31.3% | 1.9% | 0.33 | 97 |
| fast[G5]+V → llm | 4.64% | 75.1% | 23.2% | 1.7% | 0.25 | 75 |
| cache → rules → fast[G]+V → llm | 4.88% | 74.5% | 23.7% | 1.8% | 0.25 | 73 |
| fast[G]+V → llm[conf] | 0.44% | 66.9% | 0.0% | 33.1% | 0.33 | 97 |
| fast[G]+V | 0.44% | 66.9% | 0.0% | 33.1% | 0.00 | 9 |
| fast[R] | 3.28% | 93.0% | 0.0% | 7.0% | 0.00 | 9 |
| rules → llm | 6.13% | 44.9% | 53.2% | 1.9% | 0.55 | 147 |
| always-agent (reference) | 0.00% | 0.0% | 0.0% | 100.0% | 0.00 | 0 |

Traffic mix 50/30/20 (seen / never-seen groups / out of scope; an assumption, three variants shown):

| Stack | Wrong answers delivered | Answered locally | Answered by LLM | Escalated to agent | LLM calls / request | Compute ms / request |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| always-LLM | 19.00% | 0.0% | 89.9% | 10.1% | 1.00 | 267 |
| fast model, no abstention | 29.41% | 100.0% | 0.0% | 0.0% | 0.00 | 9 |
| fast[R] → llm | 22.95% | 80.8% | 14.7% | 4.5% | 0.19 | 60 |
| fast[G] → llm | 18.38% | 42.0% | 48.1% | 9.9% | 0.58 | 163 |
| fast[G]+V → llm | 18.44% | 41.3% | 48.8% | 10.0% | 0.59 | 165 |
| fast[G5]+V → llm | 17.86% | 49.9% | 40.4% | 9.7% | 0.50 | 142 |
| cache → rules → fast[G]+V → llm | 18.41% | 47.4% | 42.8% | 9.8% | 0.53 | 147 |
| fast[G]+V → llm[conf] | 1.58% | 41.3% | 0.0% | 58.7% | 0.59 | 165 |
| fast[G]+V | 1.58% | 41.3% | 0.0% | 58.7% | 0.00 | 9 |
| fast[R] | 15.84% | 80.8% | 0.0% | 19.2% | 0.00 | 9 |
| rules → llm | 18.94% | 28.1% | 61.9% | 10.0% | 0.72 | 192 |
| always-agent (reference) | 0.00% | 0.0% | 0.0% | 100.0% | 0.00 | 0 |

Per population: wrong answers delivered, and share answered locally (no LLM, no agent):

| Stack | wrong: seen groups | wrong: never-seen groups | wrong: out of scope | wrong: seen groups, 15% keyboard noise | local: seen groups | local: never-seen groups | local: out of scope | local: seen groups, 15% keyboard noise |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| always-LLM | 4.2% | 19.5% | 55.3% | 7.0% | 0% | 0% | -0% | 0% |
| fast model, no abstention | 2.5% | 27.2% | 100.0% | 37.8% | 100% | 100% | 100% | 100% |
| fast[R] → llm | 1.6% | 21.5% | 78.5% | 23.5% | 96% | 73% | 55% | 72% |
| fast[G] → llm | 2.4% | 20.1% | 55.7% | 6.7% | 74% | 16% | 2% | 16% |
| fast[G]+V → llm | 2.6% | 20.1% | 55.6% | 6.7% | 73% | 15% | 2% | 16% |
| fast[G5]+V → llm | 2.3% | 18.4% | 56.0% | 6.5% | 81% | 29% | 4% | 27% |
| cache → rules → fast[G]+V → llm | 2.4% | 20.1% | 56.0% | 6.7% | 81% | 21% | 4% | 22% |
| fast[G]+V → llm[conf] | 0.1% | 3.8% | 1.9% | 0.0% | 73% | 15% | 2% | 16% |
| fast[G]+V | 0.1% | 3.8% | 1.9% | 0.0% | 73% | 15% | 2% | 16% |
| fast[R] | 1.2% | 14.1% | 55.1% | 21.0% | 96% | 73% | 55% | 72% |
| rules → llm | 3.8% | 19.6% | 55.7% | 7.0% | 49% | 10% | 2% | 12% |
| always-agent (reference) | 0.0% | 0.0% | 0.0% | 0.0% | 0% | 0% | 0% | 0% |

**Compile from traces.** The fast model trained on the LLM's own answers for 1,350 past requests (the LLM was right on 95.0% of them; 14 answers that took the none exit were dropped) versus the same rows with human labels. For reference the LLM itself scores 94.7% on seen groups and 78.3% on never-seen groups.

| Training labels | bge-small+LR: seen | bge-small+LR: unseen | Agreement with LLM (seen) | tfidf+LR: seen |
| --- | ---: | ---: | ---: | ---: |
| human labels, all 9000 train rows | 97.5% | 72.8% | 93.6% | 97.4% |
| human labels, the same trace rows | 96.9% | 73.4% | 93.2% | 94.1% |
| LLM labels, the same trace rows | 93.3% | 76.1% | 93.1% | 91.7% |

**The same LLM on fine-grained holes** (zero-shot, the closed set is the list of intent names; the fast model is bge-small + LR trained on the task's training split, scored on the same rows):

| Task | n | LLM correct | LLM none / unparseable | Fast model correct | LLM: out-of-scope sent to the none exit |
| --- | ---: | ---: | ---: | ---: | ---: |
| clinc150 | 750 | 74.8% | 5.9% | 96.4% | 67% |
| banking77 | 539 | 59.6% | 6.1% | 91.7% | – |

**The LLM on a new language** (MASSIVE scenario router, 18 classes, English contract; fast models trained on English training groups only):

| Requests | n | LLM (Qwen3-4B) | Fast: bge-small + LR (English encoder) | Fast: multilingual-e5-small + LR |
| --- | ---: | ---: | ---: | ---: |
| English, seen groups | 400 | 83.8% | 93.5% | 91.5% |
| English, never-seen groups | 400 | 81.5% | 71.8% | 75.2% |
| Spanish, same requests as row 1 | 400 | 73.2% | 36.0% | 83.2% |
| German, same requests as row 1 | 400 | 77.8% | 40.0% | 85.0% |

**What this shows**

- In this test bed the local LLM is the weakest tier, not the compiled one. Qwen3-4B with a one-shot contract is right on 94.7% of seen-group requests and 78.3% of unseen-group requests, and it takes the "none of these" exit for only 45% of out-of-scope queries. It is robust where the embedding models are not (84% on 15% keyboard noise). A larger hosted model would very likely do better; that was not run.
- Its confidence is not a usable abstention signal: 96% of its unseen-group answers come with token probability 1.0 and are right 79% of the time. No threshold on it could be certified at a 5% budget on the calibration pools (the `llm[conf]` stack therefore sends everything it would have answered to the agent).
- Two differently built LLMs agreeing is a better signal than either one's confidence: on unseen groups they agree on 72% of requests and are wrong on 14% of those; on out-of-scope queries they agree on a wrong in-scope label for 25%.
- A guarded fast tier in front of the LLM gives the same quality for half the model calls. On the 70/20/10 mix, always-LLM delivers 12.4% wrong answers at 267 ms per request; the stack `fast[G]+V → llm` delivers 11.4% with 0.46 LLM calls and 131 ms. Almost all of its remaining errors are the LLM's: with the agent as the only fallback the same fast tier delivers 1.0% wrong answers and escalates 46% of requests.
- The naive stack is fast and wrong: `fast[R] → llm` uses 0.13 LLM calls per request and delivers 13.3% wrong answers, more than always-LLM, because it answers 55% of out-of-scope queries and 73% of unseen-group requests itself.
- Engineering of the hole matters as much as the model. Keeping the contract as a fixed prefix and reusing its KV cache cut a routing call from 6305 ms to 267 ms. A longer contract is not free even when cached (219 ms with the short contract), and batching requests that share a long cached prefix duplicated it per sequence and pushed this machine into swap.
- The fast tier and the LLM are good at opposite things. Zero-shot on fine-grained intents the LLM is far behind a model trained on the task (75% vs 96% on CLINC150, 60% vs 92% on BANKING77). On what the fast tier has not seen, it is ahead: 82% vs 72% on never-seen MASSIVE groups, and 73% / 78% on the Spanish / German versions of requests where the English fast model falls to 36% / 40%. (A multilingual encoder closes most of that gap by itself: 83% / 85%.) That is the division of labour the tiers assume, and it only pays if the gate sends the right inputs to each.
- Compiling from the LLM's own traces transfers the LLM, including its mistakes: trained on 1,350 LLM-labelled requests the fast model reaches 93.3% on seen groups (human labels for the same rows: 96.9%; the LLM itself: 94.7%) and 76.1% on unseen groups (73.4% with human labels). The student is bounded by its teacher on familiar inputs, so reviewed outputs are worth more than more traces.


## 7. Experiment 6: a production loop with scripted drift

*Replay of real outputs: every request is a held-out human-written query, fast models are really run and retrained in the loop, every LLM answer is a stored real answer (100% of requests have one). Simulated: the traffic script, and the agent (human label).* 2,790 requests in five phases.

| Phase | Requests | Mix |
| --- | ---: | --- |
| 1 steady | 420 | oos 20, seen 400 |
| 2 new groups | 610 | oos 20, seen 250, unseen 340 |
| 3 out-of-scope burst | 600 | oos 300, seen 200, unseen 100 |
| 4 noisy input | 520 | oos 20, typo 400, unseen 100 |
| 5 policy change | 640 | affected 240, seen 400 |

**Abstentions go to the local LLM, then to the agent** (the shadow oracle is the LLM):

| Policy | Wrong delivered | …of which from the fast tier | Fast | LLM | Agent | LLM calls / req (incl. shadow) | ms / req | wrong ph.1 | ph.2 | ph.3 | ph.4 | ph.5 | wrong on re-routed requests |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A always-LLM | 18.78% | 0.00% | 0.0% | 91.9% | 8.1% | 1.00 | 267 | 7.6% | 14.4% | 32.8% | 11.0% | 23.4% | 55% |
| B static, naive calibration | 26.52% | 21.43% | 82.7% | 13.8% | 3.5% | 0.17 | 55 | 6.7% | 15.6% | 42.0% | 24.0% | 37.5% | 97% |
| C static, gate + verifier | 21.00% | 6.70% | 43.1% | 49.0% | 8.0% | 0.57 | 161 | 5.7% | 13.9% | 32.3% | 10.8% | 35.5% | 90% |
| D = C + shadow sampling + class breakers | 21.00% | 6.70% | 43.1% | 49.0% | 8.0% | 0.61 | 161 | 5.7% | 13.9% | 32.3% | 10.8% | 35.5% | 90% |
| D+ = D, contract change announced | 17.96% | 1.08% | 36.3% | 55.7% | 8.0% | 0.67 | 179 | 5.7% | 13.9% | 32.3% | 10.8% | 22.2% | 55% |
| E = D + regression-gated recompiles | 20.11% | 5.41% | 45.3% | 46.7% | 8.0% | 0.59 | 155 | 5.7% | 13.9% | 32.5% | 10.6% | 31.6% | 78% |
| E' = E, regression suite left stale | 20.11% | 5.41% | 45.3% | 46.7% | 8.0% | 0.59 | 155 | 5.7% | 13.9% | 32.5% | 10.6% | 31.6% | 78% |
| E+ = E, contract change announced | 18.10% | 1.40% | 40.3% | 51.8% | 7.9% | 0.64 | 168 | 5.7% | 13.9% | 32.5% | 10.6% | 22.8% | 55% |
| F ungated self-training | 27.49% | 22.94% | 85.9% | 11.3% | 2.7% | 0.14 | 46 | 6.7% | 14.9% | 45.7% | 26.2% | 37.2% | 95% |

**Abstentions go straight to the agent** (the shadow oracle is the agent). This removes the local LLM's own errors, so what is left is the compiled tier and the loop:

| Policy | Wrong delivered | …of which from the fast tier | Fast | LLM | Agent | LLM calls / req (incl. shadow) | ms / req | wrong ph.1 | ph.2 | ph.3 | ph.4 | ph.5 | wrong on re-routed requests |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| B static, naive calibration | 21.43% | 21.43% | 82.7% | 0.0% | 17.3% | 0.00 | 9 | 4.8% | 10.7% | 29.2% | 20.2% | 36.4% | 95% |
| C static, gate + verifier | 6.70% | 6.70% | 43.1% | 0.0% | 56.9% | 0.00 | 9 | 0.0% | 1.3% | 2.7% | 1.0% | 24.7% | 65% |
| D = C + shadow sampling + class breakers | 4.41% | 4.41% | 40.4% | 0.0% | 59.6% | 0.00 | 9 | 0.0% | 1.3% | 2.7% | 1.0% | 14.7% | 39% |
| D+ = D, contract change announced | 1.08% | 1.08% | 36.3% | 0.0% | 63.7% | 0.00 | 9 | 0.0% | 1.3% | 2.7% | 1.0% | 0.2% | 0% |
| E = D + regression-gated recompiles | 3.69% | 3.69% | 46.0% | 0.0% | 54.0% | 0.00 | 9 | 0.0% | 1.1% | 2.3% | 1.0% | 12.0% | 32% |
| E' = E, regression suite left stale | 3.69% | 3.69% | 46.0% | 0.0% | 54.0% | 0.00 | 9 | 0.0% | 1.1% | 2.3% | 1.0% | 12.0% | 32% |
| E+ = E, contract change announced | 1.61% | 1.61% | 43.5% | 0.0% | 56.5% | 0.00 | 9 | 0.0% | 1.1% | 2.3% | 1.0% | 3.0% | 8% |
| F ungated self-training | 22.15% | 22.15% | 85.9% | 0.0% | 14.1% | 0.00 | 9 | 4.5% | 10.2% | 30.7% | 22.3% | 37.0% | 96% |

Shadow estimate of the fast tier's silent-error rate versus the truth (10% of fast answers re-checked; in D the checker is the LLM, in D* and E* the agent):

| Policy | Phase | Fast share | Shadow samples | Shadow estimate | 95% interval | True silent rate of fast answers |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| D | 1 steady | 67.1% | 28 | 7.1% | 2.0% – 22.6% | 0.0% |
| D | 2 new groups | 36.6% | 25 | 0.0% | 0.0% – 13.3% | 3.6% |
| D | 3 out-of-scope burst | 29.5% | 19 | 0.0% | 0.0% – 16.8% | 9.0% |
| D | 4 noisy input | 16.2% | 6 | 16.7% | 3.0% – 56.4% | 6.0% |
| D | 5 policy change | 68.1% | 39 | 17.9% | 9.0% – 32.7% | 36.2% |
| D* | 1 steady | 67.1% | 28 | 0.0% | 0.0% – 12.1% | 0.0% |
| D* | 2 new groups | 36.6% | 25 | 0.0% | 0.0% – 13.3% | 3.6% |
| D* | 3 out-of-scope burst | 29.5% | 19 | 15.8% | 5.5% – 37.6% | 9.0% |
| D* | 4 noisy input | 16.2% | 6 | 16.7% | 3.0% – 56.4% | 6.0% |
| D* | 5 policy change | 56.4% | 29 | 13.8% | 5.5% – 30.6% | 26.0% |
| E* | 1 steady | 67.1% | 28 | 0.0% | 0.0% – 12.1% | 0.0% |
| E* | 2 new groups | 43.8% | 32 | 0.0% | 0.0% – 10.7% | 2.6% |
| E* | 3 out-of-scope burst | 36.8% | 26 | 11.5% | 4.0% – 29.0% | 6.3% |
| E* | 4 noisy input | 19.6% | 9 | 11.1% | 2.0% – 43.5% | 4.9% |
| E* | 5 policy change | 64.4% | 39 | 20.5% | 10.8% – 35.5% | 18.7% |

How quickly shadow sampling notices the policy change (policy D*, agent as checker):

| Shadow rate | Requests until the breaker opens | Wrong on re-routed requests | Wrong overall | Checker calls / request | False alarms (other classes) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 5% | 400 | 39% | 4.41% | 0.017 | 0 |
| 10% | 400 | 39% | 4.41% | 0.038 | 0 |
| 20% | 200 | 17% | 2.51% | 0.071 | 0 |
| 40% | 150 | 10% | 1.90% | 0.139 | 0 |

Events, policy E (185 regression cases re-adjudicated by the agent):

- request 400: recompile (periodic) on 112 production labels: rejected (recent hold-out 0/10 wrong, regression suite 97.3% → 96.9%)
- request 800: recompile (periodic) on 304 production labels: **deployed** (recent hold-out 1/32 wrong, regression suite 97.3% → 96.6%)
- request 1200: recompile (periodic) on 443 production labels: **deployed** (recent hold-out 1/48 wrong, regression suite 96.6% → 96.5%)
- request 1600: recompile (periodic) on 568 production labels: **deployed** (recent hold-out 4/56 wrong, regression suite 96.5% → 96.2%)
- request 2000: recompile (periodic) on 769 production labels: rejected (recent hold-out 6/60 wrong, regression suite 96.2% → 96.0%)
- request 2400: recompile (periodic) on 921 production labels: **deployed** (recent hold-out 5/71 wrong, regression suite 96.2% → 96.2%)
- request 2650: breaker opens for class `credit_cards` (3 of last 8 shadow checks disagreed)
- request 2650: recompile (breaker:credit_cards) on 987 production labels: rejected (recent hold-out 8/87 wrong, regression suite 92.9% → 92.7%)

Events, policy E* (183 regression cases re-adjudicated by the agent):

- request 400: recompile (periodic) on 105 production labels: rejected (recent hold-out 0/10 wrong, regression suite 97.3% → 97.3%)
- request 800: recompile (periodic) on 293 production labels: **deployed** (recent hold-out 0/36 wrong, regression suite 97.3% → 97.2%)
- request 1200: recompile (periodic) on 391 production labels: **deployed** (recent hold-out 1/59 wrong, regression suite 97.2% → 97.1%)
- request 1600: recompile (periodic) on 444 production labels: **deployed** (recent hold-out 1/51 wrong, regression suite 97.1% → 96.9%)
- request 2000: recompile (periodic) on 642 production labels: **deployed** (recent hold-out 3/58 wrong, regression suite 96.9% → 97.1%)
- request 2400: recompile (periodic) on 812 production labels: **deployed** (recent hold-out 1/72 wrong, regression suite 97.1% → 96.8%)
- request 2750: breaker opens for class `credit_cards` (4 of last 8 shadow checks disagreed)
- request 2750: recompile (breaker:credit_cards) on 903 production labels: **deployed** (recent hold-out 4/88 wrong, regression suite 93.6% → 93.3%)

Events, policy E'*:

- request 400: recompile (periodic) on 105 production labels: rejected (recent hold-out 0/10 wrong, regression suite 97.3% → 97.3%)
- request 800: recompile (periodic) on 293 production labels: **deployed** (recent hold-out 0/36 wrong, regression suite 97.3% → 97.2%)
- request 1200: recompile (periodic) on 391 production labels: **deployed** (recent hold-out 1/59 wrong, regression suite 97.2% → 97.1%)
- request 1600: recompile (periodic) on 444 production labels: **deployed** (recent hold-out 1/51 wrong, regression suite 97.1% → 96.9%)
- request 2000: recompile (periodic) on 642 production labels: **deployed** (recent hold-out 3/58 wrong, regression suite 96.9% → 97.1%)
- request 2400: recompile (periodic) on 812 production labels: **deployed** (recent hold-out 1/72 wrong, regression suite 97.1% → 96.8%)
- request 2750: breaker opens for class `credit_cards` (4 of last 8 shadow checks disagreed)
- request 2750: recompile (breaker:credit_cards) on 903 production labels: **deployed** (recent hold-out 4/88 wrong, regression suite 96.8% → 95.9%)

Events, policy E+* (182 regression cases re-adjudicated by the agent):

- request 400: recompile (periodic) on 105 production labels: rejected (recent hold-out 0/10 wrong, regression suite 97.3% → 97.3%)
- request 800: recompile (periodic) on 293 production labels: **deployed** (recent hold-out 0/36 wrong, regression suite 97.3% → 97.2%)
- request 1200: recompile (periodic) on 391 production labels: **deployed** (recent hold-out 1/59 wrong, regression suite 97.2% → 97.1%)
- request 1600: recompile (periodic) on 444 production labels: **deployed** (recent hold-out 1/51 wrong, regression suite 97.1% → 96.9%)
- request 2000: recompile (periodic) on 642 production labels: **deployed** (recent hold-out 3/58 wrong, regression suite 96.9% → 97.1%)
- request 2150: contract change announced, class `credit_cards` invalidated
- request 2400: recompile (periodic) on 844 production labels: **deployed** (recent hold-out 5/77 wrong, regression suite 93.8% → 93.9%)

**What this shows**

- With the agent as fallback, the guards do their job on three of the four scripted drifts. The naive tier (B*) delivers 21% wrong answers; the guarded tier (C*) delivers 1.3% when new groups arrive, 2.7% in the out-of-scope burst and 1.0% on noisy input, by escalating instead of answering (57% of requests).
- A change of contract is the drift no input-side guard can see: the requests look familiar and the program is confident, so 65% of re-routed requests are answered under the old policy. Shadow sampling finds it, slowly: at a 10% rate the breaker opened after 400 requests, at 40% after 150. Announcing the change and invalidating the affected answer class at once (D+*) leaves 0% wrong. A contract version belongs in every compiled answer's guard.
- Regression-gated recompiles win back coverage without adding errors while the contract is stable (6 of 7 candidates deployed; over phases 1 to 4 the fast share is 41% against 36% for the static program, with 1.2% against 1.3% wrong answers). After the contract change the candidate looked 0.8 points worse on the stale regression suite and 0.3 points worse on the re-adjudicated one (183 cases sent to the agent): a suite that encodes the old contract penalises exactly the update that is needed, and it has to be re-adjudicated when the contract changes.
- Ungated self-training is the worst policy in the loop (22% wrong with the agent as fallback, 27% with the LLM): it has the highest fast share and learns its own mistakes.
- With the local LLM as fallback everything is noisier, because the LLM itself is wrong on 19% of this stream (33% in the out-of-scope burst) and follows the changed contract on only 45% of re-routed requests. The guarded stack (C) matches it before the policy change with 0.57 calls per request, and an LLM shadow checker is too noisy and too slow to trip the breaker in time (D equals C here). In the first phase its shadow estimate is 7% while the fast tier's true silent rate is 0%: those disagreements are the checker's errors.


## 8. Experiment 7: natural drift, nothing scripted

*Real runs on real dated data; the oracle is the editor's label (no LLM).* HuffPost headlines in date order (197,952 headlines, 77 months). Compiled on the first 12 months (20 of the 31 sections exist then); accuracy on its own temporal hold-out is 72.5%. Budget for the fast tier: 10% silent errors. Sections that were only renamed over the years (for example THE WORLDPOST, WORLDPOST and WORLD NEWS) are merged before anything is trained; sections that are new stay new.

| Policy | Fast coverage | Silent among fast answers | Wrong per request | Months over budget | Worst month | Breaker opened |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| static-R | 35.7% | 38.5% | 13.74% | 60 of 65 | 66.0% | – |
| static-T | 32.7% | 37.6% | 12.30% | 60 of 65 | 68.0% | – |
| T + breaker | 13.2% | 12.2% | 1.60% | 12 of 17 | 42.3% | 2014-05 |
| gated-quarterly | 8.7% | 8.4% | 0.73% | 3 of 15 | 11.4% | – |
| self-train | 45.1% | 18.5% | 8.32% | 57 of 65 | 43.3% | – |

| By year | 2013 | 2014 | 2015 | 2016 | 2017 | 2018 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| traffic from sections unseen at compile time | 0% | 27% | 41% | 51% | 55% | 53% |
| static-R: coverage | 52% | 38% | 32% | 29% | 28% | 26% |
| static-R: silent among fast | 11% | 31% | 53% | 59% | 58% | 64% |
| static-T: coverage | 48% | 35% | 29% | 26% | 26% | 24% |
| static-T: silent among fast | 11% | 30% | 52% | 59% | 58% | 65% |
| gated-quarterly: coverage | 25% | 10% | 0% | 4% | 6% | 0% |
| gated-quarterly: silent among fast | 9% | 7% | 0% | 9% | 10% | 0% |
| self-train: coverage | 52% | 45% | 37% | 42% | 47% | 49% |
| self-train: silent among fast | 10% | 26% | 24% | 18% | 18% | 17% |

Gated quarterly recompiles: 4 of 21 candidates could be certified on the newest labelled month; in the other quarters the fast tier stayed off.

**What this shows**

- A compiled program left alone rots, and its own confidence does not say so. The static program stays near its budget for the first year (11% silent in 2013) and is 65% wrong on what it answers by 2018, while still answering 24% of traffic. The main cause is taxonomy change: by 2018, 53% of headlines belong to sections that did not exist at compile time, so every fast answer on them is wrong by construction.
- Calibrating on the last two months instead of a random split barely helps here (38.5% vs 37.6% silent overall): a temporal hold-out protects against drift that had already started, not against sections that do not exist yet.
- Shadow sampling catches it. With 10% of fast answers re-checked, the breaker opened after 2014-05, and wrong answers per request over the whole period fell from 12.3% to 1.6%.
- Gated recompilation keeps the promise and gives up most of the speed: 8.4% silent among fast answers against a 10% budget, but only 9% coverage, because only 4 of 21 quarterly candidates could be certified on the newest labelled month. Part of that is the task (30 overlapping editorial sections), part is a real bias: the labels the system holds come from escalations and are harder than the traffic the fast tier would answer.
- Retraining on its own answers without a gate keeps coverage high (45%) and stays over budget in 57 of 65 months (18% silent overall).


## 9. Experiment 8: the compiled program run for real, and its admission test

*Live execution, one request at a time, of a `wfc.Program`: guards, cache, mined rules, sentence embedding, classifier, familiarity search and lexical verifier all computed per request; the LLM tier replays the stored real answer; the agent is the human label.* 300 seen-group, 200 unseen-group and 100 out-of-scope requests, then 152 seeded bad inputs that the program must not answer. Budget: 2% silent errors.

| Build | Admitted at 2% | at 5% | Answered locally | Escalated | Silent errors (rate; 95% upper bound) | Bad inputs answered | ms p50: rules / model tier | Answered by tier |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| guards + gate(G) + verifier | **no** | **no** | 46.8% | 11.3% | 92 (17.29%; ≤ 20.22%) | 17 of 152 | 0.11 / 6.61 | rules 184 (5 wrong), model 97 (2 wrong), llm 251 (85 wrong), agent 68 (0 wrong) |
| gate(G) + verifier, no structural guards | **no** | **no** | 46.8% | 9.2% | 101 (18.53%; ≤ 21.49%) | 21 of 152 | 0.11 / 9.63 | rules 184 (5 wrong), model 97 (2 wrong), llm 264 (94 wrong), agent 55 (0 wrong) |
| guards + gate(G), no verifier | **no** | **no** | 47.5% | 11.2% | 93 (17.45%; ≤ 20.38%) | 17 of 152 | 0.11 / 6.58 | rules 184 (5 wrong), model 101 (4 wrong), llm 248 (84 wrong), agent 67 (0 wrong) |
| guards + naive gate(R), no verifier | **no** | **no** | 84.7% | 5.5% | 104 (18.34%; ≤ 21.22%) | 71 of 152 | 0.12 / 9.02 | rules 184 (5 wrong), model 324 (74 wrong), llm 59 (25 wrong), agent 33 (0 wrong) |
| full build without the LLM tier (abstentions go to the agent) | **no** | yes | 46.8% | 53.2% | 7 (2.49%; ≤ 4.63%) | 0 of 152 | 0.11 / 6.53 | rules 184 (5 wrong), model 97 (2 wrong), agent 319 (0 wrong) |
| the same without the rule tier | yes | yes | 42.2% | 57.8% | 2 (0.79%; ≤ 2.47%) | 0 of 152 | – / 6.55 | model 253 (2 wrong), agent 347 (0 wrong) |

Seeded bad inputs that got an answer, by kind and by the tier that answered:

| Build | structural | gibberish | keyboard mash | injection text | far-domain text |
| --- | --- | --- | --- | --- | --- |
| guards + gate(G) + verifier | 0/20 | 0/60 | 0/40 | 10/12 (llm 10) | 7/20 (llm 7) |
| gate(G) + verifier, no structural guards | 3/20 (llm 3) | 0/60 | 0/40 | 11/12 (llm 11) | 7/20 (llm 7) |
| guards + gate(G), no verifier | 0/20 | 0/60 | 0/40 | 10/12 (llm 10) | 7/20 (llm 7) |
| guards + naive gate(R), no verifier | 1/20 (model 1) | 46/60 (model 46) | 0/40 | 10/12 (llm 2, model 8) | 14/20 (llm 3, model 11) |
| full build without the LLM tier (abstentions go to the agent) | 0/20 | 0/60 | 0/40 | 0/12 | 0/20 |
| the same without the rule tier | 0/20 | 0/60 | 0/40 | 0/12 | 0/20 |

Why builds were refused: *guards + gate(G) + verifier*: silent error rate 17.29% exceeds budget 2.00%, 17 bad input(s) were answered on the fast path; *gate(G) + verifier, no structural guards*: silent error rate 18.53% exceeds budget 2.00%, 21 bad input(s) were answered on the fast path; *guards + gate(G), no verifier*: silent error rate 17.45% exceeds budget 2.00%, 17 bad input(s) were answered on the fast path; *guards + naive gate(R), no verifier*: silent error rate 18.34% exceeds budget 2.00%, 71 bad input(s) were answered on the fast path; *full build without the LLM tier (abstentions go to the agent)*: silent error rate 2.49% exceeds budget 2.00%.

**What this shows**

- Run for real, a local answer takes 0.11 ms from the rule tier or 6.6 ms from the embedding tier (encode, classify, familiarity search, lexical check), and the full build answers 47% of this hard mix (a third unseen groups, a sixth out of scope) without any model call.
- The full build is **refused**: 92 silent errors in 532 program answers (17.3%) and 17 of 152 bad inputs answered. By tier: rules 5 wrong of 184, model 2 wrong of 97, llm 85 wrong of 251. Admission judges the whole program, so one weak tier fails it, and here that tier is the LLM.
- Take the LLM tier out and send abstentions to the agent: 7 silent errors in 281 answers (2.5%), refused at 2% and acceptable at 5%. Take the mined-rule tier out as well: 2 in 253 (0.8%), admitted at 2% on the point estimate, with 42% answered locally. Its 95% upper bound is 2.5%: 253 answers are not enough to *certify* 2%, which is the usual position at admission time. The rule tier added 5 points of coverage and 5 of the 7 local errors.
- The familiarity gate is what stops nonsense. With the naive gate the embedding tier answers 46 of 60 gibberish strings, 8 of 12 injection strings and 11 of 20 far-domain sentences; with the guarded gate it answers none of the 152.
- What still gets through the guarded builds gets through the LLM tier: 10 of 12 injection strings and 7 of 20 far-domain sentences were given an in-scope category by Qwen3-4B, and without structural guards 3 of 20 malformed inputs as well. A typed hole constrains the form of the answer, not the model's willingness to give one.


## 10. Experiment 9: a wrapper harness that watches the LLM work, then runs the task as a program

*Real runs.* This tests the wrapper-harness idea directly: leave the model alone, wrap it, observe it doing an open-ended task ("analyze this profile"), and turn what it did into a program whose remaining decisions are small. The uncompiled agent is three free-form calls to Qwen3-4B per profile (extract facts, assess, write a summary). The harness recorded 60 runs, searched candidate programs per output field against them, and admitted a program only if it agreed with the LLM on 40 separate recorded runs. Test: 80 profiles never seen by the compile step.

**What the compile step decided, field by field:**

| Output field | Candidates tried | Best candidate | Agreement on observed runs | On admission runs | Bar | Compiled to |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| years_experience | 29 | `y_explicit_max else y_since_first_job(now=2018)` | 57% | 55% | 80% | narrow LLM hole |
| education_level | 5 | `e_any_strict` | 45% | 32% | 85% | narrow LLM hole |
| current_title | 3 | `t_heading` | 92% | 88% | 80% | code |
| skills | 3 | `s_gazetteer` | 55% | 29% | 50% | narrow LLM hole |
| field | 2 | `f_heading_keyword` | 53% | 83% | 90% | narrow LLM hole |
| seniority | 1 | `decision tree` | 75% | 68% | 80% | narrow LLM hole |
| strengths | – | – | – | – | – | generation hole fed by compact facts (or a template) |
| gaps | – | – | – | – | – | generation hole fed by compact facts (or a template) |
| summary | – | – | – | – | – | generation hole fed by compact facts (or a template) |

One candidate was excluded before the search: `e_keyword_else_bachelor` says "bachelor" whenever the profile shows no degree. It agrees with the agent on 70% of observed runs, far more than the best grounded rule (45%), and it is wrong by the contract: the field is supposed to be read from the profile.

**The recordings are also an audit of the agent.** In 44 of 100 recorded runs the text the agent was given contains no sign of education at all (no degree word or abbreviation, no mention of school, college, university, graduation or a diploma). The agent still reported a degree in 33 of them; it answered "bachelor" in 68% of all runs.

**A rule written from the contract can be better than the agent it is compared with.** The field rule (a field name or synonym in the profile's heading) answers 66 of 100 recorded profiles and matches the human label on 83% of them; the agent matches the human label on 67% of the same profiles. Admission by agreement with the agent (79%) turned the rule down.

**Cost per profile** (single request, same machine, same model):

| Path | LLM calls | Prompt tokens | Generated tokens | Latency p50 (s) | Speed-up | Handed back to agent |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| uncompiled agent (3 free-form calls) | 3 | 1885 | 182 | 11.16 | 1.0× | – |
| compiled: structured analysis only | 1.00 | 528 | 78 | 3.82 | 2.9× | 0% |
| compiled: + prose from a small generation hole | 2.00 | 674 | 158 | 6.69 | 1.7× | 0% |
| compiled: + prose from a code template | 1.00 | 528 | 78 | 3.82 | 2.9× | 0% |

**Fidelity: does the compiled program say what the agent would have said?** The last column is a ceiling: how often the agent agrees with *itself* when the extraction instruction is merely reworded.

| Field | Produced by | Compiled agrees with agent | Agent agrees with itself (reworded) |
| --- | --- | ---: | ---: |
| years_experience(±2) | narrow hole | 71% | 90% |
| years_experience(exact) | narrow hole | 35% | 73% |
| education_level | narrow hole | 79% | 90% |
| current_title | code | 96% | 98% |
| skills(overlap) | narrow hole | 63% | 85% |
| field | narrow hole | 79% | – |
| seniority | narrow hole | 75% | – |

Against the human field label, the agent is right on 62% of test profiles and the compiled program on 66%. Prose cannot be compared exactly; as a rough indication, the embedding similarity between the compiled summary and the agent's summary is 0.84 with the generation hole and 0.85 with the template, against 0.64 between the agent's summaries of two different profiles. Guard test: 7 of 7 non-profile inputs were handed back instead of analysed.

**What this shows**

- It works, modestly: the decisions get smaller rather than disappear. Watching 60 runs turned three free-form calls (1885 prompt and 182 generated tokens, 11.2 s) into code plus one narrow typed call (528 / 78 tokens, 3.8 s). That is 2.9× faster for the structured analysis, 1.7× if the prose still comes from a small generation hole, and 2.9× with a code template for the prose. Generated tokens are the latency that is left.
- Little became pure code, and not because the agent is erratic. Reworded, it agrees with itself on years of experience (±2) for 90% of profiles and on education for 90%. The rules failed because what the agent does is not a simple function of the text: the best of 29 date rules had to assume a fixed "today" and still matched it on 55% of admission runs. Admitted as code: current_title. Left with the model as one narrow typed call: years_experience, education_level, skills, field, seniority.
- Fidelity to the agent is the wrong target when the agent is wrong. It reported a degree for 33 of 44 profiles that show no education at all, and a rule written from the contract beat it against human labels (83% vs 67%) while failing admission by agreement. Admission needs reviewed outputs, not raw traces, as its reference. The shared conversation's survey said the same of prior work (compile from the contract, treat the trace as evidence); that survey was not re-checked here, but this experiment reaches the conclusion on its own data.
- The narrow call is not a perfect stand-in for the three-call agent: it agrees with it on 71% of year counts (±2), 79% of education answers and 79% of fields, below the agent's own 90% and 90% self-agreement. It reads the first 1,800 characters instead of 4,000 and answers every field in one call; which of the two costs the fidelity was not separated. Against human labels the two are not measurably different on the one field that has them (66% compiled, 62% agent, 80 profiles).
- Not tested: an agent with tools; a compile step written by a coding agent instead of chosen from candidates I wrote after reading the observed traces (search, admission and scoring were mechanical); more than one workflow; and the obvious next step, a decision model for the closed-set fields, which needs far more than 60 observed runs to be certified.


## 11. What this changes in the design

These are the changes to the compile procedure that the results support. Each one traces to an experiment above.

1. **Calibrate against the shift, with two tests.** Selective risk on held-out rows of known groups; false-accept rate on whole held-out groups and on real out-of-scope inputs. A random split certifies the wrong thing (Exp. 3). Report both numbers.
2. **Keep familiarity separate from confidence.** Use an embedding-distance score (nearest-neighbour similarity or relative Mahalanobis) as its own gate; confidence is high on most unseen-group errors (Exp. 2, 3).
3. **Count groups, not rows, as evidence**, and treat any certificate as approximate. Keep shadow sampling on in production (Exp. 3, 6, 7).
4. **Pair the embedding model with a character n-gram model as verifier.** It costs a fraction of a millisecond, it disagrees on noise and unseen groups where the embedding model is confidently wrong, and it degrades honestly (Exp. 2, 4).
5. **Do not use a semantic cache or a single mined rule as a decision tier** without the same admission test; both were weaker than a linear head on the same data (Exp. 1).
6. **Let the compile step say no.** Some decisions cannot be certified at the budget (MASSIVE at 2%; 5 of 6 short profile fields in Exp. 9). They stay as narrow typed holes.
7. **Admit every tier, including the LLM.** The local LLM's errors dominated every stack it was in, its confidence could not be certified, and it answered more than half of the out-of-scope queries as in-scope (Exp. 5, 8).
8. **Version the contract and invalidate on change.** A contract change is invisible to input-side guards; an announced invalidation of the affected answer class removed the errors, and the regression suite has to be re-adjudicated at the same time (Exp. 6).
9. **Never retrain on unverified own answers.** Gate every update on held-out recent labels and a regression suite; remember that the labels a running system collects come from escalations and are harder than its fast-path traffic (Exp. 6, 7).
10. **Engineer the hole.** Fixed contract prefix with a reused KV cache; closed answer sets; the smallest input that suffices; batch sizes chosen with the prefix length in mind (Exp. 5, 9).

## 12. Limits of this study

- The LLMs are 4-bit local models of about 4B parameters. Every statement about "the LLM tier" is about them. The comparison between always-LLM and compiled stacks would shift with a stronger model; the fast-tier results would not.
- The agent is the human label. Human labels in these datasets are themselves imperfect (MASSIVE and the news sections visibly so), and the CLINC domain taxonomy is partly arbitrary for unseen intents.
- One frozen group split carries all LLM-in-the-loop results (Exp. 4, 5, 6, 8). The fast-tier conclusions were checked on 20 splits; the LLM ones were not.
- Intent routing and profile analysis are short-text, closed-answer tasks. Nothing here tests perceptual workflows such as video to Blender, where the verifier is the hard part.
- The production loop is a script of five phases over 2,790 requests. Detection delays are specific to that length and those rates.
- In Experiment 9 the candidate programs were hand-written by the compiling agent (me) from observed traces; search, admission and scoring were mechanical. One model, one workflow, 80 test profiles.
- Fast-tier latency was measured on CPU while a GPU job was running; LLM latency is for MLX on an M4 and includes no network.

## 13. The earlier artifact

The first version of `workflow-compiler.skill` came out of an earlier conversation and is not included in this repository (9 files: SKILL.md, a standard-library runtime, a scikit-learn decision tier, an admission harness, the expense example and its tests). I read every file. Executing the downloaded code was blocked by this session's permission layer, so **its 14 tests were not run here and that earlier claim remains unverified**. Instead the runtime and admission harness were re-authored into `wfc/` from the reviewed source and extended (independent-verifier hook, exact bounds, three threshold procedures, a false-accept-rate test, a group-aware bootstrap); `tests/` holds 21 tests, the original 11 kernel tests re-written plus 10 new ones, and all 21 pass.

The research and repository claims quoted in the shared conversation (FlashScript, TraceCompiler, PreAct, Evaporate, the video-to-blender and shader-skills repositories, the generativemedia.blog posts) were **not re-verified** in this session, and no result in this report depends on them.

## 14. Files

`REPORT.md` (this file) · `README.md` (layout and commands to reproduce) · `results/*.json` (every number) · `bench/` (experiments) · `skill/workflow-compiler/` and `workflow-compiler-v2.skill` (the skill revised with these results; 23 tests) · `wfc/` and `tests/` (kernel) · `cache/llm/*.jsonl` (stored LLM answers with raw text and token probabilities) · `logs/pip_freeze.txt` (exact package versions).
