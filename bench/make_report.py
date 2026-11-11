"""Build REPORT.md from results/*.json. Every number in the report comes from those files.

Usage: python -m bench.make_report
"""
from __future__ import annotations

import json

from .common import RESULTS, ROOT


def J(name: str):
    p = RESULTS / name
    return json.loads(p.read_text()) if p.exists() else None


def pct(x, d: int = 1) -> str:
    return "–" if x is None else f"{100 * x:.{d}f}%"


def num(x, d: int = 2) -> str:
    return "–" if x is None else f"{x:.{d}f}"


def table(header: list, rows: list, align: str | None = None) -> str:
    align = align or ("l" + "r" * (len(header) - 1))
    sep = ["---" if a == "l" else "---:" for a in align]
    out = ["| " + " | ".join(header) + " |", "| " + " | ".join(sep) + " |"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


# --------------------------------------------------------------------------- sections
def sec_setup() -> str:
    m = J("data_manifest.json")
    lat = J("exp_latency.json")
    L = J("llm_latency.json") or {}
    rows = [["CLINC150", "23,700 assistant queries, 150 intents in 10 domains; 1,200 human-written out-of-scope queries",
             f"{m['clinc150']['splits']['train']['n']:,} / {m['clinc150']['splits']['cal']['n']:,} / {m['clinc150']['splits']['test']['n']:,}"],
            ["BANKING77", "13,083 bank-support messages, 77 fine-grained intents",
             f"{m['banking77']['splits']['train']['n']:,} / {m['banking77']['splits']['cal']['n']:,} / {m['banking77']['splits']['test']['n']:,}"],
            ["MASSIVE (en-US)", "16,521 assistant utterances, 60 intents in 18 scenarios; same utterances human-translated into 5 other locales",
             f"{m['massive60']['splits']['train']['n']:,} / {m['massive60']['splits']['cal']['n']:,} / {m['massive60']['splits']['test']['n']:,}"],
            ["HuffPost news", f"{m['news']['n']:,} dated headlines with editor-assigned section, {m['news']['from']} to {m['news']['to']}", "by date"],
            ["Resume profiles", "2,484 public resume profiles in 24 human-labelled fields (opensporks/resumes)", "60 observed / 40 admission / 80 test"]]
    s = "## 1. Test bed\n\n"
    s += ("**Machine.** Apple M4 (10 cores, 24 GB), macOS. All training, inference and benchmarks ran locally. "
          "Fast models ran on CPU; the LLMs ran on the GPU through MLX.\n\n")
    s += "**Data.** Public datasets with human labels; nothing synthetic except the deliberately bad inputs.\n\n"
    s += table(["Dataset", "What it is", "train / calibration / test"], rows, "lll") + "\n\n"
    cr = m["clinc_routing"]
    s += ("**Two kinds of decision.** *Fine*: predict the intent (150 / 77 / 60 classes). *Routing*: predict the coarse class "
          "(CLINC: 10 domains; MASSIVE: 18 scenarios) while treating each intent as a **group**. Whole groups are held out, "
          "which reproduces the earlier example's unseen-merchant failure on human-written text: the label space is known, "
          f"the sub-population is new. On CLINC that is {cr['groups']['train']} training groups, {cr['groups']['cal']} groups held out "
          f"for calibration and {cr['groups']['test']} groups never seen by any model or threshold.\n\n")
    s += ("**Splits.** Models are fitted on train, thresholds are chosen on calibration, and every reported number is from test rows "
          "(or never-seen groups). Split membership is fingerprinted in `results/data_manifest.json`; one frozen group split "
          f"(seed {cr['frozen_seed']}) is used wherever a real LLM is in the loop, and 20 random group splits wherever it is not.\n\n")
    s += ("**Models actually run.**\n\n"
          "- Fast tier: TF-IDF (character and word n-grams), static embeddings (potion-base-8M), all-MiniLM-L6-v2, bge-small-en-v1.5, "
          "bge-base-en-v1.5, multilingual-e5-small; heads: logistic regression, k-NN, 1-NN (a semantic cache), class-conditional "
          "Gaussian (Mahalanobis), a 5-member MLP ensemble, mined n-gram rule committees, and MiniLM fine-tuned end to end.\n"
          "- LLM tier: Qwen3-4B-Instruct-2507 (4-bit, MLX) as the main model, Gemma-3-4B-it (4-bit QAT) as a second, differently built "
          "model, Qwen3-8B (4-bit) in a pilot.\n"
          "- Agent tier: **not a model**. Wherever the report says *agent* it means the human label, assumed correct, "
          "counted as an escalation and never timed.\n\n")
    s += ("**Terms.** *Coverage*: share of inputs a tier answers. *Silent error*: a wrong answer that was delivered without escalation; "
          "the silent-error rate is wrong answers over answered inputs. *False-accept rate*: wrong answers over **all** inputs of a "
          "population (used for populations a tier should mostly refuse). *Certified*: the threshold was chosen on calibration data by "
          "fixed-sequence testing so that the silent-error rate exceeds the budget with probability at most 5%, **if** deployment inputs "
          "are exchangeable with calibration inputs.\n\n")
    if lat:
        t = lat["tier_ms_p50"]
        rows = [["exact-match cache", num(t["exact cache"], 3)], ["mined rule committee", num(t["mined rules"], 2)],
                ["TF-IDF + logistic regression (end to end)", num(t["tfidf-char+LR end-to-end"], 2)],
                ["static embedding (potion) encode", num(t["potion encode"], 2)],
                ["MiniLM encode", num(t["minilm encode"], 1)], ["bge-small encode", num(t["bge-small encode"], 1)],
                ["bge-base encode", num(t["bge-base encode"], 1)],
                ["logistic-regression head", num(t["LR head"], 2)], ["nearest-neighbour search over 9,000 rows", num(t["knn1 over train"], 2)],
                ["MLP ensemble (5 members)", num(t["MLP ensemble x5"], 2)]]
        ft = J("exp1b_finetune.json")
        if ft:
            rows.append(["MiniLM fine-tuned end to end", num(ft["clinc150"]["latency_ms_p50"], 1)])
        for k, label in (("qwen3-4b/verify_yes_no", "LLM yes/no verification call (Qwen3-4B, cached prefix)"),
                         ("qwen3-4b/route_0shot", "LLM routing call, short contract (Qwen3-4B, cached prefix)"),
                         ("qwen3-4b/route_1shot", "LLM routing call, 1-shot contract (Qwen3-4B, cached prefix)"),
                         ("qwen3-4b/route_1shot_no_prefix_cache", "same call without the prefix cache"),
                         ("gemma3-4b/route_1shot", "LLM routing call, 1-shot contract (Gemma-3-4B)")):
            if k in L:
                rows.append([label, num(L[k]["p50"], 0)])
        s += "**Measured latency per request** (median, one request at a time on this machine):\n\n"
        s += table(["Tier", "ms"], rows) + "\n\n"
    return s


def sec_e1() -> str:
    r = J("exp1_families.json")
    ft = J("exp1b_finetune.json") or {}
    s = "## 2. Experiment 1: which fast model should take over a typed decision?\n\n"
    s += ("*Real runs. Official train / calibration / test partitions, no group shift.* Each family answers only above a confidence "
          "threshold certified on the calibration split. \"Certified coverage\" is the share of test inputs it then answers; "
          "\"silent\" is the error rate among those answers on test.\n\n")
    for ds, title in (("clinc150", "CLINC150, 150 intents"), ("banking77", "BANKING77, 77 intents"), ("massive60", "MASSIVE, 60 intents")):
        fam = r[ds]["families"]
        rows = []
        for name, v in fam.items():
            if "aurc" not in v:
                continue
            a2, a5 = v["fs@0.02"], v["fs@0.05"]
            rows.append([name, pct(v.get("accuracy")), pct(a2["coverage"]), pct(a2["silent_error_rate"], 2) if a2["answered"] else "–",
                         pct(a5["coverage"]), pct(a5["silent_error_rate"], 2) if a5["answered"] else "–",
                         num(v["fit_s"], 1) if "fit_s" in v else "–"])
        if ds in ft:
            v = ft[ds]
            a2, a5 = v["fs@0.02"], v["fs@0.05"]
            rows.append(["MiniLM fine-tuned end to end", pct(v["accuracy"]), pct(a2["coverage"]),
                         pct(a2["silent_error_rate"], 2) if a2["answered"] else "–", pct(a5["coverage"]),
                         pct(a5["silent_error_rate"], 2) if a5["answered"] else "–", num(v["fit_s"], 0)])
        s += f"**{title}** (test n = {r[ds]['n']['test']:,})\n\n"
        s += table(["Family", "Accuracy", "Coverage @2%", "Silent @2%", "Coverage @5%", "Silent @5%", "Fit (s)"], rows) + "\n\n"
        rr = []
        for name, v in fam.items():
            if name.startswith("mined-rules"):
                rr.append([name, v["n_rules"], pct(v["coverage"]), pct(v["silent_error_rate"], 2), pct(v["coverage_2votes"]),
                           pct(v["silent_error_rate_2votes"], 2)])
        s += table(["Rule committee (word n-gram → label)", "Rules", "Coverage", "Silent", "Coverage, ≥2 rules agree", "Silent"], rr) + "\n\n"
        c = fam["exact-cache"]
        s += f"Exact-match cache on test: {pct(c['coverage'], 2)} of queries repeat a training query verbatim ({c['answered']} hits, {c['silent_errors']} of them with a different label).\n\n"
    return s


def sec_e2() -> str:
    r = J("exp2_novelty.json")
    s = "## 3. Experiment 2: can the fast tier tell that an input is unfamiliar?\n\n"
    s += (f"*Real runs.* CLINC150 models; {r['n']['in_scope_test']:,} in-scope test queries against {r['n']['oos_test']:,} human-written "
          f"out-of-scope queries, {r['n']['news']:,} news headlines and {r['n']['seeded_bad']} seeded bad inputs. AUROC: 1.0 means the "
          "score separates perfectly, 0.5 means it carries no information. \"OOS answered\" is the share of out-of-scope queries the fast "
          "tier would answer at the threshold certified for a 2% budget, when calibration contains only in-scope rows (R) or also real "
          "out-of-scope rows (G).\n\n")
    D = r["detectors"]
    pick = ["bge-small/msp", "bge-small/margin", "bge-small/energy", "bge-small/knn1", "bge-small/knn5", "bge-small/centroid",
            "bge-small/maha", "bge-small/rel_maha", "bge-small/ens_msp", "bge-small/ens_agree", "bge-small/agree_tfidf",
            "bge-small/min(msp,knn1)", "bge-small/learned_trust", "tfidf/msp", "tfidf/knn1", "minilm/msp", "minilm/rel_maha",
            "bge-base/msp", "bge-base/energy", "bge-base/rel_maha"]
    label = {"msp": "confidence (max prob.)", "margin": "top-2 margin", "energy": "energy", "knn1": "nearest-neighbour similarity",
             "knn5": "mean similarity to 5 nearest", "centroid": "nearest class centroid", "maha": "Mahalanobis",
             "rel_maha": "relative Mahalanobis", "ens_msp": "MLP-ensemble confidence", "ens_agree": "MLP-ensemble unanimity",
             "agree_tfidf": "agreement with TF-IDF model", "min(msp,knn1)": "min(confidence, similarity)",
             "learned_trust": "learned trust model (uses real OOS examples)"}
    rows = []
    for k in pick:
        if k not in D:
            continue
        v = D[k]
        view, det = k.split("/", 1)
        rows.append([view, label.get(det, det), num(v["auroc_oos"], 3), pct(v["fpr95_oos"]), num(v["auroc_news"], 3),
                     num(v["auroc_error"], 3), pct(v["op_R"]["in_scope_coverage"]), pct(v["op_R"]["oos_false_accept"]),
                     pct(v["op_G"]["in_scope_coverage"]), pct(v["op_G"]["oos_false_accept"])])
    s += table(["View", "Score", "AUROC vs OOS", "FPR@95", "AUROC vs news", "AUROC vs own errors", "R: in-scope coverage",
                "R: OOS answered", "G: in-scope coverage", "G: OOS answered"], rows, "llrrrrrrrr") + "\n\n"
    # seeded bad inputs
    rows = []
    for k in ("bge-small/msp", "bge-small/knn1", "bge-small/rel_maha", "bge-small/min(msp,knn1)", "bge-small/learned_trust", "tfidf/msp", "tfidf/knn1"):
        b = D[k]["op_G"]["bad_false_accept"]
        rows.append([k] + [pct(b[c], 0) for c in ("structural", "gibberish", "keyboard_mash", "injection", "far_domain")])
    s += "Share of seeded bad inputs the score would let through (G thresholds, no structural guards in front):\n\n"
    s += table(["Score", "structural", "gibberish", "keyboard mash", "injection text", "far-domain text"], rows) + "\n\n"
    # typos
    T = r["typo_noise"]
    rows = []
    for view, v in T.items():
        for lv in ("typo00", "typo05", "typo15"):
            x = v[lv]
            rows.append([view, {"typo00": "clean", "typo05": "5% of letters", "typo15": "15% of letters"}[lv], pct(x["accuracy"]),
                         pct(x["msp"]["coverage"]), pct(x["msp"]["silent_error_rate"]),
                         pct(x["min(msp,knn1)"]["coverage"]), pct(x["min(msp,knn1)"]["silent_error_rate"])])
    s += "**Keyboard noise on the same test queries** (thresholds unchanged, certified on clean calibration data for 2%):\n\n"
    s += table(["View (+LR)", "Noise", "Accuracy", "Coverage (confidence)", "Silent", "Coverage (min of confidence, similarity)", "Silent"], rows, "llrrrrr") + "\n\n"
    # locale
    Lc = r["locale_shift"]
    rows = []
    for view, v in Lc.items():
        for loc, x in v.items():
            rows.append([view, loc, pct(x["accuracy"]), pct(x["msp"]["coverage"]), pct(x["msp"]["silent_error_rate"]),
                         pct(x["min(msp,knn1)"]["coverage"]), pct(x["min(msp,knn1)"]["silent_error_rate"]),
                         num(x.get("auroc_locale/knn1"), 3) if loc != "en-US" else "–"])
    s += (f"**Locale shift** (MASSIVE, 60 intents; trained on en-US; the same test utterances in other languages; thresholds certified "
          f"for {pct(r.get('alpha_locale', 0.05), 0)} on en-US calibration data):\n\n")
    s += table(["View (+LR)", "Locale", "Accuracy", "Coverage (confidence)", "Silent", "Coverage (min of confidence, similarity)", "Silent",
                "AUROC en vs locale (similarity)"], rows, "llrrrrrr") + "\n\n"
    return s


def sec_e3() -> str:
    r = J("exp3_group_shift.json")
    s = "## 4. Experiment 3: does a calibrated fast tier stay calibrated on groups it has never seen?\n\n"
    s += ("*Real runs, 20 random group splits per dataset.* The routing task, with a fifth of each class's intents held out entirely. "
          "Protocols: **R** calibrates on held-out rows of training groups (a random split); **G-mix** adds an equal number of rows "
          "from held-out calibration groups; **G-worst** runs two tests, selective risk on seen-group rows and false-accept rate on "
          "held-out-group rows. \"Over budget\" is the share of the 20 splits in which the realised rate exceeded the budget.\n\n")
    for ds, title in (("clinc", "CLINC routing (10 domains; 90 / 30 / 30 train / calibration / test groups)"),
                      ("massive", "MASSIVE routing (18 scenarios; 34 / 13 / 13 groups)")):
        a = r[ds]
        rows = []
        for m, v in a["models"].items():
            b = v["base"]
            rows.append([m, f"{pct(b['acc_seen']['mean'])} ± {pct(b['acc_seen']['sd'])}", f"{pct(b['acc_unseen']['mean'])} ± {pct(b['acc_unseen']['sd'])}",
                         pct(b["unseen_conf>=0.9_share"]["mean"], 0), pct(b["unseen_err_among_conf>=0.9"]["mean"], 0),
                         pct(b["unseen_errors_conf>=0.9"]["mean"], 0)])
        s += f"**{title}**\n\n"
        s += table(["Model", "Accuracy, seen groups", "Accuracy, unseen groups", "Unseen inputs with confidence ≥ 0.9",
                    "…of which wrong", "Unseen-group errors made at confidence ≥ 0.9"], rows) + "\n\n"
        if ds == "massive":
            c2 = a["models"]["bge-small+LR"]["cells"]
            s += (f"At a 2% budget this task is barely compilable with these models: with random-split calibration the confidence gate "
                  f"still answers {pct(c2['msp|R|fixed_sequence|0.02']['test_unseen']['coverage_mean'], 0)} of unseen-group inputs and gets "
                  f"{pct(c2['msp|R|fixed_sequence|0.02']['test_unseen']['silent_mean'], 0)} of those wrong, while the guarded gate (G-worst, "
                  f"confidence + familiarity) keeps {pct(c2['msp&knn1|G-worst|fixed_sequence|0.02']['test_seen']['coverage_mean'], 0)} "
                  f"seen-group coverage with {pct(c2['msp&knn1|G-worst|fixed_sequence|0.02']['test_unseen']['far_mean'], 1)} wrong answers "
                  f"per unseen input. The table shows the 5% budget.\n\n")
        for alpha in ((0.05,) if ds == "massive" else (0.02, 0.05)):
            rows = []
            for score, sl in (("msp", "confidence"), ("knn1", "familiarity only"), ("msp&knn1", "confidence + familiarity"),
                              ("learned", "learned trust model")):
                for proto in ("R", "G-mix", "G-worst"):
                    c = a["models"]["bge-small+LR"]["cells"][f"{score}|{proto}|fixed_sequence|{alpha}"]
                    u, sn = c["test_unseen"], c["test_seen"]
                    rows.append([sl, proto, pct(sn["coverage_mean"]), pct(sn["silent_mean"], 2), pct(sn["violation_rate"], 0),
                                 pct(u["coverage_mean"]), pct(u["far_mean"], 2), pct(u["far_violation_rate"], 0),
                                 pct(u["silent_mean"]), pct(c["never_rate"], 0)])
            s += f"bge-small + LR, budget {pct(alpha, 0)}, thresholds by fixed-sequence testing:\n\n"
            s += table(["Trust score", "Calibration", "Seen: coverage", "Seen: silent", "Seen: over budget", "Unseen: coverage",
                        "Unseen: wrong answers / all unseen inputs", "Unseen: over budget", "Unseen: silent among answered",
                        "Splits with no fast tier at all"], rows, "llrrrrrrrr") + "\n\n"
        rows = []
        for picker in ("empirical", "wilson_scan", "fixed_sequence", "cluster_bootstrap"):
            for proto in ("R", "G-worst"):
                c = a["models"]["bge-small+LR"]["cells"][f"msp&knn1|{proto}|{picker}|0.05"]
                u, sn = c["test_unseen"], c["test_seen"]
                rows.append([picker, proto, pct(sn["coverage_mean"]), pct(sn["silent_mean"], 2), pct(sn["violation_rate"], 0),
                             pct(u["coverage_mean"]), pct(u["far_mean"], 2), pct(u["far_violation_rate"], 0), pct(u["far_max"], 1)])
        s += "Threshold procedures compared (bge-small + LR, confidence + familiarity, budget 5%):\n\n"
        s += table(["Procedure", "Calibration", "Seen: coverage", "Seen: silent", "Seen: over budget", "Unseen: coverage",
                    "Unseen: wrong / all", "Unseen: over budget", "Unseen: worst split"], rows, "llrrrrrrr") + "\n\n"
        rows = []
        for m in a["models"]:
            c = a["models"][m]["cells"]["msp&knn1|G-worst|fixed_sequence|0.05"]
            cr = a["models"][m]["cells"]["msp|R|fixed_sequence|0.05"]
            rows.append([m, pct(cr["test_unseen"]["far_mean"], 1), pct(c["test_seen"]["coverage_mean"]), pct(c["test_seen"]["silent_mean"], 2),
                         pct(c["test_unseen"]["coverage_mean"]), pct(c["test_unseen"]["far_mean"], 2), pct(c["test_unseen"]["far_violation_rate"], 0)])
        s += "Same comparison across model families (budget 5%):\n\n"
        s += table(["Model", "R + confidence: wrong / all unseen", "G-worst gate: seen coverage", "Seen: silent", "Unseen: coverage",
                    "Unseen: wrong / all", "Unseen: over budget"], rows) + "\n\n"
    b3 = J("exp3b_cal_groups.json")
    if b3:
        rows = []
        for k, v in b3["settings"].items():
            f, c = v["fixed_sequence"], v["cluster_bootstrap"]
            rows.append([k, v["groups"]["train"], pct(f["seen_coverage"]), pct(f["unseen_far"], 2), pct(f["unseen_over_budget"], 0), pct(f["unseen_far_max"], 1),
                         pct(c["seen_coverage"]), pct(c["unseen_far"], 2), pct(c["unseen_over_budget"], 0), pct(c["unseen_far_max"], 1)])
        s += (f"**How many held-out groups does the calibration need?** (CLINC, {b3['splits']} splits per row, 30 never-seen test groups in every "
              "row, bge-small + LR, confidence + familiarity, G-worst, budget 5%; left block: row-level fixed-sequence test, right block: "
              "group-aware bootstrap)\n\n")
        s += table(["Calibration groups", "Training groups", "Seen coverage", "Unseen: wrong / all", "Over budget", "Worst split",
                    "Seen coverage (group-aware)", "Unseen: wrong / all", "Over budget", "Worst split"], rows) + "\n\n"
    return s


def sec_e4() -> str:
    r = J("exp4_verification.json")
    if not r:
        return ""
    s = "## 5. Experiment 4: what can verify a fast answer, and what does it cost?\n\n"
    n = r["n"]
    s += (f"*Real runs; LLM verifiers are stored real answers.* Frozen CLINC routing split. The fast tier (bge-small + LR) answers "
          f"everything, so the verifier is the only guard. Populations: {n['test_seen']:,} seen-group test rows, {n['test_unseen']:,} "
          f"rows of never-seen groups, {n['oos_test']:,} out-of-scope queries. *Catch* = share of the fast tier's errors rejected; "
          "*false reject* = share of its correct answers rejected; *wrong / all* = wrong answers that still get through, over all inputs.\n\n")
    lat = r.get("latency_ms", {})
    fast, llm = lat.get("fast", {}), lat.get("llm", {})
    cost = {"knn": fast.get("knn1 over train"), "maha": fast.get("Mahalanobis"), "tfidf": fast.get("tfidf-char+LR end-to-end"),
            "mlp": fast.get("MLP ensemble x5"), "minilm": (fast.get("minilm encode") or 0) + (fast.get("LR head") or 0) if fast else None,
            "bge-base": (fast.get("bge-base encode") or 0) + (fast.get("LR head") or 0) if fast else None,
            "rules": fast.get("mined rules"),
            "tfidf+minilm": (fast.get("tfidf-char+LR end-to-end") or 0) + (fast.get("minilm encode") or 0) if fast else None,
            "llm_classify": llm.get("llm_classify"), "llm_verify": llm.get("llm_verify"), "gemma": llm.get("gemma"),
            "llm_classify+gemma": (llm.get("llm_classify") or 0) + (llm.get("gemma") or 0) if llm else None}
    rows = []
    for name, v in r["verifiers"].items():
        c = cost.get(v.get("cost_key"))
        cells = [name, "0" if v.get("cost_key") is None else (num(c, 2) if c is not None and c < 10 else num(c, 0))]
        for p in ("test_seen", "test_unseen", "oos_test"):
            if p in v:
                x = v[p]
                if p == "oos_test":
                    cells += [pct(x["catch_rate"], 0), pct(x["false_accept_rate"], 1)]
                else:
                    cells += [pct(x["catch_rate"], 0), pct(x["false_reject"], 1), pct(x["false_accept_rate"], 1)]
            else:
                cells += ["–"] * (2 if p == "oos_test" else 3)
        rows.append(cells)
    s += table(["Verifier", "Added ms", "Seen: catch", "Seen: false reject", "Seen: wrong / all", "Unseen: catch",
                "Unseen: false reject", "Unseen: wrong / all", "OOS: catch", "OOS: wrong / all"], rows) + "\n\n"
    s += ("**Do verifiers repeat the fast model's mistakes?** For each second opinion B: its own error rate, how often it is wrong "
          "when the fast model is wrong, and how often it gives the *same* wrong label (the case no agreement check can catch).\n\n")
    for pop, title in (("test_seen", "seen groups"), ("test_unseen", "never-seen groups")):
        tab = r["error_correlation"].get(pop, {})
        rows = []
        for k, v in tab.items():
            if k.startswith(r["primary"] + " ->"):
                rows.append([k.split("-> ")[1], pct(v["B_err"]), pct(v["B_wrong_given_A_wrong"], 0),
                             pct(v["same_wrong_label_given_A_wrong"], 0), v["n"], v["A_errors"]])
        if rows:
            s += f"On {title} (fast model error rate {pct(next(iter(tab.values()))['A_err'])}):\n\n"
            s += table(["Second opinion", "Its error rate", "Wrong when fast model is wrong", "Same wrong label", "Rows compared",
                        "Fast-model errors among them"], rows) + "\n\n"
    return s


def sec_e5() -> str:
    r = J("exp5_tiers.json")
    if not r:
        return ""
    s = "## 6. Experiment 5: the real LLM tier, and whole tier stacks\n\n"
    s += "*LLM numbers are real outputs of local models on this machine; the agent is the human label (assumed).*\n\n"
    names = {"test_seen": "seen groups", "test_unseen": "never-seen groups", "oos_test": "out of scope", "typo15": "seen groups, 15% keyboard noise"}
    rows = []
    for model, tab in r["llm_tier"].items():
        if model == "two_llm_agreement":
            continue
        for p, title in names.items():
            if p in tab:
                x = tab[p]
                rows.append([model, title, x["n"], pct(x["accuracy"]), pct(x["none_exit_rate"]), pct(x["unparseable_rate"], 2),
                             pct(x["wrong_among_answered"]), pct(x["p_eq_1_share"], 0), pct(x["acc_when_p_eq_1"]), pct(x["acc_when_p_lt_1"])])
    s += ("**The LLM tier alone.** For out-of-scope queries the correct behaviour is the \"none of these\" exit, so accuracy there is "
          "the exit rate.\n\n")
    s += table(["Model", "Population", "n", "Correct", "Took the none exit", "Unparseable", "Wrong among answered",
                "Share answered with p = 1", "Correct when p = 1", "Correct when p < 1"], rows, "llrrrrrrrr") + "\n\n"
    ag = r["llm_tier"].get("two_llm_agreement", {})
    if ag:
        rows = [[names[p], v["n"], pct(v["agree_and_answer_rate"]), pct(v["wrong_when_agree"]), pct(v["qwen_wrong_answered"])] for p, v in ag.items()]
        s += "Two differently built LLMs as each other's check (answer only when both give the same in-scope label):\n\n"
        s += table(["Population", "n", "Both agree and answer", "Wrong when they agree", "Qwen alone: wrong answers / all"], rows) + "\n\n"
    st = r["stacks"]
    g = st["gate_alpha2"]
    s += ("**Tier stacks.** `fast[R]`: confidence threshold from random-split calibration. `fast[G]`: confidence + familiarity gate "
          "calibrated against held-out groups and real out-of-scope queries (2% budget; `G5` = 5%). `+V`: an independent lexical "
          "model must agree. `llm[conf]`: the LLM answers only when its own token probability clears a calibrated threshold. "
          f"Latency uses the measured medians ({st['latency']['source']}); escalations to the agent are counted, not timed.\n\n")
    label = {"llm": "always-LLM", "fast[all]": "fast model, no abstention", "agent": "always-agent (reference)"}
    for mix in ("70/20/10 seen/unseen/oos", "90/8/2", "50/30/20"):
        rows = []
        for pol, v in st["policies"].items():
            m = v["mix"][mix]
            sh = m["share"]
            rows.append([label.get(pol, pol.replace(" > ", " → ")), pct(m["wrong_delivered"], 2), pct(sh["cache"] + sh["rules"] + sh["model"]),
                         pct(sh["llm"]), pct(sh["agent"]), num(m["llm_calls"], 2), num(m["compute_ms_mean"], 0)])
        s += f"Traffic mix {mix} (seen / never-seen groups / out of scope; an assumption, three variants shown):\n\n"
        s += table(["Stack", "Wrong answers delivered", "Answered locally", "Answered by LLM", "Escalated to agent", "LLM calls / request",
                    "Compute ms / request"], rows) + "\n\n"
    rows = []
    for pol, v in st["policies"].items():
        pp = v["per_population"]
        rows.append([label.get(pol, pol.replace(" > ", " → "))] + [pct(pp[p]["wrong_delivered"], 1) for p in names] +
                    [pct(1 - pp[p]["share"]["agent"] - pp[p]["share"]["llm"], 0) for p in names])
    s += "Per population: wrong answers delivered, and share answered locally (no LLM, no agent):\n\n"
    s += table(["Stack"] + [f"wrong: {t}" for t in names.values()] + [f"local: {t}" for t in names.values()], rows) + "\n\n"
    tr = r.get("compile_from_traces", {})
    if "variants" in tr:
        rows = []
        for name, v in tr["variants"].items():
            rows.append([name, pct(v["bge-small+LR/test_seen"]["accuracy_vs_human"]), pct(v["bge-small+LR/test_unseen"]["accuracy_vs_human"]),
                         pct(v["bge-small+LR/test_seen"]["agreement_with_llm"]), pct(v["tfidf-char+LR/test_seen"]["accuracy_vs_human"])])
        any_v = next(iter(tr["variants"].values()))
        s += (f"**Compile from traces.** The fast model trained on the LLM's own answers for {tr['n_traces']:,} past requests "
              f"(the LLM was right on {pct(tr['llm_label_accuracy_on_traces'])} of them; {tr['dropped_none_or_invalid']} answers that took "
              f"the none exit were dropped) versus the same rows with human labels. For reference the LLM itself scores "
              f"{pct(any_v['llm/test_seen'])} on seen groups and {pct(any_v['llm/test_unseen'])} on never-seen groups.\n\n")
        s += table(["Training labels", "bge-small+LR: seen", "bge-small+LR: unseen", "Agreement with LLM (seen)", "tfidf+LR: seen"], rows) + "\n\n"
    o = J("exp5b_llm_other.json")
    if o and o.get("fine"):
        rows = [[k, v["n"], pct(v["llm_accuracy"]), pct(v["llm_none_or_unparseable"]), pct(v["fast_accuracy_same_rows"]),
                 pct(v.get("llm_oos_none_exit"), 0) if "llm_oos_none_exit" in v else "–"] for k, v in o["fine"].items()]
        s += ("**The same LLM on fine-grained holes** (zero-shot, the closed set is the list of intent names; the fast model is "
              "bge-small + LR trained on the task's training split, scored on the same rows):\n\n")
        s += table(["Task", "n", "LLM correct", "LLM none / unparseable", "Fast model correct", "LLM: out-of-scope sent to the none exit"], rows) + "\n\n"
    if o and o.get("locale"):
        lab = {"en_seen": "English, seen groups", "en_unseen": "English, never-seen groups", "es-ES_seen": "Spanish, same requests as row 1",
               "de-DE_seen": "German, same requests as row 1"}
        rows = [[lab.get(k, k), v["n"], pct(v["llm"]), pct(v["fast:bge-small"]), pct(v["fast:me5-small"])] for k, v in o["locale"].items() if "llm" in v]
        s += ("**The LLM on a new language** (MASSIVE scenario router, 18 classes, English contract; fast models trained on English "
              "training groups only):\n\n")
        s += table(["Requests", "n", "LLM (Qwen3-4B)", "Fast: bge-small + LR (English encoder)", "Fast: multilingual-e5-small + LR"], rows) + "\n\n"
    return s


def sec_e6() -> str:
    r = J("exp6_stream.json")
    if not r:
        return ""
    s = "## 7. Experiment 6: a production loop with scripted drift\n\n"
    s += (f"*Replay of real outputs: every request is a held-out human-written query, fast models are really run and retrained in the "
          f"loop, every LLM answer is a stored real answer ({pct(r['llm_answers_available'], 0)} of requests have one). Simulated: the "
          f"traffic script, and the agent (human label).* {r['n_requests']:,} requests in five phases.\n\n")
    ph = {1: "steady", 2: "new groups", 3: "out-of-scope burst", 4: "noisy input", 5: "policy change"}
    rows = [[f"{p} {ph[int(p)]}", v["n"], ", ".join(f"{k} {n}" for k, n in v["kinds"].items())] for p, v in r["phases"].items()]
    s += table(["Phase", "Requests", "Mix"], rows, "lrl") + "\n\n"
    names = {"A": "A always-LLM", "B": "B static, naive calibration", "C": "C static, gate + verifier",
             "D": "D = C + shadow sampling + class breakers", "E": "E = D + regression-gated recompiles",
             "E'": "E' = E, regression suite left stale", "F": "F ungated self-training",
             "D+": "D+ = D, contract change announced", "E+": "E+ = E, contract change announced"}
    P = r["policies"]

    def block(keys, star):
        rows = []
        for k in keys:
            kk = k + star
            if kk not in P:
                continue
            v, t = P[kk], P[kk]["total"]
            rows.append([names[k], pct(t["wrong"], 2), pct(t["wrong_via_fast"], 2), pct(t["fast"]), pct(t["llm"]), pct(t["agent"]),
                         num(t["llm_calls"], 2), num(t["ms"], 0)] +
                        [pct(v["by_phase"][p]["wrong"], 1) for p in ("1", "2", "3", "4", "5")] + [pct(v["phase5_affected"]["wrong"], 0)])
        return table(["Policy", "Wrong delivered", "…of which from the fast tier", "Fast", "LLM", "Agent", "LLM calls / req (incl. shadow)",
                      "ms / req", "wrong ph.1", "ph.2", "ph.3", "ph.4", "ph.5", "wrong on re-routed requests"], rows)
    s += "**Abstentions go to the local LLM, then to the agent** (the shadow oracle is the LLM):\n\n"
    s += block(["A", "B", "C", "D", "D+", "E", "E'", "E+", "F"], "") + "\n\n"
    s += ("**Abstentions go straight to the agent** (the shadow oracle is the agent). This removes the local LLM's own errors, "
          "so what is left is the compiled tier and the loop:\n\n")
    s += block(["B", "C", "D", "D+", "E", "E'", "E+", "F"], "*") + "\n\n"
    rows = []
    for k in ("D", "D*", "E*"):
        v = P[k]
        for p in ("1", "2", "3", "4", "5"):
            x = v["by_phase"][p]
            if x["shadow_n"]:
                rows.append([k, f"{p} {ph[int(p)]}", pct(x["fast"]), x["shadow_n"], pct(x["shadow_est"]),
                             f"{pct(x['shadow_ci'][0])} – {pct(x['shadow_ci'][1])}", pct(x["wrong_fast"])])
    s += ("Shadow estimate of the fast tier's silent-error rate versus the truth (10% of fast answers re-checked; in D the checker is "
          "the LLM, in D* and E* the agent):\n\n")
    s += table(["Policy", "Phase", "Fast share", "Shadow samples", "Shadow estimate", "95% interval", "True silent rate of fast answers"], rows, "llrrrrr") + "\n\n"
    sw = r.get("shadow_sweep", {})
    if sw:
        rows = [[pct(float(k), 0), v["requests_until_breaker"] if v["requests_until_breaker"] is not None else "never",
                 pct(v["wrong_on_rerouted"], 0), pct(v["wrong_total"], 2), num(v["agent_shadow_calls_per_request"], 3), v["other_breakers"]]
                for k, v in sw.items()]
        s += "How quickly shadow sampling notices the policy change (policy D*, agent as checker):\n\n"
        s += table(["Shadow rate", "Requests until the breaker opens", "Wrong on re-routed requests", "Wrong overall",
                    "Checker calls / request", "False alarms (other classes)"], rows) + "\n\n"
    for k in ("E", "E*", "E'*", "E+*"):
        if k not in P:
            continue
        lines = []
        for e in P[k]["events"]:
            if e["event"] == "breaker":
                lines.append(f"request {e['t']}: breaker opens for class `{e['class']}` ({e['window'][0]} of last {e['window'][1]} shadow checks disagreed)")
            elif e["event"] == "announced":
                lines.append(f"request {e['t']}: contract change announced, class `{e['class']}` invalidated")
            elif e["event"] == "recompile":
                lines.append(f"request {e['t']}: recompile ({e['reason']}) on {e['labels_used']} production labels: "
                             f"{'**deployed**' if e['admitted'] else 'rejected'} (recent hold-out {e['recent_errors']}/{e['recent_answered']} wrong, "
                             f"regression suite {pct(e['regression_acc'][0])} → {pct(e['regression_acc'][1])})")
        if lines:
            extra = P[k]["regression_readjudication_agent_calls"]
            s += f"Events, policy {k}" + (f" ({extra} regression cases re-adjudicated by the agent)" if extra else "") + ":\n\n"
            s += "\n".join("- " + ln for ln in lines) + "\n\n"
    return s


def sec_e7() -> str:
    r = J("exp7_news_drift.json")
    if not r:
        return ""
    s = "## 8. Experiment 7: natural drift, nothing scripted\n\n"
    s += (f"*Real runs on real dated data; the oracle is the editor's label (no LLM).* HuffPost headlines in date order "
          f"({r['n']:,} headlines, {len(r['months'])} months). Compiled on the first {r['compile_months']} months "
          f"({r['sections_at_compile']} of the {r['sections_total']} sections exist then); accuracy on its own temporal hold-out is "
          f"{pct(r['base_accuracy_first_year_holdout'])}. Budget for the fast tier: {pct(r['alpha'], 0)} silent errors. Sections that were only "
          "renamed over the years (for example THE WORLDPOST, WORLDPOST and WORLD NEWS) are merged before anything is trained; sections that "
          "are new stay new.\n\n")
    rows = []
    for k, v in r["policies"].items():
        x = v["summary"]
        rows.append([k, pct(x["coverage"]), pct(x["silent_among_fast"]), pct(x["wrong_per_request"], 2),
                     f"{x['months_over_budget']} of {x['months_with_fast_tier']}", pct(x["worst_month_silent"]),
                     v.get("breaker_tripped_after", "–") or "never"])
    s += table(["Policy", "Fast coverage", "Silent among fast answers", "Wrong per request", "Months over budget", "Worst month", "Breaker opened"], rows) + "\n\n"
    years = list(r["policies"]["static-T"]["by_year"])
    rows = [["traffic from sections unseen at compile time"] + [pct(r["policies"]["static-T"]["by_year"][y]["new_section_share"], 0) for y in years]]
    for k in ("static-R", "static-T", "gated-quarterly", "self-train"):
        b = r["policies"][k]["by_year"]
        rows.append([f"{k}: coverage"] + [pct(b[y]["coverage"], 0) for y in years])
        rows.append([f"{k}: silent among fast"] + [pct(b[y]["silent_among_fast"], 0) for y in years])
    s += table(["By year"] + years, rows) + "\n\n"
    ev = r["policies"]["gated-quarterly"]["events"]
    s += (f"Gated quarterly recompiles: {sum(e['admitted'] for e in ev)} of {len(ev)} candidates could be certified on the newest "
          "labelled month; in the other quarters the fast tier stayed off.\n\n")
    return s


def sec_e8() -> str:
    r = J("exp8_kernel.json")
    if not r:
        return ""
    s = "## 9. Experiment 8: the compiled program run for real, and its admission test\n\n"
    c = r["cases"]
    s += (f"*Live execution, one request at a time, of a `wfc.Program`: guards, cache, mined rules, sentence embedding, classifier, "
          f"familiarity search and lexical verifier all computed per request; the LLM tier replays the stored real answer; the agent is "
          f"the human label.* {c['seen']} seen-group, {c['unseen_groups']} unseen-group and {c['out_of_scope']} out-of-scope requests, "
          f"then {sum(r['bad_inputs'].values())} seeded bad inputs that the program must not answer. Budget: 2% silent errors.\n\n")
    rows = []
    for name, v in r["builds"].items():
        bt = v["by_tier"]
        tiers = ", ".join(f"{t} {x['n']} ({x['wrong']} wrong)" for t, x in bt.items())
        lt = v["latency_ms_p50"]
        rows.append([name, "yes" if v["admitted"] else "**no**", "yes" if v.get("admitted_at_5pct") else "**no**",
                     pct(v["no_llm_rate"]), pct(v["agent_rate"]),
                     f"{v['silent_errors']} ({pct(v['silent_error_rate'], 2)}; ≤ {pct(v['silent_error_upper95'], 2)})",
                     f"{v['guard_false_accepts']} of {v['bad_inputs']}",
                     " / ".join(num(lt[t], 2) if t in lt else "–" for t in ("rules", "model")), tiers])
    s += table(["Build", "Admitted at 2%", "at 5%", "Answered locally", "Escalated", "Silent errors (rate; 95% upper bound)",
                "Bad inputs answered", "ms p50: rules / model tier", "Answered by tier"], rows, "lllrrrrrl") + "\n\n"
    rows = []
    for name, v in r["builds"].items():
        d = v["bad_input_detail"]
        rows.append([name] + [f"{d[k]['answered']}/{d[k]['n']}" + (f" ({', '.join(f'{t} {n}' for t, n in d[k]['by_tier'].items())})" if d[k]["answered"] else "")
                              for k in ("structural", "gibberish", "keyboard_mash", "injection", "far_domain")])
    s += "Seeded bad inputs that got an answer, by kind and by the tier that answered:\n\n"
    s += table(["Build", "structural", "gibberish", "keyboard mash", "injection text", "far-domain text"], rows, "llllll") + "\n\n"
    reasons = {name: v["reasons"] for name, v in r["builds"].items() if v["reasons"]}
    if reasons:
        s += "Why builds were refused: " + "; ".join(f"*{k}*: {', '.join(v)}" for k, v in reasons.items()) + ".\n\n"
    return s


def sec_e9() -> str:
    r = J("exp9_harness.json")
    if not r:
        return ""
    sp = r["spec"]
    s = "## 10. Experiment 9: a wrapper harness that watches the LLM work, then runs the task as a program\n\n"
    s += ("*Real runs.* This tests the idea in your post directly: leave the model alone, wrap it, observe it doing an open-ended "
          "task (\"analyze this profile\"), and turn what it did into a program whose remaining decisions are small. The uncompiled "
          "agent is three free-form calls to Qwen3-4B per profile (extract facts, assess, write a summary). The harness recorded "
          f"{sp['n_observed']} runs, searched candidate programs per output field against them, and admitted a program only if it "
          f"agreed with the LLM on {sp['n_admission']} separate recorded runs. Test: {r['n_test']} profiles never seen by the compile step.\n\n")
    rows = []
    for k, v in sp["fields"].items():
        if "admitted" in v:
            rows.append([k, len(v["candidates"]), f"`{v['chosen']}`", pct(v["observed_agreement"], 0), pct(v["admission_agreement"], 0),
                         pct(v["bar"], 0), v["compiled_to"]])
        else:
            rows.append([k, "–", "–", "–", "–", "–", v["compiled_to"]])
    s += "**What the compile step decided, field by field:**\n\n"
    s += table(["Output field", "Candidates tried", "Best candidate", "Agreement on observed runs", "On admission runs", "Bar", "Compiled to"], rows, "lrlrrrl") + "\n\n"
    ung = {k: v for f in sp["fields"].values() for k, v in f.get("rejected_ungrounded", {}).items()}
    best_edu = sp["fields"]["education_level"]["observed_agreement"]
    au, fr = sp.get("agent_audit"), sp.get("field_rule_vs_human")
    if ung:
        k, v = next(iter(ung.items()))
        s += (f"One candidate was excluded before the search: `{k}` says \"bachelor\" whenever the profile shows no degree. It agrees with the "
              f"agent on {pct(v, 0)} of observed runs, far more than the best grounded rule ({pct(best_edu, 0)}), and it is wrong by the contract: "
              "the field is supposed to be read from the profile.\n\n")
    if au:
        s += (f"**The recordings are also an audit of the agent.** In {au['profiles_with_no_education_evidence_in_input']} of {au['runs_audited']} "
              f"recorded runs the text the agent was given contains no sign of education at all (no degree word or abbreviation, no mention of "
              f"school, college, university, graduation or a diploma). The agent still reported a degree in "
              f"{au['of_those_agent_still_reported_a_degree']} of them; it answered \"bachelor\" in {pct(au['agent_reported_bachelor_overall'], 0)} of all runs.\n\n")
    if fr:
        s += (f"**A rule written from the contract can be better than the agent it is compared with.** The field rule (a field name or synonym in the "
              f"profile's heading) answers {fr['rule_answers']} of {fr['rows']} recorded profiles and matches the human label on "
              f"{pct(fr['rule_matches_human_label'], 0)} of them; the agent matches the human label on {pct(fr['agent_matches_human_label_same_rows'], 0)} "
              f"of the same profiles. Admission by agreement with the agent ({pct(fr['rule_matches_agent'], 0)}) turned the rule down.\n\n")
    a = r["agent"]
    rows = [["uncompiled agent (3 free-form calls)", num(a["llm_calls"], 0), num(a["prompt_tokens"], 0), num(a["generated_tokens"], 0),
             num(a["latency_s"]["p50"], 2), "1.0×", "–"]]
    lab = {"structured": "compiled: structured analysis only", "prose_hole": "compiled: + prose from a small generation hole",
           "template": "compiled: + prose from a code template"}
    for m, v in r["modes"].items():
        rows.append([lab[m], num(v["llm_calls"], 2), num(v["prompt_tokens"], 0), num(v["generated_tokens"], 0), num(v["latency_s"]["p50"], 2),
                     f"{a['latency_s']['p50'] / v['latency_s']['p50']:.1f}×", pct(v["fallback_to_agent"], 0)])
    s += "**Cost per profile** (single request, same machine, same model):\n\n"
    s += table(["Path", "LLM calls", "Prompt tokens", "Generated tokens", "Latency p50 (s)", "Speed-up", "Handed back to agent"], rows) + "\n\n"
    v = r["modes"]["structured"]
    sa = r["agent_self_agreement_reworded_prompt"]
    rows = []
    for k, x in v["agreement_with_agent"].items():
        base = k.split("(")[0]
        via = "code" if v["code_fields"].get(base, 0) >= 0.5 * r["n_test"] else ("mixed" if v["code_fields"].get(base, 0) else "narrow hole")
        rows.append([k, via, pct(x, 0), pct(sa.get(k), 0) if k in sa else "–"])
    s += ("**Fidelity: does the compiled program say what the agent would have said?** The last column is a ceiling: how often the "
          "agent agrees with *itself* when the extraction instruction is merely reworded.\n\n")
    s += table(["Field", "Produced by", "Compiled agrees with agent", "Agent agrees with itself (reworded)"], rows, "llrr") + "\n\n"
    s += (f"Against the human field label, the agent is right on {pct(v['agent_field_accuracy_vs_human'], 0)} of test profiles and the "
          f"compiled program on {pct(v['field_accuracy_vs_human'], 0)}. ")
    if "summary_cosine_to_agent" in r["modes"]["prose_hole"]:
        ph, tp = r["modes"]["prose_hole"], r["modes"]["template"]
        s += (f"Prose cannot be compared exactly; as a rough indication, the embedding similarity between the compiled summary and the "
              f"agent's summary is {num(ph['summary_cosine_to_agent'])} with the generation hole and {num(tp['summary_cosine_to_agent'])} "
              f"with the template, against {num(ph['summary_cosine_floor_other_profile'])} between the agent's summaries of two different profiles. ")
    g = r["guard_test"]
    s += f"Guard test: {g['rejected']} of {g['bad_inputs']} non-profile inputs were handed back instead of analysed.\n\n"
    return s


SECTIONS = [sec_setup, sec_e1, sec_e2, sec_e3, sec_e4, sec_e5, sec_e6, sec_e7, sec_e8, sec_e9]


def build() -> str:
    from . import report_prose as P
    parts = [P.title_and_summary()]
    for fn in SECTIONS:
        body = fn()
        if not body:
            continue
        parts.append(body + P.findings(fn.__name__))
    parts.append(P.closing())
    return "\n".join(parts)


if __name__ == "__main__":
    out = ROOT / "REPORT.md"
    out.write_text(build())
    print("wrote", out, len(out.read_text()), "chars")
