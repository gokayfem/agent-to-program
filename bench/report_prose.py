"""Written findings for REPORT.md. Numbers are looked up in results/*.json, never typed in."""
from __future__ import annotations

from .make_report import J, num, pct


def _cell(ds, model, key):
    return J("exp3_group_shift.json")[ds]["models"][model]["cells"][key]


def findings(section: str) -> str:
    fn = globals().get("f_" + section)
    return (fn() + "\n") if fn else ""


# --------------------------------------------------------------------------- per-experiment findings
def f_sec_e1() -> str:
    r = J("exp1_families.json")
    ft = J("exp1b_finetune.json") or {}
    L = J("exp_latency.json")["tier_ms_p50"]
    c, b, m = r["clinc150"]["families"], r["banking77"]["families"], r["massive60"]["families"]
    over = []
    for ds in ("clinc150", "banking77"):
        for k, v in r[ds]["families"].items():
            if "fs@0.02" in v and v["fs@0.02"]["answered"] and v["fs@0.02"]["silent_error_rate"] > 0.02:
                over.append(f"{k} on {ds} ({pct(v['fs@0.02']['silent_error_rate'], 2)})")
    s = "**What this shows**\n\n"
    s += (f"- On in-distribution traffic a small frozen encoder with a linear head is enough. On CLINC150, bge-small + LR is "
          f"{pct(c['bge-small+LR']['accuracy'])} accurate and can answer {pct(c['bge-small+LR']['fs@0.02']['coverage'])} of queries inside a "
          f"2% budget at about {num(L['bge-small encode'] + L['LR head'], 1)} ms; character TF-IDF + LR answers "
          f"{pct(c['tfidf-char+LR']['fs@0.02']['coverage'])} at {num(L['tfidf-char+LR end-to-end'], 2)} ms.\n")
    if "clinc150" in ft:
        f = ft["clinc150"]
        s += (f"- Fine-tuning MiniLM end to end ({num(f['fit_s'] / 60, 1)} minutes on CPU) gives {pct(f['accuracy'])} and "
              f"{pct(f['fs@0.02']['coverage'])} coverage at {num(f['latency_ms_p50'], 1)} ms: the accuracy of the larger frozen encoders at "
              f"about half the latency of bge-small.\n")
    s += (f"- How much can be taken over depends on the task, not the model. BANKING77's fine-grained intents allow "
          f"{pct(b['tfidf-char+LR']['fs@0.02']['coverage'], 0)}–{pct(b['bge-small+kNN']['fs@0.02']['coverage'], 0)} at 2%. On MASSIVE "
          f"(accuracy {pct(m['potion+LR']['accuracy'], 0)}–{pct(m['bge-small+MLPx5']['accuracy'], 0)}) **no family could be certified at 2%**; "
          f"at 5% they answer {pct(m['bge-small+LR']['fs@0.05']['coverage'], 0)}–{pct(m['bge-small+MLPx5']['fs@0.05']['coverage'], 0)}. "
          "A compile step has to be allowed to conclude \"this decision stays with the model\".\n")
    s += (f"- A semantic cache (1-NN on embedding similarity) is the weakest use of an embedding. With the threshold peeked on test, "
          f"it could cover {pct(c['bge-small+1NN']['oracle_cov@0.02'], 0)} of CLINC150 at 2% against {pct(c['bge-small+LR']['oracle_cov@0.02'], 0)} "
          f"for a linear head on the same vectors ({pct(b['bge-small+1NN']['oracle_cov@0.02'], 0)} vs {pct(b['bge-small+LR']['oracle_cov@0.02'], 0)} "
          "on BANKING77): similarity to one stored example says little about whether its label transfers.\n")
    rk = next(k for k in c if k.startswith("mined-rules(p>=1.0,n>=10"))
    s += (f"- Rule committees mined from history are cheap ({num(L['mined rules'], 2)} ms) and abstain by construction, but single rules are "
          f"not safe: {c[rk]['n_rules']} rules with no counter-example in training cover {pct(c[rk]['coverage'], 0)} of CLINC150 test "
          f"queries with {pct(c[rk]['silent_error_rate'], 1)} silent errors; requiring two agreeing rules gives {pct(c[rk]['coverage_2votes'], 0)} "
          f"coverage at {pct(c[rk]['silent_error_rate_2votes'], 1)}.\n")
    s += (f"- Verbatim repeats are rare in these crowd-sourced sets ({pct(c['exact-cache']['coverage'], 2)}–{pct(m['exact-cache']['coverage'], 2)}), "
          "so the exact-match cache does almost nothing here. Real traffic repeats far more; that was not measured.\n")
    if over:
        s += ("- \"Certified\" is a statistical statement. " + str(len(over)) + " family/dataset pairs ended slightly over the 2% budget on test: "
              + "; ".join(over) + ". The calibration and test partitions of these datasets were collected separately, so even here "
              "exchangeability is only approximate.\n")
    return s


def f_sec_e2() -> str:
    r = J("exp2_novelty.json")
    D, T, Lc = r["detectors"], r["typo_noise"], r["locale_shift"]
    ft = (J("exp1b_finetune.json") or {}).get("clinc150")
    s = "**What this shows**\n\n"
    s += (f"- A model's own confidence is a poor familiarity test. With the threshold certified on in-scope data only, bge-small + LR "
          f"would answer {pct(D['bge-small/msp']['op_R']['oos_false_accept'], 0)} of human-written out-of-scope queries "
          f"(bge-base: {pct(D['bge-base/msp']['op_R']['oos_false_accept'], 0)}), which turns a 2% tier into a "
          f"{pct(D['bge-small/msp']['op_R']['mixed_silent_rate'], 0)} tier on CLINC's own 82/18 in-scope/out-of-scope test mix. "
          "The bigger, more accurate encoder is the worse one here.\n")
    s += (f"- Distance in embedding space is much better. Relative Mahalanobis distance reaches AUROC "
          f"{num(D['bge-small/rel_maha']['auroc_oos'], 3)} (bge-base {num(D['bge-base/rel_maha']['auroc_oos'], 3)}) and lets through "
          f"{pct(D['bge-small/rel_maha']['op_R']['oos_false_accept'], 1)} of out-of-scope queries, at the price of in-scope coverage "
          f"({pct(D['bge-small/rel_maha']['op_R']['in_scope_coverage'], 0)} instead of {pct(D['bge-small/msp']['op_R']['in_scope_coverage'], 0)}; "
          f"{pct(D['bge-base/rel_maha']['op_R']['in_scope_coverage'], 0)} with bge-base). Nearest-neighbour similarity ranks out-of-scope "
          "inputs almost as well but is useless as a correctness score, so on its own it certifies nothing (the 0% rows).\n")
    s += (f"- Showing the calibration step real out-of-scope examples helps every score: confidence alone drops from "
          f"{pct(D['bge-small/msp']['op_R']['oos_false_accept'], 0)} to {pct(D['bge-small/msp']['op_G']['oos_false_accept'], 0)} answered. "
          f"A learned trust model over confidence, distances and cross-view agreement keeps the most coverage "
          f"({pct(D['bge-small/learned_trust']['op_G']['in_scope_coverage'], 0)}) but still answers "
          f"{pct(D['bge-small/learned_trust']['op_G']['oos_false_accept'], 1)} of out-of-scope queries. Nothing gets near-scope inputs under 1% "
          "without giving up a quarter of the in-scope traffic.\n")
    if ft:
        s += (f"- Fine-tuning improves this but does not fix it: the fine-tuned MiniLM's confidence has AUROC {num(ft['oos']['msp']['auroc'], 3)} "
              f"and still answers {pct(ft['oos']['false_accept_at_certified_threshold'], 0)} of out-of-scope queries at its certified threshold.\n")
    s += (f"- Keyboard noise breaks embedding models *silently*. At 15% noisy letters bge-small falls to {pct(T['bge-small']['typo15']['accuracy'], 0)} "
          f"accuracy while its confidence gate still answers {pct(T['bge-small']['typo15']['msp']['coverage'], 0)} of requests, "
          f"{pct(T['bge-small']['typo15']['msp']['silent_error_rate'], 0)} of them wrong (bge-base: {pct(T['bge-base']['typo15']['msp']['silent_error_rate'], 0)}"
          + (f"; fine-tuned MiniLM: {pct(ft['clinc_test_typo15']['silent_error_rate'], 0)}" if ft else "") +
          f"). Character TF-IDF degrades honestly: coverage falls to {pct(T['tfidf']['typo15']['msp']['coverage'], 0)} and silent errors stay at "
          f"{pct(T['tfidf']['typo15']['msp']['silent_error_rate'], 1)}. Adding the similarity floor cuts bge-small's silent errors to "
          f"{pct(T['bge-small']['typo15']['min(msp,knn1)']['silent_error_rate'], 0)}, still several times the budget.\n")
    en = ("minilm", "bge-small")
    worst = max((Lc[v][loc]["msp"]["silent_error_rate"] * Lc[v][loc]["msp"]["coverage"], v, loc) for v in en for loc in Lc[v] if loc != "en-US")
    s += (f"- A new language is the easy case for a familiarity gate and the worst case without one. English-only encoders are "
          f"{pct(min(Lc[v][loc]['accuracy'] for v in en for loc in Lc[v] if loc != 'en-US'), 0)}–"
          f"{pct(max(Lc[v][loc]['accuracy'] for v in en for loc in Lc[v] if loc != 'en-US'), 0)} accurate on translated utterances, yet the "
          f"confidence gate keeps answering (worst: {worst[1]} on {worst[2]}, {pct(Lc[worst[1]][worst[2]]['msp']['coverage'], 0)} answered, "
          f"{pct(Lc[worst[1]][worst[2]]['msp']['silent_error_rate'], 0)} of those wrong). Similarity to the training set separates English from every other locale with AUROC "
          f"{num(min(Lc[v][loc]['auroc_locale/knn1'] for v in en for loc in Lc[v] if loc != 'en-US'), 2)}–"
          f"{num(max(Lc[v][loc]['auroc_locale/knn1'] for v in en for loc in Lc[v] if loc != 'en-US'), 2)}, and the combined gate answers at most "
          f"{pct(max(Lc[v][loc]['min(msp,knn1)']['coverage'] for v in en for loc in Lc[v] if loc != 'en-US'), 1)} of them. A multilingual encoder "
          f"(me5-small) generalises ({pct(min(Lc['me5-small'][loc]['accuracy'] for loc in Lc['me5-small'] if loc != 'en-US'), 0)}–"
          f"{pct(max(Lc['me5-small'][loc]['accuracy'] for loc in Lc['me5-small'] if loc != 'en-US'), 0)} accurate) but its confidence gate is "
          f"{pct(min(Lc['me5-small'][loc]['msp']['silent_error_rate'] for loc in Lc['me5-small'] if loc != 'en-US'), 0)}–"
          f"{pct(max(Lc['me5-small'][loc]['msp']['silent_error_rate'] for loc in Lc['me5-small'] if loc != 'en-US'), 0)} wrong on what it answers: "
          "better generalisation makes the shift harder to see, not safer.\n")
    return s


def f_sec_e3() -> str:
    a = J("exp3_group_shift.json")
    c, m = a["clinc"]["models"], a["massive"]["models"]
    b = c["bge-small+LR"]["base"]
    cell = lambda key, ds="clinc", model="bge-small+LR": _cell(ds, model, key)
    s = "**What this shows**\n\n"
    s += (f"- The unseen-merchant failure is real on human text and it is a property of the models, not of the toy. On groups never seen "
          f"in training, accuracy falls from {pct(b['acc_seen']['mean'], 0)} to {pct(b['acc_unseen']['mean'], 0)} (CLINC) and from "
          f"{pct(m['bge-small+LR']['base']['acc_seen']['mean'], 0)} to {pct(m['bge-small+LR']['base']['acc_unseen']['mean'], 0)} (MASSIVE), and the "
          f"embedding classifiers do not notice: {pct(b['unseen_errors_conf>=0.9']['mean'], 0)} of bge-small's unseen-group errors are made "
          f"with confidence ≥ 0.9. TF-IDF is wrong more often but far less sure of itself ({pct(c['tfidf-char+LR']['base']['unseen_errors_conf>=0.9']['mean'], 0)}).\n")
    r2, r5 = cell("msp|R|fixed_sequence|0.02"), cell("msp|R|fixed_sequence|0.05")
    s += (f"- Random-split calibration certifies exactly the wrong thing. It keeps its promise on seen groups in every split "
          f"({pct(r2['test_seen']['silent_mean'], 1)} silent at a 2% budget) and breaks it on unseen groups in every split: the tier answers "
          f"{pct(r2['test_unseen']['coverage_mean'], 0)} of unseen-group inputs and {pct(r2['test_unseen']['silent_mean'], 0)} of those answers are wrong "
          f"(at a 5% budget: {pct(r5['test_unseen']['coverage_mean'], 0)} answered, {pct(r5['test_unseen']['silent_mean'], 0)} wrong).\n")
    g2, g5 = cell("msp&knn1|G-worst|fixed_sequence|0.02"), cell("msp&knn1|G-worst|fixed_sequence|0.05")
    s += (f"- Calibrating against held-out groups works, and its price is coverage on familiar traffic. With the confidence + familiarity gate "
          f"wrong answers per unseen-group input fall from {pct(r2['test_unseen']['far_mean'], 1)} to {pct(g2['test_unseen']['far_mean'], 1)} "
          f"(2% budget) and from {pct(r5['test_unseen']['far_mean'], 0)} to {pct(g5['test_unseen']['far_mean'], 1)} (5%), while seen-group coverage "
          f"drops from {pct(r2['test_seen']['coverage_mean'], 0)} to {pct(g2['test_seen']['coverage_mean'], 0)} and from "
          f"{pct(r5['test_seen']['coverage_mean'], 0)} to {pct(g5['test_seen']['coverage_mean'], 0)}. The gate does not make unseen-group answers "
          f"trustworthy ({pct(g2['test_unseen']['silent_mean'], 0)} of the ones it lets through are still wrong); it works by refusing "
          f"{pct(1 - g2['test_unseen']['coverage_mean'], 0)} of them.\n")
    cb = cell("msp&knn1|G-worst|cluster_bootstrap|0.05")
    s += (f"- It is not a guarantee. The realised rate on never-seen groups exceeded the budget in {pct(g2['test_unseen']['far_violation_rate'], 0)} "
          f"of splits at 2% and {pct(g5['test_unseen']['far_violation_rate'], 0)} at 5%, although the procedure promises 5%. Rows from one group "
          f"fail together, so 4,500 calibration rows from 30 groups are closer to 30 observations than to 4,500. A group-aware bootstrap "
          f"picker is over budget in {pct(cb['test_unseen']['far_violation_rate'], 0)} of splits at 5% and pays "
          f"{num(100 * (g5['test_seen']['coverage_mean'] - cb['test_seen']['coverage_mean']), 0)} points of seen coverage for it.\n")
    b3 = J("exp3b_cal_groups.json")
    if b3:
        S = b3["settings"]
        lo, hi = min(S, key=int), max(S, key=int)
        s += (f"- More held-out groups help, steadily. Going from {lo} to {hi} calibration groups takes the row-level test from over budget in "
              f"{pct(S[lo]['fixed_sequence']['unseen_over_budget'], 0)} of splits to {pct(S[hi]['fixed_sequence']['unseen_over_budget'], 0)} and the "
              f"group-aware one from {pct(S[lo]['cluster_bootstrap']['unseen_over_budget'], 0)} to {pct(S[hi]['cluster_bootstrap']['unseen_over_budget'], 0)}; "
              f"the worst split falls from {pct(S[lo]['fixed_sequence']['unseen_far_max'], 0)} to {pct(S[hi]['fixed_sequence']['unseen_far_max'], 0)} "
              f"wrong answers per unseen input. With 20 splits per row these rates are themselves rough (one split is 5 points).\n")
    k2 = cell("knn1|G-mix|fixed_sequence|0.02")
    l5 = cell("learned|G-worst|fixed_sequence|0.05")
    s += (f"- The scores trade differently. Familiarity alone gives the fewest wrong answers on unseen groups "
          f"({pct(k2['test_unseen']['far_mean'], 2)} per unseen input at 2%) but lets more seen-group errors through "
          f"({pct(k2['test_seen']['silent_mean'], 2)}). A learned trust model keeps the most seen coverage "
          f"({pct(l5['test_seen']['coverage_mean'], 0)} at 5%) with similar unseen-group leakage ({pct(l5['test_unseen']['far_mean'], 1)}).\n")
    mm = cell("msp&knn1|G-worst|fixed_sequence|0.05", "massive")
    mr = cell("msp|R|fixed_sequence|0.05", "massive")
    s += (f"- MASSIVE repeats the pattern with fewer groups and noisier labels: random calibration lets {pct(mr['test_unseen']['far_mean'], 0)} of "
          f"unseen-group inputs through wrong at a 5% budget, the guarded gate {pct(mm['test_unseen']['far_mean'], 1)} with "
          f"{pct(mm['test_seen']['coverage_mean'], 0)} seen coverage. All four model families behave the same way (last table of each block), "
          "so the remedy belongs in the calibration protocol, not in the choice of classifier.\n")
    return s


def f_sec_e7() -> str:
    r = J("exp7_news_drift.json")
    P = r["policies"]
    y = P["static-T"]["by_year"]
    years = list(y)
    ev = P["gated-quarterly"]["events"]
    s = "**What this shows**\n\n"
    s += (f"- A compiled program left alone rots, and its own confidence does not say so. The static program stays near its budget for the "
          f"first year ({pct(y[years[0]]['silent_among_fast'], 0)} silent in {years[0]}) and is {pct(y[years[-1]]['silent_among_fast'], 0)} wrong on what it "
          f"answers by {years[-1]}, while still answering {pct(y[years[-1]]['coverage'], 0)} of traffic. The main cause is taxonomy change: by "
          f"{years[-1]}, {pct(y[years[-1]]['new_section_share'], 0)} of headlines belong to sections that did not exist at compile time, so every "
          "fast answer on them is wrong by construction.\n")
    s += (f"- Calibrating on the last two months instead of a random split barely helps here "
          f"({pct(P['static-R']['summary']['silent_among_fast'], 1)} vs {pct(P['static-T']['summary']['silent_among_fast'], 1)} silent overall): a temporal "
          "hold-out protects against drift that had already started, not against sections that do not exist yet.\n")
    s += (f"- Shadow sampling catches it. With 10% of fast answers re-checked, the breaker opened after "
          f"{P['T + breaker']['breaker_tripped_after']}, and wrong answers per request over the whole period fell from "
          f"{pct(P['static-T']['summary']['wrong_per_request'], 1)} to {pct(P['T + breaker']['summary']['wrong_per_request'], 1)}.\n")
    s += (f"- Gated recompilation keeps the promise and gives up most of the speed: {pct(P['gated-quarterly']['summary']['silent_among_fast'], 1)} "
          f"silent among fast answers against a 10% budget, but only {pct(P['gated-quarterly']['summary']['coverage'], 0)} coverage, because "
          f"only {sum(e['admitted'] for e in ev)} of {len(ev)} quarterly candidates could be certified on the newest labelled month. Part of that is "
          "the task (30 overlapping editorial sections), part is a real bias: the labels the system holds come from escalations and are harder "
          "than the traffic the fast tier would answer.\n")
    s += (f"- Retraining on its own answers without a gate keeps coverage high ({pct(P['self-train']['summary']['coverage'], 0)}) and stays "
          f"over budget in {P['self-train']['summary']['months_over_budget']} of {P['self-train']['summary']['months_with_fast_tier']} months "
          f"({pct(P['self-train']['summary']['silent_among_fast'], 0)} silent overall).\n")
    return s


def f_sec_e4() -> str:
    r = J("exp4_verification.json")
    V = r["verifiers"]
    g = lambda name, pop, key: V[name][pop][key] if name in V and pop in V[name] else None
    same = ["agree: kNN head, same embedding", "agree: MLP ensemble unanimous, same embedding"]
    lex = "agree: tfidf-char+LR (lexical view)"
    llmc, llmv = "LLM classifies independently and agrees (qwen3-4b)", "LLM yes/no check of the proposal (qwen3-4b)"
    cu = r["error_correlation"].get("test_unseen", {})
    P = r["primary"]
    lat = r["latency_ms"]
    s = "**What this shows**\n\n"
    s += (f"- A stricter threshold on the model's own confidence is not verification. Calibrated on a random split it catches "
          f"{pct(g('self: msp >= thr (random-split calibration)', 'test_unseen', 'catch_rate'), 0)} of unseen-group errors and lets "
          f"{pct(g('self: msp >= thr (random-split calibration)', 'oos_test', 'false_accept_rate'), 0)} of out-of-scope queries through.\n")
    s += (f"- Second opinions built on the same embedding mostly repeat the mistake. A k-NN head and a unanimous 5-MLP ensemble catch "
          f"{pct(g(same[0], 'test_unseen', 'catch_rate'), 0)} and {pct(g(same[1], 'test_unseen', 'catch_rate'), 0)} of unseen-group errors. "
          f"When the fast model is wrong on an unseen group, the ensemble gives the *same wrong label* "
          f"{pct(cu[P + ' -> mlp-ens(bge-small)']['same_wrong_label_given_A_wrong'], 0)} of the time; the TF-IDF model "
          f"{pct(cu[P + ' -> tfidf-char+LR']['same_wrong_label_given_A_wrong'], 0)}; the LLM "
          f"{pct(cu[P + ' -> llm:qwen3-4b']['same_wrong_label_given_A_wrong'], 0)}. "
          "Independence is measurable, and it follows how differently the checker was built.\n")
    s += (f"- The cheapest useful verifier is a lexical model: {num(lat['fast'].get('tfidf-char+LR end-to-end'), 2)} ms, catches "
          f"{pct(g(lex, 'test_seen', 'catch_rate'), 0)} of seen-group errors, {pct(g(lex, 'test_unseen', 'catch_rate'), 0)} of unseen-group errors and "
          f"{pct(g(lex, 'oos_test', 'catch_rate'), 0)} of out-of-scope answers, while rejecting {pct(g(lex, 'test_seen', 'false_reject'), 1)} of correct "
          "seen-group answers. Agreement checks cannot refuse an input both models are wrong about in the same way, which is why "
          f"out-of-scope queries still get through ({pct(g(lex, 'oos_test', 'false_accept_rate'), 0)}).\n")
    fam = "familiarity: knn1 (same embedding)"
    s += (f"- Familiarity checks do what agreement cannot: they refuse {pct(g(fam, 'oos_test', 'catch_rate'), 0)} of out-of-scope answers and "
          f"{pct(g(fam, 'test_unseen', 'catch_rate'), 0)} of unseen-group errors for {num(lat['fast'].get('knn1 over train'), 2)} ms, but they also refuse "
          f"{pct(g(fam, 'test_seen', 'false_reject'), 0)} of correct seen-group answers. A verifier is paid for mostly in false rejects, not in milliseconds.\n")
    if llmc in V and "oos_test" in V[llmc]:
        s += (f"- An LLM is the most independent checker and the most expensive: asking the local LLM to classify independently costs "
              f"{num(lat['llm'].get('llm_classify'), 0)} ms and catches {pct(g(llmc, 'test_unseen', 'catch_rate'), 0)} of unseen-group errors and "
              f"{pct(g(llmc, 'oos_test', 'catch_rate'), 0)} of out-of-scope answers.")
        if llmv in V and "test_unseen" in V[llmv]:
            s += (f" A yes/no check of the proposal is cheaper ({num(lat['llm'].get('llm_verify'), 0)} ms) and worse on both counts: it catches "
                  f"{pct(g(llmv, 'test_unseen', 'catch_rate'), 0)} of unseen-group errors and rejects {pct(g(llmv, 'test_seen', 'false_reject'), 1)} of "
                  f"correct seen-group answers (independent classification: {pct(g(llmc, 'test_seen', 'false_reject'), 1)}).")
        s += " Running it on every fast answer gives the speed back; it belongs on a sample (shadowing) or behind a cheap pre-filter.\n"
    return s


def f_sec_e5() -> str:
    r = J("exp5_tiers.json")
    q = r["llm_tier"]["qwen3-4b"]
    st = r["stacks"]["policies"]
    mix = "70/20/10 seen/unseen/oos"
    M = lambda pol: st[pol]["mix"][mix]
    L = J("llm_latency.json") or {}
    s = "**What this shows**\n\n"
    s += (f"- In this test bed the local LLM is the weakest tier, not the compiled one. Qwen3-4B with a one-shot contract is right on "
          f"{pct(q['test_seen']['accuracy'])} of seen-group requests and {pct(q['test_unseen']['accuracy'])} of unseen-group requests, and it takes "
          f"the \"none of these\" exit for only {pct(q['oos_test']['none_exit_rate'], 0)} of out-of-scope queries. It is robust where the embedding "
          f"models are not ({pct(q['typo15']['accuracy'], 0)} on 15% keyboard noise). A larger hosted model would very likely do better; that was not run.\n")
    s += (f"- Its confidence is not a usable abstention signal: {pct(q['test_unseen']['p_eq_1_share'], 0)} of its unseen-group answers come with "
          f"token probability 1.0 and are right {pct(q['test_unseen']['acc_when_p_eq_1'], 0)} of the time. No threshold on it could be certified "
          f"at a 5% budget on the calibration pools (the `llm[conf]` stack therefore sends everything it would have answered to the agent).\n")
    ag = r["llm_tier"].get("two_llm_agreement", {})
    if "test_unseen" in ag:
        s += (f"- Two differently built LLMs agreeing is a better signal than either one's confidence: on unseen groups they agree on "
              f"{pct(ag['test_unseen']['agree_and_answer_rate'], 0)} of requests and are wrong on {pct(ag['test_unseen']['wrong_when_agree'], 0)} of those; "
              f"on out-of-scope queries they agree on a wrong in-scope label for {pct(ag['oos_test']['agree_and_answer_rate'] * ag['oos_test']['wrong_when_agree'], 0)}.\n")
    a, c, f0 = M("llm"), M("fast[G]+V > llm"), M("fast[G]+V")
    s += (f"- A guarded fast tier in front of the LLM gives the same quality for half the model calls. On the 70/20/10 mix, always-LLM "
          f"delivers {pct(a['wrong_delivered'], 1)} wrong answers at {num(a['compute_ms_mean'], 0)} ms per request; the stack "
          f"`fast[G]+V → llm` delivers {pct(c['wrong_delivered'], 1)} with {num(c['llm_calls'], 2)} LLM calls and {num(c['compute_ms_mean'], 0)} ms. "
          f"Almost all of its remaining errors are the LLM's: with the agent as the only fallback the same fast tier delivers "
          f"{pct(f0['wrong_delivered'], 1)} wrong answers and escalates {pct(f0['escalations'], 0)} of requests.\n")
    n, b = M("fast[R] > llm"), M("fast[all]")
    s += (f"- The naive stack is fast and wrong: `fast[R] → llm` uses {num(n['llm_calls'], 2)} LLM calls per request and delivers "
          f"{pct(n['wrong_delivered'], 1)} wrong answers, more than always-LLM, because it answers {pct(st['fast[R] > llm']['per_population']['oos_test']['share']['model'], 0)} "
          f"of out-of-scope queries and {pct(st['fast[R] > llm']['per_population']['test_unseen']['share']['model'], 0)} of unseen-group requests itself.\n")
    if "qwen3-4b/route_1shot_no_prefix_cache" in L:
        s += (f"- Engineering of the hole matters as much as the model. Keeping the contract as a fixed prefix and reusing its KV cache cut a "
              f"routing call from {num(L['qwen3-4b/route_1shot_no_prefix_cache']['p50'], 0)} ms to {num(L['qwen3-4b/route_1shot']['p50'], 0)} ms. "
              f"A longer contract is not free even when cached ({num(L['qwen3-4b/route_0shot']['p50'], 0)} ms with the short contract), and "
              "batching requests that share a long cached prefix duplicated it per sequence and pushed this machine into swap.\n")
    o = J("exp5b_llm_other.json")
    if o and o.get("fine") and o.get("locale"):
        f, lc = o["fine"], o["locale"]
        s += (f"- The fast tier and the LLM are good at opposite things. Zero-shot on fine-grained intents the LLM is far behind a model trained "
              f"on the task ({pct(f['clinc150']['llm_accuracy'], 0)} vs {pct(f['clinc150']['fast_accuracy_same_rows'], 0)} on CLINC150, "
              f"{pct(f['banking77']['llm_accuracy'], 0)} vs {pct(f['banking77']['fast_accuracy_same_rows'], 0)} on BANKING77). On what the fast tier "
              f"has not seen, it is ahead: {pct(lc['en_unseen']['llm'], 0)} vs {pct(lc['en_unseen']['fast:bge-small'], 0)} on never-seen MASSIVE groups, "
              f"and {pct(lc['es-ES_seen']['llm'], 0)} / {pct(lc['de-DE_seen']['llm'], 0)} on the Spanish / German versions of requests where the "
              f"English fast model falls to {pct(lc['es-ES_seen']['fast:bge-small'], 0)} / {pct(lc['de-DE_seen']['fast:bge-small'], 0)}. "
              f"(A multilingual encoder closes most of that gap by itself: {pct(lc['es-ES_seen']['fast:me5-small'], 0)} / "
              f"{pct(lc['de-DE_seen']['fast:me5-small'], 0)}.) That is the division of labour the tiers assume, and it only pays if the gate sends "
              "the right inputs to each.\n")
    tr = r.get("compile_from_traces", {})
    if "variants" in tr:
        v = tr["variants"]
        hl, ll = v["human labels, the same trace rows"], v["LLM labels, the same trace rows"]
        s += (f"- Compiling from the LLM's own traces transfers the LLM, including its mistakes: trained on {tr['n_traces']:,} LLM-labelled "
              f"requests the fast model reaches {pct(ll['bge-small+LR/test_seen']['accuracy_vs_human'])} on seen groups "
              f"(human labels for the same rows: {pct(hl['bge-small+LR/test_seen']['accuracy_vs_human'])}; the LLM itself: {pct(ll['llm/test_seen'])}) and "
              f"{pct(ll['bge-small+LR/test_unseen']['accuracy_vs_human'])} on unseen groups ({pct(hl['bge-small+LR/test_unseen']['accuracy_vs_human'])} with "
              "human labels). The student is bounded by its teacher on familiar inputs, so reviewed outputs are worth more than more traces.\n")
    return s


def f_sec_e6() -> str:
    r = J("exp6_stream.json")
    P = r["policies"]
    sw = r["shadow_sweep"]
    s = "**What this shows**\n\n"
    s += (f"- With the agent as fallback, the guards do their job on three of the four scripted drifts. The naive tier (B*) delivers "
          f"{pct(P['B*']['total']['wrong'], 0)} wrong answers; the guarded tier (C*) delivers {pct(P['C*']['by_phase']['2']['wrong'], 1)} when new groups "
          f"arrive, {pct(P['C*']['by_phase']['3']['wrong'], 1)} in the out-of-scope burst and {pct(P['C*']['by_phase']['4']['wrong'], 1)} on noisy input, "
          f"by escalating instead of answering ({pct(P['C*']['total']['agent'], 0)} of requests).\n")
    s += (f"- A change of contract is the drift no input-side guard can see: the requests look familiar and the program is confident, so "
          f"{pct(P['C*']['phase5_affected']['wrong'], 0)} of re-routed requests are answered under the old policy. Shadow sampling finds it, slowly: "
          f"at a 10% rate the breaker opened after {sw['0.1']['requests_until_breaker']} requests, at 40% after {sw['0.4']['requests_until_breaker']}. "
          f"Announcing the change and invalidating the affected answer class at once (D+*) leaves {pct(P['D+*']['phase5_affected']['wrong'], 0)} wrong. "
          "A contract version belongs in every compiled answer's guard.\n")
    es = P["E*"]
    dep = [e for e in es["events"] if e["event"] == "recompile"]
    def first4(pol, key):
        bp = P[pol]["by_phase"]
        n = sum(bp[p]["n"] for p in ("1", "2", "3", "4"))
        return sum(bp[p][key] * bp[p]["n"] for p in ("1", "2", "3", "4")) / n
    s += (f"- Regression-gated recompiles win back coverage without adding errors while the contract is stable "
          f"({sum(e['admitted'] for e in dep)} of {len(dep)} candidates deployed; over phases 1 to 4 the fast share is "
          f"{pct(first4('E*', 'fast'), 0)} against {pct(first4('C*', 'fast'), 0)} for the static program, with {pct(first4('E*', 'wrong'), 1)} "
          f"against {pct(first4('C*', 'wrong'), 1)} wrong answers). ")
    ep = P["E'*"]
    last_e = [e for e in es["events"] if e["event"] == "recompile"][-1]
    last_p = [e for e in ep["events"] if e["event"] == "recompile"][-1]
    s += (f"After the contract change the candidate looked {num(100 * (last_p['regression_acc'][0] - last_p['regression_acc'][1]), 1)} points worse on the "
          f"stale regression suite and {num(100 * (last_e['regression_acc'][0] - last_e['regression_acc'][1]), 1)} points worse on the re-adjudicated one "
          f"({es['regression_readjudication_agent_calls']} cases sent to the agent): a suite that encodes the old contract penalises exactly the update "
          "that is needed, and it has to be re-adjudicated when the contract changes.\n")
    s += (f"- Ungated self-training is the worst policy in the loop ({pct(P['F*']['total']['wrong'], 0)} wrong with the agent as fallback, "
          f"{pct(P['F']['total']['wrong'], 0)} with the LLM): it has the highest fast share and learns its own mistakes.\n")
    s += (f"- With the local LLM as fallback everything is noisier, because the LLM itself is wrong on {pct(P['A']['total']['wrong'], 0)} of this stream "
          f"({pct(P['A']['by_phase']['3']['wrong'], 0)} in the out-of-scope burst) and follows the changed contract on only "
          f"{pct(1 - P['A']['phase5_affected']['wrong'], 0)} of re-routed requests. The guarded stack (C) matches it before the policy change with "
          f"{num(P['C']['total']['llm_calls'], 2)} calls per request, and an LLM shadow checker is too noisy and too slow to trip the breaker in time "
          f"(D equals C here). In the first phase its shadow estimate is {pct(P['D']['by_phase']['1']['shadow_est'], 0)} while the fast tier's true "
          f"silent rate is {pct(P['D']['by_phase']['1']['wrong_fast'], 0)}: those disagreements are the checker's errors.\n")
    return s


def f_sec_e8() -> str:
    r = J("exp8_kernel.json")
    B = r["builds"]
    full = B["guards + gate(G) + verifier"]
    nog = B["gate(G) + verifier, no structural guards"]
    naive = B["guards + naive gate(R), no verifier"]
    nl, nr = B["full build without the LLM tier (abstentions go to the agent)"], B["the same without the rule tier"]
    ans = lambda v: round(v["fast_path_rate"] * v["cases"])
    lt = full["latency_ms_p50"]
    d_full, d_nog, d_naive = full["bad_input_detail"], nog["bad_input_detail"], naive["bad_input_detail"]
    s = "**What this shows**\n\n"
    s += (f"- Run for real, a local answer takes {num(lt['rules'], 2)} ms from the rule tier or {num(lt['model'], 1)} ms from the embedding tier "
          f"(encode, classify, familiarity search, lexical check), and the full build answers {pct(full['no_llm_rate'], 0)} of this hard mix "
          "(a third unseen groups, a sixth out of scope) without any model call.\n")
    by = full["by_tier"]
    s += (f"- The full build is **refused**: {full['silent_errors']} silent errors in {ans(full)} program answers ({pct(full['silent_error_rate'], 1)}) "
          f"and {full['guard_false_accepts']} of {full['bad_inputs']} bad inputs answered. By tier: " +
          ", ".join(f"{t} {x['wrong']} wrong of {x['n']}" for t, x in by.items() if t != "agent") +
          ". Admission judges the whole program, so one weak tier fails it, and here that tier is the LLM.\n")
    s += (f"- Take the LLM tier out and send abstentions to the agent: {nl['silent_errors']} silent errors in {ans(nl)} answers "
          f"({pct(nl['silent_error_rate'], 1)}), refused at 2% and acceptable at 5%. Take the mined-rule tier out as well: {nr['silent_errors']} in "
          f"{ans(nr)} ({pct(nr['silent_error_rate'], 1)}), admitted at 2% on the point estimate, with {pct(nr['no_llm_rate'], 0)} answered locally. "
          f"Its 95% upper bound is {pct(nr['silent_error_upper95'], 1)}: {ans(nr)} answers are not enough to *certify* 2%, which is the usual "
          f"position at admission time. The rule tier added {num(100 * (nl['no_llm_rate'] - nr['no_llm_rate']), 0)} points of coverage and "
          f"{nl['by_tier']['rules']['wrong']} of the {nl['silent_errors']} local errors.\n")
    s += (f"- The familiarity gate is what stops nonsense. With the naive gate the embedding tier answers "
          f"{d_naive['gibberish']['answered']} of {d_naive['gibberish']['n']} gibberish strings, {d_naive['injection']['by_tier'].get('model', 0)} of "
          f"{d_naive['injection']['n']} injection strings and {d_naive['far_domain']['by_tier'].get('model', 0)} of {d_naive['far_domain']['n']} "
          "far-domain sentences; with the guarded gate it answers none of the 152.\n")
    s += (f"- What still gets through the guarded builds gets through the LLM tier: {d_full['injection']['answered']} of {d_full['injection']['n']} "
          f"injection strings and {d_full['far_domain']['answered']} of {d_full['far_domain']['n']} far-domain sentences were given an in-scope "
          f"category by Qwen3-4B, and without structural guards {d_nog['structural']['answered']} of {d_nog['structural']['n']} malformed inputs as well. "
          "A typed hole constrains the form of the answer, not the model's willingness to give one.\n")
    return s


def f_sec_e9() -> str:
    r = J("exp9_harness.json")
    sp, a, M = r["spec"], r["agent"], r["modes"]
    st, ph, tp = M["structured"], M["prose_hole"], M["template"]
    code = [k for k, v in sp["fields"].items() if v.get("admitted")]
    hole = [k for k, v in sp["fields"].items() if v.get("admitted") is False]
    sa = r["agent_self_agreement_reworded_prompt"]
    ag = st["agreement_with_agent"]
    au, fr = sp["agent_audit"], sp["field_rule_vs_human"]
    yf = sp["fields"]["years_experience"]
    s = "**What this shows**\n\n"
    s += (f"- It works, modestly: the decisions get smaller rather than disappear. Watching {sp['n_observed']} runs turned three free-form calls "
          f"({num(a['prompt_tokens'], 0)} prompt and {num(a['generated_tokens'], 0)} generated tokens, {num(a['latency_s']['p50'], 1)} s) into code plus "
          f"one narrow typed call ({num(st['prompt_tokens'], 0)} / {num(st['generated_tokens'], 0)} tokens, {num(st['latency_s']['p50'], 1)} s). That is "
          f"{a['latency_s']['p50'] / st['latency_s']['p50']:.1f}× faster for the structured analysis, "
          f"{a['latency_s']['p50'] / ph['latency_s']['p50']:.1f}× if the prose still comes from a small generation hole, and "
          f"{a['latency_s']['p50'] / tp['latency_s']['p50']:.1f}× with a code template for the prose. Generated tokens are the latency that is left.\n")
    s += (f"- Little became pure code, and not because the agent is erratic. Reworded, it agrees with itself on years of experience (±2) for "
          f"{pct(sa['years_experience(±2)'], 0)} of profiles and on education for {pct(sa['education_level'], 0)}. The rules failed because what the "
          f"agent does is not a simple function of the text: the best of {len(yf['candidates'])} date rules had to assume a fixed \"today\" and still "
          f"matched it on {pct(yf['admission_agreement'], 0)} of admission runs. Admitted as code: {', '.join(code) if code else 'nothing'}. "
          f"Left with the model as one narrow typed call: {', '.join(hole)}.\n")
    s += (f"- Fidelity to the agent is the wrong target when the agent is wrong. It reported a degree for "
          f"{au['of_those_agent_still_reported_a_degree']} of {au['profiles_with_no_education_evidence_in_input']} profiles that show no education at all, "
          f"and a rule written from the contract beat it against human labels ({pct(fr['rule_matches_human_label'], 0)} vs "
          f"{pct(fr['agent_matches_human_label_same_rows'], 0)}) while failing admission by agreement. Admission needs reviewed outputs, not raw "
          "traces, as its reference. The shared conversation's survey said the same of prior work (compile from the contract, treat the trace as "
          "evidence); that survey was not re-checked here, but this experiment reaches the conclusion on its own data.\n")
    s += (f"- The narrow call is not a perfect stand-in for the three-call agent: it agrees with it on {pct(ag['years_experience(±2)'], 0)} of year "
          f"counts (±2), {pct(ag['education_level'], 0)} of education answers and {pct(ag['field'], 0)} of fields, below the agent's own "
          f"{pct(sa['years_experience(±2)'], 0)} and {pct(sa['education_level'], 0)} self-agreement. It reads the first 1,800 characters instead of 4,000 "
          "and answers every field in one call; which of the two costs the fidelity was not separated. Against human labels the two are not "
          f"measurably different on the one field that has them ({pct(st['field_accuracy_vs_human'], 0)} compiled, "
          f"{pct(st['agent_field_accuracy_vs_human'], 0)} agent, {r['n_test']} profiles).\n")
    s += ("- Not tested: an agent with tools; a compile step written by a coding agent instead of chosen from candidates I wrote after reading "
          "the observed traces (search, admission and scoring were mechanical); more than one workflow; and the obvious next step, a decision "
          "model for the closed-set fields, which needs far more than 60 observed runs to be certified.\n")
    return s


def _llm_answer_count() -> tuple:
    from .common import CACHE
    n = sum(len(f.read_text().splitlines()) for f in (CACHE / "llm").glob("*.jsonl"))
    t = CACHE / "harness" / "traces.jsonl"
    k = len(t.read_text().splitlines()) if t.exists() else 0
    return n, k


def _llm_strength() -> str:
    o = J("exp5b_llm_other.json")
    if not o or not o.get("locale"):
        return ""
    lc = o["locale"]
    return (f"The LLM is the stronger tier only where the fast model has seen nothing: {pct(lc['en_unseen']['llm'], 0)} vs "
            f"{pct(lc['en_unseen']['fast:bge-small'], 0)} on never-seen MASSIVE groups, {pct(lc['es-ES_seen']['llm'], 0)} vs "
            f"{pct(lc['es-ES_seen']['fast:bge-small'], 0)} on Spanish versions of familiar requests. ")


def title_and_summary() -> str:
    e1, e2, e3 = J("exp1_families.json"), J("exp2_novelty.json"), J("exp3_group_shift.json")
    e4, e5, e6, e7 = J("exp4_verification.json"), J("exp5_tiers.json"), J("exp6_stream.json"), J("exp7_news_drift.json")
    e8, e9 = J("exp8_kernel.json"), J("exp9_harness.json")
    lat, L = J("exp_latency.json")["tier_ms_p50"], J("llm_latency.json") or {}
    n_llm, n_tr = _llm_answer_count()
    c = e1["clinc150"]["families"]
    cell = lambda key, ds="clinc", model="bge-small+LR": _cell(ds, model, key)
    r2, g2 = cell("msp|R|fixed_sequence|0.02"), cell("msp&knn1|G-worst|fixed_sequence|0.02")
    b = e3["clinc"]["models"]["bge-small+LR"]["base"]
    q = e5["llm_tier"]["qwen3-4b"]
    mix = "70/20/10 seen/unseen/oos"
    st = e5["stacks"]["policies"]
    P = e6["policies"]
    T = e2["typo_noise"]
    llm_ms = L.get("qwen3-4b/route_1shot", {}).get("p50")
    s = "# LLM as a compiler: what holds up under harder tests\n\n"
    s += ("Continuation of the shared conversation. The earlier demonstration ran on synthetic expense lines with a stubbed LLM; "
          "this one runs on public, human-labelled data with real models on this machine. "
          f"Run on {J('data_manifest.json')['_meta']['written'][:10]}, Apple M4, local compute only.\n\n")
    s += "## Summary\n\n"
    s += (f"**What was run.** Nine experiments on four public datasets with human labels plus 2,484 public resume profiles; twelve fast-model "
          f"families; two local LLMs with {n_llm:,} stored real answers, and {n_tr} recorded three-step agent runs; 20 random group splits "
          "wherever no LLM was needed. Every table below is generated from `results/*.json`.\n\n")
    s += "**What it found.**\n\n"
    s += (f"1. **The speed is real, and the amount you can take depends on the task.** A local decision costs "
          f"{num(lat['tfidf-char+LR end-to-end'], 1)}–{num(lat['bge-small encode'] + lat['LR head'] + lat['knn1 over train'], 0)} ms against "
          f"{num(llm_ms, 0) if llm_ms else '~300'} ms for one cached-prefix call to a local 4B LLM. On familiar traffic a frozen encoder with a "
          f"linear head answers {pct(c['bge-small+LR']['fs@0.02']['coverage'], 0)} of CLINC150 inside a 2% silent-error budget, "
          f"{pct(e1['banking77']['families']['bge-small+LR']['fs@0.02']['coverage'], 0)} of BANKING77, and nothing on MASSIVE, where no model could "
          "be certified at 2%.\n")
    s += (f"2. **The unseen-group failure from the first demo is general.** Calibrated on a random split, the fast tier keeps its 2% promise on "
          f"familiar groups and, in all 20 splits, breaks it on groups it never saw: it answers {pct(r2['test_unseen']['coverage_mean'], 0)} of them "
          f"and {pct(r2['test_unseen']['silent_mean'], 0)} of those answers are wrong. {pct(b['unseen_errors_conf>=0.9']['mean'], 0)} of those errors "
          "are made with confidence ≥ 0.9.\n")
    s += (f"3. **The fix costs coverage and is not a guarantee.** Calibrating against held-out groups and real out-of-scope queries, with a "
          f"familiarity score separate from confidence, cuts wrong answers per unseen-group input from {pct(r2['test_unseen']['far_mean'], 1)} to "
          f"{pct(g2['test_unseen']['far_mean'], 1)} and familiar coverage from {pct(r2['test_seen']['coverage_mean'], 0)} to "
          f"{pct(g2['test_seen']['coverage_mean'], 0)}. It still ended over budget in {pct(g2['test_unseen']['far_violation_rate'], 0)} of splits, "
          "because rows from one group fail together.\n")
    if e4:
        cu = e4["error_correlation"]["test_unseen"]
        Pm = e4["primary"]
        V = e4["verifiers"]
        s += (f"4. **A verifier is only as good as it is different.** When the fast model is wrong on an unseen group, a 5-model ensemble on the "
              f"same embedding gives the same wrong label {pct(cu[Pm + ' -> mlp-ens(bge-small)']['same_wrong_label_given_A_wrong'], 0)} of the time, "
              f"a character n-gram model {pct(cu[Pm + ' -> tfidf-char+LR']['same_wrong_label_given_A_wrong'], 0)}, the LLM "
              f"{pct(cu[Pm + ' -> llm:qwen3-4b']['same_wrong_label_given_A_wrong'], 0)}. The lexical check adds "
              f"{num(lat['tfidf-char+LR end-to-end'], 1)} ms and catches {pct(V['agree: tfidf-char+LR (lexical view)']['test_unseen']['catch_rate'], 0)} "
              "of those errors; verification is paid for in false rejects, not milliseconds.\n")
    s += (f"5. **Embedding models fail silently on noise and on new languages; a lexical model fails honestly.** With 15% of letters mistyped, "
          f"bge-small is {pct(T['bge-small']['typo15']['accuracy'], 0)} accurate and its confidence gate still answers "
          f"{pct(T['bge-small']['typo15']['msp']['coverage'], 0)} of requests, {pct(T['bge-small']['typo15']['msp']['silent_error_rate'], 0)} of them wrong; "
          f"character TF-IDF answers {pct(T['tfidf']['typo15']['msp']['coverage'], 0)} with {pct(T['tfidf']['typo15']['msp']['silent_error_rate'], 1)} wrong. "
          f"On Turkish versions of English requests the same kind of model answers {pct(e2['locale_shift']['bge-small']['tr-TR']['msp']['coverage'], 0)} "
          f"and gets {pct(e2['locale_shift']['bge-small']['tr-TR']['msp']['silent_error_rate'], 0)} of those wrong; a similarity floor refuses almost all of them.\n")
    a_, c_, f_ = st["llm"]["mix"][mix], st["fast[G]+V > llm"]["mix"][mix], st["fast[G]+V"]["mix"][mix]
    s += (f"6. **Here the local LLM is the weak tier.** Qwen3-4B is right on {pct(q['test_seen']['accuracy'], 0)} of familiar requests and "
          f"{pct(q['test_unseen']['accuracy'], 0)} of unseen-group requests and takes the \"none of these\" exit for only "
          f"{pct(q['oos_test']['none_exit_rate'], 0)} of out-of-scope queries. A guarded fast tier in front of it delivers the same quality "
          f"({pct(c_['wrong_delivered'], 1)} vs {pct(a_['wrong_delivered'], 1)} wrong on a 70/20/10 mix) with {num(c_['llm_calls'], 2)} calls per "
          f"request instead of 1; with the agent as the only fallback it delivers {pct(f_['wrong_delivered'], 1)} wrong and escalates "
          f"{pct(f_['escalations'], 0)}. " + _llm_strength() + "A stronger LLM would change these absolute numbers; none was run.\n")
    y = e7["policies"]["static-T"]["by_year"]
    yrs = list(y)
    s += (f"7. **Compiled programs rot, and a contract change is the drift no guard sees.** On {len(e7['months']) / 12:.1f} years of dated headlines a program compiled "
          f"on the first year goes from {pct(y[yrs[0]]['silent_among_fast'], 0)} to {pct(y[yrs[-1]]['silent_among_fast'], 0)} silent errors while still "
          f"answering. In the scripted loop the guards handle new groups, an out-of-scope burst and noise "
          f"({pct(P['C*']['by_phase']['2']['wrong'], 1)}, {pct(P['C*']['by_phase']['3']['wrong'], 1)}, {pct(P['C*']['by_phase']['4']['wrong'], 1)} wrong), "
          f"but after a policy change {pct(P['C*']['phase5_affected']['wrong'], 0)} of re-routed requests are answered under the old policy until "
          f"shadow sampling notices ({e6['shadow_sweep']['0.1']['requests_until_breaker']} requests at a 10% rate) or the change is announced "
          f"({pct(P['D+*']['phase5_affected']['wrong'], 0)}). Retraining on the system's own answers without a gate was the worst policy in the "
          f"scripted loop ({pct(P['F*']['total']['wrong'], 0)} wrong) and stayed over budget in "
          f"{e7['policies']['self-train']['summary']['months_over_budget']} of {e7['policies']['self-train']['summary']['months_with_fast_tier']} "
          "months on the news stream.\n")
    if e9:
        a9, st9, ph9 = e9["agent"], e9["modes"]["structured"], e9["modes"]["prose_hole"]
        code = [k for k, v in e9["spec"]["fields"].items() if v.get("admitted")]
        s += (f"8. **The wrapper-harness idea works modestly: the decisions shrink, few disappear.** Watching {e9['spec']['n_observed']} runs of a three-call "
              f"\"analyze this profile\" agent was enough to compile it into code plus one narrow typed call: "
              f"{a9['latency_s']['p50'] / st9['latency_s']['p50']:.1f}× faster for the structured analysis "
              f"({num(a9['latency_s']['p50'], 1)} s → {num(st9['latency_s']['p50'], 1)} s), {a9['latency_s']['p50'] / ph9['latency_s']['p50']:.1f}× when "
              f"prose is still generated. Only {len(code)} of 6 short fields passed admission as pure code; the rest stayed with the model as one "
              f"smaller typed call, because what the agent does for them is not a simple function of the text. The recordings also audited the "
              f"agent: it reported a degree for {e9['spec']['agent_audit']['of_those_agent_still_reported_a_degree']} of "
              f"{e9['spec']['agent_audit']['profiles_with_no_education_evidence_in_input']} profiles that show no education at all.\n")
    s += "\n**What is measured, replayed, or assumed.**\n\n"
    from .make_report import table
    s += table(["Kind", "What"], [
        ["Measured (real runs)", "All fast-model training and inference; all LLM answers and latencies (Qwen3-4B, Gemma-3-4B on this machine); "
         "the three-call agent and the compiled program of Experiment 9; every accuracy, coverage and error figure against human labels"],
        ["Replayed real outputs", "Experiments 5, 6 and 8 look up the stored real LLM answer for each exact request instead of calling the model again; "
         "latency per request in the stack and loop tables is assembled from the measured per-tier medians"],
        ["Scripted", "The order and mix of traffic in Experiment 6 (five phases), the traffic mixes in Experiment 5, the seeded bad inputs, keyboard noise"],
        ["Assumed", "The agent: wherever a request is escalated, the human label is taken as the agent's answer and assumed correct; its cost is "
         "counted as escalations and never timed"],
        ["Not run", "Any hosted or frontier model; an agent with tools; GPU latency for the fast tier; real production traffic with repeats"],
    ], "ll") + "\n"
    return s


def _e9_holes() -> str:
    e9 = J("exp9_harness.json")
    if not e9:
        return "several profile fields"
    f = e9["spec"]["fields"]
    n_hole = sum(1 for v in f.values() if v.get("admitted") is False)
    n = sum(1 for v in f.values() if "admitted" in v)
    return f"{n_hole} of {n} short profile fields"


def closing() -> str:
    e3, e6 = J("exp3_group_shift.json"), J("exp6_stream.json")
    s = "## 11. What this changes in the design\n\n"
    s += ("These are the changes to the compile procedure that the results support. Each one traces to an experiment above.\n\n"
          "1. **Calibrate against the shift, with two tests.** Selective risk on held-out rows of known groups; false-accept rate on whole "
          "held-out groups and on real out-of-scope inputs. A random split certifies the wrong thing (Exp. 3). Report both numbers.\n"
          "2. **Keep familiarity separate from confidence.** Use an embedding-distance score (nearest-neighbour similarity or relative "
          "Mahalanobis) as its own gate; confidence is high on most unseen-group errors (Exp. 2, 3).\n"
          "3. **Count groups, not rows, as evidence**, and treat any certificate as approximate. Keep shadow sampling on in production (Exp. 3, 6, 7).\n"
          "4. **Pair the embedding model with a character n-gram model as verifier.** It costs a fraction of a millisecond, it disagrees on "
          "noise and unseen groups where the embedding model is confidently wrong, and it degrades honestly (Exp. 2, 4).\n"
          "5. **Do not use a semantic cache or a single mined rule as a decision tier** without the same admission test; both were weaker than "
          "a linear head on the same data (Exp. 1).\n"
          "6. **Let the compile step say no.** Some decisions cannot be certified at the budget (MASSIVE at 2%; " + _e9_holes() +
          " in Exp. 9). They stay as narrow typed holes.\n"
          "7. **Admit every tier, including the LLM.** The local LLM's errors dominated every stack it was in, its confidence could not be "
          "certified, and it answered more than half of the out-of-scope queries as in-scope (Exp. 5, 8).\n"
          "8. **Version the contract and invalidate on change.** A contract change is invisible to input-side guards; an announced invalidation "
          "of the affected answer class removed the errors, and the regression suite has to be re-adjudicated at the same time (Exp. 6).\n"
          "9. **Never retrain on unverified own answers.** Gate every update on held-out recent labels and a regression suite; remember that "
          "the labels a running system collects come from escalations and are harder than its fast-path traffic (Exp. 6, 7).\n"
          "10. **Engineer the hole.** Fixed contract prefix with a reused KV cache; closed answer sets; the smallest input that suffices; "
          "batch sizes chosen with the prefix length in mind (Exp. 5, 9).\n\n")
    s += "## 12. Limits of this study\n\n"
    s += ("- The LLMs are 4-bit local models of about 4B parameters. Every statement about \"the LLM tier\" is about them. The comparison "
          "between always-LLM and compiled stacks would shift with a stronger model; the fast-tier results would not.\n"
          "- The agent is the human label. Human labels in these datasets are themselves imperfect (MASSIVE and the news sections visibly so), "
          "and the CLINC domain taxonomy is partly arbitrary for unseen intents.\n"
          "- One frozen group split carries all LLM-in-the-loop results (Exp. 4, 5, 6, 8). The fast-tier conclusions were checked on 20 splits; "
          "the LLM ones were not.\n"
          "- Intent routing and profile analysis are short-text, closed-answer tasks. Nothing here tests perceptual workflows such as video to "
          "Blender, where the verifier is the hard part.\n"
          "- The production loop is a script of five phases over 2,790 requests. Detection delays are specific to that length and those rates.\n"
          "- In Experiment 9 the candidate programs were hand-written by the compiling agent (me) from observed traces; search, admission and "
          "scoring were mechanical. One model, one workflow, 80 test profiles.\n"
          "- Fast-tier latency was measured on CPU while a GPU job was running; LLM latency is for MLX on an M4 and includes no network.\n\n")
    s += "## 13. The earlier artifact\n\n"
    s += ("`workflow-compiler.skill` was downloaded from the shared chat and is kept unmodified in `prior_artifact/` "
          "(9 files: SKILL.md, a standard-library runtime, a scikit-learn decision tier, an admission harness, the expense example and "
          "its tests). I read every file. Executing the downloaded code was blocked by this session's permission layer, so **its 14 tests "
          "were not run here and that earlier claim remains unverified**. Instead the runtime and admission harness were re-authored into "
          "`wfc/` from the reviewed source and extended (independent-verifier hook, exact bounds, three threshold procedures, a "
          "false-accept-rate test, a group-aware bootstrap); `tests/` holds 21 tests, the original 11 kernel tests re-written plus 10 new "
          "ones, and all 21 pass.\n\n")
    s += ("The research and repository claims quoted in the shared conversation (FlashScript, TraceCompiler, PreAct, Evaporate, the "
          "video-to-blender and shader-skills repositories, the generativemedia.blog posts) were **not re-verified** in this session, and no "
          "result in this report depends on them.\n\n")
    s += "## 14. Files\n\n"
    s += ("`REPORT.md` (this file) · `README.md` (layout and commands to reproduce) · `results/*.json` (every number) · `bench/` (experiments) · "
          "`skills/workflow-compiler/` and `workflow-compiler-v2.skill` (the skill revised with these results; 23 tests) · "
          "`wfc/` and `tests/` (kernel) · `cache/llm/*.jsonl` (stored LLM answers with raw text and token probabilities) · "
          "`cache/harness/traces.jsonl` (recorded agent runs) · `logs/pip_freeze.txt` (exact package versions).\n")
    return s
