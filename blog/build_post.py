"""Write the blog post into the blog checkout: article entry + media manifest.

Usage:  .venv/bin/python blog/build_post.py

The prose is fixed here; every number in it is formatted from results/*.json so the post
cannot drift from the report. Re-running replaces the existing entry (same articleId) and
keeps its original publication time.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import math
import os
import pathlib
import struct

ROOT = pathlib.Path(__file__).resolve().parents[1]
RES = ROOT / "results"
BLOG = ROOT / "blog" / "gokaygokay-blog"
ARTICLES = BLOG / "src" / "content" / "blog-articles.json"
MANIFEST = BLOG / "src" / "content" / "blog-media-manifest.json"
PUBLIC = BLOG / "public"

ARTICLE_ID = "blog-agent-to-program"
TITLE = "Agent 2 Program: What Survives When You Compile an LLM Workflow"
SLUG = "agent-2-program-what-survives-when-you-compile-an-llm-workflow"
SOURCE = "https://github.com/gokayfem/agent-to-program"
TAGS = ["Agents", "Research", "Creative Engineering", "Open Source"]
EXCERPT = ("An agent solves a task once and a program handles it after that. I tested the idea on real data with local "
           "models: where the speed is real, where it fails silently, and what a wrapper harness actually saves.")
M = "/articles/agent-to-program-"


def J(name):
    return json.loads((RES / name).read_text())


def pct(x, d=0):
    return f"{100 * x:.{d}f}%"


# --------------------------------------------------------------------------- numbers
def numbers() -> dict:
    lat, L = J("exp_latency.json")["tier_ms_p50"], J("llm_latency.json")
    e1, e2, e3 = J("exp1_families.json"), J("exp2_novelty.json"), J("exp3_group_shift.json")
    e4, e5, e6, e7 = J("exp4_verification.json"), J("exp5_tiers.json"), J("exp6_stream.json"), J("exp7_news_drift.json")
    e8, e9 = J("exp8_kernel.json"), J("exp9_harness.json")
    c = e3["clinc"]["models"]["bge-small+LR"]
    r, g = c["cells"]["msp|R|fixed_sequence|0.02"], c["cells"]["msp&knn1|G-worst|fixed_sequence|0.02"]
    T = e2["typo_noise"]
    cu = e4["error_correlation"]["test_unseen"]
    same = lambda b: cu[f"bge-small+LR -> {b}"]["same_wrong_label_given_A_wrong"]
    q = e5["llm_tier"]["qwen3-4b"]
    mix = "70/20/10 seen/unseen/oos"
    st = e5["stacks"]["policies"]
    P = e6["policies"]
    y = e7["policies"]["static-T"]["by_year"]
    yrs = list(y)
    full = e8["builds"]["guards + gate(G) + verifier"]["bad_input_detail"]["injection"]
    sp, a, m = e9["spec"], e9["agent"], e9["modes"]
    n_llm = sum(len(f.read_text().splitlines()) for f in (ROOT / "cache" / "llm").glob("*.jsonl"))
    return {
        "n_llm": f"{n_llm:,}",
        "fast_lo": f"{lat['tfidf-char+LR end-to-end']:.1f}",
        "fast_hi": f"{lat['bge-small encode'] + lat['LR head'] + lat['knn1 over train']:.0f}",
        "llm_ms": f"{L['qwen3-4b/route_1shot']['p50']:.0f}",
        "llm_nocache_s": f"{L['qwen3-4b/route_1shot_no_prefix_cache']['p50'] / 1000:.1f}",
        "cov_clinc": pct(e1["clinc150"]["families"]["bge-small+LR"]["fs@0.02"]["coverage"]),
        "cov_bank": pct(e1["banking77"]["families"]["bge-small+LR"]["fs@0.02"]["coverage"]),
        "splits": e3["clinc"]["splits"],
        "r_ans": pct(r["test_unseen"]["coverage_mean"]), "r_sil": pct(r["test_unseen"]["silent_mean"]),
        "r_viol": round(r["test_unseen"]["violation_rate"] * e3["clinc"]["splits"]),
        "conf9": pct(c["base"]["unseen_errors_conf>=0.9"]["mean"]),
        "r_far": pct(r["test_unseen"]["far_mean"], 1), "g_far": pct(g["test_unseen"]["far_mean"], 1),
        "r_cov": pct(r["test_seen"]["coverage_mean"]), "g_cov": pct(g["test_seen"]["coverage_mean"]),
        "g_viol": pct(g["test_unseen"]["far_violation_rate"]),
        "ty_acc": pct(T["bge-small"]["typo15"]["accuracy"]), "ty_ans": pct(T["bge-small"]["typo15"]["msp"]["coverage"]),
        "ty_wrong": pct(T["bge-small"]["typo15"]["msp"]["silent_error_rate"], 1),
        "tf_ans": pct(T["tfidf"]["typo15"]["msp"]["coverage"]), "tf_wrong": pct(T["tfidf"]["typo15"]["msp"]["silent_error_rate"], 1),
        "same_ens": pct(same("mlp-ens(bge-small)")), "same_tf": pct(same("tfidf-char+LR")), "same_llm": pct(same("llm:qwen3-4b")),
        "tf_ms": f"{lat['tfidf-char+LR end-to-end']:.1f}",
        "tf_catch": pct(e4["verifiers"]["agree: tfidf-char+LR (lexical view)"]["test_unseen"]["catch_rate"]),
        "llm_seen": pct(q["test_seen"]["accuracy"]), "llm_unseen": pct(q["test_unseen"]["accuracy"]),
        "llm_none": pct(q["oos_test"]["none_exit_rate"]),
        "inj": full["answered"], "inj_n": full["n"],
        "stack_wrong": pct(st["fast[G]+V > llm"]["mix"][mix]["wrong_delivered"], 1),
        "llm_wrong": pct(st["llm"]["mix"][mix]["wrong_delivered"], 1),
        "stack_calls": f"{st['fast[G]+V > llm']['mix'][mix]['llm_calls']:.2f}",
        "agent_wrong": pct(st["fast[G]+V"]["mix"][mix]["wrong_delivered"], 1),
        "agent_esc": pct(st["fast[G]+V"]["mix"][mix]["escalations"]),
        "news_first": pct(y[yrs[0]]["silent_among_fast"]), "news_last": pct(y[yrs[-1]]["silent_among_fast"]),
        "news_y0": yrs[0], "news_y1": yrs[-1], "news_new": pct(y[yrs[-1]]["new_section_share"]),
        "n_req": f"{e6['n_requests']:,}",
        "c2": pct(P["C*"]["by_phase"]["2"]["wrong"], 1), "c3": pct(P["C*"]["by_phase"]["3"]["wrong"], 1),
        "c4": pct(P["C*"]["by_phase"]["4"]["wrong"], 1), "naive": pct(P["B*"]["total"]["wrong"]),
        "pol": pct(P["C*"]["phase5_affected"]["wrong"]), "sh_req": e6["shadow_sweep"]["0.1"]["requests_until_breaker"],
        "ann": pct(P["D+*"]["phase5_affected"]["wrong"]),
        "n_obs": sp["n_observed"], "n_adm": sp["n_admission"], "n_test": e9["n_test"],
        "ag_s": f"{a['latency_s']['p50']:.1f}", "st_s": f"{m['structured']['latency_s']['p50']:.1f}",
        "pr_s": f"{m['prose_hole']['latency_s']['p50']:.1f}",
        "n_code": sum(1 for v in sp["fields"].values() if v.get("admitted")),
        "n_short": sum(1 for v in sp["fields"].values() if "admitted" in v),
        "st_pt": f"{m['structured']['prompt_tokens']:,.0f}", "ag_pt": f"{a['prompt_tokens']:,.0f}",
        "st_gt": f"{m['structured']['generated_tokens']:.0f}", "ag_gt": f"{a['generated_tokens']:.0f}",
        "au_none": sp["agent_audit"]["profiles_with_no_education_evidence_in_input"], "au_runs": sp["agent_audit"]["runs_audited"],
        "au_claim": sp["agent_audit"]["of_those_agent_still_reported_a_degree"],
        "ung": pct(next(iter(sp["fields"]["education_level"]["rejected_ungrounded"].values()))),
        "honest": pct(sp["fields"]["education_level"]["observed_agreement"]),
        "rule_h": pct(sp["field_rule_vs_human"]["rule_matches_human_label"]),
        "agent_h": pct(sp["field_rule_vs_human"]["agent_matches_human_label_same_rows"]),
    }


# --------------------------------------------------------------------------- blocks
def p(*parts):
    """A paragraph. Parts are strings or (text, href) tuples."""
    return {"type": "unstyled", "spans": [{"text": x, "styles": []} if isinstance(x, str) else
                                          {"text": x[0], "styles": [], "href": x[1]} for x in parts]}


def h2(text):
    return {"type": "header-two", "spans": [{"text": text, "styles": []}]}


def png_size(path: pathlib.Path):
    with open(path, "rb") as f:
        head = f.read(24)
    return struct.unpack(">II", head[16:24])


def image(name, caption):
    w, h = png_size(PUBLIC / (M + name).lstrip("/"))
    return {"type": "media", "media": [{"type": "image", "src": M + name, "width": w, "height": h}], "caption": caption}


def blocks(n: dict) -> list:
    return [
        p("Every time I hand an agent a job it has done before, it starts from zero."),
        p("It reads the input, plans, calls its tools, checks itself and writes the answer. The tenth run costs as much as the "
          "first, and most of what happens in the tenth run already happened in the first."),
        p("The idea I posted was simple. I don't want to change the model. I want a wrapper harness that watches the LLM do a "
          "task like \"analyze my profile\", learns the process, and turns it into a program. The next profile should be much "
          "faster, and the decisions left for the model should be smaller than the first time."),
        p("That is using an LLM as a compiler instead of a runtime. I asked Fable 5.2 in Claude Code to test it properly on my "
          "own machine: public datasets with human labels, real local models, and training, calibration and test data kept "
          f"apart. It designed nine experiments, ran them on an M4 and stored {n['n_llm']} real model answers along the way. The ",
          ("code, results and full report", SOURCE), " are open."),
        {"type": "media", "media": [{"type": "video", "src": M + "film.mp4", "poster": M + "film-poster.jpg"}],
         "caption": f"The film: what each tier costs, the unseen-group failure, a replay of {n['n_req']} real requests through the "
                    "production loop, and the harness result. Every number in it is read from the result files."},

        h2("A program is a ladder of cheaper answers"),
        p("Compiling does not remove the model. It moves it. Arithmetic, lookups and formatting become code. A judgment such as "
          "\"which department does this request belong to\" becomes a typed decision, answered by the cheapest tier that is sure: "
          "an exact cache, mined rules, a small decision model, then one narrow LLM call, and finally the full agent."),
        p(f"On my M4 a local decision takes between {n['fast_lo']} and {n['fast_hi']} milliseconds. One typed call to a local 4B "
          f"model takes {n['llm_ms']} ms, and only because the task contract sits in a cached prompt prefix. Re-reading the "
          f"contract on every call takes {n['llm_nocache_s']} seconds."),
        image("tier-latency.png", "Median time for one answer at each tier, on a log scale. The gap between the embedding model "
                                  "and the LLM call is where the speed comes from."),
        p(f"How much the fast tiers can take over depends on the task. With a 2% budget for silent errors, a small embedding "
          f"model could answer {n['cov_clinc']} of one assistant dataset and {n['cov_bank']} of a banking dataset. On a third "
          "dataset no model could be certified at 2% at all. A compile step has to be allowed to say: this decision stays with "
          "the model."),

        h2("Confidence is not familiarity"),
        p("The dangerous failure showed up when I held out whole groups of requests. The fast model was trained on some intents, "
          "then tested on intents it had never seen, with the same set of possible answers."),
        p(f"Calibrated the usual way, on a random split, it kept its 2% promise on familiar groups. On the never-seen groups it "
          f"answered {n['r_ans']} of requests and {n['r_sil']} of those answers were wrong. That happened in {n['r_viol']} of "
          f"{n['splits']} random splits. {n['conf9']} of those mistakes were made with a confidence of 0.9 or higher."),
        image("unseen-groups.png", "Left: wrong answers per request from groups the model never saw. Right: what the fix costs on "
                                   f"familiar traffic. Means over {n['splits']} random group splits."),
        p("The fix is to calibrate against the shift you expect. Hold whole groups out, test them separately, and add a "
          "familiarity score that does not depend on the model's confidence. I used the distance to the nearest training example. "
          f"That cut wrong answers on unseen groups from {n['r_far']} to {n['g_far']} of requests. It also cut the share of "
          f"familiar requests the fast tier answers from {n['r_cov']} to {n['g_cov']}, and it still went over budget in "
          f"{n['g_viol']} of the splits, because rows from one group fail together. It is an estimate, not a guarantee."),
        p(f"Typing noise showed the same problem from another side. With 15% of letters mistyped, the embedding model was "
          f"{n['ty_acc']} accurate and still answered {n['ty_ans']} of requests, {n['ty_wrong']} of them wrong. A plain character "
          f"n-gram model answered {n['tf_ans']} and stayed at {n['tf_wrong']} wrong. Its confidence dropped with its accuracy."),
        image("typing-noise.png", "The same test requests with more and more mistyped letters. Both thresholds were certified on "
                                  "clean data and never changed."),

        h2("A verifier has to fail differently"),
        p("A fast answer should not be its own verifier. But a second model trained on the same features is almost the same model."),
        p(f"When the fast model was wrong on a never-seen group, an ensemble of five networks on the same embedding gave the same "
          f"wrong answer {n['same_ens']} of the time. A character n-gram model did so {n['same_tf']} of the time, and a local LLM "
          f"{n['same_llm']}. The n-gram check costs {n['tf_ms']} ms and catches {n['tf_catch']} of those errors, so it became the "
          "default verifier."),
        image("same-wrong-answer.png", "How often each second opinion repeats the fast model's mistake on never-seen groups. "
                                       "The more differently a checker is built, the less it agrees with the error."),
        p(f"The local LLM was not a safe fallback either. Qwen3-4B was right on {n['llm_seen']} of familiar requests and "
          f"{n['llm_unseen']} of unseen-group requests. It took its \"none of these\" exit for only {n['llm_none']} of out-of-scope "
          f"queries, and it gave an in-scope answer to {n['inj']} of {n['inj_n']} prompt-injection strings."),
        p(f"Putting the guarded fast tier in front of it gave the same quality, {n['stack_wrong']} wrong against {n['llm_wrong']} "
          f"for always asking the LLM, with {n['stack_calls']} model calls per request instead of one. With the human label as the "
          f"fallback instead of the LLM, the same fast tier delivered {n['agent_wrong']} wrong answers and escalated {n['agent_esc']} "
          "of requests."),

        h2("Programs rot"),
        p(f"I compiled a headline router on one year of HuffPost headlines and let it run on the next five. In {n['news_y0']} it "
          f"was wrong on {n['news_first']} of what it answered. In {n['news_y1']} it was wrong on {n['news_last']}, and it was still "
          f"answering. By then {n['news_new']} of the headlines belonged to sections that did not exist when it was compiled."),
        image("five-years-of-headlines.png", "Month by month: the share of wrong answers among the headlines the static program "
                                             "still answers, and the share of traffic from sections created after it was compiled."),
        p(f"Then a scripted loop over {n['n_req']} held-out requests with four kinds of drift. Anything the program refused went "
          f"to the human label. The guarded program handled new "
          f"groups, a burst of out-of-scope requests and noisy input with {n['c2']}, {n['c3']} and {n['c4']} wrong answers. The "
          f"naive one was wrong on {n['naive']} of the whole stream."),
        p(f"The fourth drift was a policy change: requests about physical cards moved from one department to another. The inputs "
          f"looked familiar and the program was confident, so {n['pol']} of the re-routed requests were answered under the old "
          f"policy. Re-checking 10% of the fast answers took {n['sh_req']} requests to notice. Announcing the change, and switching "
          f"off the affected answer class at once, brought it to {n['ann']}."),
        image("production-loop.png", "Wrong answers per 100 requests across the five phases. The guards handle unfamiliar inputs. "
                                     "They cannot see a change of rules."),
        p("Two rules came out of this. A compiled answer has to carry the version of the contract it was compiled for. And a "
          "program must never retrain on its own unverified answers: that was the worst policy in the loop."),

        h2("The wrapper harness, tested"),
        p(f"This is the experiment closest to my original post. The agent analyzed public résumé profiles in three free-form "
          f"calls: extract the facts, assess them, write a summary. The harness recorded {n['n_obs']} runs. For every output field "
          f"it searched a small family of candidate programs, and admitted one only if it agreed with the agent on {n['n_adm']} "
          "other recorded runs."),
        p(f"On {n['n_test']} profiles it had never seen, the compiled version took {n['st_s']} seconds instead of {n['ag_s']}. "
          f"If the prose is still generated by the model, {n['pr_s']} seconds."),
        image("seconds-per-profile.png", "Seconds per profile for the three-call agent and the two compiled versions, on the same "
                                         "local model."),
        p(f"The decisions shrank more than they disappeared. Only {n['n_code']} of {n['n_short']} short fields became pure code. "
          f"The rest stayed with the model as one narrow, typed call that reads {n['st_pt']} tokens instead of {n['ag_pt']} and "
          f"writes {n['st_gt']} instead of {n['ag_gt']}."),
        p(f"The recordings also audited the agent. In {n['au_none']} of {n['au_runs']} runs the profile text showed no education "
          f"at all, and the agent still reported a degree in {n['au_claim']} of them. A rule that copied this habit agreed with the "
          f"agent on {n['ung']} of runs. The best honest rule agreed on {n['honest']}."),
        p(f"It went the other way too. A rule written from the task description matched the human label on {n['rule_h']} of the "
          f"profiles it answered. The agent matched on {n['agent_h']} of the same profiles. Judged by agreement with the agent, "
          "that rule was rejected."),
        p("So the trace is evidence, not the specification. Compile from the contract, and check against reviewed outputs "
          "wherever you have them."),

        h2("What this does not show"),
        p("The LLMs here are 4-bit local models of about 4B parameters. A larger hosted model would change the numbers for the "
          "LLM tier; the fast-tier results would stay. The last tier in these tests is the human label standing in for an agent. "
          "The harness was tested on one workflow, with candidate programs written by the coding agent after reading the recorded "
          "runs. And none of this touches perceptual work like video to Blender, where the verifier is the hard part."),
        p("I like where it landed. The idea survives with less magic than I hoped and more structure: a ladder of cheap answers, "
          "a gate that knows what it has not seen, a checker that is built differently, and a contract that outranks the trace."),
        p(("Code, the full report and the revised skill →", SOURCE)),
    ]


# --------------------------------------------------------------------------- write
def manifest_entry(local: str) -> dict:
    data = (PUBLIC / local.lstrip("/")).read_bytes()
    film = local.endswith(".mp4") or "poster" in local
    commit = os.environ.get("SOURCE_COMMIT")       # the agent-to-program commit that holds the scripts and results
    entry = {"local": local, "source": f"{SOURCE}/blob/{commit or 'main'}/blog/make_media.py"}
    if commit:
        entry["sourceCommit"] = commit
    return {**entry,
            "capturedAt": datetime.date.today().isoformat(),
            "method": ("Rendered frame by frame with matplotlib from the experiment result files and encoded once with ffmpeg (H.264)"
                       if local.endswith(".mp4") else
                       "Frame of the film rendered with matplotlib from the experiment result files" if film else
                       "Rendered with matplotlib from the experiment result files; not resized or recompressed"),
            "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def main():
    n = numbers()
    bl = blocks(n)
    content = "\n\n".join("".join(s["text"] for s in b["spans"]) for b in bl if b["type"] != "media")
    articles = json.loads(ARTICLES.read_text())
    old = next((a for a in articles if a["articleId"] == ARTICLE_ID), None)
    published = old["publishedAt"] if old else datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
    post = {"id": SLUG, "articleId": ARTICLE_ID, "title": TITLE, "excerpt": EXCERPT, "content": content,
            "date": published[:10], "publishedAt": published, "tags": TAGS, "featuredImage": M + "tier-latency.png",
            "readTime": f"{max(1, math.ceil(len(content.split()) / 220))} min", "sourceUrl": SOURCE, "publication": "blog",
            "blocks": bl}
    articles = [post] + [a for a in articles if a["articleId"] != ARTICLE_ID]
    ARTICLES.write_text(json.dumps(articles, indent=2, ensure_ascii=False) + "\n")

    used = {post["featuredImage"]}
    for b in bl:
        for m in b.get("media", []):
            used.add(m["src"])
            if m.get("poster"):
                used.add(m["poster"])
    manifest = [a for a in json.loads(MANIFEST.read_text()) if not a["local"].startswith(M)]
    manifest += [manifest_entry(u) for u in sorted(used)]
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(f"post: {len(content.split())} words, {post['readTime']}, {len(bl)} blocks, {len(used)} media, published {published}")


if __name__ == "__main__":
    main()
