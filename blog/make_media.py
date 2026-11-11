"""Figures and the short film for the blog post, drawn from results/*.json.

Usage:  .venv/bin/python blog/make_media.py [figures] [film] [stills]

Nothing here contains a typed-in result: every plotted value is read from the result files
of the experiments. Output goes to blog/gokaygokay-blog/public/articles/.

Chart conventions (dark surface, validated palette, thin marks, rounded data-ends, hairline
solid grid, selective direct labels, text in ink tokens) follow the dataviz skill; the three
categorical slots used were checked with its validator for the dark surface.
"""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyBboxPatch, PathPatch  # noqa: E402
from matplotlib.path import Path  # noqa: E402
from matplotlib.transforms import IdentityTransform  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = ROOT / "blog" / "gokaygokay-blog" / "public" / "articles"
PRE = "agent-to-program-"

SURFACE, PLANE = "#1a1a19", "#0d0d0d"
INK, INK2, MUTED = "#f0efec", "#c3c2b7", "#898781"
GRID, AXIS = "#2c2c2a", "#383835"
S1, S2, S3 = "#3987e5", "#d95926", "#199e70"       # categorical slots 1-3, dark steps (validated)
LIME = "#c4f400"                                      # the blog's accent; used for chrome, never for data series

plt.rcParams.update({
    "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": AXIS, "axes.facecolor": SURFACE, "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "xtick.major.size": 0, "ytick.major.size": 0, "xtick.major.pad": 5, "ytick.major.pad": 5,
    "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
})


def J(name):
    return json.loads((RES / name).read_text())


def new_fig(h_in: float, title: str, sub: str = "", w_in: float = 8.0, dpi: int = 200):
    fig = plt.figure(figsize=(w_in, h_in), dpi=dpi)
    top = 1 - 0.34 / h_in
    fig.text(0.055, top, title, fontsize=13.5, fontweight="bold", color=INK, va="top", ha="left")
    if sub:
        fig.text(0.055, top - 0.30 / h_in, sub, fontsize=9.6, color=INK2, va="top", ha="left")
    return fig


def bar(ax, a, b, lo, hi, color, end="top", r=9.0):
    """A bar with a rounded data-end and a square baseline. (a, b) span the thickness, (lo, hi)
    the length; `end` says where the data-end is. Drawn in display pixels after layout."""
    fig = ax.figure
    fig.canvas.draw()
    if end == "top":
        (x0, y0), (x1, y1) = ax.transData.transform([(a, lo), (b, hi)])
        r = min(r, abs(x1 - x0) / 2, abs(y1 - y0))
        v = [(x0, y0), (x0, y1 - r), (x0, y1), (x0 + r, y1), (x1 - r, y1), (x1, y1), (x1, y1 - r), (x1, y0), (x0, y0)]
    else:  # right
        (x0, y0), (x1, y1) = ax.transData.transform([(lo, a), (hi, b)])
        r = min(r, abs(y1 - y0) / 2, abs(x1 - x0))
        v = [(x0, y0), (x1 - r, y0), (x1, y0), (x1, y0 + r), (x1, y1 - r), (x1, y1), (x1 - r, y1), (x0, y1), (x0, y0)]
    codes = [Path.MOVETO, Path.LINETO, Path.CURVE3, Path.CURVE3, Path.LINETO, Path.CURVE3, Path.CURVE3, Path.LINETO, Path.CLOSEPOLY]
    fig.add_artist(PathPatch(Path(v, codes), facecolor=color, edgecolor="none", transform=IdentityTransform(), zorder=3))


def legend(fig, x, y, items, fontsize=9.6, gap=0.21):
    """Swatch + name pairs in ink; identity never rests on color alone."""
    for name, color in items:
        hh = 0.014 * fig.get_figwidth() / fig.get_figheight()
        fig.add_artist(FancyBboxPatch((x, y - hh / 2 - 0.002), 0.014, hh,
                                      boxstyle="round,pad=0,rounding_size=0.003", facecolor=color, edgecolor="none",
                                      transform=fig.transFigure))
        fig.text(x + 0.022, y, name, fontsize=fontsize, color=INK2, va="center", ha="left")
        x += gap


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / (PRE + name)
    fig.savefig(path, dpi=fig.dpi)
    plt.close(fig)
    print("wrote", path.name)
    return path


# --------------------------------------------------------------------------- data
def tiers():
    t = J("exp_latency.json")["tier_ms_p50"]
    L = J("llm_latency.json")
    return [("Exact cache", t["exact cache"]),
            ("Mined rules", t["mined rules"]),
            ("Character n-gram model", t["tfidf-char+LR end-to-end"]),
            ("Embedding model + familiarity check", t["bge-small encode"] + t["LR head"] + t["knn1 over train"]),
            ("Typed LLM call, cached contract", L["qwen3-4b/route_1shot"]["p50"]),
            ("The same call, contract re-read", L["qwen3-4b/route_1shot_no_prefix_cache"]["p50"])]


def fmt_ms(v):
    if v < 0.01:
        return f"{v:.3f} ms"
    if v < 1:
        return f"{v:.2f} ms"
    if v < 100:
        return f"{v:.1f} ms"
    if v < 1000:
        return f"{v:.0f} ms"
    return f"{v / 1000:.1f} s"


# --------------------------------------------------------------------------- figures
def draw_ladder(ax, items, upto=None, label_size=10.5, k=1.0):
    """Dot plot on a log axis (bars would lie on a log scale): one row per tier."""
    n = len(items)
    ax.set_xscale("log")
    ax.set_xlim(3e-4, 4e4)
    ax.set_ylim(-0.7, n - 0.3)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xticks([1e-3, 1e-2, 1e-1, 1, 10, 100, 1e3, 1e4])
    ax.set_xticklabels(["0.001 ms", "0.01", "0.1", "1 ms", "10", "100", "1 s", "10 s"], fontsize=9)
    ax.minorticks_off()
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    dx, up, dn = 12 * k, 0.72 * label_size, -0.80 * label_size
    for i, (name, ms) in enumerate(items):
        y = n - 1 - i
        if upto is not None and i >= upto:
            continue
        ax.plot([3e-4, ms], [y, y], color=AXIS, lw=0.8 * k, zorder=2, solid_capstyle="round")
        ax.plot([ms], [y], marker="o", ms=9.5 * k, color=S1, mec=SURFACE, mew=2 * k, zorder=4)
        right = ms < 30
        ax.annotate(f"{name}", (ms, y), xytext=(dx if right else -dx, up), textcoords="offset points",
                    fontsize=label_size, color=INK, ha="left" if right else "right", va="center", fontweight="bold")
        ax.annotate(fmt_ms(ms), (ms, y), xytext=(dx if right else -dx, dn), textcoords="offset points",
                    fontsize=label_size - 1, color=INK2, ha="left" if right else "right", va="center")


def fig_tiers():
    fig = new_fig(4.5, "What one answer costs, tier by tier",
                  "Median time per request on an Apple M4. A request is answered by the cheapest tier that is sure of it.")
    ax = fig.add_axes([0.055, 0.12, 0.89, 0.66])
    draw_ladder(ax, tiers())
    return save(fig, "tier-latency.png")


def unseen_numbers():
    c = J("exp3_group_shift.json")["clinc"]["models"]["bge-small+LR"]["cells"]
    r, g = c["msp|R|fixed_sequence|0.02"], c["msp&knn1|G-worst|fixed_sequence|0.02"]
    return {"r_far": r["test_unseen"]["far_mean"], "g_far": g["test_unseen"]["far_mean"],
            "r_cov": r["test_seen"]["coverage_mean"], "g_cov": g["test_seen"]["coverage_mean"],
            "r_ans": r["test_unseen"]["coverage_mean"], "r_sil": r["test_unseen"]["silent_mean"],
            "r_seen_sil": r["test_seen"]["silent_mean"], "viol": g["test_unseen"]["far_violation_rate"]}


def two_bar_panel(ax, vals, colors, ymax, title, fmt, ref=None, ref_label=""):
    ax.set_xlim(-0.75, 1.75)
    ax.set_ylim(0, ymax)
    ax.set_xticks([])
    ax.spines["left"].set_visible(False)
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", labelsize=9)
    ax.set_yticks(np.arange(0, ymax + 0.01, 5 if ymax <= 30 else 20))
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    ax.set_title(title, fontsize=10.5, color=INK, loc="left", pad=12, fontweight="bold")
    if ref is not None:
        ax.axhline(ref, color=MUTED, lw=0.8, zorder=2)
        ax.text(1.74, ref + ymax * 0.012, ref_label, fontsize=8.8, color=MUTED, ha="right", va="bottom")
    for i, (v, col) in enumerate(zip(vals, colors)):
        bar(ax, i - 0.115, i + 0.115, 0, v, col, "top")
        ax.text(i, v + ymax * 0.025, fmt(v), fontsize=11.5, color=INK, ha="center", va="bottom", fontweight="bold")


def fig_unseen():
    u = unseen_numbers()
    fig = new_fig(4.7, "Calibrating on a random split certifies the wrong thing",
                  "A fifth of the intents were never shown to the fast model. Mean of 20 random group splits, 2% error budget.")
    a1 = fig.add_axes([0.075, 0.10, 0.39, 0.56])
    a2 = fig.add_axes([0.565, 0.10, 0.39, 0.56])
    two_bar_panel(a1, [100 * u["r_far"], 100 * u["g_far"]], [S1, S2], 21, "Wrong answers per request from never-seen groups",
                  lambda v: f"{v:.1f}%", ref=2, ref_label="2% budget")
    two_bar_panel(a2, [100 * u["r_cov"], 100 * u["g_cov"]], [S1, S2], 110, "Familiar requests the fast tier still answers",
                  lambda v: f"{v:.0f}%")
    legend(fig, 0.075, 0.775, [("Calibrated on a random split", S1), ("Calibrated against held-out groups, with a familiarity check", S2)], gap=0.33)
    return save(fig, "unseen-groups.png")


def verifier_rows():
    tab = J("exp4_verification.json")["error_correlation"]["test_unseen"]
    names = {"mlp-ens(bge-small)": "Ensemble of 5 networks, same embedding", "minilm+LR": "A different small encoder",
             "bge-small+kNN": "Nearest-neighbour vote, same embedding", "bge-base+LR": "A bigger encoder",
             "tfidf-char+LR": "Character n-gram model", "llm:qwen3-4b": "Local LLM (Qwen3-4B)", "llm:gemma3-4b": "Second local LLM (Gemma-3-4B)"}
    rows = [(names[k.split("-> ")[1]], 100 * v["same_wrong_label_given_A_wrong"]) for k, v in tab.items()
            if k.startswith("bge-small+LR ->") and k.split("-> ")[1] in names]
    return sorted(rows, key=lambda r: -r[1])


def fig_verifier():
    rows = verifier_rows()
    fig = new_fig(4.6, "A second opinion helps only if it fails differently",
                  "When the fast model is wrong on a never-seen group, how often does the checker give the same wrong answer?")
    ax = fig.add_axes([0.40, 0.09, 0.545, 0.70])
    n = len(rows)
    ax.set_xlim(0, 80)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_yticks([])
    ax.spines["left"].set_color(AXIS)
    ax.spines["bottom"].set_visible(False)
    ax.set_xticks([0, 20, 40, 60, 80])
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    ax.tick_params(axis="x", labelsize=9)
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    for i, (name, v) in enumerate(rows):
        y = n - 1 - i
        bar(ax, y - 0.19, y + 0.19, 0, v, S1, "right")
        ax.text(v + 1.4, y, f"{v:.0f}%", fontsize=11, color=INK, va="center", ha="left", fontweight="bold")
        ax.text(-1.6, y, name, fontsize=10.2, color=INK2, va="center", ha="right")
    return save(fig, "same-wrong-answer.png")


def typo_series():
    T = J("exp2_novelty.json")["typo_noise"]
    lv = ["typo00", "typo05", "typo15"]
    return {v: {"wrong": [100 * T[v][k]["msp"]["silent_error_rate"] for k in lv],
                "answered": [100 * T[v][k]["msp"]["coverage"] for k in lv],
                "acc": [100 * T[v][k]["accuracy"] for k in lv]} for v in ("bge-small", "tfidf")}


def fig_typos():
    s = typo_series()
    fig = new_fig(4.6, "Typing noise: one model fails silently, the other backs off",
                  "Same 4,500 test requests with 0%, 5% and 15% of letters mistyped. Thresholds unchanged, certified for 2% on clean data.")
    ax = fig.add_axes([0.075, 0.13, 0.66, 0.52])
    x = [0, 5, 15]
    ax.set_xlim(-0.8, 15.8)
    ax.set_ylim(0, 28)
    ax.set_xticks(x)
    ax.set_xticklabels(["clean", "5% of letters", "15% of letters"], fontsize=9.5)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    ax.tick_params(axis="y", labelsize=9)
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.set_title("Wrong answers among the requests each model chose to answer", fontsize=10.5, color=INK, loc="left", pad=12, fontweight="bold")
    ax.axhline(2, color=MUTED, lw=0.8, zorder=2)
    ax.text(10.0, 1.5, "2% budget", fontsize=8.8, color=MUTED, ha="center", va="top")
    for key, col, name in (("bge-small", S1, "Embedding model"), ("tfidf", S2, "Character n-gram model")):
        y = s[key]["wrong"]
        ax.plot(x, y, color=col, lw=2, solid_capstyle="round", solid_joinstyle="round", zorder=3)
        ax.plot(x, y, ls="none", marker="o", ms=8.5, color=col, mec=SURFACE, mew=2, zorder=4)
        ax.annotate(f"{name}\n{y[-1]:.1f}% wrong, answering {s[key]['answered'][-1]:.0f}%", (15, y[-1]), xytext=(14, 0),
                    textcoords="offset points", fontsize=9.8, color=INK, va="center", ha="left", annotation_clip=False, linespacing=1.35)
    legend(fig, 0.075, 0.775, [("Embedding model (bge-small)", S1), ("Character n-gram model (TF-IDF)", S2)], gap=0.30)
    return save(fig, "typing-noise.png")


def news_series():
    r = J("exp7_news_drift.json")
    months = r["months"][r["compile_months"]:]
    rows = r["policies"]["static-T"]["monthly"]
    return months, [100 * m["silent"] for m in rows], [100 * m["new_section_share"] for m in rows], 100 * r["alpha"]


def fig_news():
    months, wrong, new, budget = news_series()
    fig = new_fig(5.6, "A compiled program left alone, over five years of real headlines",
                  "HuffPost headlines routed to their section. Compiled on the first 12 months (2012), then never updated.")
    x = np.arange(len(months))
    ticks = [i for i, m in enumerate(months) if m.endswith("-01")]
    a1 = fig.add_axes([0.075, 0.47, 0.87, 0.32])
    a2 = fig.add_axes([0.075, 0.09, 0.87, 0.24])
    for ax, y, title, ymax in ((a1, wrong, "Wrong answers among the headlines it still answers", 80),
                               (a2, new, "Headlines from sections that did not exist when it was compiled", 80)):
        ax.set_xlim(-0.5, len(months) - 0.5)
        ax.set_ylim(0, ymax)
        ax.set_xticks(ticks)
        ax.set_xticklabels([months[i][:4] for i in ticks], fontsize=9)
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
        ax.set_yticks([0, 20, 40, 60, 80])
        ax.tick_params(axis="y", labelsize=9)
        ax.grid(axis="y")
        ax.set_axisbelow(True)
        ax.spines["left"].set_visible(False)
        ax.set_title(title, fontsize=10.5, color=INK, loc="left", pad=9, fontweight="bold")
        ax.fill_between(x, 0, y, color=S1, alpha=0.10, lw=0, zorder=2)
        ax.plot(x, y, color=S1, lw=2, solid_capstyle="round", solid_joinstyle="round", zorder=3)
        ax.plot([x[-1]], [y[-1]], marker="o", ms=8.5, color=S1, mec=SURFACE, mew=2, zorder=4)
        ax.annotate(f"{y[-1]:.0f}%", (x[-1], y[-1]), xytext=(-8, 11), textcoords="offset points", fontsize=10.5, color=INK,
                    ha="right", va="bottom", fontweight="bold")
    a1.axhline(budget, color=MUTED, lw=0.8, zorder=2)
    a1.text(len(months) - 0.6, budget + 1.5, f"{budget:.0f}% budget", fontsize=8.8, color=MUTED, ha="right", va="bottom")
    return save(fig, "five-years-of-headlines.png")


def loop_series():
    r = J("exp6_stream.json")
    P = r["policies"]
    bounds = np.cumsum([v["n"] for v in r["phases"].values()])
    series = {k: [100 * w["wrong"] for w in P[k]["windows"]] for k in ("B*", "C*", "D+*")}
    xs = [min((i + 1) * 100, r["n_requests"]) - 50 if (i + 1) * 100 <= r["n_requests"] else (i * 100 + r["n_requests"]) / 2
          for i in range(len(series["B*"]))]
    return xs, series, bounds, r["n_requests"], {k: 100 * P[k]["total"]["wrong"] for k in series}


LOOP_NAMES = {"B*": "Naive: confidence threshold from a random split", "C*": "Guarded: familiarity gate + lexical check",
              "D+*": "Guarded, and the contract change is announced"}
LOOP_COL = {"B*": S1, "C*": S2, "D+*": S3}
PHASES = ["steady", "new groups", "out-of-scope burst", "noisy input", "policy change"]


def draw_loop(ax, upto=None, label_size=9.2, show_phase=None, k=1.0):
    xs, series, bounds, n, totals = loop_series()
    ax.set_xlim(0, n)
    ax.set_ylim(0, 62)
    ax.set_yticks([0, 20, 40, 60])
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    ax.set_xticks([0, 500, 1000, 1500, 2000, 2500])
    ax.set_xticklabels(["0", "500", "1,000", "1,500", "2,000", "2,500 requests"], fontsize=label_size - 0.2)
    ax.tick_params(axis="y", labelsize=label_size - 0.2)
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    starts = [0] + list(bounds[:-1])
    for i, (a, b) in enumerate(zip(starts, bounds)):
        if show_phase is not None and a > show_phase:
            continue
        if i:
            ax.axvline(a, color=AXIS, lw=0.8, zorder=1)
        ax.text((a + b) / 2, 60.5, PHASES[i], fontsize=label_size, color=INK2, ha="center", va="top")
    for key in ("B*", "D+*", "C*"):          # the two guarded lines coincide until the policy change
        x, y = np.array(xs), np.array(series[key])
        if upto is not None:
            keep = x <= upto
            x, y = x[keep], y[keep]
        if len(x):
            ax.plot(x, y, color=LOOP_COL[key], lw=2 * k, solid_capstyle="round", solid_joinstyle="round", zorder=3)
            ax.plot([x[-1]], [y[-1]], marker="o", ms=8 * k, color=LOOP_COL[key], mec=SURFACE, mew=2 * k, zorder=4)
    return totals


def fig_loop():
    fig = new_fig(5.0, "A production loop, replayed on 2,790 real requests",
                  "Wrong answers per 100 requests. Refused requests go to the agent (the human label). The two guarded lines coincide until the last phase.")
    ax = fig.add_axes([0.075, 0.11, 0.87, 0.56])
    totals = draw_loop(ax)
    y = 0.80
    for k in ("B*", "C*", "D+*"):
        legend(fig, 0.075, y, [(f"{LOOP_NAMES[k]}  ·  {totals[k]:.1f}% wrong overall", LOOP_COL[k])])
        y -= 0.042
    return save(fig, "production-loop.png")


def harness_rows():
    r = J("exp9_harness.json")
    a, m = r["agent"], r["modes"]
    return [("The agent: three free-form calls", a["latency_s"]["p50"], a["prompt_tokens"], a["generated_tokens"]),
            ("Compiled, prose still generated", m["prose_hole"]["latency_s"]["p50"], m["prose_hole"]["prompt_tokens"], m["prose_hole"]["generated_tokens"]),
            ("Compiled, structured analysis", m["structured"]["latency_s"]["p50"], m["structured"]["prompt_tokens"], m["structured"]["generated_tokens"])]


def fig_harness():
    rows = harness_rows()
    r = J("exp9_harness.json")
    fig = new_fig(3.9, "After watching 60 runs: seconds per profile",
                  f"Same local model, same {r['n_test']} unseen profiles, one request at a time. Token counts are per profile.")
    ax = fig.add_axes([0.335, 0.12, 0.61, 0.60])
    n = len(rows)
    ax.set_xlim(0, 14.5)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_yticks([])
    ax.spines["bottom"].set_visible(False)
    ax.set_xticks([0, 4, 8, 12])
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0f} s")
    ax.tick_params(axis="x", labelsize=9)
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    for i, (name, s, pt, gt) in enumerate(rows):
        y = n - 1 - i
        bar(ax, y - 0.17, y + 0.17, 0, s, S1, "right")
        ax.text(s + 0.25, y + 0.01, f"{s:.1f} s", fontsize=11.5, color=INK, va="center", ha="left", fontweight="bold")
        ax.text(-0.3, y + 0.13, name, fontsize=10.2, color=INK, va="center", ha="right")
        ax.text(-0.3, y - 0.17, f"reads {pt:,.0f} tokens, writes {gt:,.0f}", fontsize=9, color=MUTED, va="center", ha="right")
    return save(fig, "seconds-per-profile.png")


# --------------------------------------------------------------------------- the film
W, H, FPS = 1920, 1080, 30


def ease(t):
    t = min(1.0, max(0.0, t))
    return t * t * (3 - 2 * t)


def film_frame(fig, t):
    """Draw the frame at time t (seconds). Five scenes; every number is read from results/."""
    fig.clf()
    fig.patch.set_facecolor(SURFACE)
    u, hr = unseen_numbers(), harness_rows()

    def fade(t0, t1, dur=0.5):
        return min(ease((t - t0) / dur), ease((t1 - t) / dur))

    def header(kicker, title, a):
        fig.text(0.06, 0.905, kicker, fontsize=17, color=LIME, alpha=a, fontweight="bold")
        fig.text(0.06, 0.845, title, fontsize=34, color=INK, alpha=a, fontweight="bold")

    if t < 4.0:                                        # 1 title
        a = fade(0.0, 4.0, 0.7)
        fig.text(0.06, 0.60, "Agent 2 Program", fontsize=78, color=INK, alpha=a, fontweight="bold")
        fig.text(0.06, 0.475, "What survives when you compile an LLM workflow", fontsize=30, color=INK2, alpha=a)
        fig.add_artist(plt.Line2D([0.06, 0.06 + 0.16 * ease(t / 1.2)], [0.42, 0.42], color=LIME, lw=5, alpha=a, transform=fig.transFigure))
        fig.text(0.06, 0.34, "Nine experiments · public human-labelled data · local models on one M4", fontsize=19, color=MUTED, alpha=a)
    elif t < 11.0:                                     # 2 tier ladder
        a = fade(4.0, 11.0)
        header("THE PROGRAM IS A LADDER OF CHEAPER ANSWERS", "What one answer costs, tier by tier", a)
        ax = fig.add_axes([0.06, 0.13, 0.88, 0.60])
        items = tiers()
        upto = int(np.clip((t - 4.5) / 0.75 + 1, 0, len(items)))
        draw_ladder(ax, items, upto=upto, label_size=19, k=1.7)
        ax.tick_params(axis="x", labelsize=16)
        for art in ax.get_children():
            if hasattr(art, "set_alpha") and art is not ax.patch:
                art.set_alpha(a)
    elif t < 19.0:                                     # 3 unseen groups
        a = fade(11.0, 19.0)
        header("CONFIDENCE IS NOT FAMILIARITY", "Requests from groups the fast model never saw", a)
        ax = fig.add_axes([0.10, 0.15, 0.50, 0.52])
        g1 = ease((t - 11.6) / 1.2)
        g2 = ease((t - 14.6) / 1.2)
        vals = [100 * u["r_far"] * g1, 100 * u["g_far"] * g2]
        ax.set_xlim(-0.75, 1.75)
        ax.set_ylim(0, 21)
        ax.set_xticks([])
        ax.spines["left"].set_visible(False)
        ax.grid(axis="y")
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", labelsize=16)
        ax.set_yticks([0, 5, 10, 15, 20])
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
        ax.axhline(2, color=MUTED, lw=1.2)
        ax.text(1.74, 2.3, "2% budget", fontsize=16, color=MUTED, ha="right", va="bottom", alpha=a)
        for i, (v, col, lab) in enumerate(zip(vals, [S1, S2], ["calibrated on\na random split", "calibrated against\nheld-out groups"])):
            if v > 0.05:
                bar(ax, i - 0.11, i + 0.11, 0, v, col, "top", r=10)
                ax.text(i, v + 0.6, f"{v:.1f}%", fontsize=26, color=INK, ha="center", va="bottom", fontweight="bold", alpha=a)
            ax.text(i, -1.2, lab, fontsize=17, color=INK2, ha="center", va="top", alpha=a, linespacing=1.3)
        fig.text(0.10, 0.725, "Wrong answers per request", fontsize=19, color=INK, alpha=a, fontweight="bold")
        if t > 12.6:
            b = a * ease((t - 12.6) / 0.6)
            fig.text(0.66, 0.60, f"{100 * u['r_ans']:.0f}% answered,", fontsize=30, color=INK, alpha=b, fontweight="bold")
            fig.text(0.66, 0.545, f"{100 * u['r_sil']:.0f}% of those wrong", fontsize=30, color=INK, alpha=b, fontweight="bold")
            fig.text(0.66, 0.49, "in 20 of 20 random group splits", fontsize=19, color=INK2, alpha=b)
        if t > 15.6:
            b = a * ease((t - 15.6) / 0.6)
            fig.text(0.66, 0.36, "The fix costs coverage:", fontsize=19, color=INK2, alpha=b)
            fig.text(0.66, 0.305, f"{100 * u['r_cov']:.0f}% → {100 * u['g_cov']:.0f}%", fontsize=30, color=INK, alpha=b, fontweight="bold")
            fig.text(0.66, 0.255, "of familiar requests still answered", fontsize=19, color=INK2, alpha=b)
    elif t < 31.0:                                     # 4 the loop
        a = fade(19.0, 31.0)
        header("PROGRAMS ROT", "A production loop, replayed on 2,790 real requests", a)
        ax = fig.add_axes([0.06, 0.12, 0.88, 0.50])
        n = loop_series()[3]
        cur = n * ease((t - 19.8) / 8.0)
        totals = draw_loop(ax, upto=cur, label_size=16, show_phase=cur, k=1.7)
        ax.axvline(cur, color=LIME, lw=1.5, alpha=0.8 * a, zorder=5)
        fig.text(0.06, 0.655, "Wrong answers per 100 requests", fontsize=17, color=INK2, alpha=a)
        y = 0.775
        for k in ("B*", "C*", "D+*"):
            fig.add_artist(FancyBboxPatch((0.06, y - 0.011), 0.012, 0.022, boxstyle="round,pad=0,rounding_size=0.003",
                                          facecolor=LOOP_COL[k], edgecolor="none", transform=fig.transFigure, alpha=a))
            fig.text(0.08, y, LOOP_NAMES[k], fontsize=18, color=INK2, va="center", alpha=a)
            if t > 28.2:
                fig.text(0.60, y, f"{totals[k]:.1f}% wrong overall", fontsize=18, color=INK, va="center",
                         alpha=a * ease((t - 28.2) / 0.6), fontweight="bold")
            y -= 0.037
    else:                                              # 5 the harness
        a = fade(31.0, 38.0, 0.6)
        header("THE WRAPPER HARNESS", "After watching the agent analyze 60 profiles", a)
        ax = fig.add_axes([0.40, 0.33, 0.52, 0.36])
        ax.set_xlim(0, 14.5)
        ax.set_ylim(-0.6, 2.6)
        ax.set_yticks([])
        ax.spines["bottom"].set_visible(False)
        ax.set_xticks([0, 4, 8, 12])
        ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0f} s")
        ax.tick_params(axis="x", labelsize=16)
        ax.grid(axis="x")
        ax.set_axisbelow(True)
        for i, (name, s, pt, gt) in enumerate(hr):
            y = 2 - i
            g = ease((t - 31.6 - 0.5 * i) / 1.0)
            if g > 0.01:
                bar(ax, y - 0.17, y + 0.17, 0, s * g, S1, "right", r=10)
                ax.text(s * g + 0.25, y, f"{s * g:.1f} s", fontsize=24, color=INK, va="center", ha="left", fontweight="bold", alpha=a)
            ax.text(-0.3, y, name, fontsize=18, color=INK2, va="center", ha="right", alpha=a)
        if t > 34.0:
            b = a * ease((t - 34.0) / 0.6)
            fig.text(0.06, 0.205, "The decisions got smaller. Only 1 of 6 short fields became pure code.", fontsize=22, color=INK, alpha=b, fontweight="bold")
            fig.text(0.06, 0.15, "Code, the full report and the revised skill: generativemedia.blog", fontsize=19, color=INK2, alpha=b)


FILM_SECONDS = 38.0


def film():
    out = OUT / (PRE + "film.mp4")
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
           "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    n = int(FILM_SECONDS * FPS)
    for i in range(n):
        film_frame(fig, i / FPS)
        fig.canvas.draw()
        p.stdin.write(bytes(fig.canvas.buffer_rgba()))
        if i % 150 == 0:
            print(f"  film frame {i}/{n}", flush=True)
    p.stdin.close()
    p.wait()
    plt.close(fig)
    print("wrote", out.name)


def stills(times=(2.0, 9.5, 17.5, 27.0, 29.5, 36.5)):
    """A few frames of the film, for checking layout before rendering all of it."""
    d = ROOT / "blog" / "film_stills"
    d.mkdir(exist_ok=True)
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    for t in times:
        film_frame(fig, t)
        fig.savefig(d / f"t{t:05.1f}.png", dpi=100)
    plt.close(fig)
    print("wrote stills to", d)


def poster():
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    film_frame(fig, 29.6)
    path = OUT / (PRE + "film-poster.jpg")
    fig.savefig(path, dpi=100, pil_kwargs={"quality": 92})
    plt.close(fig)
    print("wrote", path.name)


if __name__ == "__main__":
    what = sys.argv[1:] or ["figures"]
    if "figures" in what:
        for f in (fig_tiers, fig_unseen, fig_verifier, fig_typos, fig_news, fig_loop, fig_harness):
            f()
    if "stills" in what:
        stills()
    if "film" in what:
        film()
        poster()
