"""Experiment 3b: how many held-out groups does grouped calibration need?

CLINC routing, 3 never-seen test groups per domain (30) in every setting; the number of groups
held out for calibration varies from 1 to 5 per domain (10 to 50 groups); the rest are training
groups. 20 random splits per setting. bge-small + LR, confidence + familiarity gate, G-worst
protocol, 5% budget, with the row-level (fixed-sequence) and the group-aware (bootstrap) picker.
"""
from __future__ import annotations

from joblib import Parallel, delayed

from .common import save_json
from .exp3_group_shift import aggregate, one_split

SPLITS = 20


def run() -> dict:
    out = {"splits": SPLITS, "settings": {}}
    for n_cal in (1, 2, 3, 4, 5):
        runs = Parallel(n_jobs=5)(delayed(one_split)("clinc", s, frac_cal=n_cal / 15, frac_test=0.2) for s in range(SPLITS))
        agg = aggregate(runs)
        row = {"groups": agg["groups"], "rows": agg["n"]}
        for picker in ("fixed_sequence", "cluster_bootstrap"):
            c = agg["models"]["bge-small+LR"]["cells"][f"msp&knn1|G-worst|{picker}|0.05"]
            row[picker] = {"seen_coverage": c["test_seen"]["coverage_mean"], "seen_silent": c["test_seen"]["silent_mean"],
                           "unseen_far": c["test_unseen"]["far_mean"], "unseen_far_max": c["test_unseen"]["far_max"],
                           "unseen_over_budget": c["test_unseen"]["far_violation_rate"], "never": c["never_rate"]}
        out["settings"][str(n_cal * 10)] = row
        print(f"  {n_cal * 10:2d} calibration groups (train {agg['groups']['train']}): " + " | ".join(
            f"{p}: seen cov {row[p]['seen_coverage']:.3f} unseen FAR {row[p]['unseen_far']:.4f} over budget {row[p]['unseen_over_budget']:.2f} "
            f"worst {row[p]['unseen_far_max']:.3f}" for p in ("fixed_sequence", "cluster_bootstrap")), flush=True)
    return out


if __name__ == "__main__":
    save_json("exp3b_cal_groups.json", run())
