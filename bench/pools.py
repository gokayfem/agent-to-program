"""Named, deterministic row pools on the frozen CLINC routing split.

Every experiment that needs a real LLM answer draws from these pools, so the LLM is run
once per (model, hole, text) and the answers are replayed. Pools are subsets of the frozen
split's partitions, chosen by sorted uid with a fixed seed; nothing here looks at labels
other than to stratify by group.

  test_seen     20 per training group      (1800)  official test partition, groups seen in training
  test_unseen   50 per held-out test group (1500)  groups never seen by any fast model or threshold
  cal_seen      10 per training group      (900)   official val partition
  cal_unseen    30 per calibration group   (900)   held-out calibration groups
  traces        15 per training group      (1350)  official train partition: the "past runs" a
                                                   compile-from-traces build learns from
  oos_test      all 1000 human-written out-of-scope test queries
  oos_val       all 100 out-of-scope validation queries
  typo15        600 test_seen-pool rows with 15% keyboard noise
  policy        rows used after the contract change (see holes.POLICY_NOTE)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import data
from .perturb import typo_noise

# Groups the policy change really re-routes: requests about a physical card. All three are
# training groups, so the stale fast tier answers them confidently.
POLICY_GROUPS = ("damaged_card", "replacement_card_duration", "report_lost_card")
# Rows of these groups make up the policy pool. `new_card` is in the pool but is NOT re-routed:
# in CLINC it means "apply for a new credit card", which the policy text does not cover. An
# earlier version of this file relabelled it to banking; a closed model that correctly kept it
# in credit_cards exposed the mistake (see exp10 and the correction note in REPORT.md).
POLICY_POOL_GROUPS = POLICY_GROUPS + ("new_card",)
POLICY_FROM, POLICY_TO = "credit_cards", "banking"


def _per_group(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    parts = []
    for g, sub in df.sort_values("uid").groupby("group", sort=True):
        parts.append(sub.sample(min(n, len(sub)), random_state=seed))
    return pd.concat(parts)


def frozen() -> pd.DataFrame:
    return data.clinc_routing(data.FROZEN_SEED)


def pools() -> dict:
    f = frozen()
    p = {
        "test_seen": _per_group(f[f.split == "test_seen"], 20, 11),
        "test_unseen": _per_group(f[f.split == "test_unseen"], 50, 12),
        "cal_seen": _per_group(f[f.split == "cal_seen"], 10, 13),
        "cal_unseen": _per_group(f[f.split == "cal_unseen"], 30, 14),
        "traces": _per_group(f[f.split == "train"], 15, 15),
    }
    oos = data.clinc_oos().copy()
    oos["label"], oos["group"], oos["split"] = "none_of_these", "oos", "oos"
    p["oos_test"] = oos[oos.part == "test"]
    p["oos_val"] = oos[oos.part == "val"]
    # Full calibration partitions for the fast tiers (no LLM answers needed for these).
    # oos_cal = oos-val plus the oos-train rows that are NOT quoted in the LLM contract.
    p["cal_seen_full"] = f[f.split == "cal_seen"]
    p["cal_unseen_full"] = f[f.split == "cal_unseen"]
    quoted = set(oos[oos.part == "train"].sort_values("uid").uid[:20])
    p["oos_cal"] = oos[(oos.part != "test") & (~oos.uid.isin(quoted))]
    # typo pool: a fixed subset of the test_seen pool, noised deterministically
    ty = p["test_seen"].sort_values("uid").sample(600, random_state=16).copy()
    ty["clean_text"] = ty["text"]
    ty["text"] = [typo_noise(t, 0.15, seed=i) for i, t in enumerate(ty.clean_text)]
    p["typo15"] = ty
    # policy-change pool: every test-partition / unseen row of the affected groups + a control sample
    aff = f[f.group.isin(POLICY_POOL_GROUPS) & f.split.isin(["test_seen", "test_unseen"])].copy()
    ctrl = p["test_seen"][~p["test_seen"].group.isin(POLICY_POOL_GROUPS)].sort_values("uid").sample(400, random_state=17)
    pol = pd.concat([aff, ctrl]).copy()
    pol["old_label"] = pol["label"]
    pol.loc[pol.group.isin(POLICY_GROUPS), "label"] = POLICY_TO
    p["policy"] = pol
    return p


if __name__ == "__main__":
    f = frozen()
    print("policy groups:", {g: (f[f.group == g].label.iloc[0], f[f.group == g].role.iloc[0]) for g in POLICY_GROUPS})
    for k, v in pools().items():
        print(k, len(v), v.split.value_counts().to_dict())
