"""Experiment 5b: the same local LLM on two more holes.

  fine     zero-shot 150-way (CLINC150) and 77-way (BANKING77) intent holes, names only, against
           the fast models trained on the task's training split.
  locale   the MASSIVE scenario router (18 classes, 1-shot contract in English) on English
           requests and on the same requests human-translated into Spanish and German, against
           the English-trained fast tier.
"""
from __future__ import annotations

import numpy as np

from . import data, holes, kit
from . import models as M
from .common import save_json
from .label_llm import fine_subsets, massive_subsets
from .llm import AnswerStore

NONE = holes.NONE


def answers(hole, texts):
    st = AnswerStore("qwen3-4b", hole)
    have = np.array([t in st.rows for t in texts])
    lab = np.array([st.rows[t]["label"] if t in st.rows else None for t in texts], dtype=object)
    return lab, have


def run() -> dict:
    out = {"fine": {}, "locale": {}}
    for name, (sub, oos) in fine_subsets().items():
        h = holes.fine_hole(name)
        lab, have = answers(h, list(sub.text))
        if have.mean() < 0.99:
            continue
        frame = data.fine_frame(name)
        tag = {"clinc150": "clinc", "banking77": "banking"}[name]
        tr = frame.index[frame.split == "train"].values
        views = kit.build_views(tag, None, tr, ("bge-small",))
        fm = kit.fit_family("bge-small+LR", views, tr, frame.loc[tr, "label"].values)
        pred = fm.run(sub.index.values)["pred"]
        row = {"n": int(len(sub)), "llm_accuracy": float((lab == sub.label.values).mean()),
               "llm_none_or_unparseable": float(np.mean([l is None or l == NONE for l in lab])),
               "fast_accuracy_same_rows": float((pred == sub.label.values).mean())}
        if oos is not None:
            lo, ho = answers(h, list(oos.text))
            if ho.mean() >= 0.99:
                row["oos_n"] = int(len(oos))
                row["llm_oos_none_exit"] = float(np.mean([l == NONE or l is None for l in lo]))
        out["fine"][name] = row
        print(f"  fine {name}: LLM {row['llm_accuracy']:.3f} vs bge-small+LR {row['fast_accuracy_same_rows']:.3f} "
              f"(n={row['n']}); oos none-exit {row.get('llm_oos_none_exit')}", flush=True)
    h = holes.massive_routing_hole()
    f = data.massive_routing()
    tr = f.index[f.split == "train"].values
    subs = massive_subsets()
    for enc in ("bge-small", "me5-small"):
        X = {loc: kit.dense_view(enc, f"massive_{loc}") for loc in ("en-US", "es-ES", "de-DE")}
        head = M.LRHead(C=20.0).fit(X["en-US"][tr], f.loc[tr, "label"].values)
        for key, sub in subs.items():
            loc = "en-US" if key.startswith("en_") else key.split("_")[0]
            pred = head.classes_[head.proba(X[loc][sub.index.values]).argmax(1)]
            out["locale"].setdefault(key, {})[f"fast:{enc}"] = float((pred == sub.label.values).mean())
    for key, sub in subs.items():
        lab, have = answers(h, list(sub.text))
        if have.mean() < 0.99:
            out["locale"].pop(key, None)
            continue
        out["locale"][key].update({"n": int(len(sub)), "llm": float((lab == sub.label.values).mean()),
                                   "llm_none_or_unparseable": float(np.mean([l is None or l == NONE for l in lab]))})
        print(f"  locale {key}: LLM {out['locale'][key]['llm']:.3f}  fast bge-small {out['locale'][key]['fast:bge-small']:.3f} "
              f"me5 {out['locale'][key]['fast:me5-small']:.3f}", flush=True)
    return out


if __name__ == "__main__":
    save_json("exp5b_llm_other.json", run())
