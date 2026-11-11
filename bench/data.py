"""Public datasets with human labels, and the frozen splits every experiment uses.

  CLINC150   23.7k assistant queries, 150 intents in 10 domains, plus 1.2k human-written
             out-of-scope queries.            (Larson et al. 2019, github.com/clinc/oos-eval)
  BANKING77  13k banking support queries, 77 fine-grained intents.   (Casanueva et al. 2020)
  MASSIVE    16.5k assistant utterances, 60 intents in 18 scenarios, human-translated
             into 51 locales with shared ids. (FitzGerald et al. 2022)
  NEWS       210k HuffPost headlines 2012-2022 with editor-assigned category, author, date.
             (Misra 2022, News Category Dataset)

Two kinds of decision are built from them:

  fine     the label is the intent itself (150 / 77 / 60 classes).
  routing  the label is the coarse class (domain / scenario) and each intent is a
           *group*. Whole groups are held out of training, so "unseen group, known
           label space" can be tested: the same shape as the unseen-merchant failure
           in the original expense example, but on human-written text.

Run `python -m bench.data` to download everything and write results/data_manifest.json
with a fingerprint of every split.
"""
from __future__ import annotations

import io
import json
import tarfile
import urllib.request

import numpy as np
import pandas as pd

from .common import MASTER_SEED, PROC, RAW, save_json, sha_ids

URLS = {
    "clinc_full.json": "https://raw.githubusercontent.com/clinc/oos-eval/master/data/data_full.json",
    "clinc_domains.json": "https://raw.githubusercontent.com/clinc/oos-eval/master/data/domains.json",
    "banking_train.csv": "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/train.csv",
    "banking_test.csv": "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/test.csv",
    "massive.tar.gz": "https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.0.tar.gz",
    "news.json": "https://huggingface.co/datasets/heegyu/news-category-dataset/resolve/main/data.json",
    "resumes.csv": "https://huggingface.co/datasets/opensporks/resumes/resolve/main/Resume/Resume.csv",
}
MASSIVE_LOCALES = ("en-US", "de-DE", "es-ES", "fr-FR", "tr-TR", "ja-JP")


def fetch(name: str) -> "pathlib.Path":
    path = RAW / name
    if not path.exists():
        print(f"downloading {name} ...", flush=True)
        req = urllib.request.Request(URLS[name], headers={"User-Agent": "python-urllib"})
        with urllib.request.urlopen(req, timeout=120) as r:
            path.write_bytes(r.read())
    return path


# --------------------------------------------------------------------------- loaders
def load_clinc() -> pd.DataFrame:
    out = PROC / "clinc.parquet"
    if out.exists():
        return pd.read_parquet(out)
    full = json.loads(fetch("clinc_full.json").read_text())
    domains = json.loads(fetch("clinc_domains.json").read_text())
    intent2domain = {i: d for d, ints in domains.items() for i in ints}
    rows = []
    for key, items in full.items():
        part = key.replace("oos_", "")
        for j, (text, intent) in enumerate(items):
            oos = intent == "oos"
            rows.append({"uid": f"clinc/{key}/{j}", "text": text, "intent": intent,
                         "domain": "oos" if oos else intent2domain[intent], "part": part, "is_oos": oos})
    df = pd.DataFrame(rows)
    df.to_parquet(out)
    return df


def load_banking() -> pd.DataFrame:
    out = PROC / "banking.parquet"
    if out.exists():
        return pd.read_parquet(out)
    frames = []
    for part in ("train", "test"):
        d = pd.read_csv(fetch(f"banking_{part}.csv"))
        d = d.rename(columns={"category": "intent"})
        d["part"] = part
        d["uid"] = [f"banking/{part}/{j}" for j in range(len(d))]
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df["text"] = df["text"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    # carve a calibration partition out of train, stratified, fixed seed
    rng = np.random.default_rng(MASTER_SEED)
    for intent, idx in df[df.part == "train"].groupby("intent").groups.items():
        idx = np.array(sorted(idx))
        rng.shuffle(idx)
        df.loc[idx[: max(1, round(0.2 * len(idx)))], "part"] = "val"
    df.to_parquet(out)
    return df


def load_massive(locale: str = "en-US") -> pd.DataFrame:
    out = PROC / f"massive_{locale}.parquet"
    if out.exists():
        return pd.read_parquet(out)
    with tarfile.open(fetch("massive.tar.gz")) as tar:
        member = next(m for m in tar.getmembers() if m.name.endswith(f"data/{locale}.jsonl") and m.isfile())
        raw = tar.extractfile(member).read().decode("utf-8")
    rows = []
    for line in io.StringIO(raw):
        r = json.loads(line)
        rows.append({"uid": f"massive/{locale}/{r['id']}", "mid": int(r["id"]), "text": r["utt"],
                     "intent": r["intent"], "scenario": r["scenario"],
                     "part": {"dev": "val"}.get(r["partition"], r["partition"])})
    df = pd.DataFrame(rows)
    df.to_parquet(out)
    return df


NEWS_MERGE = {"THE WORLDPOST": "WORLD NEWS", "WORLDPOST": "WORLD NEWS", "ARTS": "ARTS & CULTURE",
              "CULTURE & ARTS": "ARTS & CULTURE", "PARENTS": "PARENTING", "STYLE": "STYLE & BEAUTY",
              "GREEN": "ENVIRONMENT", "TASTE": "FOOD & DRINK", "HEALTHY LIVING": "WELLNESS",
              "COLLEGE": "EDUCATION", "SCIENCE": "SCIENCE & TECH", "TECH": "SCIENCE & TECH"}


def load_news() -> pd.DataFrame:
    """HuffPost headlines in date order. Renamed/duplicate sections are merged (NEWS_MERGE)."""
    out = PROC / "news.parquet"
    if out.exists():
        return pd.read_parquet(out)
    path = fetch("news.json")
    head = path.read_text()[:1].strip()
    if head == "[":
        df = pd.DataFrame(json.loads(path.read_text()))
    else:
        df = pd.read_json(path, lines=True)
    df = df.rename(columns={"category": "label_raw"})
    df["text"] = df["headline"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    df = df[df.text.str.len() >= 15].copy()
    df["date"] = pd.to_datetime(df["date"])
    df["label"] = df["label_raw"].replace(NEWS_MERGE)
    df = df.sort_values(["date", "text"], kind="stable").drop_duplicates("text").reset_index(drop=True)
    df["uid"] = [f"news/{j}" for j in range(len(df))]
    df = df[["uid", "text", "label", "label_raw", "authors", "date"]]
    df.to_parquet(out)
    return df


# --------------------------------------------------------------------------- group-held-out splits
def group_roles(df: pd.DataFrame, class_col: str, group_col: str, seed: int,
                frac_test: float = 0.2, frac_cal: float = 0.2, min_groups: int = 3) -> dict:
    """Assign every group (intent) a role: train, cal (held out for calibration) or test (never seen)."""
    rng = np.random.default_rng(seed)
    roles = {}
    for c in sorted(df[class_col].unique()):
        gs = sorted(df.loc[df[class_col] == c, group_col].unique())
        rng.shuffle(gs)
        if len(gs) < min_groups:
            roles.update({g: "train" for g in gs})
            continue
        nt = max(1, int(round(len(gs) * frac_test)))
        nc = max(1, int(round(len(gs) * frac_cal)))
        while nt + nc >= len(gs):                 # keep at least one training group per class
            if nc > 1:
                nc -= 1
            elif nt > 1:
                nt -= 1
            else:
                nc = 0
                break
        for j, g in enumerate(gs):
            roles[g] = "test" if j < nt else ("cal" if j < nt + nc else "train")
    return roles


def routing_frame(df: pd.DataFrame, class_col: str, group_col: str, seed: int, **kw) -> pd.DataFrame:
    """Row-level split for a routing task with whole groups held out.

    train        official train rows of training groups          -> fit models
    cal_seen     official val rows of training groups            -> calibration (in-distribution)
    cal_unseen   all rows of calibration groups                  -> calibration (group shift)
    test_seen    official test rows of training groups           -> report
    test_unseen  all rows of test groups (never seen anywhere)   -> report
    """
    roles = group_roles(df, class_col, group_col, seed, **kw)
    d = df.copy()
    d["label"], d["group"] = d[class_col], d[group_col]
    d["role"] = d[group_col].map(roles)
    split = np.where(d.role == "test", "test_unseen",
             np.where(d.role == "cal", "cal_unseen",
              np.where(d.part == "train", "train",
               np.where(d.part == "val", "cal_seen", "test_seen"))))
    d["split"] = split
    return d


FROZEN_SEED = 7     # the single split used wherever a real LLM is in the loop


def clinc_routing(seed: int = FROZEN_SEED) -> pd.DataFrame:
    df = load_clinc()
    return routing_frame(df[~df.is_oos], "domain", "intent", seed)


def massive_routing(seed: int = FROZEN_SEED) -> pd.DataFrame:
    return routing_frame(load_massive("en-US"), "scenario", "intent", seed)


def clinc_oos() -> pd.DataFrame:
    df = load_clinc()
    return df[df.is_oos].copy()


def fine_frame(name: str) -> pd.DataFrame:
    """Fine-grained task with official partitions: split in {train, cal, test}."""
    if name == "clinc150":
        df = load_clinc()
        df = df[~df.is_oos].copy()
    elif name == "banking77":
        df = load_banking().copy()
    elif name == "massive60":
        df = load_massive("en-US").copy()
    else:
        raise KeyError(name)
    df["label"] = df["intent"]
    df["split"] = df["part"].map({"train": "train", "val": "cal", "test": "test"})
    return df


def manifest() -> dict:
    m = {}
    for name in ("clinc150", "banking77", "massive60"):
        d = fine_frame(name)
        m[name] = {"classes": int(d.label.nunique()),
                   "splits": {s: {"n": int((d.split == s).sum()), "sha": sha_ids(d.uid[d.split == s])}
                              for s in ("train", "cal", "test")}}
    oos = clinc_oos()
    m["clinc_oos"] = {p: {"n": int((oos.part == p).sum()), "sha": sha_ids(oos.uid[oos.part == p])}
                      for p in ("train", "val", "test")}
    for name, fn in (("clinc_routing", clinc_routing), ("massive_routing", massive_routing)):
        d = fn()
        roles = d.groupby("group").role.first()
        m[name] = {"frozen_seed": FROZEN_SEED, "classes": int(d.label.nunique()),
                   "groups": {r: int((roles == r).sum()) for r in ("train", "cal", "test")},
                   "test_groups": sorted(roles[roles == "test"].index),
                   "cal_groups": sorted(roles[roles == "cal"].index),
                   "splits": {s: {"n": int((d.split == s).sum()), "sha": sha_ids(d.uid[d.split == s])}
                              for s in ("train", "cal_seen", "cal_unseen", "test_seen", "test_unseen")}}
    for loc in MASSIVE_LOCALES[1:]:
        d = load_massive(loc)
        m[f"massive_{loc}"] = {"n_test": int((d.part == "test").sum())}
    news = load_news()
    m["news"] = {"n": int(len(news)), "from": str(news.date.min().date()), "to": str(news.date.max().date()),
                 "classes": int(news.label.nunique()), "sha": sha_ids(news.uid)}
    return m


if __name__ == "__main__":
    m = manifest()
    save_json("data_manifest.json", m)
    for k, v in m.items():
        print(k, json.dumps(v)[:400])
