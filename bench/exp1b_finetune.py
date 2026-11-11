"""Experiment 1b: the heaviest fast-tier option, a small encoder fine-tuned end to end.

all-MiniLM-L6-v2 (22M parameters) with a linear head, trained on the task's own training
split on CPU. Reported next to the frozen-embedding heads of exp1: accuracy, risk-coverage,
the operational point certified on the calibration split, out-of-scope detection (CLINC),
robustness to keyboard noise, per-request latency and training time.
"""
from __future__ import annotations

import sys
import time

import numpy as np
import torch

from . import data, kit, precompute
from . import models as M
from .common import save_json

NAME = "sentence-transformers/all-MiniLM-L6-v2"


class Net(torch.nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()
        from transformers import AutoModel
        self.enc = AutoModel.from_pretrained(NAME)
        self.head = torch.nn.Linear(self.enc.config.hidden_size, n_classes)

    def forward(self, **b):
        h = self.enc(**b).last_hidden_state
        m = b["attention_mask"].unsqueeze(-1).float()
        return self.head((h * m).sum(1) / m.sum(1))


def logits_for(net, tok, texts, bs: int = 256) -> np.ndarray:
    net.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), bs):
            b = tok(list(texts[i:i + bs]), padding=True, truncation=True, max_length=64, return_tensors="pt")
            out.append(net(**b).numpy())
    return np.vstack(out)


def run(name: str, epochs: int = 4, lr: float = 1e-4, seed: int = 0) -> dict:
    from transformers import AutoTokenizer
    torch.manual_seed(seed)
    torch.set_num_threads(6)
    frame = data.fine_frame(name)
    parts = {s: frame[frame.split == s] for s in ("train", "cal", "test")}
    classes = sorted(frame.label.unique())
    cid = {c: i for i, c in enumerate(classes)}
    tok = AutoTokenizer.from_pretrained(NAME)
    net = Net(len(classes))
    tr_texts, tr_y = list(parts["train"].text), torch.tensor([cid[c] for c in parts["train"].label])
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.01)
    steps = epochs * ((len(tr_texts) + 31) // 32)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    t0 = time.perf_counter()
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(tr_texts))
        for i in range(0, len(perm), 32):
            idx = perm[i:i + 32]
            b = tok([tr_texts[j] for j in idx], padding=True, truncation=True, max_length=64, return_tensors="pt")
            loss = torch.nn.functional.cross_entropy(net(**b), tr_y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        print(f"  {name} epoch {ep + 1}/{epochs} loss {loss.item():.3f}  {time.perf_counter() - t0:.0f}s", flush=True)
    fit_s = time.perf_counter() - t0

    def sig(texts):
        z = logits_for(net, tok, texts)
        P = M.softmax(z, axis=1)
        return {"pred": np.array(classes, dtype=object)[P.argmax(1)], **M.confidence_signals(P, z)}
    cal, te = sig(list(parts["cal"].text)), sig(list(parts["test"].text))
    c_cal, c_te = cal["pred"] == parts["cal"].label.values, te["pred"] == parts["test"].label.values
    res = {"accuracy": float(c_te.mean()), "fit_s": round(fit_s, 1), "epochs": epochs,
           **kit.selective_summary(te["msp"], c_te, cal["msp"], c_cal)}
    if name == "clinc150":
        oos = data.clinc_oos()
        o = sig(list(oos[oos.part == "test"].text))
        res["oos"] = {d: {"auroc": M.auroc(te[d], o[d]), "fpr95": M.fpr_at_tpr(te[d], o[d])} for d in ("msp", "energy", "max_logit")}
        thr = res["fs@0.02"]["thr"]
        res["oos"]["false_accept_at_certified_threshold"] = float((o["msp"] >= thr).mean())
        for tag in ("clinc_test_typo05", "clinc_test_typo15"):
            s = sig(precompute.corpora()[tag])
            c = s["pred"] == parts["test"].label.values
            a = s["msp"] >= thr
            res[tag] = {"accuracy": float(c.mean()), "coverage": float(a.mean()),
                        "silent_error_rate": float((a & ~c).sum() / max(1, a.sum()))}
    # single-request latency
    net.eval()
    ms = []
    with torch.no_grad():
        for t in list(parts["test"].text[:205]):
            s = time.perf_counter()
            b = tok([t], truncation=True, max_length=64, return_tensors="pt")
            net(**b)
            ms.append((time.perf_counter() - s) * 1e3)
    res["latency_ms_p50"] = float(np.median(ms[5:]))
    print(f"  {name} fine-tuned minilm: acc {res['accuracy']:.4f} aurc {res['aurc']:.4f} cov@2% {res['fs@0.02']['coverage']:.3f} "
          f"err {res['fs@0.02']['silent_error_rate']:.4f} fit {fit_s:.0f}s latency {res['latency_ms_p50']:.1f} ms", flush=True)
    if "oos" in res:
        print("   oos:", {k: v for k, v in res["oos"].items()}, " typo15:", res["clinc_test_typo15"], flush=True)
    return res


if __name__ == "__main__":
    names = sys.argv[1:] or ["clinc150", "banking77"]
    save_json("exp1b_finetune.json", {n: run(n) for n in names})
