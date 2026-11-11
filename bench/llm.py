"""The LLM tier: a real local model answering a typed, closed-set question.

Runs an MLX-quantised instruct model on this machine's GPU. A "hole" is a fixed
instruction prefix (the task contract and the closed answer set) followed by
one query. The prefix is processed once and its KV cache reused for every
query, so each call only pays for the query tokens and the few answer tokens.

Every answer is stored in cache/llm/<model>__<hole>.jsonl keyed by the exact
query text, so experiments replay real model outputs instead of re-running or
inventing them. Stored per answer: raw text, parsed label (None if the reply is
not in the closed set), the model's own probability for the generated answer
(product of token probabilities), and how the call was timed.

Two timings are recorded and must not be confused:
  batch_s_per_item   wall time of a batch / batch size (throughput labelling)
  single-call latency measured separately by `latency_probe` with batch size 1,
                     which is what one production request would wait for.
"""
from __future__ import annotations

import json
import math
import re
import time
from typing import Optional, Sequence

from .common import CACHE, sha_text

LLM_DIR = CACHE / "llm"
LLM_DIR.mkdir(parents=True, exist_ok=True)

MODELS = {
    "qwen3-4b": "mlx-community/Qwen3-4B-Instruct-2507-4bit",
    "gemma3-4b": "mlx-community/gemma-3-4b-it-qat-4bit",
    "llama32-3b": "mlx-community/Llama-3.2-3B-Instruct-4bit",
    "qwen3-8b": "mlx-community/Qwen3-8B-4bit",
}


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


class Hole:
    """A typed question: instruction text + closed answer set (+ optional 'none' exit)."""

    def __init__(self, name: str, instruction: str, choices: Sequence[str], none_label: Optional[str] = None,
                 query_prefix: str = "Query: ", answer_cue: str = "", max_tokens: int = 12):
        self.name, self.instruction = name, instruction
        self.choices, self.none_label = list(choices), none_label
        self.query_prefix, self.answer_cue, self.max_tokens = query_prefix, answer_cue, max_tokens
        self._lookup = {_norm(c): c for c in self.choices}
        if none_label:
            self._lookup[_norm(none_label)] = none_label
        self.version = sha_text(instruction + "|" + "|".join(self.choices) + f"|{none_label}|{query_prefix}|{answer_cue}")

    def user_text(self, query: str) -> str:
        return f"{self.instruction}\n\n{self.query_prefix}{query}{self.answer_cue}"

    def parse(self, raw: str) -> Optional[str]:
        """Exact closed-set match on the first line; anything else is None (the hole abstains)."""
        first = raw.strip().split("\n")[0]
        first = re.sub(r"^(answer|label|category|domain|intent)\s*[:\-]\s*", "", first.strip(), flags=re.I)
        return self._lookup.get(_norm(first))


class LocalLLM:
    def __init__(self, key: str):
        from mlx_lm import load
        self.key, self.repo = key, MODELS.get(key, key)
        t0 = time.perf_counter()
        self.model, self.tok = load(self.repo)
        self.load_s = time.perf_counter() - t0
        self._prefix = {}

    # -- prompt assembly ----------------------------------------------------
    def _tokens(self, hole: Hole, query: str) -> list:
        msgs = [{"role": "user", "content": hole.user_text(query)}]
        kw = {}
        try:
            return list(self.tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                                     enable_thinking=False))
        except TypeError:
            return list(self.tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, **kw))

    def _prefix_cache(self, hole: Hole):
        """KV cache for the token prefix shared by every query of this hole."""
        if hole.version in self._prefix:
            return self._prefix[hole.version]
        import mlx.core as mx
        from mlx_lm.models.cache import make_prompt_cache
        a, b = self._tokens(hole, "alpha one"), self._tokens(hole, "zebra 2 ?")
        n = 0
        while n < min(len(a), len(b)) and a[n] == b[n]:
            n += 1
        n = max(0, n - 1)                        # leave a token so merges at the boundary stay stable
        cache = make_prompt_cache(self.model)
        if n > 0:
            self.model(mx.array(a[:n])[None], cache=cache)
            mx.eval([c.state for c in cache])
        self._prefix[hole.version] = (n, a[:n], cache)
        return self._prefix[hole.version]

    # -- generation ---------------------------------------------------------
    def ask(self, hole: Hole, queries: Sequence[str], batch_size: int = 16, use_prefix_cache: bool = True) -> list:
        """Answer every query. Returns dicts in order; nothing is cached here."""
        from mlx_lm.generate import batch_generate
        n, prefix, cache = self._prefix_cache(hole) if use_prefix_cache else (0, [], None)
        out = []
        for i in range(0, len(queries), batch_size):
            chunk = list(queries[i:i + batch_size])
            toks = [self._tokens(hole, q) for q in chunk]
            for t in toks:
                assert t[:n] == prefix, "prompt prefix changed; cannot reuse the KV cache"
            t0 = time.perf_counter()
            resp = batch_generate(self.model, self.tok, [t[n:] for t in toks],
                                  prompt_caches=[cache] * len(chunk) if n else None,
                                  max_tokens=hole.max_tokens, return_logprobs=True,
                                  completion_batch_size=max(batch_size, 8), prefill_batch_size=max(1, min(batch_size, 16)))
            dt = time.perf_counter() - t0
            for q, t, text, lps in zip(chunk, toks, resp.texts, resp.logprobs):
                label = hole.parse(text)
                out.append({"q": q, "raw": text.strip()[:80], "label": label,
                            "p": float(math.exp(sum(lps))) if lps else 0.0, "n_gen": len(lps),
                            "n_prompt": len(t), "n_new_prompt": len(t) - n,
                            "batch": len(chunk), "batch_s_per_item": dt / len(chunk)})
        return out

    def chat(self, user_texts: Sequence[str], max_tokens: int = 128, batch_size: int = 4) -> list:
        """Free-form generation for arbitrary prompts (no shared prefix). Used by the agent runs
        the harness observes. Returns text plus token counts and per-item batch time."""
        from mlx_lm.generate import batch_generate
        out = []
        for i in range(0, len(user_texts), batch_size):
            chunk = list(user_texts[i:i + batch_size])
            toks = []
            for u in chunk:
                msgs = [{"role": "user", "content": u}]
                try:
                    toks.append(list(self.tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                                                  enable_thinking=False)))
                except TypeError:
                    toks.append(list(self.tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)))
            t0 = time.perf_counter()
            resp = batch_generate(self.model, self.tok, toks, max_tokens=max_tokens, return_token_ids=True,
                                  completion_batch_size=max(batch_size, 8), prefill_batch_size=max(1, batch_size))
            dt = time.perf_counter() - t0
            for t, text, ids in zip(toks, resp.texts, resp.token_ids):
                out.append({"text": text.strip(), "n_prompt": len(t), "n_gen": len(ids),
                            "batch": len(chunk), "batch_s_per_item": dt / len(chunk)})
        return out

    def latency_probe(self, hole: Hole, queries: Sequence[str], use_prefix_cache: bool = True) -> list:
        """Single-request latency (batch size 1), in milliseconds per call."""
        ms = []
        for q in queries:
            t0 = time.perf_counter()
            self.ask(hole, [q], batch_size=1, use_prefix_cache=use_prefix_cache)
            ms.append((time.perf_counter() - t0) * 1e3)
        return ms


# --------------------------------------------------------------------------- answer store
class AnswerStore:
    """Append-only store of real LLM answers for one (model, hole)."""

    def __init__(self, model_key: str, hole: Hole):
        self.path = LLM_DIR / f"{model_key}__{hole.name}__{hole.version}.jsonl"
        self.meta_path = self.path.with_suffix(".meta.json")
        self.rows = {}
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                r = json.loads(line)
                self.rows[r["q"]] = r
        if not self.meta_path.exists():
            self.meta_path.write_text(json.dumps({"model": model_key, "hole": hole.name, "version": hole.version,
                                                  "choices": hole.choices, "none_label": hole.none_label,
                                                  "instruction": hole.instruction}, indent=1))

    def missing(self, queries: Sequence[str]) -> list:
        seen, out = set(), []
        for q in queries:
            if q not in self.rows and q not in seen:
                seen.add(q)
                out.append(q)
        return out

    def add(self, rows: Sequence[dict]) -> None:
        with open(self.path, "a") as f:
            for r in rows:
                self.rows[r["q"]] = r
                f.write(json.dumps(r) + "\n")

    def get(self, queries: Sequence[str]) -> list:
        return [self.rows[q] for q in queries]

    def has_all(self, queries: Sequence[str]) -> bool:
        return all(q in self.rows for q in queries)


def label_queries(llm: LocalLLM, hole: Hole, queries: Sequence[str], batch_size: int = 16, log_every: int = 200) -> AnswerStore:
    store = AnswerStore(llm.key, hole)
    todo = store.missing(queries)
    t0 = time.perf_counter()
    for i in range(0, len(todo), log_every):
        chunk = todo[i:i + log_every]
        store.add(llm.ask(hole, chunk, batch_size=batch_size))
        done = i + len(chunk)
        print(f"  [{llm.key}/{hole.name}] {done}/{len(todo)}  {(time.perf_counter() - t0) / done * 1e3:.0f} ms/item",
              flush=True)
    return store
