"""Scorer seam and candidate registry for the System 1 bake-off (issues #10, #12).

A `Scorer` is the one interface between the bake-off driver
(`scripts/run_system1_bakeoff.py`) and a model: it turns a list of row texts
into one probability in [0, 1] per row, in input order. Everything above this
seam is model-agnostic; everything below it is a thin adapter.

Adding a candidate = one small adapter class (a `__init__(keywords)`, the
five attributes below, and `score`) plus ONE line in `CANDIDATES`. Model
runtimes (torch, transformers, ...) must be imported lazily inside the adapter
(first `score()` call), never at module top level, so importing this module --
and the whole test suite -- never needs the optional `bakeoff` extra
(`pip install -e ".[bakeoff]"`).

Per CONTEXT.md these are System 1 models: zero-shot, probability out, no
generated rationale, never trained or tuned on this project's data. Each
adapter's `template` is frozen before its first scoring run and recorded in
the report -- editing it after seeing Ground-truth scores would be hidden
tuning, so don't.
"""

from __future__ import annotations

from typing import Callable, Protocol


class Scorer(Protocol):
    name: str  # display name in the report; also the registry key
    model_id: str  # e.g. the Hugging Face repo id
    revision: str  # exact pinned checkpoint revision (supply-chain audit trail)
    template: str  # frozen criteria template, with a `{keywords}` placeholder
    gate_eligible: bool  # False for AUROC-only references that can never get a GO

    def score(self, texts: list[str]) -> list[float]:
        """One probability in [0, 1] per text, same order. May raise."""
        ...


def fill_template(template: str, keywords: list[str]) -> str:
    """The one place config keywords are folded into a template."""
    return template.format(keywords="; ".join(keywords))


class Qwen3RerankerScorer:
    """Qwen3-Reranker-0.6B: one (instruction, query, document) prompt per row;
    probability = softmax over the final-position logits of the "yes"/"no"
    tokens, i.e. sigmoid(yes - no). Follows the model card's usage."""

    name = "qwen3-reranker-0.6b"
    model_id = "Qwen/Qwen3-Reranker-0.6B"
    revision = "e61197ed45024b0ed8a2d74b80b4d909f1255473"
    gate_eligible = True
    template = (
        "<Instruct>: Given a shipment record's product description, judge whether it describes "
        "any of the listed products, a clear synonym of one, or an obvious sub-component, "
        "variant or accessory of one.\n"
        "<Query>: Products: {keywords}"
    )
    _PREFIX = (
        "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query "
        'and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n'
        "<|im_start|>user\n"
    )
    _SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

    def __init__(self, keywords: list[str], batch_size: int = 8, max_length: int = 1024):
        self._criteria = fill_template(self.template, keywords)
        self._batch_size = batch_size
        self._max_length = max_length
        self._loaded = None

    def _load(self):
        if self._loaded is None:
            import torch  # lazy: optional `bakeoff` extra
            from transformers import AutoModelForCausalLM, AutoTokenizer

            tok = AutoTokenizer.from_pretrained(self.model_id, revision=self.revision, padding_side="left")
            model = AutoModelForCausalLM.from_pretrained(self.model_id, revision=self.revision).eval()
            device = "mps" if torch.backends.mps.is_available() else "cpu"
            model.to(device)
            self._loaded = (torch, tok, model, device)
        return self._loaded

    def score(self, texts: list[str]) -> list[float]:
        torch, tok, model, device = self._load()
        no_id, yes_id = tok.convert_tokens_to_ids("no"), tok.convert_tokens_to_ids("yes")
        prefix_ids = tok.encode(self._PREFIX, add_special_tokens=False)
        suffix_ids = tok.encode(self._SUFFIX, add_special_tokens=False)
        room = self._max_length - len(prefix_ids) - len(suffix_ids)

        out: list[float] = []
        for i in range(0, len(texts), self._batch_size):
            pairs = [f"{self._criteria}\n<Document>: {t}" for t in texts[i : i + self._batch_size]]
            enc = tok(pairs, padding=False, truncation=True, max_length=room, add_special_tokens=False)
            enc["input_ids"] = [prefix_ids + ids + suffix_ids for ids in enc["input_ids"]]
            batch = tok.pad({"input_ids": enc["input_ids"]}, padding=True, return_tensors="pt").to(device)
            with torch.no_grad():
                logits = model(**batch).logits[:, -1, :]
            pair_logits = torch.stack([logits[:, no_id], logits[:, yes_id]], dim=1)
            out.extend(torch.softmax(pair_logits.float(), dim=1)[:, 1].cpu().tolist())
        return out


class GteModernBertRerankerScorer:
    """gte-reranker-modernbert-base: a cross-encoder with a single-logit head;
    one (query, document) pair per row, probability = sigmoid(logit). Follows
    the model card's AutoModelForSequenceClassification usage."""

    name = "gte-reranker-modernbert-base"
    model_id = "Alibaba-NLP/gte-reranker-modernbert-base"
    revision = "f7481e6055501a30fb19d090657df9ec1f79ab2c"
    gate_eligible = True
    template = (
        "Given a shipment record's product description, judge whether it describes "
        "any of the listed products, a clear synonym of one, or an obvious sub-component, "
        "variant or accessory of one. Products: {keywords}"
    )

    def __init__(self, keywords: list[str], batch_size: int = 8, max_length: int = 1024):
        self._query = fill_template(self.template, keywords)
        self._batch_size = batch_size
        self._max_length = max_length
        self._loaded = None

    def _load(self):
        if self._loaded is None:
            import torch  # lazy: optional `bakeoff` extra
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            tok = AutoTokenizer.from_pretrained(self.model_id, revision=self.revision)
            model = AutoModelForSequenceClassification.from_pretrained(self.model_id, revision=self.revision).eval()
            device = "mps" if torch.backends.mps.is_available() else "cpu"
            model.to(device)
            self._loaded = (torch, tok, model, device)
        return self._loaded

    def score(self, texts: list[str]) -> list[float]:
        torch, tok, model, device = self._load()
        out: list[float] = []
        for i in range(0, len(texts), self._batch_size):
            docs = texts[i : i + self._batch_size]
            # only_second: truncate the document, never the criteria
            batch = tok([self._query] * len(docs), docs, padding=True, truncation="only_second",
                        max_length=self._max_length, return_tensors="pt").to(device)
            with torch.no_grad():
                logits = model(**batch).logits.view(-1)
            out.extend(torch.sigmoid(logits.float()).cpu().tolist())
        return out


def max_per_row(flat: list[float], n_keywords: int) -> list[float]:
    """Collapse row-major (row, keyword) probabilities to one max per row."""
    return [max(flat[i : i + n_keywords]) for i in range(0, len(flat), n_keywords)]


class DebertaNliScorer:
    """DeBERTa-v3-large zeroshot v2.0 (NLI): one hypothesis per keyword, the row
    text is the premise; row score = max entailment probability over keywords.
    Entailment probability = softmax over the (entailment, not_entailment) logits,
    as in the model card's multi_label=True pipeline."""

    name = "deberta-v3-large-zeroshot-v2.0"
    model_id = "MoritzLaurer/deberta-v3-large-zeroshot-v2.0"
    revision = "cf44676c28ba7312e5c5f8f8d2c22b3e0c9cdae2"
    gate_eligible = True
    template = "This shipment record describes {keywords}."  # filled with ONE keyword per hypothesis

    def __init__(self, keywords: list[str], batch_size: int = 16, max_length: int = 512):
        self._hypotheses = [fill_template(self.template, [k]) for k in keywords]
        self._batch_size = batch_size
        self._max_length = max_length
        self._loaded = None

    def _load(self):
        if self._loaded is None:
            import torch  # lazy: optional `bakeoff` extra
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            tok = AutoTokenizer.from_pretrained(self.model_id, revision=self.revision)
            model = AutoModelForSequenceClassification.from_pretrained(self.model_id, revision=self.revision).eval()
            device = "mps" if torch.backends.mps.is_available() else "cpu"
            model.to(device)
            self._loaded = (torch, tok, model, device)
        return self._loaded

    def _entail(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Entailment probability per (premise, hypothesis) pair; the only model-touching step."""
        torch, tok, model, device = self._load()
        ent = model.config.label2id["entailment"]
        out: list[float] = []
        for i in range(0, len(pairs), self._batch_size):
            chunk = pairs[i : i + self._batch_size]
            enc = tok([p for p, _ in chunk], [h for _, h in chunk], padding=True, truncation="only_first",
                      max_length=self._max_length, return_tensors="pt").to(device)
            with torch.no_grad():
                logits = model(**enc).logits
            out.extend(torch.softmax(logits.float(), dim=1)[:, ent].cpu().tolist())
        return out

    def score(self, texts: list[str]) -> list[float]:
        pairs = [(t, h) for t in texts for h in self._hypotheses]
        return max_per_row(self._entail(pairs), len(self._hypotheses))


# Registry: candidate name -> factory(keywords) -> Scorer. One line per candidate.
CANDIDATES: dict[str, Callable[[list[str]], Scorer]] = {
    Qwen3RerankerScorer.name: Qwen3RerankerScorer,
    GteModernBertRerankerScorer.name: GteModernBertRerankerScorer,
    DebertaNliScorer.name: DebertaNliScorer,
}


def build_scorers(names: list[str], keywords: list[str]) -> list[Scorer]:
    unknown = [n for n in names if n not in CANDIDATES]
    if unknown:
        raise ValueError(f"unknown candidate(s) {unknown}; known: {sorted(CANDIDATES)}")
    return [CANDIDATES[n](keywords) for n in names]
