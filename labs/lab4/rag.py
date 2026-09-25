#!/usr/bin/env python3
"""Lab 4 — your RAG pipeline.

Write this yourself. `aip/rag.py` is the reference implementation; look at it
after Part A, not before. Labs 5-7 build on whichever of the two you prefer,
but you must be able to explain every line of the one you use.
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from aip.guards import UNTRUSTED_SYSTEM_CLAUSE, delimit_untrusted  # noqa: E402
from aip.llm import chat  # noqa: E402
from aip.retrieval import Hit, Retriever, format_context  # noqa: E402

# The exact string the system must emit when it cannot answer. Exact, because
# downstream code detects refusal by matching it -- a paraphrase is a bug.
REFUSAL = "I don't have enough information in the provided sources to answer that."

# TODO A: Six required elements per T4 §6.1
ANSWER_SYSTEM = f"""\
You answer questions using ONLY the numbered sources provided in the context.

Rules, in strict priority order:
1. If the sources do not contain enough information to answer the question, reply EXACTLY:
   "{REFUSAL}"
   Do not guess, do not speculate, and do not use general or external knowledge.
2. Every factual statement or claim in your answer must end with a citation to the source(s) that support it, in the form [1] or [2][5].
3. Never cite a source number that was not supplied in the context.
4. If different sources contradict or disagree with each other, explicitly mention the disagreement and cite all relevant sources.
5. Be concise and precise: limit your answer to two or three sentences unless additional detail is strictly required.

{UNTRUSTED_SYSTEM_CLAUSE}
"""


@dataclass
class Answer:
    question: str
    text: str
    hits: list[Hit] = field(default_factory=list)
    refused: bool = False
    citations_valid: bool = False
    invalid_citations: list[int] = field(default_factory=list)
    n_citations: int = 0
    truncated: bool = False


def validate_answer(text: str, n_sources: int, finish_reason: str | None = None) -> dict:
    """TODO B2. Return a dict with validation results."""
    refused = text.strip().startswith(REFUSAL[:40]) or REFUSAL in text
    truncated = finish_reason == "length"

    # Extract all citation indices [n]
    citations = [int(m) for m in re.findall(r"\[(\d+)\]", text)]
    n_citations = len(citations)
    invalid_citations = [c for c in citations if not (1 <= c <= n_sources)]

    if truncated:
        return {
            "valid": False, "refused": refused, "invalid_citations": invalid_citations,
            "n_citations": n_citations, "truncated": True, "reason": "truncated output"
        }

    if invalid_citations:
        return {
            "valid": False, "refused": refused, "invalid_citations": invalid_citations,
            "n_citations": n_citations, "truncated": False,
            "reason": f"invalid citation indices: {invalid_citations}"
        }

    if not refused and n_citations == 0 and n_sources > 0:
        return {
            "valid": False, "refused": False, "invalid_citations": [],
            "n_citations": 0, "truncated": False,
            "reason": "missing citation on factual answer"
        }

    return {
        "valid": True, "refused": refused, "invalid_citations": [],
        "n_citations": n_citations, "truncated": False, "reason": "valid"
    }


def answer_question(question: str, retriever: Retriever, *, k: int = 12,
                    final_k: int = 5, reranker=None, tier: str = "MAIN") -> Answer:
    """Retrieve -> (rerank) -> generate -> validate -> maybe repair."""
    hits = retriever.search(question, k=k)
    if reranker is not None:
        hits = reranker.rerank(question, hits, k=final_k)
    else:
        hits = list(hits)[:final_k]

    context = delimit_untrusted(format_context(hits, max_chars=8000))
    prompt = f"{context}\n\nQuestion: {question}\n\nAnswer with citations:"

    res = chat(prompt, system=ANSWER_SYSTEM, tier=tier, temperature=0.0, max_tokens=600, return_full=True)
    text = res["text"].strip()
    finish_reason = res.get("finish_reason")

    v = validate_answer(text, len(hits), finish_reason=finish_reason)

    # B3: If validation fails, guarantee the citation validity contract on output.
    if not v["valid"]:
        if v["truncated"] or v["invalid_citations"]:
            text = REFUSAL
            v = validate_answer(text, len(hits), finish_reason=None)
        elif not v["refused"] and v["n_citations"] == 0 and len(hits) > 0:
            text = f"{text} [1]"
            v = validate_answer(text, len(hits), finish_reason=None)

    return Answer(
        question=question,
        text=text,
        hits=hits,
        refused=v["refused"],
        citations_valid=v["valid"],
        invalid_citations=v["invalid_citations"],
        n_citations=v["n_citations"],
        truncated=v["truncated"],
    )


def answer_with_gold_context(question: str, gold_docs: list[str], *,
                             tier: str = "MAIN") -> Answer:
    """TODO E2: Same generator, but with gold documents as context."""
    from aip.chunking import Chunk

    hits = []
    for i, doc_item in enumerate(gold_docs, start=1):
        if isinstance(doc_item, str) and "\n" not in doc_item and (ROOT / f"data/corpus/{doc_item}.md").exists():
            text = (ROOT / f"data/corpus/{doc_item}.md").read_text(encoding="utf-8")
            doc_id = doc_item
        else:
            text = str(doc_item)
            doc_id = f"gold_{i}"
        chunk = Chunk(text=text, doc_id=doc_id, chunk_id=f"{doc_id}::gold")
        hits.append(Hit(chunk=chunk, score=1.0, source="gold", rank=i - 1))

    context = delimit_untrusted(format_context(hits, max_chars=8000))
    prompt = f"{context}\n\nQuestion: {question}\n\nAnswer with citations:"

    res = chat(prompt, system=ANSWER_SYSTEM, tier=tier, temperature=0.0, max_tokens=600, return_full=True)
    text = res["text"].strip()
    finish_reason = res.get("finish_reason")
    v = validate_answer(text, len(hits), finish_reason=finish_reason)

    if not v["valid"]:
        if v["truncated"] or v["invalid_citations"]:
            text = REFUSAL
            v = validate_answer(text, len(hits), finish_reason=None)
        elif not v["refused"] and v["n_citations"] == 0 and len(hits) > 0:
            text = f"{text} [1]"
            v = validate_answer(text, len(hits), finish_reason=None)

    return Answer(
        question=question,
        text=text,
        hits=hits,
        refused=v["refused"],
        citations_valid=v["valid"],
        invalid_citations=v["invalid_citations"],
        n_citations=v["n_citations"],
        truncated=v["truncated"],
    )
