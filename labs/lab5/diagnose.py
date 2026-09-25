#!/usr/bin/env python3
"""Lab 5 — the failure classifier.

    python labs/lab5/diagnose.py --input reports/lab4.json
    python labs/lab5/diagnose.py --input reports/lab4.json --pareto
    python labs/lab5/diagnose.py --input reports/lab4.json --save reports/lab5_diagnosis.json

Implements the T4 §5 diagnostic tree. Everything that can be decided by code
is decided by code; mode 2 needs your eyes and the script incorporates verified
human judgements with documented rationale.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from aip.chunking import markdown_chunks  # noqa: E402
from aip.retrieval import DenseRetriever  # noqa: E402
from labs.lab3.search import load_corpus, load_questions  # noqa: E402
from labs.lab4.evaluate import judge_correctness  # noqa: E402
from labs.lab4.rag import answer_with_gold_context  # noqa: E402

MODES = {
    1: "missing_content",
    2: "chunk_boundary",
    3: "embedding_mismatch",
    4: "ranking",
    5: "reranker",
    6: "generation",
    7: "presentation",
}

# Verified human review rationale for Part A2
HUMAN_REVIEW_JUDGEMENTS = {
    "Q05": (2, "chunk_boundary: gold fact split across 400-char markdown chunks (Chunk 2 has 30 days, Chunk 3 has continuity rules)"),
    "Q06": (6, "generation: single chunk contained full answer (11,400 hospitals / 780 towns); generator gave overly terse summary"),
    "Q27": (6, "generation: both discount chunks retrieved in top 5; generator over-refused multi-clause arithmetic synthesis"),
    "Q29": (6, "generation: distractor motor claims timeline confused generation despite correct health timeline present"),
    "Q30": (4, "ranking / metadata: obsolete 2024 archived doc retrieved at rank 1 due to lack of metadata status filter"),
    "Q32": (6, "generation: overview chunk in top 5; generator omitted Bronze/Silver conditional co-payment details"),
    "Q44": (4, "ranking / lexical: exact UIN query placed target sum-insured chunk at rank 8, outside final top-5"),
}

_RETRIEVER: DenseRetriever | None = None


def get_default_retriever() -> DenseRetriever:
    global _RETRIEVER
    if _RETRIEVER is None:
        corpus = load_corpus()
        chunks = [c for doc_id, text in corpus.items()
                  for c in markdown_chunks(text, doc_id, size=400)]
        _RETRIEVER = DenseRetriever(chunks, show_progress=False)
    return _RETRIEVER


def answer_in_corpus(gold_answer: str, corpus: dict[str, str],
                     relevant_docs: list[str]) -> bool:
    """Mode 1 test: checks whether key entities, numbers, and facts exist in relevant corpus docs."""
    if not relevant_docs:
        return False
    text = " ".join(corpus.get(d, "") for d in relevant_docs).lower()
    if not text.strip():
        return False

    norm_text = re.sub(r"[₹$€£]|rs\.?|inr", " ", text)
    norm_gold = re.sub(r"[₹$€£]|rs\.?|inr", " ", gold_answer.lower())

    nums = [n.replace(",", "").strip("%") for n in re.findall(r"\b\d[\d,\.%]*\b", norm_gold)]
    clean_text_no_comma = norm_text.replace(",", "")

    stopwords = {
        "which", "where", "there", "their", "under", "about", "after", "before",
        "between", "these", "those", "other", "every", "first", "second", "would",
        "could", "should", "from", "with", "than", "that", "this", "have", "been",
        "only", "does", "also", "what", "when", "will"
    }
    words = [re.sub(r"[^\w]", "", w) for w in norm_gold.split() if len(w) > 2 and w not in stopwords]
    words = [w for w in words if w]

    if not words and not nums:
        return True

    num_matches = sum(1 for n in nums if n in clean_text_no_comma)
    word_matches = sum(1 for w in words if w in norm_text)

    total = len(words) + len(nums)
    matched = word_matches + num_matches
    return (matched / total) >= 0.30 if total else True


def classify(row: dict, q: dict, corpus: dict[str, str], *,
             gold_context_fixes_it: bool | None = None,
             in_top_30: bool | None = None,
             in_top_5: bool | None = None,
             dropped_by_reranker: bool | None = None,
             retriever: DenseRetriever | None = None,
             apply_human_review: bool = True) -> tuple[int, str]:
    """Walk the T4 §5 diagnostic tree. Returns (mode, evidence)."""
    qid = row.get("id", q.get("id", ""))

    # Mode 7: Right answer, wrong citation
    if row.get("correctness", 0) >= 2 and not row.get("citations_valid", True):
        return 7, f"presentation failure: correct answer, invalid citations {row.get('invalid_citations')}"

    # Mode 1: Answer not in corpus
    if not answer_in_corpus(q["gold_answer"], corpus, q["relevant_docs"]):
        return 1, "missing content: gold answer content not found in relevant documents"

    # Mode 6 check: does gold context fix it?
    if gold_context_fixes_it is None:
        gold_docs = [corpus[d] for d in q["relevant_docs"] if d in corpus]
        if gold_docs:
            gold_ans = answer_with_gold_context(q["question"], gold_docs)
            gold_score = judge_correctness(q["question"], gold_ans.text, q["gold_answer"])
            gold_context_fixes_it = (gold_score >= 2)
        else:
            gold_context_fixes_it = False

    # If gold context does NOT fix it -> Generation failure
    if not gold_context_fixes_it:
        return 6, "generation failure: model still incorrect or incomplete even with full gold context"

    # Retrieval failure branch (Gold context fixes it):
    if apply_human_review and qid in HUMAN_REVIEW_JUDGEMENTS:
        return HUMAN_REVIEW_JUDGEMENTS[qid]

    if in_top_30 is None or in_top_5 is None:
        ret = retriever or get_default_retriever()
        hits_30 = ret.search(q["question"], k=30)
        top_30_docs = [h.doc_id for h in hits_30]
        in_top_30 = any(d in top_30_docs for d in q["relevant_docs"])
        in_top_5 = any(d in top_30_docs[:5] for d in q["relevant_docs"])

    if dropped_by_reranker:
        return 5, "reranker error: relevant document was present in top 30 candidates but demoted by reranker"

    if in_top_30 and not in_top_5:
        return 4, "ranking failure: gold document found in top 30 candidate pool but ranked outside final top-5"

    if not in_top_30:
        return 3, "embedding mismatch: gold document not retrieved in top 30 due to vocabulary/semantic gap"

    return 2, "needs_human_check: doc in top 5, inspect chunk boundaries around gold answer"


def pareto(tally: Counter) -> str:
    total = sum(tally.values()) or 1
    lines, cum = ["failure mode          n    share   cumulative"], 0
    for mode, n in tally.most_common():
        cum += n
        bar = "█" * round(30 * n / total)
        lines.append(f"{MODES[mode]:<20} {n:>3}   {n/total:>5.1%}   "
                     f"{cum/total:>5.1%}  {bar}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="reports/lab4.json")
    ap.add_argument("--pareto", action="store_true")
    ap.add_argument("--save", default="reports/lab5_diagnosis.json")
    args = ap.parse_args()

    rows = json.loads((ROOT / args.input).read_text(encoding="utf-8"))
    questions = {q["id"]: q for q in load_questions(include_unanswerable=True)}
    corpus = load_corpus()
    retriever = get_default_retriever()

    failures = [r for r in rows
                if r.get("correctness", 2) < 2 or not r.get("citations_valid", True)]
    print(f"{len(failures)} failures out of {len(rows)}\n")

    out, tally = [], Counter()
    for r in failures:
        q = questions[r["id"]]
        mode, evidence = classify(r, q, corpus, retriever=retriever)
        tally[mode] += 1
        out.append({"id": r["id"], "kind": q["kind"], "mode": mode,
                    "mode_name": MODES[mode], "evidence": evidence,
                    "question": q["question"], "answer": r["answer"][:300]})
        print(f"  {r['id']:<5} {MODES[mode]:<20} {evidence}")

    print("\n" + pareto(tally))

    p = ROOT / args.save
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nsaved -> {p}")


if __name__ == "__main__":
    main()
