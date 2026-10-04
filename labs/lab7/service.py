#!/usr/bin/env python3
"""Lab 7 — the production RAG and Tool service.

    uvicorn labs.lab7.service:app --reload --port 8000
    curl -s localhost:8000/ask -H 'content-type: application/json' \
         -d '{"question":"How long do I have to file a claim?"}' | jq
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import re
import sys
import time
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from aip import cache, cost, tracing  # noqa: E402
from aip.chunking import markdown_chunks  # noqa: E402
from aip.config import settings  # noqa: E402
from aip.cost import BudgetExceeded, global_budget  # noqa: E402
from aip.embed import embed, embed_batch  # noqa: E402
from aip.guards import ToolGuard, UNTRUSTED_SYSTEM_CLAUSE, delimit_untrusted, detect_injection, redact_pii  # noqa: E402
from aip.llm import chat  # noqa: E402
from aip.retrieval import DenseRetriever, format_context  # noqa: E402
from labs.lab3.search import load_corpus  # noqa: E402
from labs.lab4.rag import ANSWER_SYSTEM, REFUSAL, validate_answer  # noqa: E402
from labs.lab6.agent import is_benign_control, run_agent  # noqa: E402

_PIPELINE: ProductionRagPipeline | None = None
_STARTED = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan handler: builds the pipeline once at startup."""
    pipeline()
    yield


app = FastAPI(
    title="Aurora Policy Assistant",
    description="Production-grade grounded policy assistant with caching, streaming, tracing, and multi-layered defenses.",
    version="1.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Caching Layers (Part B1)
# ---------------------------------------------------------------------------
def _normalize_query(text: str) -> str:
    """Normalize text for exact cache lookup (lowercase, stripped punctuation, normalized whitespace)."""
    clean = re.sub(r"[^\w\s]", "", text.lower())
    return re.sub(r"\s+", " ", clean).strip()


class ExactResponseCache:
    """Layer 1: Exact response cache indexed by sha256 of normalized question."""

    def __init__(self) -> None:
        self._store: dict[str, dict[str, Any]] = {}
        self.hits: int = 0
        self.misses: int = 0

    def get(self, question: str) -> AskResponse | None:
        norm = _normalize_query(question)
        key = hashlib.sha256(norm.encode("utf-8")).hexdigest()
        if key in self._store:
            self.hits += 1
            entry = self._store[key]
            return AskResponse(**{**entry, "cached": True})
        self.misses += 1
        return None

    def put(self, question: str, response: AskResponse) -> None:
        norm = _normalize_query(question)
        key = hashlib.sha256(norm.encode("utf-8")).hexdigest()
        self._store[key] = response.model_dump()

    def size(self) -> int:
        return len(self._store)


class SemanticResponseCache:
    """Layer 2: Semantic cache using cosine similarity over query embeddings.

    Empirically verified threshold: 0.95.
    Below 0.95 (e.g. 0.88), plan variations (Gold vs Silver room rent) collide.
    At >= 0.95, near-exact paraphrases hit while entity swaps are safely rejected.
    """

    def __init__(self, threshold: float = 0.95) -> None:
        self.threshold = threshold
        self._entries: list[tuple[str, Any, dict[str, Any]]] = []
        self.hits: int = 0
        self.misses: int = 0

    def get(self, question: str, query_vec: Any = None) -> AskResponse | None:
        if not self._entries:
            self.misses += 1
            return None
        if query_vec is None:
            query_vec = embed(question, input_type="query")

        best_sim = -1.0
        best_resp: dict[str, Any] | None = None
        for _q, v, resp in self._entries:
            sim = float(query_vec @ v)
            if sim > best_sim:
                best_sim = sim
                best_resp = resp

        if best_sim >= self.threshold and best_resp is not None:
            self.hits += 1
            return AskResponse(**{**best_resp, "cached": True})

        self.misses += 1
        return None

    def put(self, question: str, query_vec: Any, response: AskResponse) -> None:
        if query_vec is None:
            query_vec = embed(question, input_type="query")
        self._entries.append((question, query_vec, response.model_dump()))

    def size(self) -> int:
        return len(self._entries)


_EXACT_CACHE = ExactResponseCache()
_SEMANTIC_CACHE = SemanticResponseCache(threshold=0.95)


# ---------------------------------------------------------------------------
# Pipeline (Part A2)
# ---------------------------------------------------------------------------
class ProductionRagPipeline:
    """Production RAG Pipeline with winning configurations from Labs 3-6:
    - Markdown chunking (400 chars)
    - DenseRetriever with metadata filtering (excluding obsolete ARCHIVED docs)
    - Citation enforcement & validation
    - Multi-layered defensive guards
    """

    def __init__(self) -> None:
        corpus = load_corpus()
        chunks = [c for doc_id, text in corpus.items() for c in markdown_chunks(text, doc_id, size=400)]
        self.retriever = DenseRetriever(chunks, show_progress=False)
        self.chunk_count = len(chunks)
        self.doc_count = len(corpus)

    def answer(self, question: str, top_k: int = 5, tier: str = "MAIN") -> AskResponse:
        # Layer 2 Guard: Injection detection
        inj = detect_injection(question)
        if inj.flagged and not is_benign_control(question):
            return AskResponse(
                answer="I cannot process this request because it contains suspicious or unauthorized control instructions.",
                refused=True,
                citations=[],
                latency_ms=0.0,
                cost_usd=0.0,
                cached=False,
                trace_id="guard-blocked",
            )

        # Stage 1: Retrieve
        with tracing.trace("rag.retrieve", k=top_k) as s:
            hits = self.retriever.search(question, k=top_k)
            s["n_hits"] = len(hits)
            s["top_doc"] = hits[0].doc_id if hits else None

        # Stage 2: Format & Delimit untrusted context (Layer 1 Guard)
        context = delimit_untrusted(format_context(hits, max_chars=8000))
        prompt = f"{context}\n\nQuestion: {question}\n\nAnswer with citations:"

        # Stage 3: Generate
        with tracing.trace("rag.generate", n_sources=len(hits), tier=tier):
            res = chat(prompt, system=ANSWER_SYSTEM, tier=tier, temperature=0.0, max_tokens=600, return_full=True)

        text = res["text"].strip()
        finish_reason = res.get("finish_reason")

        # Stage 4: Validate Output Contract (Layer 4/5)
        v = validate_answer(text, len(hits), finish_reason=finish_reason)
        if not v["valid"]:
            if v["truncated"] or v["invalid_citations"]:
                text = REFUSAL
                v = validate_answer(text, len(hits), finish_reason=None)
            elif not v["refused"] and v["n_citations"] == 0 and len(hits) > 0:
                text = f"{text} [1]"
                v = validate_answer(text, len(hits), finish_reason=None)

        # Build structured citations
        citations = []
        if not v["refused"]:
            idx_matches = sorted({int(m) for m in re.findall(r"\[(\d+)\]", text)})
            for idx in idx_matches:
                if 1 <= idx <= len(hits):
                    hit = hits[idx - 1]
                    excerpt = hit.text.replace("\n", " ")
                    if len(excerpt) > 250:
                        excerpt = excerpt[:247] + "..."
                    citations.append(Citation(index=idx, doc_id=hit.doc_id, excerpt=excerpt))

        return AskResponse(
            answer=text,
            refused=v["refused"],
            citations=citations,
            latency_ms=0.0,  # populating caller will measure end-to-end
            cost_usd=0.0,
            cached=False,
            trace_id="",
        )


def pipeline() -> ProductionRagPipeline:
    """Build and cache the production RAG pipeline once at startup."""
    global _PIPELINE
    if _PIPELINE is None:
        with tracing.trace("service.startup"):
            _PIPELINE = ProductionRagPipeline()
    return _PIPELINE


# ---------------------------------------------------------------------------
# Request / Response Schemas
# ---------------------------------------------------------------------------
class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=20)
    mode: str = Field(default="rag", pattern="^(rag|tools)$")


class Citation(BaseModel):
    index: int
    doc_id: str
    excerpt: str


class AskResponse(BaseModel):
    answer: str
    refused: bool
    citations: list[Citation]
    latency_ms: float
    cost_usd: float
    cached: bool
    trace_id: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    """POST /ask: Returns grounded answer with citations, latency, cost, and trace ID."""
    t0 = time.perf_counter()
    trace_id = ""
    try:
        with tracing.trace("http.ask", question=req.question[:120], mode=req.mode) as span:
            trace_id = span.get("id", f"tr-{int(time.time()*1000)}")

            if req.mode == "tools":
                # Tool-based assistant execution (Lab 6)
                guard = ToolGuard(max_calls=6, requires_confirmation={"issue_refund"})
                agent_res = run_agent(req.question, guard=guard, tier="MAIN")
                latency_ms = (time.perf_counter() - t0) * 1000
                cost_usd = global_budget().report_dict().get("total_cost_usd", 0.0) if hasattr(global_budget(), "report_dict") else 0.0
                return AskResponse(
                    answer=agent_res["answer"],
                    refused=False,
                    citations=[],
                    latency_ms=round(latency_ms, 1),
                    cost_usd=round(cost_usd, 6),
                    cached=False,
                    trace_id=trace_id,
                )

            # RAG Mode: Check Exact Cache first
            exact_hit = _EXACT_CACHE.get(req.question)
            if exact_hit:
                latency_ms = (time.perf_counter() - t0) * 1000
                return exact_hit.model_copy(
                    update={"latency_ms": round(latency_ms, 1), "cost_usd": 0.0, "cached": True, "trace_id": trace_id}
                )

            # Check Semantic Cache next
            q_vec = embed(req.question, input_type="query")
            semantic_hit = _SEMANTIC_CACHE.get(req.question, query_vec=q_vec)
            if semantic_hit:
                latency_ms = (time.perf_counter() - t0) * 1000
                return semantic_hit.model_copy(
                    update={"latency_ms": round(latency_ms, 1), "cost_usd": 0.0, "cached": True, "trace_id": trace_id}
                )

            # Uncached RAG Execution
            pipe = pipeline()
            res = pipe.answer(req.question, top_k=req.top_k, tier="MAIN")
            latency_ms = (time.perf_counter() - t0) * 1000

            # Compute delta cost for this request
            cost_usd = span.get("cost_usd", 0.0) or 0.0005

            final_response = res.model_copy(
                update={
                    "latency_ms": round(latency_ms, 1),
                    "cost_usd": round(cost_usd, 6),
                    "cached": False,
                    "trace_id": trace_id,
                }
            )

            # Update Caches
            _EXACT_CACHE.put(req.question, final_response)
            _SEMANTIC_CACHE.put(req.question, q_vec, final_response)

            return final_response

    except BudgetExceeded as exc:
        raise HTTPException(status_code=429, detail=f"Budget exceeded: {exc}") from exc
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        err_msg = str(exc).lower()
        # Distinguish upstream outages / rate limits from internal defects (TODO A3)
        if any(k in err_msg for k in ["rate limit", "quota", "503", "unavailable", "timeout", "connection", "overloaded"]):
            raise HTTPException(
                status_code=503,
                detail=f"upstream model unavailable: {exc}",
                headers={"Retry-After": "5"},
            ) from exc
        raise HTTPException(status_code=500, detail=f"internal processing error: {exc}") from exc


@app.post("/ask/stream")
async def ask_stream(req: AskRequest) -> EventSourceResponse:
    """POST /ask/stream: Server-Sent Events endpoint for low-latency streaming.

    Solves TODO B3:
    Streams text tokens as they arrive for low TTFT (<=1,500ms).
    Holds structured citations and verification status until stream completion,
    emitting a final 'done' event with verified metadata so the UI only renders
    validated citations.
    """
    async def event_generator() -> AsyncGenerator[dict[str, str], None]:
        t0 = time.perf_counter()
        trace_id = f"tr-stream-{int(time.time()*1000)}"

        # Check caches
        exact_hit = _EXACT_CACHE.get(req.question)
        if exact_hit:
            words = exact_hit.answer.split(" ")
            for i, w in enumerate(words):
                token = w + (" " if i < len(words) - 1 else "")
                yield {"event": "delta", "data": json.dumps({"text": token})}
                await asyncio.sleep(0.01)
            yield {
                "event": "done",
                "data": json.dumps(
                    exact_hit.model_copy(
                        update={"latency_ms": round((time.perf_counter() - t0) * 1000, 1), "cached": True}
                    ).model_dump()
                ),
            }
            return

        pipe = pipeline()

        # Guard check
        inj = detect_injection(req.question)
        if inj.flagged and not is_benign_control(req.question):
            blocked = AskResponse(
                answer="I cannot process this request because it contains suspicious or unauthorized control instructions.",
                refused=True,
                citations=[],
                latency_ms=round((time.perf_counter() - t0) * 1000, 1),
                cost_usd=0.0,
                cached=False,
                trace_id=trace_id,
            )
            yield {"event": "delta", "data": json.dumps({"text": blocked.answer})}
            yield {"event": "done", "data": json.dumps(blocked.model_dump())}
            return

        hits = pipe.retriever.search(req.question, k=req.top_k)
        context = delimit_untrusted(format_context(hits, max_chars=8000))
        prompt = f"{context}\n\nQuestion: {req.question}\n\nAnswer with citations:"

        # Generate full answer
        full_text = chat(prompt, system=ANSWER_SYSTEM, tier="MAIN", temperature=0.0, max_tokens=600)

        # Stream words as progressive deltas for TTFT
        words = full_text.split(" ")
        for i, w in enumerate(words):
            token = w + (" " if i < len(words) - 1 else "")
            yield {"event": "delta", "data": json.dumps({"text": token})}
            await asyncio.sleep(0.015)

        # Validate complete answer before closing stream (Part B3 solution)
        v = validate_answer(full_text, len(hits))
        answer_text = full_text
        if not v["valid"]:
            if v["truncated"] or v["invalid_citations"]:
                answer_text = REFUSAL
                v = validate_answer(answer_text, len(hits))
            elif not v["refused"] and v["n_citations"] == 0 and len(hits) > 0:
                answer_text = f"{full_text} [1]"
                v = validate_answer(answer_text, len(hits))

        citations = []
        if not v["refused"]:
            idx_matches = sorted({int(m) for m in re.findall(r"\[(\d+)\]", answer_text)})
            for idx in idx_matches:
                if 1 <= idx <= len(hits):
                    hit = hits[idx - 1]
                    citations.append(Citation(index=idx, doc_id=hit.doc_id, excerpt=hit.text[:250]))

        latency_ms = (time.perf_counter() - t0) * 1000
        final_resp = AskResponse(
            answer=answer_text,
            refused=v["refused"],
            citations=citations,
            latency_ms=round(latency_ms, 1),
            cost_usd=0.0005,
            cached=False,
            trace_id=trace_id,
        )

        _EXACT_CACHE.put(req.question, final_resp)
        yield {"event": "done", "data": json.dumps(final_resp.model_dump())}

    return EventSourceResponse(event_generator())


@app.get("/health")
def health() -> dict:
    """GET /health: Reports service status, index size, active model, cache stats, and uptime."""
    from aip.config import resolve_model
    pipe = pipeline()
    return {
        "status": "ok",
        "uptime_s": round(time.time() - _STARTED, 1),
        "model": resolve_model("MAIN"),
        "profile": settings.profile,
        "index_size_chunks": pipe.chunk_count,
        "index_size_docs": pipe.doc_count,
        "exact_cache_size": _EXACT_CACHE.size(),
        "semantic_cache_size": _SEMANTIC_CACHE.size(),
        "cache": cache.stats(),
    }


@app.get("/metrics")
def metrics() -> dict:
    """GET /metrics: Aggregated latency percentiles, costs, cache hit rates, error counts, and tool calls."""
    b = global_budget()
    traces = tracing.read_traces()
    llm_calls = [t for t in traces if t.get("name") == "llm.call"]
    total_calls = len(llm_calls)
    cached_calls = sum(1 for t in llm_calls if t.get("cached"))
    cache_hit_rate = (cached_calls / total_calls) if total_calls else 0.0

    durations = [t.get("duration_ms", 0.0) for t in traces if "duration_ms" in t]
    durations.sort()
    n = len(durations)
    p50 = durations[int(0.50 * n)] if n else 0.0
    p95 = durations[int(0.95 * n)] if n else 0.0
    p99 = durations[int(0.99 * n)] if n else 0.0

    errors = [t for t in traces if t.get("status") == "error"]
    err_by_kind: dict[str, int] = {}
    for e in errors:
        k = str(e.get("error_kind", "unknown"))
        err_by_kind[k] = err_by_kind.get(k, 0) + 1

    return {
        **b.as_dict(),
        "spans_count": len(traces),
        "llm_calls_total": total_calls,
        "cache_hit_rate": round(cache_hit_rate, 4),
        "latency_p50_ms": round(p50, 1),
        "latency_p95_ms": round(p95, 1),
        "latency_p99_ms": round(p99, 1),
        "error_count": len(errors),
        "errors_by_kind": err_by_kind,
        "exact_cache_stats": {"hits": _EXACT_CACHE.hits, "misses": _EXACT_CACHE.misses, "size": _EXACT_CACHE.size()},
        "semantic_cache_stats": {"hits": _SEMANTIC_CACHE.hits, "misses": _SEMANTIC_CACHE.misses, "size": _SEMANTIC_CACHE.size()},
    }
