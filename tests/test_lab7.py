from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient

from labs.lab7.gate import measure
from labs.lab7.service import app, _EXACT_CACHE, _SEMANTIC_CACHE


def test_service_health():
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "index_size_chunks" in data
    assert data["index_size_chunks"] > 0
    assert "cache" in data


def test_service_ask_grounded():
    client = TestClient(app)
    req_body = {
        "question": "How many days do I have to submit a reimbursement claim after discharge?",
        "top_k": 5,
        "mode": "rag",
    }
    resp = client.post("/ask", json=req_body)
    assert resp.status_code == 200
    data = resp.json()
    assert "answer" in data
    assert not data["refused"]
    assert len(data["citations"]) > 0
    assert data["trace_id"] != ""
    assert data["latency_ms"] >= 0.0


def test_service_caching():
    client = TestClient(app)
    q = "What is the room rent limit on the Silver plan?"

    # Initial call (or cache miss)
    resp1 = client.post("/ask", json={"question": q})
    assert resp1.status_code == 200

    # Exact cache hit
    resp2 = client.post("/ask", json={"question": q})
    assert resp2.status_code == 200
    data2 = resp2.json()
    assert data2["cached"] is True

    # High similarity semantic cache hit (threshold >= 0.95)
    q_paraphrase = "What is the room rent limit for the Silver plan?"
    resp3 = client.post("/ask", json={"question": q_paraphrase})
    assert resp3.status_code == 200
    data3 = resp3.json()
    assert data3["cached"] is True


def test_service_error_validation_422():
    client = TestClient(app)
    # Question too short (< 3 chars)
    resp = client.post("/ask", json={"question": "no"})
    assert resp.status_code == 422

    # Top_k out of bounds
    resp2 = client.post("/ask", json={"question": "Valid query here?", "top_k": 99})
    assert resp2.status_code == 422


def test_service_metrics():
    client = TestClient(app)
    resp = client.get("/metrics")
    assert resp.status_code == 200
    data = resp.json()
    assert "latency_p50_ms" in data
    assert "latency_p95_ms" in data
    assert "exact_cache_stats" in data
    assert "semantic_cache_stats" in data


def test_service_streaming():
    client = TestClient(app)
    q = "How many days do I have to submit a reimbursement claim after discharge?"
    with client.stream("POST", "/ask/stream", json={"question": q}) as resp:
        assert resp.status_code == 200
        lines = [line for line in resp.iter_lines() if line]
        assert len(lines) > 0
        has_delta = any("event: delta" in l for l in lines)
        has_done = any("event: done" in l for l in lines)
        assert has_delta or has_done


def test_regression_gate_measure(monkeypatch):
    monkeypatch.setenv("AIP_OFFLINE", "1")
    metrics = measure()
    assert "correctness" in metrics
    assert "faithfulness" in metrics
    assert "citation_validity" in metrics
    assert "refusal_recall" in metrics
    assert "refusal_precision" in metrics
    assert "hit_rate_at_5" in metrics
    assert metrics["citation_validity"] >= 0.98
