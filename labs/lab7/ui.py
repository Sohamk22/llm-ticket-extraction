#!/usr/bin/env python3
"""Lab 7 — Streamlit front end for Aurora Policy Assistant.

    streamlit run labs/lab7/ui.py

Requires the service to be running:
    uvicorn labs.lab7.service:app --port 8000

The one non-negotiable UI requirement: citations must be expandable to show
the source text. Grounding the user cannot check is decoration.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import requests
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

REVIEW_QUEUE = ROOT / ".aip_traces/review_queue.jsonl"

st.set_page_config(page_title="Aurora Policy Assistant", page_icon="🛡️", layout="wide")

st.sidebar.title("Configuration")
API = st.sidebar.text_input("Service URL", "http://localhost:8000")
mode = st.sidebar.radio("Assistant Mode", options=["rag", "tools"], format_func=lambda x: "RAG Q&A" if x == "rag" else "Tool-Using Agent")
top_k = st.sidebar.slider("Top K Retrieved", min_value=1, max_value=10, value=5)

st.title("🛡️ Aurora Policy Assistant")
st.caption(
    "Production-grade insurance assistant. Answers are strictly grounded in Aurora's "
    "policy documentation with verified citations. When coverage is not specified in "
    "the sources, the assistant safely refuses rather than hallucinating."
)

# Example questions
with st.expander("💡 Example questions to try"):
    st.markdown(
        "- **Standard Query**: *How many days do I have to submit a reimbursement claim after discharge?*\n"
        "- **Plan Coverage**: *What is the room rent limit on the Silver plan?*\n"
        "- **Exclusions**: *Is maternity covered on the Bronze plan?*\n"
        "- **Waiting Period**: *What is the waiting period for pre-existing diseases?*\n"
        "- **Unanswerable / Refusal**: *Does Aurora cover laser treatment for cosmetic tattoos?*\n"
        "- **Tool Mode**: *What is the remaining balance on policy AUR-1234567?*\n"
        "- **Tool Mode Calculation**: *Compute annual premium for Gold plan with eldest age 52 and 3 members.*"
    )

q = st.text_input("Ask a question", placeholder="How long do I have to file a reimbursement claim?")

if st.button("Ask Assistant", type="primary") and q:
    with st.spinner("Processing request..."):
        try:
            r = requests.post(f"{API}/ask", json={"question": q, "top_k": top_k, "mode": mode}, timeout=60)
            r.raise_for_status()
            data = r.json()
        except requests.HTTPError as exc:
            st.error(f"HTTP Error {exc.response.status_code}: {exc.response.text[:300]}")
            st.stop()
        except requests.RequestException as exc:
            st.error(f"Service unreachable at {API}: {exc}")
            st.stop()

    st.markdown("### Answer")
    if data.get("refused"):
        st.warning(data["answer"])
    else:
        st.success(data["answer"])

    # Part A4: Render citations as expanders showing the source text
    citations = data.get("citations", [])
    if citations:
        st.markdown("### 📚 Grounding & Citations")
        for c in citations:
            with st.expander(f"[{c['index']}] Source Document: `{c['doc_id']}`"):
                st.info(c["excerpt"])

    st.divider()
    cols = st.columns(5)
    cols[0].metric("Latency", f"{data.get('latency_ms', 0):.1f} ms")
    cols[1].metric("Cost", f"${data.get('cost_usd', 0):.6f}")
    cols[2].metric("Cached", "Yes" if data.get("cached") else "No")
    cols[3].metric("Sources Used", len(citations))
    cols[4].metric("Status", "Refused" if data.get("refused") else "Grounded")

    st.caption(f"Trace ID: `{data.get('trace_id', '')}` | Endpoint: `{API}/ask` | Mode: `{mode}`")

    # Feedback loop (Stretch feature)
    st.markdown("#### Was this response helpful?")
    c_fb1, c_fb2, c_fb3 = st.columns([1, 1, 6])
    if c_fb1.button("👍 Helpful"):
        st.toast("Thank you for your feedback!")
    if c_fb2.button("👎 Report Issue"):
        REVIEW_QUEUE.parent.mkdir(parents=True, exist_ok=True)
        with REVIEW_QUEUE.open("a", encoding="utf-8") as f:
            f.write(json.dumps({
                "ts": time.time(),
                "question": q,
                "answer": data.get("answer"),
                "trace_id": data.get("trace_id"),
                "flagged": True,
            }) + "\n")
        st.warning("Case appended to review queue (`.aip_traces/review_queue.jsonl`) for golden set triage.")
