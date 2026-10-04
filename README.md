# Aurora Policy Assistant & LLM Engineering System

A production-grade, grounded insurance policy intelligence service and structured information extraction pipeline built across Labs 1–7 of the AI-in-Practice curriculum.

---

## ⚡ 5-Minute Quickstart

### 1. Environment Setup

```bash
# Clone and enter directory
cd aip-lab1

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

Set your API key in `.env` (optional for offline testing):
```bash
GEMINI_API_KEY=your_key_here
AIP_PROFILE=gemini
```

---

### 2. Run the Production Service (FastAPI)

Start the Uvicorn server:
```bash
uvicorn labs.lab7.service:app --reload --port 8000
```

#### Test Endpoints via `curl`:
```bash
# 1. Health & Cache Status
curl -s http://localhost:8000/health | jq

# 2. Grounded Q&A with Citations & Trace ID
curl -s http://localhost:8000/ask -H 'Content-Type: application/json' \
     -d '{"question":"How many days do I have to submit a reimbursement claim after discharge?"}' | jq

# 3. Observability & Latency Percentiles
curl -s http://localhost:8000/metrics | jq
```

---

### 3. Launch the User Interfaces

In separate terminal tabs:

```bash
# 1. Policy Assistant Chat UI (with expandable citation sources)
streamlit run labs/lab7/ui.py

# 2. Operations & Observability Dashboard (stage latency, cost tracking, SLO alerts)
streamlit run labs/lab7/dashboard.py
```

---

### 4. Run the CI Regression Gate

Run the automated evaluation gate locally against the calibrated golden set:

```bash
# Run deterministic regression gate offline (replays committed cache at $0.00 cost)
AIP_OFFLINE=1 python labs/lab7/gate.py --config labs/lab7/thresholds.yml

# Test deliberate regression failure (drops top_k candidates from 5 to 1)
AIP_OFFLINE=1 python labs/lab7/gate.py --config labs/lab7/thresholds.yml --break-k 1
```

---

### 5. Run the Full Test Suite

```bash
# Run all unit tests across Labs 1-7 (39/39 passing)
AIP_OFFLINE=1 pytest tests/
```

---

## 📐 Architecture & Key Components

```text
aip-lab1/
├── aip/                     # Core LLM engineering toolkit
│   ├── chunking.py          # Fixed, sliding, recursive & markdown chunkers
│   ├── retrieval.py         # Dense, BM25, Chroma, and Hybrid fusion retrievers
│   ├── guards.py            # Injection detection, PII redaction, ToolGuard
│   ├── llm.py               # LiteLLM client with structured caching and retries
│   ├── cost.py              # Real-time token metering & Budget enforcement
│   └── tracing.py           # Structured JSONL spans per pipeline stage
├── labs/
│   ├── lab1/                # Ticket extraction & schema validation
│   ├── lab2/                # Few-shot prompts, golden evals & cost curve
│   ├── lab3/                # Retrieval sweeps (Markdown-400 + Dense winner)
│   ├── lab4/                # RAG pipeline, LLM judges & citation enforcement
│   ├── lab5/                # Systematic failure diagnosis & Pareto ranking
│   ├── lab6/                # Tool-using assistant & multi-layered defense
│   └── lab7/                # Production HTTP service, caching, gate & dashboard
├── data/
│   ├── corpus/              # Aurora Insurance markdown policies
│   └── eval/                # Calibrated golden test questions (rag_golden.jsonl)
├── tests/
│   ├── test_aip.py          # Core framework tests
│   └── test_lab7.py         # Service, streaming, caching, and gate tests
├── EVALUATION_REPORT.md     # Full 2-page production evaluation report
└── README.md                # Quickstart and project reference
```

---

## 📊 Summary Evaluation Results

| Metric | Target / Gate | Achieved Score | Status |
|---|---|---|---|
| **Answer Correctness** | $\ge 0.75$ | **$0.8500$** | **PASS** |
| **Faithfulness** | $\ge 0.90$ | **$0.9778$** | **PASS** |
| **Citation Validity** | $\ge 0.98$ | **$1.0000$** | **PASS** |
| **Refusal Recall** | $\ge 0.80$ | **$1.0000$** | **PASS** |
| **Retrieval Hit Rate @ 5** | $\ge 0.85$ | **$0.9762$** | **PASS** |
| **Cost Per Query** | $\le \$0.01$ | **$\$0.0005$** | **PASS** |
| **p95 Latency (Cached)** | $\le 800\text{ ms}$ | **$2.3\text{ ms}$** | **PASS** |
| **p95 Latency (Uncached)** | $\le 6,000\text{ ms}$ | **$1,240\text{ ms}$** | **PASS** |
| **Unit Tests Passing** | $100\%$ | **$39 / 39$ passed** | **PASS** |
