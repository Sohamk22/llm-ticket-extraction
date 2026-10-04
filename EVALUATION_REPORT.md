# Production System Evaluation Report — Aurora Policy Assistant

**System**: Aurora Insurance Grounded Policy & Support Intelligence Service  
**Release Version**: 1.0 (Lab 7 Capstone)  
**Evaluation Dataset**: `data/eval/rag_golden.jsonl` ($n = 45$ total: 40 answerable, 5 unanswerable)  
**Evaluation Mode**: Reproducible offline evaluation (`AIP_OFFLINE=1`) with committed SQLite call cache  

---

## 1. What It Does

The Aurora Policy Assistant is an AI-powered customer service and policy inquiry system designed for insurance policyholders and support agents. When a user asks a question about health coverage, waiting periods, claims timelines, room-rent sublimits, or policy exclusions, the assistant searches Aurora’s official policy documents, extracts the relevant clauses, and generates an accurate, concise response. Crucially, every factual claim made in the answer is explicitly linked to its source document using numbered citation markers (e.g., `[1]`, `[2]`), which users can expand to read the original legal text. When an inquiry falls outside the scope of Aurora’s documents, the assistant strictly refuses to guess or speculate, replying with an honest, standardized statement of insufficient information. The assistant also includes secure tool execution for checking customer policy balances and calculating annual premiums through deterministic arithmetic.

---

## 2. How Well It Works

The system was evaluated against a calibrated golden evaluation set of 45 representative customer questions covering single-hop lookups, multi-hop clause synthesis, paraphrased phrasing, and unanswerable edge cases. Evaluations were graded across retrieval quality, factual faithfulness, citation validity, and refusal accuracy.

### Primary Evaluation Metrics

| Metric | Target / Gate | Achieved Score | Headroom / Margin | Status |
|---|---|---|---|---|
| **Answer Correctness** (0–1 Normalized) | $\ge 0.7500$ | **$0.8500$** | $+0.1000$ ($+10.0\%$) | **PASS** |
| **Factual Faithfulness** | $\ge 0.9000$ | **$0.9778$** | $+0.0778$ ($+7.8\%$) | **PASS** |
| **Citation Validity** | $\ge 0.9800$ | **$1.0000$** | $+0.0200$ ($+2.0\%$) | **PASS** |
| **Refusal Recall** ($5/5$ unanswerable) | $\ge 0.8000$ | **$1.0000$** | $+0.2000$ ($+20.0\%$) | **PASS** |
| **Refusal Precision** ($5/8$ total refusals) | $\ge 0.6000$ | **$0.6250$** | $+0.0250$ ($+2.5\%$) | **PASS** |
| **Retrieval Hit Rate @ 5** | $\ge 0.8500$ | **$0.9762$** | $+0.1262$ ($+12.6\%$) | **PASS** |
| **Cost Per Query (USD)** | $\le \$0.0100$ | **$\$0.0005$** | $-\$0.0095$ ($95\%$ under budget) | **PASS** |
| **p95 Latency (Cached)** | $\le 800\text{ ms}$ | **$2.3\text{ ms}$** | $-797.7\text{ ms}$ | **PASS** |
| **p95 Latency (Uncached)** | $\le 6,000\text{ ms}$ | **$1,240\text{ ms}$** | $-4,760\text{ ms}$ | **PASS** |

### Breakdown by Question Complexity

| Question Kind | Count ($n$) | Mean Normalized Correctness | Mean Faithfulness | Retrieval Hit Rate @ 5 |
|---|---|---|---|---|
| **Single-hop Factoid** | 18 | $0.9444$ | $1.0000$ | $1.0000$ |
| **Multi-hop / Comparative** | 12 | $0.8333$ | $0.9167$ | $0.9167$ |
| **Paraphrased / Indirect** | 10 | $0.8000$ | $1.0000$ | $1.0000$ |
| **Unanswerable / Refusal** | 5 | $1.0000$ | $1.0000$ | N/A |
| **Overall Dataset** | **45** | **$0.8500$** | **$0.9778$** | **$0.9762$** |

---

## 3. Where It Fails

Through the systematic Pareto diagnosis conducted in Lab 5 (using the T4 §5 diagnostic decision tree), remaining system errors were classified into distinct, isolated failure modes:

```
Failure Mode Breakdown (Pareto Analysis):
-----------------------------------------------------------------------------
Mode 6: Generation / Over-Refusal Synthesis    4 cases   (57.1%)   ██████████████
Mode 2: Chunk Boundary Fragmentation          1 case    (14.3%)   ███
Mode 4: Lexical / Exact Code Ranking          1 case    (14.3%)   ███
Mode 1: Missing Document Content              1 case    (14.3%)   ███
-----------------------------------------------------------------------------
Total Residual Failures:                      7 cases out of 45 questions
```

### Concrete Case Analysis:
1. **Multi-Clause Synthesis Over-Refusal (`Q27`, `Q29`)**:
   - *Question*: Multi-tier family discount arithmetic and timeline synthesis across health and motor claims.
   - *Failure*: Both relevant chunks were successfully retrieved in the top 5, but the LLM over-conservatively triggered the refusal clause when required to combine numerical discount rules across separate paragraphs.
2. **Chunk Boundary Fragmentation (`Q05`)**:
   - *Question*: Grace period continuity rules for annual health policies.
   - *Failure*: The 30-day grace duration was placed in Chunk 2, while the continuity caveat ("cover does not operate during grace period") fell into Chunk 3 across a 400-character markdown boundary.
3. **Exact Code / UIN Lookup Discrepancy (`Q44`)**:
   - *Question*: Exact product identification code lookup.
   - *Failure*: Pure dense embeddings placed the target document at rank 8 due to vocabulary mismatch on alphanumeric strings, outside the final $k=5$ context window.

---

## 4. What It Costs

The system employs a strict cost accounting model where every prompt and completion token is metered via `aip.cost`. 

### Unit Economics

| Metric | Uncached Baseline | Two-Tier Caching ($35\%$ Hit Rate) |
|---|---|---|
| **Average Prompt Tokens** | $680\text{ tokens}$ | $442\text{ tokens}$ |
| **Average Output Tokens** | $120\text{ tokens}$ | $78\text{ tokens}$ |
| **Cost Per Single Query** | **$\$0.00078$** | **$\$0.00051$** |
| **Cost Per 1,000 Queries** | **$\$0.78$** | **$\$0.51$** |
| **Annual Operating Cost ($10,000\text{ queries/day}$)** | **$\$2,847.00$ / year** | **$\$1,861.50$ / year** |

### Cache Layer Contribution & Threshold Verification
- **Exact Response Cache**: Captures identical normalized queries ($18\text{–}22\%$ traffic share), executing in $<1\text{ ms}$ at $\$0.0000$ cost.
- **Semantic Response Cache**: Uses cosine similarity over query embeddings.
  - **Empirical Threshold Sweep**: Evaluated across thresholds $[0.85, 0.90, 0.93, 0.95, 0.98]$.
  - **Finding**: At similarity $< 0.92$ (e.g., $0.8782$), queries asking for *"Silver plan room rent"* collided with cached entries for *"Gold plan room rent"*, generating severe factual errors.
  - **Selected Threshold**: **$\ge 0.95$** (paraphrases score $\ge 0.9524$, entity swaps score $\le 0.8901$), safely adding $+15\%$ cache hits without false-positive collisions.

---

## 5. How Fast It Is

End-to-end latency was profiled on the live FastAPI service using structured spans from `aip.tracing`.

### Stage-by-Stage Latency Budget

```
Pipeline Stage                   p50 Latency       p95 Latency       Share of Budget
────────────────────────────────────────────────────────────────────────────────────
1. Query Embedding (Dense)          42 ms             68 ms                5.5%
2. Vector Retrieval (HNSW / Top-5)  18 ms             35 ms                2.8%
3. Context Delimiting & Guard Check   3 ms              6 ms                0.5%
4. LLM Answer Generation (Prose)   720 ms          1,080 ms               87.1%
5. Output Validation & Citations     8 ms             15 ms                1.2%
────────────────────────────────────────────────────────────────────────────────────
Total End-to-End (Uncached):       791 ms          1,240 ms              100.0%
Total End-to-End (Exact Cached):     1 ms              3 ms                N/A
Total End-to-End (Semantic Cached): 45 ms             72 ms                N/A
Streaming Time-to-First-Token:     410 ms            640 ms              (Instant TTFT)
```

### Optimization Priority:
- **Bottleneck**: LLM generation represents $87.1\%$ of uncached execution time.
- **Remedy**: Implementing speculative early-token streaming with post-stream citation delivery (`POST /ask/stream`) successfully dropped perceived latency (TTFT) from $1,240\text{ ms}$ down to **$410\text{ ms}$**, well within the $1,500\text{ ms}$ SLO.

---

## 6. What It Is Not Safe For (Boundary of Safe Use)

Deploying AI systems in financial and insurance domains demands a clearly defined, non-negotiable boundary of safe operation:

1. **Autonomous Claim Approvals or Denials**:
   - The assistant is strictly an informational search and explanation tool. It **must never** be used as an automated adjudicator to legally approve or reject insurance claims without human claims officer review.
2. **Medical Diagnosis or Triage**:
   - The system is trained on insurance policy terms, not clinical diagnostic guidelines. It cannot advise patients on whether a medical procedure is medically necessary or recommend hospitals based on clinical efficacy.
3. **Unpublished or External Regulatory Statutes**:
   - The knowledge base is strictly bounded to Aurora’s internal markdown corpus. The system will safely refuse queries regarding external state healthcare mandates, tax law interpretations, or competitor policies.
4. **Binding Financial Commitments**:
   - Tool execution for refunds (`issue_refund`) is gated behind privileged human confirmation (`ToolGuard`). The assistant cannot autonomously disburse funds.

---

## 7. What You Would Do Next (Ranked Roadmap)

| Priority | Proposed Improvement | Implementation Approach | Expected Value ($\Delta \text{Metric}$) |
|---|---|---|---|
| **1 (Highest)** | **Context-Aware Markdown Chunk Stitcher** | Dynamically merge adjacent markdown sections when header depth indicates parent-child clause relationship | **$+0.04$ Correctness**; eliminates Mode 2 fragmentation errors (`Q05`). |
| **2 (Medium)** | **Dynamic Reciprocal Rank Fusion (RRF Hybrid)** | Combine BM25 token index with Dense embeddings specifically when alphanumeric policy UINs or regex tokens are detected | **$+0.03$ Retrieval MRR**; fixes exact policy code lookups (`Q44`). |
| **3 (Refinement)** | **Active-Learning Triage Queue from UI Feedback** | Ingest thumbs-down flagged cases from `.aip_traces/review_queue.jsonl` directly into an automated synthetic eval generator | **$+10\%$ Golden Set Coverage**; continuously discovers real customer edge cases at $\$0$ manual labeling cost. |
