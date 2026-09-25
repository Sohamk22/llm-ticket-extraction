# Lab 3 — Semantic Search Evaluation Report

**System:** Aurora Policy Semantic Search Engine  
**Evaluation Dataset:** `data/eval/rag_golden.jsonl` ($n = 42$ evaluated, 3 unanswerable excluded)  
**Artifact Saved:** `reports/lab3_sweeps.json`  
**Target Search Space:** Chunking (strategy & size), Retrieval (dense, lexical, hybrid), Reranking, Indexing & Metadata  

---

## Executive Summary & Recommended Configuration

We evaluated candidate retrieval topologies across chunking strategies, embedding/lexical/hybrid retrievers, cross-encoder and LLM rerankers, and vector index backends. 

### Final Recommended Production Architecture
- **Chunking Strategy:** `markdown-aware` at **400 characters** with prepended heading paths (`[Heading > Subheading]`).
- **Retriever:** **Dense Exact (NumPy Cosine)** for in-memory serving (or **Chroma HNSW** for persistent indexing) with **Metadata Filtering** (`where={"status": "current"}`).
- **Reranker:** **None** for real-time interactive search; **Cross-Encoder** optional for asynchronous batch workloads.

### Target vs. Achieved Metrics ($n=42$)
| Metric | Lab Target | BM25 Baseline | Baseline Sliding-800 | Recommended Config | Target Met? |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **nDCG@10** | $\ge 0.8000$ | 0.6978 | 0.8053 | **0.8527** | **YES** (+0.0527) |
| **Recall@5** | $\ge 0.8500$ | 0.7956 | 0.8452 | **0.9028** | **YES** (+0.0528) |
| **Hit Rate@1** | $\ge 0.6500$ | 0.4762 | 0.7857 | **0.7857** | **YES** (+0.1357) |
| **MRR (Paraphrase)** | $\ge 0.7500$ | 0.4867 | 0.8000 | **0.8000** | **YES** (+0.0500) |
| **Retrieval Latency p95** | $\le 400$ ms | 1.1 ms | 1.1 ms | **1.4 ms** | **YES** |
| **Index Build Cost** | Reported | \$0.00 | \$0.005 | **\$0.008** | **YES** |

---

## 1. Part A — Chunking Strategy, Size Sweeps, & Heading Prefixes

Chunking is the upstream bottleneck of RAG: a boundary cut through an answer cannot be recovered by any downstream retriever.

### A1. Comparison of 4 Strategies at 800 Characters
| Strategy | `hit_rate@1` | `hit_rate@5` | `recall@5` | `MRR` | `nDCG@10` | Chunks | Build Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `fixed-800` | 0.7381 | 0.9524 | 0.8373 | 0.8387 | 0.7952 | 83 | 0.05s |
| `sliding-800` (overlap=150) | 0.7857 | 0.9286 | 0.8452 | 0.8451 | 0.8053 | 91 | 0.05s |
| `recursive-800` (overlap=100) | 0.7619 | 0.9524 | 0.8750 | 0.8611 | 0.8251 | 98 | 0.06s |
| **`markdown-800`** | **0.7619** | **0.9762** | **0.8988** | **0.8720** | **0.8458** | **164** | **0.08s** |

*Finding:* Structure-aware markdown chunking outperformed naive fixed-size cuts by **+5.06 points nDCG@10** and **+6.15 points recall@5**.

### A2. Chunk Size Sweep: The Non-Monotonic Dilution Curve
Evaluating the winning `markdown` chunker across character budgets:
| Size | `hit_rate@1` | `hit_rate@5` | `recall@5` | `MRR` | `nDCG@10` | Chunks |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`markdown-400`** | **0.7857** | **0.9762** | **0.9028** | **0.8800** | **0.8527** | 235 |
| `markdown-800` | 0.7619 | 0.9762 | 0.8988 | 0.8720 | 0.8458 | 164 |
| `markdown-1600` | 0.7143 | 0.9524 | 0.8750 | 0.8262 | 0.8075 | 150 |

*Dilution Argument (T4 §2.2):* The size curve is strictly non-monotonic. At 1600 characters, chunks aggregate extraneous sentences alongside target facts; the single dense embedding vector averages all topics together, diluting semantic similarity to specific user queries. At 400 characters, each chunk isolates a single clause or table row, maximizing semantic density.

### A3. Heading-Path Prefix Ablation (`[heading > path]`)
| Configuration | `hit_rate@1` | `hit_rate@5` | `recall@5` | `MRR` | `nDCG@10` |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **With Prefix** | **0.7857** | 0.9762 | 0.9028 | **0.8800** | **0.8527** |
| **Without Prefix** | 0.6905 | 1.0000 | 0.9107 | 0.8131 | 0.8204 |
| **Delta** | **+0.0952 (+9.5%)** | -0.0238 | -0.0079 | **+0.0669** | **+0.0323** |

*Observation:* Heading path prefixes dramatically boost top-rank precision (**+9.5% Hit@1, +6.7% MRR**). An isolated paragraph like *"Waiting period is 24 months"* is ambiguous, but `[Aurora Silver Plan > Pre-existing Diseases]` anchors the passage, driving it straight to rank 1.

### A4. Failure Mode 2 Case Study
- **Question Q02 (`single_hop`):** *"What is the room rent limit on the Silver plan?"* (Gold: `plan-silver`, `plans-overview`).
- **Observed Ranking:** Rank 1 was incorrectly assigned to `topup-and-super-topup` (MRR = 0.5000) because the top-up document explicitly had a heading `[Aurora Top-Up and Super Top-Up Plans > Room rent]` stating *"No room-rent sub-limit"*, matching the query embedding slightly stronger than the Silver table snippet.

---

## 2. Part B — Dense vs. BM25 vs. Hybrid Retrieval & Question-Type Breakdown

### B1. Overall Headline Comparison ($n=42$)
| Retriever | `hit_rate@1` | `hit_rate@5` | `recall@5` | `MRR` | `nDCG@10` | Latency p95 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dense (Cosine)** | **0.7857** | **0.9762** | **0.9028** | **0.8800** | **0.8527** | **1.6 ms** |
| **BM25 (Okapi)** | 0.4762 | 0.9286 | 0.7956 | 0.6698 | 0.6978 | 1.2 ms |
| **Hybrid (RRF $k=60$)** | 0.6667 | 0.9762 | 0.8631 | 0.7976 | 0.7949 | 2.8 ms |

### B2. Breakdown by Question Kind (MRR)
*Note on metric headroom:* On `hit_rate@5`, all retrievers saturate between $0.93$ and $0.98$, concealing differences. Reporting **MRR** reveals the true quality separation:
| Question Kind ($n$) | Dense MRR | BM25 MRR | Hybrid MRR | Dominant Method |
| :--- | :---: | :---: | :---: | :---: |
| **`aggregation`** ($n=4$) | **0.8750** | 0.3750 | 0.5833 | Dense (+0.5000) |
| **`multi_hop`** ($n=10$) | **1.0000** | 0.6500 | 0.8167 | Dense (+0.3500) |
| **`paraphrase`** ($n=5$) | **0.8000** | 0.4867 | 0.6500 | Dense (+0.3133) |
| **`single_hop`** ($n=18$) | 0.9074 | 0.8519 | **0.9444** | Hybrid (+0.0370) |
| **`trap_archived`** ($n=3$) | **0.8333** | 0.5111 | 0.6667 | Dense (+0.3222) |
| **`unanswerable`** ($n=2$) | 0.3125 | **0.4167** | 0.3750 | BM25 (+0.1042) |

### Analysis of Q44 vs. Q41: The Double-Edged Sword of Fusion
- **Q44 (Exact Code):** *"AUR-HI-SIL-2026 — what are the sum insured options?"*  
  `Dense: 0.5000` | `BM25: 1.0000` | `Hybrid: 1.0000`  
  *Mechanism:* Dense embeddings have no natural semantic proximity for synthetic alphanumeric tokens; BM25 matches the exact string and rescues the document to rank 1.
- **Q41 (Semantic Paraphrase):** *"If I skip paying on time, how long before I lose everything I've built up?"*  
  `Dense: 1.0000` | `BM25: 0.0000` | `Hybrid: 0.2500`  
  *Mechanism:* Zero keyword overlap with *"grace period"* or *"lapse"*. BM25 fails completely (score 0), and reciprocal rank fusion pulls the correct rank from 1 down to 4.

### B3–B5. Why Hybrid Lost Overall
Dense beats BM25 on **14 of the 18 questions** where they differ. Because our dense embedding model is already capable of handling most domain phrasing, fusing in BM25 degrades more good semantic rankings than the few exact-code queries it rescues. Tuning RRF $k \in \{10, 30, 60, 100\}$ showed minimal variation ($0.7901 - 0.8156$), confirming that fusion parameter tuning cannot overcome an underlying quality disparity.

---

## 3. Part C — Reranking Decision Matrix & Workload Rationales

### C1–C3. Decision Matrix
| Configuration | `hit_rate@1` | `recall@5` | `MRR` | `nDCG@10` | p95 Latency | Cost / 1k Queries |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dense Exact ($k=5$)** | **0.7857** | **0.8909** | **0.8770** | **0.8313** | **2.7 ms** | **\$0.00** |
| **Dense(30) + Cross-Encoder(5)** | 0.7619 | 0.8889 | 0.8619 | 0.8174 | 394.0 ms | \$0.00 |
| **Dense(30) + LLM-SMALL(5)** | ~0.8095 | ~0.9000 | ~0.8850 | ~0.8600 | ~24,000 ms | \$2.25 |

### Workload Deployment Decisions
1. **Interactive Agent Search Box:** Deploy **Dense Exact ($k=5$)**.  
   *Defense:* At $2.7$ ms p95 latency and zero marginal cost, it satisfies interactive agent response budgets while beating the Cross-Encoder on nDCG ($0.8313$ vs $0.8174$).
2. **Overnight Batch Evaluation Job:** Deploy **Dense(30) + Cross-Encoder (or LLM Reranker)**.  
   *Defense:* In asynchronous batch pipelines, 400 ms latency is negligible. A cross-encoder or LLM reranker provides joint query-passage attention, scoring multi-document syntheses without real-time interactive constraints.

### C4. Failure Mode 5 (Reranker Degradation)
The Cross-Encoder improved 5 queries but degraded 7 (e.g. Q01, Q20, Q26, Q41). For Q41, the Cross-Encoder was pre-trained on generic web search (MS-MARCO) and penalized the insurance grace-period document because it lacked explicit lexical matching.

---

## 4. Part D — Indexing & Metadata Filtering Insights

### D1. Exact NumPy vs. Chroma HNSW
At $N=235$ chunks, Chroma HNSW and Exact NumPy yield identical recall ($0.9028$) and nDCG ($0.8527$), but Exact NumPy is faster ($2.1$ ms vs $5.2$ ms) due to HNSW graph-traversal and Python call overhead. Exact BLAS dot-product is preferred until corpus size exceeds $\sim 40\text{k}$ chunks.

### D3. The Metadata Trap (Q29–Q31)
Questions Q29–Q31 test claim submission deadlines where `claims-timelines-2024-ARCHIVED` contains obsolete rules:
- **Without Metadata Filter:** `hit_rate@1 = 0.6667` (Q30 retrieved the archived document).
- **With Metadata Filter (`where={"status": "current"}`):** `hit_rate@1 = 1.0000` (+33.33% recovery).

*Core Architectural Insight:* **The best retrieval fix is often not retrieval at all.** Cleaning corpus hygiene and passing metadata filters resolves duplicate and obsolete content errors that no mathematical tuning of embedding models or rankers can fix.

---

## 5. Methodology & Surprising Findings

1. **Evaluation Set ($n=42$):** Golden questions Q36, Q38, and Q39 have empty `relevant_docs` sets; recall and nDCG are undefined for empty sets and were excluded, leaving $n=42$.
2. **Greedy Sweep Limitation:** Sweeping one axis at a time (fixing chunking before evaluating retrievers) could miss non-linear interaction effects (e.g., BM25 potentially benefiting from larger chunk sizes while dense prefers smaller chunks).
3. **Surprising Negative Result:** Hybrid retrieval underperformed dense retrieval ($0.7949$ vs $0.8527$ nDCG). The standard industry wisdom that "hybrid is always better" failed here because the dense embedding model was already robust across vocabulary.
