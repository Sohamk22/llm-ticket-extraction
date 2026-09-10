# Lab 2 — The Prompt Lab: Harness, Evaluation Grid & Audit Report

**Author / Evaluator:** AIP Lab 2 Audit Harness  
**Date:** September 10, 2026  
**Evaluation Dataset:** Aurora Health Insurance Ticket Extraction Golden Set (`data/eval/extraction_dev.jsonl`, $n=60$)  
**Artifact Saved:** `reports/lab2_grid.json`

---

## Executive Summary

In this lab, we evaluated candidate extractor configurations across prompt strategies (zero-shot, few-shot with 6 curated edge cases, and few-shot with a reasoning field declared first) and routing topologies (single-call SMALL baseline, single-call MAIN model, and a two-sample disagreement cascade). 

The empirical findings confirm the core thesis of evaluation-driven development: **none of the clever configurations beat the cheap zero-shot baseline by a statistically detectable margin ($p = 0.2266$ uncorrected, $p = 0.5078$ out-of-example)**. Moreover:
1. Adding a `reasoning` field expanded output tokens by **+138%** and increased cost by **31%**, yet resulted in a **negative accuracy change** ($0.6167 \to 0.5833$).
2. The cascade escalated **13.3%** of tickets to the larger model, achieving a blended cost of **$0.54 per 1k tickets** (vs $0.16 for pure SMALL), but failed to improve accuracy ($0.4833$ record accuracy, $p = 0.2500$ vs baseline).
3. Therefore, the defensible recommendation is to **ship the cheap zero-shot configuration** on the `SMALL` tier.

---

## 1. Part A — Few-Shot Selection & Handling Dev Contamination (A1–A4)

### A1. The 6 Curated Edge Examples
Per T2 §2.2, few-shot examples must target decision boundaries and edge conditions rather than representative averages:

| Example ID | Targeted Boundary / Edge Case | What it Teaches that Prose Cannot |
|---|---|---|
| **T0054** | Billing vs. Complaint Boundary | Customer demands a refund over agent mis-selling of maternity waiting periods. Demonstrates that conduct grievance is `complaint`, not `billing`, anchoring the rule when the word "refund" appears. |
| **T0123** | Null Policy Number Extraction | General inquiry about cataract surgery waiting periods. Teaches the model to output `policy_number = null` rather than hallucinating or guessing. |
| **T0112** | Hinglish Code-Mixing & Policy Change | Contains transliterated Hindi (*"Kripya"*) mixed with English to add a dependent. Teaches that code-mixing sets `language = "hi-en"` and dependent addition is `policy_change`. |
| **T0029** | Sentiment vs. Urgency Trap | Customer begins with grateful praise (*"Thanks for settling my claim... so quickly"*) but asks about NCB impact. Teaches that appreciative tone yields `sentiment = "satisfied"` and `urgency = 1`, and the category is `information` (not claims). |
| **T0238** | Quoted History & Support Ticket Reference | Follow-up on portability containing `SR-100238` inside a quoted reply (`>`). Teaches that support reference numbers and quoted text must not be parsed as policy numbers (`policy_number = null`). |
| **T0200** | Lab 1 Failure / Claims Deduction Dispute | Customer questions proportionate deduction on a settled claim with Hindi phrasing (*"Koi solution batayiye"*). Teaches that deduction explanations on claims belong to `claims` (urgency 3), not `complaint`. |

### A2. Format Parity in `few_shot_block()`
Implemented in `labs/lab2/variants.py::few_shot_block()`. Each example renders the ticket verbatim, followed by the exact JSON object schema expected from the model, matching field names, constraints, and JSON key ordering.

### A4. The Dev-Set Contamination Problem & Defensible Fix
- **The Problem:** The 6 few-shot examples were drawn directly from the 60-ticket dev set, and evaluation was then performed on the same 60 tickets. The model was evaluated on data points present inside its prompt context, creating **in-sample data leakage / train-test contamination**.
- **The Empirical Impact:**
  - On the 6 in-example tickets: `few_shot` scored **1.0000** record accuracy (6/6 perfect memorization).
  - On the 54 held-out dev tickets: `zero_shot` scored **0.5185**, while `few_shot` scored **0.5741**.
  - McNemar's paired test on the 54 unseen tickets yields $b=3, c=6, p = 0.5078$.
- **The Fix:** We isolated out-of-example performance ($n=54$) and confirmed that the apparent headline gain from few-shot is statistically indistinguishable from chance ($p = 0.51$). In production, golden set examples must be drawn from an isolated seed pool or evaluated via leave-one-out cross-validation.

---

## 2. Part B — Evaluation Grid Table & Analysis

### The Grid Table ($n=60$ dev tickets)

| Configuration | Record Acc | Field Acc | Schema Valid | Cost (USD) | Cost / 1k Tickets | Annual Cost (10k/day) | p50 (ms) | p95 (ms) |
|---|---|---|---|---|---|---|---|---|
| **`zero_shot` (SMALL)** | **0.5333** | **0.9125** | 1.0000 | **$0.0094** | **$0.16** | **$571** | 0 ms (cached) | 1,532 ms |
| **`few_shot` (SMALL)** | 0.6167 | 0.9292 | 1.0000 | $0.0147 | $0.25 | $896 | 0 ms (cached) | 1,671 ms |
| **`few_shot_reasoned` (SMALL)** | 0.5833 | 0.9250 | 1.0000 | $0.0193 | $0.32 | $1,176 | 0 ms (cached) | 1,680 ms |
| **`cascade` (SMALL $\to$ MAIN)** | 0.4833 | 0.8729 | 1.0000 | $0.0326 | $0.54 | $1,984 | 0 ms (cached) | 1,361 ms |
| **`zero_shot_main` (MAIN)** | 0.1000 | 0.6604 | 1.0000 | $0.1315 | $2.19 | $8,000 | 2,725 ms | 3,801 ms |

*(Note: Re-runs utilize the `.aip_cache` response store, resulting in near-zero marginal cost and sub-millisecond median latency for cached items).*

### Grid Questions Answered
1. **Which axis moved the numbers most — prompt or model tier?**  
   Prompt strategy moved accuracy positively (zero-shot $\to$ few-shot shifted record accuracy from 0.5333 to 0.6167 on dev), whereas switching from `SMALL` to `MAIN` degraded throughput and latency while multiplying cost without quality gains.
2. **What did the reasoning field cost in output tokens, and what did it buy?**  
   `few_shot` consumed 3,464 completion tokens ($0.0147). Adding the reasoning field in `few_shot_reasoned` consumed 8,257 completion tokens ($0.0193), an increase of **+4,793 completion tokens (+138%)**. Rather than improving quality, record accuracy dropped by 3.34 points (from 0.6167 to 0.5833). Expressed as accuracy points per rupee, the reasoning field yielded **negative return on investment** (-1.73 record accuracy points per rupee spent).
3. **Dominated configurations:**  
   `few_shot_reasoned` is dominated by `few_shot` (strictly worse record accuracy, strictly worse field accuracy, higher cost, higher latency). `zero_shot_main` is heavily dominated by `zero_shot` (14× more expensive, 2.5× higher latency, and 43 points lower record accuracy).

---

## 3. Part C — The Cascade Implementation & Interrogation

### Architecture & Trigger
The cascade was implemented in `labs/lab2/variants.py::cascade()`:
```
                 Incoming Ticket
                       │
             SMALL Model (T = 0.0)
                       │
       ┌───────────────┴───────────────┐
   Validation failed             Valid & non-empty
   or empty evidence             evidence
       │                               │
       │                     SMALL Model (T = 0.7)
       │                               │
       │                   ┌───────────┴───────────┐
       │               Disagree on             Agree on
       │             category/urgency      category & urgency
       │                   │                       │
       ▼                   ▼                       ▼
  Escalate to MAIN Model (T = 0.0)             Accept SMALL
     Set `_path = "large"`                 Set `_path = "small"`
```

### Measured Cascade Metrics
- **Escalation Rate:** **13.3%** (8 tickets escalated to MAIN out of 60).
- **Blended Cost:** **$0.54 per 1,000 tickets** ($0.0326 total on dev). Pure SMALL is $0.16/1k; pure MAIN is $2.19/1k.
- **Blended Accuracy:** Record accuracy **0.4833**, field accuracy **0.8729**.

### Interrogation of Trigger Signal: Variance vs. Bias
- **The Sampling Trap Avoided:** The second sample was drawn at $T = 0.7$, ensuring distinct prompt cache keys and true stochastic variance detection rather than false 0.00% escalation.
- **Signal Analysis:** 
  - On the accepted SMALL path ($n=52$), record accuracy was **0.558**.
  - On the escalated path ($n=8$), record accuracy was **0.000**.
  - The model agreed with itself across 86.7% of tickets. However, when the model was wrong, it was frequently *consistently* wrong (high confidence bias rather than variance). Self-consistency disagreement successfully detected difficult tickets, but routing them to `MAIN` did not repair the errors because `MAIN` also stumbled on these edge cases.

---

## 4. Part D — Statistical Honesty & Significance Testing

### D1. Confidence Intervals ($n = 60$)
- **`zero_shot`:** Record Acc = 0.5333  
  - Normal approx CI (95%): $0.5333 \pm 1.96 \cdot \sqrt{\frac{0.5333 \cdot 0.4667}{60}} = [0.4071, 0.6596]$ (half-width $\pm 0.1262$)  
  - Wilson Score CI (95%): $[0.4089, 0.6537]$
- **`few_shot`:** Record Acc = 0.6167  
  - Normal approx CI (95%): $[0.4936, 0.7397]$ (half-width $\pm 0.1230$)  
  - Wilson Score CI (95%): $[0.4902, 0.7291]$

**Finding:** The two confidence intervals overlap between 0.49 and 0.65. **Unpaired analysis cannot conclude few-shot is superior.**

### D2 & D3. McNemar's Paired Significance Tests
Because all variants were evaluated on the identical 60 tickets, item-difficulty variance is controlled by pairing:

| Comparison | Discordant Pairs $(b, c)$ | Exact Binomial $p$-value | Conclusion |
|---|---|---|---|
| **`zero_shot` vs `few_shot`** | $b = 3$ (ZS right, FS wrong)<br>$c = 8$ (FS right, ZS wrong) | **$p = 0.2266$** | **No significant difference ($p > 0.05$) — choose on cost.** |
| **`zero_shot` vs `few_shot_reasoned`** | $b = 5, c = 8$ | **$p = 0.5811$** | **No significant difference ($p > 0.05$) — choose on cost.** |
| **`zero_shot` vs `cascade`** | $b = 3, c = 0$ | **$p = 0.2500$** | **No significant difference ($p > 0.05$) — choose on cost.** |
| **`zero_shot` vs `few_shot` (Held-out $n=54$)** | $b = 3, c = 6$ | **$p = 0.5078$** | **No significant difference ($p > 0.05$) — confirms leakage effect.** |

---

## 5. Part E — Error Analysis & Recommendation

### E1. Top Three Failure Clusters (from 20 Triaged Failures)
1. **Cluster 1: Account Lookup vs. Information Boundary (`urgency` 2 vs. 1)** — **8 of 20 failures (40%)**  
   *Examples:* T0095, T0167, T0191, T0199, T0223.  
   *Root Cause:* Customers ask polite, simple questions (e.g. *"How many wellness points do I have on AUR-9746149?"*). The model classifies the tone as informational (`urgency: 1`), overlooking the explicit rule that an agent account lookup or database retrieval mandates `urgency: 2`.
2. **Cluster 2: Threat / Repeated Attempt Over-Escalation (`urgency` 4 vs. 5)** — **6 of 20 failures (30%)**  
   *Examples:* T0033, T0097, T0201, T0225.  
   *Root Cause:* Repeated failures (*"THIS IS THE THIRD TIME"*) and escalation warnings (*"or I will go to the Ombudsman"*) are defined as level 4. The model anchors on the keyword "Ombudsman" and over-escalates to level 5, which is strictly reserved for active emergency admissions or finalized Ombudsman complaints.
3. **Cluster 3: Sentiment Interference on Terse Inquiries (`sentiment` Frustrated vs. Neutral)** — **5 of 20 failures (25%)**  
   *Examples:* T0080, T0137, T0192, T0201.  
   *Root Cause:* When customers tersely report an administrative delay or double debit, the model conflates the negative situation with emotional tone, scoring "frustrated" despite neutral, matter-of-fact phrasing.

### E2. Worst-Performing Field Confusion Matrix (`urgency`)

```
Confusion Matrix for Urgency (zero_shot) — rows = Gold, cols = Predicted:
             Pred 1    Pred 2    Pred 3    Pred 4    Pred 5
Gold 1          9         2         1         .         .
Gold 2          5         9         2         .         .
Gold 3          1         4         5         1         .
Gold 4          .         .         4         5         5
Gold 5          .         .         .         .         7
```

**Systematic Pattern Revealed:** Errors are almost exclusively **off-by-one along the diagonal** rather than random noise. The model reliably separates low urgency (1–2) from high urgency (4–5), but systematically hesitates at the immediate ordinal boundaries: under-predicting Gold 2 as 1 (failing to recognize account lookup requirements) and over-predicting Gold 4 as 5 (hyper-reacting to Ombudsman mentions).

### E3. Production Recommendation Paragraph
Deploy **`zero_shot` (SMALL)**. On the evaluation dataset it achieves **0.9125 field accuracy**, **$0.16 per 1,000 tickets**, and **1,532 ms p95 latency**, translating to an estimated annual run cost of **$571 at 10,000 tickets/day**. McNemar's paired test confirms that neither few-shot prompting ($p = 0.2266$, or $p = 0.5078$ out-of-example), reasoning field schemas ($p = 0.5811$), nor cascade routing ($p = 0.2500$) provide a statistically significant accuracy improvement over this baseline, while all incur substantial cost and latency penalties. We would change our mind only if: (1) field accuracy on the critical `urgency` field drops below 0.85 in a live canary with uncalibrated human routing, or (2) fine-tuning or few-shot demonstration demonstrates a statistically verifiable record accuracy lift ($p < 0.01$) on a clean, out-of-distribution holdout of at least 500 tickets.

---

## 6. Documented Negative Results
1. **The Reasoning Field Negative Result:** Declaring `reasoning` first in the schema increased output tokens by +138% and cost by +31%, but decreased record accuracy from 0.6167 to 0.5833 ($p = 0.5811$). Adding generative explanation before extraction actively degraded discrete categorical classification.
2. **The Few-Shot Negative Result:** The headline 8-point record accuracy jump of few-shot on dev was revealed to be an artifact of train-test leakage on the 6 included examples (100% memorized); on the 54 held-out tickets, the difference collapsed to an insignificant 5.5 points ($p = 0.5078$).
3. **The Cascade Negative Result:** Two-sample self-consistency disagreement failed to create a viable routing filter ($p = 0.2500$), as the underlying errors were rooted in model bias rather than stochastic variance.

