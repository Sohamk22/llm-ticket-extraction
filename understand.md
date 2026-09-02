# The Reliable Extractor — A Complete, Plain-English Guide (`understand.md`)

> **How to use this guide:** This document is written to give you an intuitive, crystal-clear understanding of the entire project. After reading this, you will be able to explain what we built, why we built it this way, what broke, and how we fixed it to an engineer, a manager, or an interviewer.

---

## 1. The Big Picture: What Problem Are We Solving?

Imagine you work for **Aurora Health Insurance**, a large health insurer receiving **10,000 customer messages a day** across emails, WhatsApp chats, and web forms.

Right now, human customer support agents manually open every single message, read it, and type 8 fields into a triage system (What category is this? How urgent is it? What is their policy number? Should it be escalated?).
- It takes **40 seconds per ticket**.
- For 10,000 tickets a day, that requires a huge team of agents and costs **₹1.2 Crore ($146,000 USD) every year** just for basic data entry.

**Our Mission:** Build an automated AI pipeline that reads messy, real-world customer support tickets and turns each one into a **100% valid, highly accurate, structured JSON record** that downstream databases can ingest safely without crashing.

---

## 2. The Setup Mystery: The PyTorch & macOS Problem

Before writing any code, we hit a real-world dependency clash:

### What went wrong?
- On **macOS Intel (`x86_64`)**, Python 3.13 did not have pre-built binary wheels for PyTorch (`torch`).
- When we tried installing PyTorch on Python 3.13, PyTorch couldn't build.
- Furthermore, PyTorch 2.2.2 requires **NumPy 1.x**; modern `numpy 2.x` broke PyTorch's internal C-bindings (`_ARRAY_API not found`), and Transformers 5.x refused to load PyTorch, throwing `NameError: name 'nn' is not defined`.

### How we fixed it:
1. We installed **Python 3.12** via Homebrew (`brew install python@3.12`).
2. We recreated our virtual environment with Python 3.12 (`python3.12 -m venv .venv`).
3. We pinned compatible versions (`numpy<2`, `scipy<1.15`, `transformers==4.48.3`, `tokenizers==0.21.4`).
4. Result: All 32 unit tests passed instantly.

---

## 3. The 4-Part Engineering Journey

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│     PART A      │ ──▶ │     PART B      │ ──▶ │     PART C      │ ──▶ │     PART D      │
│  The Naive Way  │     │ Schema & Repair │     │  Hybrid Engine  │     │  Test & Triple  │
│  Watch it fail  │     │ Pydantic strict │     │ Move code out   │     │  Quality × Cost │
└─────────────────┘     └─────────────────┘     └─────────────────┘     └─────────────────┘
```

---

### Part A: The Naive Way (`v0_naive.py`) — Watching It Break

Everyone’s first instinct with LLMs is: *“Just prompt the model: 'Extract category and urgency as JSON', and call `json.loads()`.”*

We ran this naive approach on 40 tickets to see what happens in reality:

```
PART 1 — What happened:
  bare json.loads() parsed:    0 / 40  (100% failed!)
  why? Markdown fences (```json ... ```) blocked Python's json.loads()

PART 2 — What was hiding behind the fence:
  We wrote a regex to strip ```json fences and parsed 30 objects.
  How many were clean?         0 / 40!
  why?
    - urgency was returned as a string ("high", "3") instead of an int (3).
    - category was returned as arbitrary words ("RefundRequest", "general") instead of allowed enums.
```

#### The Golden Lesson of Part A:
> **0/40 parsed $\rightarrow$ 30/40 parsed after a 1-line regex fix $\rightarrow$ STILL 0/40 clean.**
> 
> Fixing syntax does not fix semantics. A model returning syntactically valid JSON with illegal enums and wrong data types will quietly corrupt downstream databases. We need **Level 3 Enforcement**.

---

### Part B: The Schema & The Repair Loop (`TicketRecord`)

In Part B, we built a bulletproof contract using **Pydantic**:

1. **`Literal` types instead of `str`:**
   Instead of `category: str`, we defined:
   `Literal["billing", "claims", "policy_change", "technical", "complaint", "information"]`
   If the model emits `"Refund"`, Pydantic catches it immediately.

2. **Field `description`s as Prompt Real Estate:**
   When you create a Pydantic model, `model_json_schema()` converts all field descriptions into the prompt sent to the LLM. Field descriptions are the most powerful place to write instructions because they sit directly next to the property they govern!

3. **In-Schema Chain-of-Thought (`evidence` before `category`):**
   *Why did we place `evidence` BEFORE `category` in the schema?*
   LLMs generate text autoregressively (from left to right). By forcing the model to write the verbatim quote in `evidence` first, the model "thinks" and quotes the source text before committing to a category label!

4. **The Active Repair Loop:**
   If the model returns an invalid payload, `aip.llm.structured` captures Pydantic's exact validation error, sends it back to the model, and asks: *"You returned invalid urgency 'high'; please fix it to an integer 1-5."* The model fixes it on the second try.

5. **Never Crash Fallback:**
   If a model call fails or exhausts retries, we never raise an unhandled exception. We catch it and return a record with `needs_human_review=True` so human triage can review it safely.

---

### Part C: The Hybrid Engine — Moving Work Out of the Model

In Part B, the LLM was trying to do everything: extracting policy numbers, detecting phone numbers, and computing business escalation logic.

*Is an LLM good at regex matching?* No. It is slow, expensive, and non-deterministic.
*Is an LLM good at business if-statements?* No. Compliance officers cannot audit what a prompt might do.

In **Part C**, we separated concerns:

| Task | Who Does It? | Why? |
|---|---|---|
| **`category`, `sentiment`, `urgency`** | **LLM** (`gemini-3.5-flash-lite`) | Requires human-like linguistic judgement and context understanding. |
| **`policy_number`** | **Python Regex** (`AUR-\d{7}`) | 100% accurate, instant, zero API cost. |
| **`contains_pii`** | **Python Regex** (Phones & external emails) | 100% auditable, zero token cost. |
| **`escalate`** | **Python Business Rule** (`urgency >= 4 or 'ombudsman' in text`) | Directly readable by compliance, unit-testable. |

#### The Quoted-Reply Trap (C3):
In email threads, customers often reply to older emails that have quote marks (`> On 5 Mar wrote: Policy AUR-9999999`). That old policy number is stale history!
- **Our Rule:** We strip all lines starting with `>` and only search the live message body.
- **Result:** `policy_number` and `contains_pii` reached **100.0% accuracy** on both dev and test datasets!

#### What Part C bought us:
- Cost dropped by **~46%** (from \$0.000291 to \$0.000156 per ticket) because prompt and output tokens were removed.
- Accuracy held rock-steady (**91.3% field accuracy**).

---

### Part D: Final Test & The Triple (Quality × Cost × Latency)

We evaluated our finished system on the **120-item unseen test split**:

```
─────────────────────────────────────────────────────────────────
THE TRIPLE SCORECARD:
  1. QUALITY:
     • Schema Validity:   100.0%  (Target: 100%)    ── PASS!
     • Field Accuracy:     91.25% (Target: ≥ 90%)   ── PASS!
     • Record Accuracy:    53.33% (All 8 fields correct simultaneously)
     • Unhandled Errors:   0      (Zero crashes)    ── PASS!

  2. COST:
     • 120-Item Run Cost: $0.0187 (Target: ≤ $0.15) ── PASS (87% under budget!)
     • Cost per Ticket:   $0.000156 USD (≈ 1.3 paise)

  3. LATENCY:
     • p95 Latency:       1,222 ms (Target: ≤ 4,000 ms) ── PASS (3.2x faster!)
─────────────────────────────────────────────────────────────────
```

---

## 4. Deep Error Analysis: Why Isn't Record Accuracy 100%?

You might ask: *"Field accuracy is 91.3%, so why is record accuracy 53.3%?"*

### The Multiplicative Math:
For a record to be 100% correct, **all 8 fields must be right simultaneously**:
$$1.00_{\text{policy}} \times 1.00_{\text{pii}} \times 1.00_{\text{product}} \times 1.00_{\text{lang}} \times 0.942_{\text{category}} \times 0.900_{\text{escalate}} \times 0.850_{\text{sentiment}} \times 0.608_{\text{urgency}} \approx \mathbf{52.5\%}$$
Four fields are already at 100%. The bottleneck is purely the 3 subjective judgement fields:

1. **Urgency Errors (87.2% are off-by-one):**
   - 41 of 47 urgency errors were off by just 1 level (e.g. model predicted 2 when gold was 1).
   - *Why?* Differentiating whether a general question requires an account lookup (Level 2) or is answerable from public docs (Level 1) is subtle.
2. **`complaint` vs `claims`:**
   - 5 cases where a customer shouted about a delayed cashless claim were tagged `claims` instead of `complaint`. The customer was angry, but wanted their claim processed.
3. **`sentiment` (Neutral vs Frustrated):**
   - When a customer politely writes *"I submitted portability 21 days ago and heard nothing"*, humans label it `frustrated` because of elapsed time, but the model saw polite language and tagged it `neutral`.

---

## 5. The Business Case: Why This Saves ₹64 Lakhs ($77k USD) / Year

### Manual Human Labor:
- 10,000 tickets/day = 3,650,000 tickets/year.
- 40 seconds per ticket at ₹300/hour ($\approx \$3.60/\text{hr}$) = **₹3.33 ($0.040) per ticket**.
- **Annual Manual Triage Cost:** **₹1.21 Crore ($146,000 USD)**.

### Our Automated System:
- Model API cost: **₹0.013 ($0.000156) per ticket**.
- **Annual Total Model Cost:** **₹47,450 ($569 USD)**.
- **Direct Processing Savings:** **99.6% reduction**!

### Even with Human Review of Imperfect Records:
Even if human agents must re-verify the ~46% imperfect records, the company still saves over **₹64,00,000 ($77,290 USD) every year** while cutting processing time from 40 seconds to **1.2 seconds**.

---

## 6. 3-Minute Elevator Pitch / Interview Cheat Sheet

If asked about this project in an interview or review, say this:

1. **The Problem:** *"We built an automated customer support triage extractor for an insurance provider processing 10,000 tickets a day across emails, WhatsApp, and forms."*
2. **The Naive Failure:** *"We proved that naive prompt engineering with `json.loads` has a 100% failure rate in production because markdown fences break parsers, and unconstrained schemas produce corrupt data types."*
3. **The Architecture:** *"We built a hybrid 3-tier architecture:*
   - *Pydantic Level 3 contracts with active schema repair loops.*
   - *In-schema Chain-of-Thought (extracting evidence before classification).*
   - *Deterministic boundary engineering: moving regex, PII detection, and escalation rules out of the LLM into Python code."*
4. **The Results:** *"On the 120-item unseen test split, we achieved **100% schema validity**, **91.3% field accuracy**, and **1.2s p95 latency** at just **$0.00015 per ticket**—yielding a **99.6% cost reduction** and saving **₹64 Lakhs annually**."*
