# Answers and Observations

## Setup

The first issue I faced was with the Python environment on my Intel Mac.

The project initially used Python 3.13, but some of the required PyTorch dependencies were not available properly for the setup. I switched to Python 3.12 and used compatible versions of the required packages.

The final environment used:

* Python 3.12
* `numpy<2`
* `scipy<1.15`
* `transformers==4.48.3`
* `tokenizers==0.21.4`

After fixing the environment, the Pydantic primer passed all 8/8 checks and the complete test suite passed 34/34 tests.

---

# Part A — Naive LLM Approach

## What was the first approach?

The first version was intentionally simple. I gave the support ticket to the LLM and asked it to return the required information as JSON.

The output was then passed through `json.loads()`.

The problem is that getting something that *looks* like JSON from an LLM does not mean it is actually valid or follows the required schema.

## What went wrong?

When I tested the naive approach on 40 tickets, I saw several different types of failures:

| Problem                      | Observation |
| ---------------------------- | ----------: |
| Markdown code fences         |    30 cases |
| Invalid category values      |    30 cases |
| Urgency returned as a string |    30 cases |
| Rate-limit exceptions        |    10 cases |

For example, the model could return something like:

````text
```json
{"category": "complaint", ...}
````

````

This looks fine to a person, but `json.loads()` cannot directly parse the Markdown fences.

The model could also return `"4"` instead of `4`, or produce a category outside the allowed set.

## Why is `json.loads()` not enough?

`json.loads()` only checks whether the output is valid JSON.

It does **not** check whether:

- all required fields are present
- values have the correct types
- categories belong to the allowed set
- urgency is within the required range
- the output follows the expected structure

So valid JSON can still be an invalid ticket record.

### Main takeaway from Part A

The main thing I learned here is that **valid JSON is not the same as valid structured data**.

---

# Part B — Pydantic + Validation

## Why did I use Pydantic?

Pydantic gives the output an actual schema.

Instead of only asking the model to return JSON, I define what a valid `TicketRecord` should look like and let Pydantic validate the result.

This checks things like:

- required fields
- data types
- allowed values
- urgency range
- overall structure

## What happens when validation fails?

I added a repair loop.

The general flow became:

```text
Ticket
   ↓
LLM
   ↓
Structured output
   ↓
Pydantic validation
   ↓
Valid? ── Yes → Return record
   │
   No
   ↓
Repair attempt
   ↓
Validate again
   ↓
Still invalid → Human review
````

This is much safer than assuming that the first model response will always be correct.

## Why is the repair loop useful?

LLMs can make small formatting or schema mistakes even when they understand the ticket correctly.

Instead of throwing away the whole result, the system can send the invalid output back for correction.

If the system still cannot produce a valid record, it does not silently continue. It marks the record as:

```text
needs_human_review = True
```

This gives us a safe fallback instead of allowing a bad record into the pipeline.

## Why did I put evidence before category?

I placed the evidence field earlier in the structured output.

The idea was to make the model identify useful evidence from the ticket before deciding on the category. Since the model generates the structured output sequentially, this can help provide useful context for later fields.

---

# Part C — Deterministic Extraction

This was probably the most important architectural change.

Not every field needs an LLM.

Some information follows very clear rules, so using an LLM for it is unnecessary.

## Policy Number

Policy numbers follow a fixed pattern:

```text
AUR-1234567
```

So I used a regular expression:

```text
\bAUR-\d{7}\b
```

This is more reliable than asking the LLM to find the policy number.

## PII Detection

I also moved PII detection into deterministic Python logic using patterns for things such as phone numbers and email addresses.

This means that the system does not need to "reason" about something that can be detected using a predictable pattern.

## Escalation

Escalation was also implemented as a Python business rule.

The rule used was:

```text
urgency >= 4
OR
"ombudsman" appears in the ticket
```

If either condition is true, the ticket is escalated.

## Why is this better?

The final architecture became:

```text
                 ┌── Deterministic rules
Ticket ──────────┤   Policy number
                 │   PII
                 │   Escalation
                 │
                 └── LLM
                     Category
                     Urgency
                     Sentiment
                     Product
                     Language
```

The main idea is:

**Use the LLM where judgement is required and use normal code where the rules are explicit.**

This also reduces cost and makes the system easier to reason about.

One issue I had to be careful about was quoted replies. A ticket can contain an older message starting with `>`, and extracting information from that quoted text can lead to incorrect results. The pipeline therefore removes those quoted reply lines before processing.

---

# Part D — Evaluation

The final system was evaluated on the 120-ticket test set.

The results were:

| Metric               |   Result |
| -------------------- | -------: |
| Schema validity      |     100% |
| Field accuracy       |   91.25% |
| Record accuracy      |   53.33% |
| Total cost           |  $0.0187 |
| p95 latency          | 1,222 ms |
| Unhandled exceptions |        0 |

The deterministic fields performed very well, while most of the remaining errors came from judgement-based fields.

## Per-field accuracy

| Field         | Accuracy |
| ------------- | -------: |
| Policy number |     100% |
| Contains PII  |     100% |
| Language      |     100% |
| Product       |     100% |
| Category      |    94.2% |
| Escalate      |    90.0% |
| Sentiment     |    85.0% |
| Urgency       |    60.8% |

The biggest weakness was urgency classification.

This makes sense because deciding whether a ticket is urgency 3 or 4, for example, can be subjective.

---

# Error Analysis

I looked at the incorrect predictions instead of only looking at the overall accuracy.

The main error patterns were:

### 1. Complaint vs Claims

Some tickets contained language that could reasonably fit both categories. These were difficult for the model to separate consistently.

### 2. Urgency 3 vs 4

This was one of the most common problems.

The difference between a serious issue and an immediately urgent issue is not always explicitly stated in the ticket.

### 3. Urgency 1 vs 2

Similar problems appeared at the lower end of the urgency scale.

### 4. Urgency 4 vs 5

The model sometimes recognized that an issue was highly urgent but did not consistently distinguish between the two highest urgency levels.

### 5. Neutral vs Satisfied

Some messages contained polite or positive language while still mainly describing a problem. This caused occasional sentiment confusion.

## What does this tell me?

The errors were not mainly caused by the system producing invalid output. The schema was valid.

The bigger problem was **semantic judgement**.

That is an important distinction:

```text
Reliability problem → Pydantic / validation
Judgement problem  → LLM / prompting / better labels
```

---

# Cost Analysis

The deterministic changes also reduced the model cost.

The reported cost per ticket went from approximately:

```text
$0.000291 → $0.000156
```

This is around a **46.4% reduction**.

For a larger workload, this difference becomes significant.

For example, assuming 10,000 tickets per day:

```text
10,000 × 365 = 3,650,000 tickets/year
```

Using the assumptions in the analysis, processing everything manually would cost roughly:

```text
$146,000/year
```

while the model-based system would cost approximately:

```text
$569/year
```

under the measured model cost.

The important point is not that the LLM completely replaces humans. Instead, it can reduce the amount of manual work required, while uncertain records can still be sent for human review.

---

# Main Takeaways

The biggest things I learned from this project were:

1. **LLM output should not be trusted just because it looks like JSON.**
2. **Pydantic provides an actual validation layer around LLM output.**
3. **A repair loop is useful for recovering from small schema errors.**
4. **Deterministic tasks should be handled with normal Python whenever possible.**
5. **LLMs are more useful for judgement-based tasks such as category, sentiment, and urgency.**
6. **Evaluation should look at individual fields and actual failure cases, not just one overall accuracy number.**
7. **A system can be structurally reliable while still having semantic classification errors.**
8. **The best architecture is not "LLM everywhere"; it is a combination of LLMs, deterministic code, validation, and human review.**

Overall, the project helped me understand how to build a more reliable LLM pipeline rather than simply prompting a model and trusting whatever it returns.
