# LLM Ticket Extraction

This repository contains my work on building a structured information extraction pipeline for messy customer support tickets. The main focus was combining LLMs with Pydantic validation and deterministic Python logic.

## Structured Ticket Extraction

The system converts unstructured support messages into structured records containing category, urgency, sentiment, product, language, policy number, PII detection, and escalation status.

I first implemented a naive LLM-based extractor and then introduced Pydantic schemas and a repair loop to validate and correct model outputs.

For deterministic fields such as policy numbers, PII, and escalation, I moved the logic from the LLM into Python using regular expressions and business rules.

## Evaluation

The final system was evaluated across quality, cost, and latency on 120 unseen tickets.

| Metric          |   Result |
| --------------- | -------: |
| Schema validity |     100% |
| Field accuracy  |   91.25% |
| Record accuracy |   53.33% |
| Test cost       |  $0.0187 |
| p95 latency     | 1,222 ms |

## What I Learned

The main takeaway was that not every task needs an LLM. Using the model for judgement, deterministic code for explicit rules, and Pydantic for validation makes the overall system more reliable and cost-efficient.

## Repository Structure

```text
├── labs/
│   └── lab1/
│       ├── extract.py
│       ├── v0_naive.py
│       └── run_eval.py
├── data/
├── tests/
├── aip/
├── report.md
├── requirements.txt
└── README.md
```

`extract.py` contains the main extraction pipeline, while `v0_naive.py` and `run_eval.py` contain the baseline and evaluation workflow.

## Running

Install the dependencies:

```bash
pip install -r requirements.txt
```

Run the tests:

```bash
make test
```

Run the evaluator:

```bash
python labs/lab1/run_eval.py --split dev --variant c --compare b --workers 1
```
