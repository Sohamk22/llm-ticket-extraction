#!/usr/bin/env python3
"""Lab 2 — the configurations under test.

Each variant is a callable `str -> dict`. `grid.py` runs them all through the
same harness, so the only thing that differs between rows of your table is the
thing you intended to differ.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from pydantic import Field

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from aip.llm import StructuredOutputError, structured  # noqa: E402
from labs.lab1.extract import (  # noqa: E402
    SYSTEM_PROMPT, TicketRecord, TicketRecordC, apply_business_rules, extract_deterministic,
)

# ---------------------------------------------------------------------------
# A1 — your six chosen examples.
# ---------------------------------------------------------------------------
# Chosen dev-set tickets covering key boundaries and edge cases (T2 §2.2):
#   - T0054: the billing/complaint boundary (mis-selling refund is complaint, not billing)
#   - T0123: no policy number present (teaches null/None, preventing hallucination)
#   - T0112: Hinglish code-mixing ("Kripya") tagging hi-en; dependent addition is policy_change
#   - T0029: satisfied praise about past claim is information inquiry, not claims (urgency 1)
#   - T0238: quoted reply containing 'SR-...' support ID must be ignored for policy_number
#   - T0200: settled claim deduction dispute is claims (urgency 3), not complaint; Hinglish
FEW_SHOT_IDS: list[str] = [
    "T0054",  # teaches: mis-selling/grievance is complaint, not billing, even when refund demanded
    "T0123",  # teaches: general inquiry with no policy mentioned yields policy_number=null
    "T0112",  # teaches: Hinglish code-mixing ("Kripya") tags hi-en; dependent addition is policy_change
    "T0029",  # teaches: satisfied praise about past claim does not make it a claim; inquiry is information
    "T0238",  # teaches: quoted history ('>') and support ticket numbers ('SR-...') must be ignored
    "T0200",  # teaches: deduction disputes on settled claims are claims (urgency 3), not complaints
]

FEW_SHOT_METADATA: dict[str, dict[str, str]] = {
    "T0054": {
        "evidence": "Your agent mis-sold me this policy.",
        "reasoning": (
            "Customer alleges agent mis-selling of the maternity waiting period and demands a refund. "
            "Because the grievance is directly about Aurora's conduct and sales practices, it is classified "
            "as 'complaint' rather than 'billing'. The tone is hostile with refund demands, giving sentiment 'angry' "
            "and urgency 4."
        ),
    },
    "T0123": {
        "evidence": "What is the waiting period for cataract surgery?",
        "reasoning": (
            "Customer asks a routine informational question about waiting periods for cataract surgery "
            "under the Bronze plan. No transaction or dispute is active. Category is 'information', urgency is 1, "
            "and sentiment is 'neutral'."
        ),
    },
    "T0112": {
        "evidence": "Kripya ADD MY MOTHER AS a dependent on my Aurora Bronze policy.",
        "reasoning": (
            "Customer requests adding a dependent to their Aurora Bronze policy, which alters the contract terms "
            "('policy_change'). The text includes transliterated Hindi ('Kripya'), so language is 'hi-en'. "
            "Standard account action request yields urgency 2 and neutral sentiment."
        ),
    },
    "T0029": {
        "evidence": "Just wanted to confirm whether my no-claim bonus is affected by this claim.",
        "reasoning": (
            "Customer expresses appreciation for a fast claim settlement but is only inquiring whether NCB is affected. "
            "Because no claim or transaction is pending or requested, category is 'information' rather than 'claims'. "
            "Tone is appreciative ('satisfied') with urgency 1."
        ),
    },
    "T0238": {
        "evidence": "I submitted a portability request 21 days ago and heard nothing.",
        "reasoning": (
            "Customer follows up on a pending portability request from 21 days ago. Porting alters coverage, "
            "so category is 'policy_change'. Customer is waiting with urgency 3 and sentiment 'frustrated'. "
            "The quoted reply contains SR-100238 which is a support ticket reference, not an AUR- policy number."
        ),
    },
    "T0200": {
        "evidence": "Nobody explained the deduction. What is proportionate deduction and why does it apply to me?",
        "reasoning": (
            "Customer questions a deduction made on a settled claim and asks for clarification on proportionate deduction. "
            "This pertains directly to claim settlement and reimbursement ('claims'). Urgency is 3 because the customer "
            "is waiting on an unresolved deduction dispute. Phrasing 'Koi solution batayiye' indicates mixed Hindi ('hi-en') "
            "with 'frustrated' sentiment."
        ),
    },
}


def load_examples(ids: list[str]) -> list[dict]:
    rows = [json.loads(l) for l in
            (ROOT / "data/eval/extraction_dev.jsonl").open(encoding="utf-8")]
    by_id = {r["id"]: r for r in rows}
    missing = [i for i in ids if i not in by_id]
    if missing:
        raise KeyError(f"unknown example ids: {missing}")
    return [by_id[i] for i in ids]


def few_shot_block(ids: list[str], reasoned: bool = False) -> str:
    """A2: render the examples into the prompt.

    The example output format must be byte-identical to the format you are
    asking the model to produce. A mismatch here is a classic own goal.
    """
    examples = load_examples(ids)
    blocks = []
    for ex in examples:
        eid = ex["id"]
        meta = FEW_SHOT_METADATA.get(eid, {})
        exp = ex["expected"]
        rec: dict[str, Any] = {}
        if reasoned:
            rec["reasoning"] = meta.get("reasoning", "")
        rec["evidence"] = meta.get("evidence", "")
        rec["category"] = exp["category"]
        rec["urgency"] = exp["urgency"]
        rec["sentiment"] = exp["sentiment"]
        rec["product"] = exp["product"]
        rec["language"] = exp["language"]

        block = (
            f"--- Example Ticket ({eid}) ---\n"
            f"{ex['input'].strip()}\n\n"
            f"Output JSON:\n"
            f"{json.dumps(rec, indent=2)}"
        )
        blocks.append(block)
    return "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# The variants
# ---------------------------------------------------------------------------
def zero_shot(ticket: str, tier: str = "SMALL") -> dict:
    """TODO B: Lab 1 Part C, no examples. This is your baseline."""
    try:
        rec = structured(ticket, schema=TicketRecordC, system=SYSTEM_PROMPT, tier=tier, max_tokens=600)
        rec_dict = rec.model_dump()
    except StructuredOutputError as e:
        rec_dict = {
            "evidence": "",
            "category": "information",
            "urgency": 1,
            "sentiment": "neutral",
            "product": "unknown",
            "language": "en",
            "needs_human_review": True,
            "review_reason": f"StructuredOutputError: {e}",
        }
    except Exception as e:  # noqa: BLE001
        rec_dict = {
            "evidence": "",
            "category": "information",
            "urgency": 1,
            "sentiment": "neutral",
            "product": "unknown",
            "language": "en",
            "needs_human_review": True,
            "review_reason": f"UnhandledException: {type(e).__name__}: {e}",
        }

    det_fields = extract_deterministic(ticket)
    rec_dict.update(det_fields)
    return apply_business_rules(rec_dict, ticket)


def few_shot(ticket: str, tier: str = "SMALL") -> dict:
    """TODO B: zero_shot + the few-shot block."""
    sys_prompt = f"{SYSTEM_PROMPT}\n\n### FEW-SHOT EXAMPLES:\n\n{few_shot_block(FEW_SHOT_IDS, reasoned=False)}"
    try:
        rec = structured(ticket, schema=TicketRecordC, system=sys_prompt, tier=tier, max_tokens=600)
        rec_dict = rec.model_dump()
    except StructuredOutputError as e:
        rec_dict = {
            "evidence": "",
            "category": "information",
            "urgency": 1,
            "sentiment": "neutral",
            "product": "unknown",
            "language": "en",
            "needs_human_review": True,
            "review_reason": f"StructuredOutputError: {e}",
        }
    except Exception as e:  # noqa: BLE001
        rec_dict = {
            "evidence": "",
            "category": "information",
            "urgency": 1,
            "sentiment": "neutral",
            "product": "unknown",
            "language": "en",
            "needs_human_review": True,
            "review_reason": f"UnhandledException: {type(e).__name__}: {e}",
        }

    det_fields = extract_deterministic(ticket)
    rec_dict.update(det_fields)
    return apply_business_rules(rec_dict, ticket)


class TicketRecordReasoned(TicketRecordC):
    """TODO B: add a `reasoning: str` field FIRST (T2 §3.3).

    Pydantic keeps declaration order, and field order in the JSON Schema
    influences generation order. Putting reasoning first makes it condition the
    answer; putting it last makes it a post-hoc rationalisation. You want the
    first. Measure the difference in output tokens.
    """
    reasoning: str = Field(
        description="Step-by-step chain-of-thought analyzing the customer issue, evidence quote, category, urgency, and sentiment before deciding."
    )

    @classmethod
    def model_json_schema(cls, *args, **kwargs) -> dict[str, Any]:
        schema = super().model_json_schema(*args, **kwargs)
        if "properties" in schema and "reasoning" in schema["properties"]:
            props = {"reasoning": schema["properties"]["reasoning"]}
            props.update({k: v for k, v in schema["properties"].items() if k != "reasoning"})
            schema["properties"] = props
        if "required" in schema:
            schema["required"] = ["reasoning"] + [k for k in schema["required"] if k != "reasoning"]
        return schema


def few_shot_reasoned(ticket: str, tier: str = "SMALL") -> dict:
    """TODO B: few_shot with TicketRecordReasoned."""
    sys_prompt = f"{SYSTEM_PROMPT}\n\n### FEW-SHOT EXAMPLES:\n\n{few_shot_block(FEW_SHOT_IDS, reasoned=True)}"
    try:
        rec = structured(ticket, schema=TicketRecordReasoned, system=sys_prompt, tier=tier, max_tokens=1000)
        rec_dict = rec.model_dump()
    except StructuredOutputError as e:
        rec_dict = {
            "reasoning": "",
            "evidence": "",
            "category": "information",
            "urgency": 1,
            "sentiment": "neutral",
            "product": "unknown",
            "language": "en",
            "needs_human_review": True,
            "review_reason": f"StructuredOutputError: {e}",
        }
    except Exception as e:  # noqa: BLE001
        rec_dict = {
            "reasoning": "",
            "evidence": "",
            "category": "information",
            "urgency": 1,
            "sentiment": "neutral",
            "product": "unknown",
            "language": "en",
            "needs_human_review": True,
            "review_reason": f"UnhandledException: {type(e).__name__}: {e}",
        }

    det_fields = extract_deterministic(ticket)
    rec_dict.update(det_fields)
    return apply_business_rules(rec_dict, ticket)


def cascade(ticket: str) -> dict:
    """TODO C: SMALL first; escalate to MAIN on a trigger you choose.

    Triggers, roughly in ascending order of how well they work:
      - validation failed                      (free, weak: misses confident errors)
      - evidence field empty or very short     (free, surprisingly decent)
      - urgency >= 4                           (free, but it is not a confidence signal)
      - two SMALL samples at T=0.7 disagree    (2x small cost, much the best)

    Record which path each ticket took -- set rec['_path'] = 'small' | 'large'
    so grid.py can report the escalation rate.
    """
    escalate = False
    rec1 = None

    # Step 1: Draw primary sample from SMALL model at temperature=0.0
    try:
        rec1 = structured(
            ticket,
            schema=TicketRecordC,
            system=SYSTEM_PROMPT,
            tier="SMALL",
            temperature=0.0,
            max_tokens=600,
        )
        if not rec1.evidence or len(rec1.evidence.strip()) < 5:
            escalate = True
    except (StructuredOutputError, Exception):
        escalate = True

    # Step 2: Test self-consistency against second sample at T=0.7 (different cache key)
    if not escalate and rec1 is not None:
        try:
            rec2 = structured(
                ticket,
                schema=TicketRecordC,
                system=SYSTEM_PROMPT,
                tier="SMALL",
                temperature=0.7,
                max_tokens=600,
            )
            # Escalation triggered if category or urgency disagree
            if rec1.category != rec2.category or rec1.urgency != rec2.urgency:
                escalate = True
        except (StructuredOutputError, Exception):
            escalate = True

    # Step 3: Route
    if escalate:
        res = zero_shot(ticket, tier="MAIN")
        res["_path"] = "large"
        return res

    rec_dict = rec1.model_dump()
    det_fields = extract_deterministic(ticket)
    rec_dict.update(det_fields)
    res = apply_business_rules(rec_dict, ticket)
    res["_path"] = "small"
    return res


VARIANTS = {
    "zero_shot": lambda t: zero_shot(t, "SMALL"),
    "zero_shot_main": lambda t: zero_shot(t, "MAIN"),
    "few_shot": lambda t: few_shot(t, "SMALL"),
    "few_shot_main": lambda t: few_shot(t, "MAIN"),
    "few_shot_reasoned": lambda t: few_shot_reasoned(t, "SMALL"),
    "few_shot_reasoned_main": lambda t: few_shot_reasoned(t, "MAIN"),
    "cascade": cascade,
}

