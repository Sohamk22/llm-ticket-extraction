#!/usr/bin/env python3
"""Lab 1, Parts B and C — the extractor you actually ship.

Complete the TODOs. `run_eval.py` imports `extract_b` and `extract_c` from
here, so keep those two function names.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from aip.guards import _PII_PATTERNS
from aip.llm import StructuredOutputError, structured

CATEGORIES = Literal["billing", "claims", "policy_change",
                     "technical", "complaint", "information"]


# ===========================================================================
# PART B — the schema
# ===========================================================================
class TicketRecord(BaseModel):
    """The contract. Everything the model is allowed to say, and nothing else.

    Remember from T2 §3.2: field `description`s are shipped to the model as
    part of the JSON Schema. They are the highest-leverage place to put an
    instruction, because they sit next to the thing they govern. Write them as
    instructions to the model, not as documentation for a human.
    """

    # B1a: Placed `evidence` BEFORE `category`: acts as reasoning / chain-of-thought in schema (T2 §3.3)
    # so the model outputs the supporting verbatim quote first, conditioning and improving category classification.
    evidence: str = Field(
        max_length=200,
        description="The exact span of the ticket that determined the category, quoted verbatim. One sentence at most."
    )

    category: CATEGORIES = Field(
        description=(
            "billing = money in (premium, debits, refunds, invoices, 80D tax certificate, instalments). "
            "claims = actual or intended claim (cashless, reimbursement, settlement, deduction, rejection). "
            "policy_change = altering contract (add/remove member, upgrade, port, change contact details). "
            "technical = broken app, portal, OTP, login, locator, document upload. "
            "complaint = Aurora's conduct is the subject (mis-selling, kept on hold, ignored grievance). "
            "information = general question with no pending transaction behind it. "
            "CRITICAL: An angry message about a claim is 'claims' if they want the claim processed; "
            "it is 'complaint' only when Aurora's conduct itself is the grievance."
        )
    )

    urgency: int = Field(
        ge=1, le=5,
        description=(
            "Urgency scale 1-5. "
            "1 = answerable from general product knowledge or self-service how-to, Aurora need not look anything up. "
            "2 = requires Aurora to look up customer account, act on it, or fix a defect, or transaction is in flight. "
            "3 = something already went wrong or is stuck and customer is waiting (double debit, portability delayed). "
            "4 = repeated failure to resolve ('THIRD TIME'), money/access at risk now, or threat of escalation. "
            "5 = active emergency (ICU admission denied) or customer states they ARE escalating to Ombudsman. "
            "Add +1 (capped at 5) if message specifies same-day or next-morning deadline. Judge situation, not shouting."
        )
    )

    sentiment: Literal["angry", "frustrated", "neutral", "satisfied"] = Field(
        description=(
            "angry = hostile, shouting, threats. "
            "frustrated = references a prior failure, repeat attempt, delay, or unanswered request, but still civil. "
            "neutral = matter-of-fact first-time request (however terse). "
            "satisfied = explicit thanks or praise. Tone only, independent of urgency."
        )
    )

    product: Literal["bronze", "silver", "gold", "platinum", "unknown"] = Field(
        description=(
            "The specific Aurora plan named in the message: 'bronze', 'silver', 'gold', 'platinum'. "
            "Use 'unknown' if no plan is explicitly named; never guess or infer from sum insured or context."
        )
    )

    language: Literal["en", "hi-en"] = Field(
        description=(
            "'hi-en' if Hindi words or transliterated Hindi in Latin script (e.g. jaldi, kripya, bahut, turant) "
            "are mixed into English; 'en' for pure English."
        )
    )

    # Part B only: the model decides these. In Part C you will delete them
    # from this schema and compute them in code instead.
    policy_number: str | None = Field(
        default=None,
        pattern=r"^AUR-\d{7}$",
        description=(
            "Format AUR- followed by exactly 7 digits (e.g. AUR-1234567), copied verbatim from the live message body. "
            "Must be null if no policy number appears or if it only appears in a quoted reply ('>'). "
            "Never invent, infer, or reformat."
        )
    )

    contains_pii: bool = Field(
        default=False,
        description=(
            "True if the ticket contains a phone number or an external personal email address "
            "(not Aurora's own published addresses like support@aurorahealth.example). "
            "A personal name alone is False."
        )
    )

    # Set by our code, never by the model.
    needs_human_review: bool = False
    review_reason: str = ""

    @field_validator("policy_number", mode="before")
    @classmethod
    def _policy_format(cls, v: str | None) -> str | None:
        if v is None:
            return None
        v_str = str(v).strip()
        if v_str.lower() in {"", "null", "none", "n/a", "undefined", "not provided"}:
            return None
        m = re.search(r"AUR-\d{7}", v_str)
        if m:
            return m.group(0)
        return v_str


SYSTEM_PROMPT = """\
You are an expert customer support ticket extraction and triage system for Aurora Health Insurance.
Your task is to analyze support tickets across email, WhatsApp, and web forms and return a strictly validated JSON record.

Guidelines:
1. Extract the verbatim span justifying the category into `evidence` first (Chain-of-Thought in schema).
2. Classify `category` into one of the 6 allowed values:
   - billing: premium, debits, refunds, invoices, 80D tax certificate, payment instalments.
   - claims: actual or intended claim (cashless, reimbursement, settlement, deduction, rejection).
   - policy_change: altering contract (add/remove member, upgrade, port, contact details change).
   - technical: broken app, portal, OTP, login, hospital locator, upload issues.
   - complaint: Aurora's conduct is the grievance (mis-selling, kept on hold, ignored grievance).
   - information: general questions with no active transaction behind them.
   * Note: An angry ticket about a claim is `claims` if the customer wants it processed; `complaint` only when Aurora's conduct itself is the grievance.
3. Score `urgency` on 1-5 scale based on situation:
   - 1: General product knowledge / self-service how-to (no account lookup needed).
   - 2: Account lookup / action / bug fix required, or transaction in flight.
   - 3: Something already went wrong / stuck and customer waiting (double debit, delayed portability).
   - 4: Repeated failure ('THIRD TIME'), money/access at risk now, or threat of escalation.
   - 5: Active emergency (ICU cashless denied) or customer states they ARE filing with Ombudsman.
   * Add +1 (max 5) if explicit same-day or next-morning deadline mentioned.
4. Detect `sentiment` based strictly on customer tone:
   - angry: hostile, shouting, overt threats.
   - frustrated: references prior failure / delays / unanswered requests, but still civil.
   - neutral: matter-of-fact first-time request.
   - satisfied: thanks or praise.
5. Identify `product` (bronze, silver, gold, platinum) only if explicitly named; otherwise 'unknown'. Never guess.
6. Set `language` to 'hi-en' if Hindi words / transliteration appear (e.g. jaldi, kripya, turant, bahut), else 'en'.
7. Output must strictly conform to the required JSON schema.
"""


def extract_b(ticket: str) -> TicketRecord:
    """Part B: the model decides everything."""
    try:
        return structured(ticket, schema=TicketRecord, system=SYSTEM_PROMPT, tier="SMALL", max_tokens=600)
    except StructuredOutputError as e:
        return TicketRecord(
            evidence="",
            category="information",
            urgency=1,
            sentiment="neutral",
            product="unknown",
            language="en",
            policy_number=None,
            contains_pii=False,
            needs_human_review=True,
            review_reason=f"StructuredOutputError: {e}",
        )
    except Exception as e:  # noqa: BLE001
        return TicketRecord(
            evidence="",
            category="information",
            urgency=1,
            sentiment="neutral",
            product="unknown",
            language="en",
            policy_number=None,
            contains_pii=False,
            needs_human_review=True,
            review_reason=f"UnhandledException: {type(e).__name__}: {e}",
        )


# ===========================================================================
# PART C — move the deterministic work out of the model
# ===========================================================================
POLICY_RE = re.compile(r"\bAUR-\d{7}\b")

# The quoted-reply marker. Everything after this is history, not the current
# message. Part C3 asks you to decide what that means for policy extraction.
QUOTE_MARKER = re.compile(r"^\s*>", re.MULTILINE)


def extract_deterministic(ticket: str) -> dict:
    """TODO C1: return {'policy_number', 'contains_pii'} without a model call.

    policy_number:
        Find AUR-<7 digits>.

    TODO C3 -- the trap. Some tickets contain TWO policy-number-shaped strings:
        one in the live body, and one in a quoted reply below a '>' line from
        an earlier thread. They are not always the same number.

        Rule: Quoted lines starting with '>' represent prior reply history.
        We exclude lines starting with '>' and search only the unquoted live
        body for the active policy number. If absent from live body, return None.

    contains_pii:
        True if the ticket contains a phone number or an external email address
        (excluding Aurora's official support addresses).
    """
    live_lines = [line for line in ticket.splitlines() if not line.strip().startswith(">")]
    live_text = "\n".join(live_lines)

    pn_match = POLICY_RE.search(live_text)
    policy_number = pn_match.group(0) if pn_match else None

    has_pii = False
    if _PII_PATTERNS["PHONE_IN"].search(ticket):
        has_pii = True
    else:
        for email in _PII_PATTERNS["EMAIL"].findall(ticket):
            if not email.endswith("@aurorahealth.example"):
                has_pii = True
                break

    return {"policy_number": policy_number, "contains_pii": has_pii}


def apply_business_rules(rec_fields: dict, ticket: str) -> dict:
    """TODO C1b: compute `escalate` in code.

        escalate = urgency >= 4 or 'ombudsman' appears in the ticket
    """
    urgency = rec_fields.get("urgency", 1)
    escalate = bool((urgency >= 4) or ("ombudsman" in ticket.lower()))
    return {**rec_fields, "escalate": escalate}


class TicketRecordC(BaseModel):
    """TODO C2: the reduced schema the model sees in Part C.

    Copy TicketRecord and delete the fields you now compute in code. Fewer
    fields means a shorter prompt, fewer output tokens, and three fields at
    100% accuracy. Measure all three effects.
    """

    evidence: str = Field(
        max_length=200,
        description="The exact span of the ticket that determined the category, quoted verbatim. One sentence at most."
    )

    category: CATEGORIES = Field(
        description=(
            "billing = money in (premium, debits, refunds, invoices, 80D tax certificate, instalments). "
            "claims = actual or intended claim (cashless, reimbursement, settlement, deduction, rejection). "
            "policy_change = altering contract (add/remove member, upgrade, port, change contact details). "
            "technical = broken app, portal, OTP, login, locator, document upload. "
            "complaint = Aurora's conduct is the subject (mis-selling, kept on hold, ignored grievance). "
            "information = general question with no pending transaction behind it. "
            "CRITICAL: An angry message about a claim is 'claims' if they want the claim processed; "
            "it is 'complaint' only when Aurora's conduct itself is the grievance."
        )
    )

    urgency: int = Field(
        ge=1, le=5,
        description=(
            "Urgency scale 1-5. "
            "1 = answerable from general product knowledge or self-service how-to, Aurora need not look anything up. "
            "2 = requires Aurora to look up customer account, act on it, or fix a defect, or transaction is in flight. "
            "3 = something already went wrong or is stuck and customer is waiting (double debit, portability delayed). "
            "4 = repeated failure to resolve ('THIRD TIME'), money/access at risk now, or threat of escalation. "
            "5 = active emergency (ICU admission denied) or customer states they ARE escalating to Ombudsman. "
            "Add +1 (capped at 5) if message specifies same-day or next-morning deadline. Judge situation, not shouting."
        )
    )

    sentiment: Literal["angry", "frustrated", "neutral", "satisfied"] = Field(
        description=(
            "angry = hostile, shouting, threats. "
            "frustrated = references a prior failure, repeat attempt, delay, or unanswered request, but still civil. "
            "neutral = matter-of-fact first-time request (however terse). "
            "satisfied = explicit thanks or praise. Tone only, independent of urgency."
        )
    )

    product: Literal["bronze", "silver", "gold", "platinum", "unknown"] = Field(
        description=(
            "The specific Aurora plan named in the message: 'bronze', 'silver', 'gold', 'platinum'. "
            "Use 'unknown' if no plan is explicitly named; never guess or infer from sum insured or context."
        )
    )

    language: Literal["en", "hi-en"] = Field(
        description=(
            "'hi-en' if Hindi words or transliterated Hindi in Latin script (e.g. jaldi, kripya, bahut, turant) "
            "are mixed into English; 'en' for pure English."
        )
    )

    # Set by our code, never by the model.
    needs_human_review: bool = False
    review_reason: str = ""


def extract_c(ticket: str) -> dict:
    """Part C: model for judgement, code for everything else.

    Returns a plain dict (model fields + deterministic fields + business rules)
    so that run_eval.py can score it against the gold labels directly.
    """
    try:
        rec = structured(ticket, schema=TicketRecordC, system=SYSTEM_PROMPT, tier="SMALL", max_tokens=500)
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

    # Deterministic extraction
    det_fields = extract_deterministic(ticket)
    rec_dict.update(det_fields)

    # Business rules
    final_dict = apply_business_rules(rec_dict, ticket)
    return final_dict


if __name__ == "__main__":
    import json

    root = Path(__file__).resolve().parents[2]
    sample = json.loads(
        (root / "data/eval/extraction_dev.jsonl").open(encoding="utf-8").readline()
    )
    print("--- ticket ---")
    print(sample["input"][:600])
    print("\n--- gold ---")
    print(sample["expected"])
    print("\n--- yours ---")
    print(extract_c(sample["input"]))
