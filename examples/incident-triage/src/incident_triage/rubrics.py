"""The rubrics the service classifies with.

Every shape the library documents appears here, because a consumer has to use them the way
the API describes them: a ``Choice`` with plain and JSON option descriptions, a ``Score`` over
an ordered scale, a ``Noul`` with and without criteria, few-shot examples at all three
levels, and the same questions again as raw mappings.
"""

from __future__ import annotations

from jevper import Choice, Example, Noul, Score

__all__ = [
    "CATALOG_OPTIONS",
    "INTENT",
    "NEEDS_HUMAN",
    "PLAIN_RUBRIC",
    "PRODUCT_AREAS",
    "URGENCY",
    "WIDE_26",
    "build_rubric",
    "catalog_choice",
    "raw_rubric",
    "wide_26",
]

INTENT = Choice(
    instructions="Pick the intent of the support ticket. Treat the customer's words, not the product area.",
    criteria={
        "billing": "money: charges, invoices, refunds, plan changes, payment failures",
        "technical": "errors, crashes, login problems, slow or wrong results",
        "sales": "pricing, plans, purchasing, upgrades, seats, quotes",
        # A JSON option description: the library renders a non-string criteria value as JSON.
        "abuse": {"summary": "abuse or policy reports", "examples": ["harassment", "scraping", "spam"]},
        "other": "anything else that does not fit the four above",
    },
    examples=(
        Example(
            state="I was charged twice for the same subscription this month.",
            answer="billing",
            probabilities={"billing": 0.9, "technical": 0.02, "sales": 0.02, "abuse": 0.01, "other": 0.05},
        ),
        Example(state="The dashboard takes forty seconds to load every morning.", answer="technical"),
    ),
)

URGENCY = Score(
    instructions="Rate how urgent this ticket is for a human responder.",
    criteria=["no action needed today", "answer within a business day", "same day", "production is down"],
    examples=(
        Example(
            state="Your invoice PDF is missing the purchase order number we need for our records.",
            answer=1,
            probabilities={0: 0.05, 1: 0.8, 2: 0.1, 3: 0.05},
        ),
    ),
)

NEEDS_HUMAN = Noul(
    instructions="Is a human needed, or can the self-serve documentation answer this?",
    criteria={
        "true": "the customer needs a person: a refund decision, an outage, a data question",
        "false": "the documentation or a status page answers this without a person",
    },
    examples=(Example(state="Everything is down for our whole company since 09:00.", answer=True),),
)

# Thirty product areas: past 26 the labels are two letters, so this only works on the JSON
# methods. A real catalogue routing rubric looks exactly like this.
PRODUCT_AREAS = Choice(
    instructions="Pick the product area the ticket is about.",
    criteria={
        f"area-{index:02d}": f"tickets about product area {index:02d}"
        for index in range(1, 31)
    },
)

# Twenty-six options: the widest a label readout can be asked for.
WIDE_26 = Choice(
    instructions="Pick the queue the ticket belongs to.",
    criteria={f"queue-{chr(ord('A') + index)}": f"queue {chr(ord('A') + index)}" for index in range(26)},
)

# The Jev API's documented maximum for a Choice.
CATALOG_OPTIONS = 255


def catalog_choice() -> Choice:
    """A 255-option routing rubric — the documented maximum, asked as one question."""
    return Choice(
        instructions="Pick the catalogue entry the ticket refers to.",
        criteria={f"sku-{index:04d}": f"catalogue entry {index:04d}" for index in range(CATALOG_OPTIONS)},
    )


def wide_26() -> Choice:
    """A fresh 26-option rubric (the module-level one carries no examples)."""
    return WIDE_26


def build_rubric(*, few_shot: bool = True) -> dict[str, Choice | Score | Noul]:
    """The rubric the service runs: intent, urgency and the human-handoff question."""
    if not few_shot:
        return {
            "intent": Choice(instructions=INTENT.instructions, criteria=INTENT.criteria),
            "urgency": Score(instructions=URGENCY.instructions, criteria=URGENCY.criteria),
            "needs_human": Noul(instructions=NEEDS_HUMAN.instructions, criteria=NEEDS_HUMAN.criteria),
        }
    return {"intent": INTENT, "urgency": URGENCY, "needs_human": NEEDS_HUMAN}


def raw_rubric() -> dict[str, dict]:
    """The same three questions as raw mappings, which the library validates identically."""
    return {
        "intent": {
            "type": "choice",
            "instructions": "Pick the intent of the support ticket.",
            "criteria": {"billing": "money", "technical": "errors", "sales": "pricing", "other": "anything else"},
        },
        "urgency": {
            "type": "score",
            "instructions": "Rate how urgent this ticket is.",
            "criteria": ["low", "normal", "high", "critical"],
        },
        "needs_human": {"type": "noul", "instructions": "Does this need a human?"},
    }


PLAIN_RUBRIC = build_rubric(few_shot=False)
"""The rubric without few-shot examples — the one a traced call uses, so the trace shows the
question block and the state and nothing else."""
