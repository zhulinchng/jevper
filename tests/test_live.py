"""Optional live check against a real provider.

Skipped unless both ``LLM_MODEL`` and ``OPENAI_API_KEY`` are set::

    LLM_MODEL=gpt-5.6-terra OPENAI_API_KEY=... pytest -q tests/test_live.py

Runs one ``choice`` question with ``method="structured"`` and one with ``method="logprobs"``.
"""

from __future__ import annotations

import os

import pytest

from jevper import Choice, SystemOneClient

MODEL = os.environ.get("LLM_MODEL")
API_KEY = os.environ.get("OPENAI_API_KEY")

pytestmark = pytest.mark.skipif(
    not (MODEL and API_KEY), reason="set LLM_MODEL and OPENAI_API_KEY to run the live check"
)

CRITERIA = {
    "billing": "money, invoices, refunds, charges",
    "technical": "errors, crashes, login or performance problems",
    "sales": "pricing, plans, purchasing, upgrades",
}


def _client(**kwargs):
    from openai import OpenAI

    return SystemOneClient(OpenAI(api_key=API_KEY), model=MODEL, **kwargs)


def test_choice_structured_against_a_live_model():
    response = _client(method="structured", temperature=0.0).system_one(
        state="I was charged twice for the same subscription this month.",
        questions={"intent": Choice(instructions="Pick the intent of the message.", criteria=CRITERIA)},
    )

    answer = response.answers["intent"]
    assert answer.choice in CRITERIA
    assert set(answer.probabilities) == set(CRITERIA)
    assert sum(answer.probabilities.values()) == pytest.approx(1.0, abs=1e-6)
    assert 0.0 <= answer.confidence <= 1.0
    assert response.usage.n_calls >= 1


def test_choice_logprobs_against_a_live_model():
    response = _client(method="logprobs", api="chat_completions").system_one(
        state="I was charged twice for the same subscription this month.",
        questions={"intent": Choice(instructions="Pick the intent of the message.", criteria=CRITERIA)},
    )

    answer = response.answers["intent"]
    assert answer.choice in CRITERIA
    assert sum(answer.probabilities.values()) == pytest.approx(1.0, abs=1e-6)
