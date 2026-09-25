"""jevper's arithmetic and answer types against answers the real Jev service actually gave.

``tests/fixtures/jev-1.13-free.json`` holds three responses captured with raw HTTP from the System One
endpoint opencode Zen serves on 2026-09-26, and ``docs/jev-comparison.md`` is the write-up. What is
pinned here is the half of that comparison which is a claim about *jevper*: that its confidence
reproduces the hosted service's confidence from the hosted service's own probabilities, that its
score expectation reproduces the hosted score, and that its response model accepts the hosted
payloads verbatim. The service's answers are a fixture, not a promise — they drift with the model —
so nothing here needs a network, a key, or a GPU.

The tolerance is 0.02 rather than exact because the service prints probabilities and confidence at
two decimals, computing the second from a distribution it does not print. Recomputing from the
printed numbers cannot always land on the printed answer: ``tone_5_vague`` needs a peak near 0.43 to
produce the 0.29 it reports, and 0.43 would have printed as ``0.43``. The service does not even
reproduce itself — the same noul asked twice in one call came back 0.42 and 0.41.
"""

from __future__ import annotations

import json
import math
from functools import cache
from pathlib import Path
from typing import Any

import pytest

from jevper.normalize import choice_confidence, rescale, score_confidence
from jevper.types import NoulAnswer, ScoreAnswer, SystemOneResponse

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "jev-1.13-free.json"

#: The service prints two decimals; see the module docstring for why exactness is not available.
TOLERANCE = 0.02


@cache
def cases() -> tuple[dict[str, Any], ...]:
    """The recorded request/response pairs, in the order the fixture lists them."""
    document = json.loads(FIXTURE.read_text())
    return tuple(document["cases"])


@cache
def recorded() -> tuple[tuple[str, str, dict[str, Any]], ...]:
    """Every answer the service gave, as ``(case name, question id, answer)``."""
    return tuple(
        (case["name"], question_id, answer)
        for case in cases()
        for question_id, answer in case["response"]["answers"].items()
    )


def _levels(answer: dict[str, Any]) -> list[float]:
    """A score answer's probabilities in level order; the wire keys are strings, the levels are ints."""
    levels = sorted(int(key) for key in answer["probabilities"])
    return [answer["probabilities"][str(level)] for level in levels]


@pytest.mark.parametrize(
    ("case", "question", "answer"),
    [(case, question, answer) for case, question, answer in recorded() if answer["type"] == "choice"],
    ids=[f"{case}-{question}" for case, question, answer in recorded() if answer["type"] == "choice"],
)
def test_jevpers_choice_confidence_reproduces_the_hosted_answer(
    case: str, question: str, answer: dict[str, Any]
) -> None:
    """The peak-scaled formula is the service's own, to within the precision the service prints."""
    computed = choice_confidence(list(answer["probabilities"].values()))

    assert computed == pytest.approx(answer["confidence"], abs=TOLERANCE), (
        f"{case}/{question}: jevper computes {computed:.4f} from the probabilities the service "
        f"printed, and the service reported {answer['confidence']}"
    )


@pytest.mark.parametrize(
    ("case", "question", "answer"),
    [(case, question, answer) for case, question, answer in recorded() if answer["type"] == "score"],
    ids=[f"{case}-{question}" for case, question, answer in recorded() if answer["type"] == "score"],
)
def test_jevpers_score_confidence_reproduces_the_hosted_answer(
    case: str, question: str, answer: dict[str, Any]
) -> None:
    """Concentration around the modal level, the service's own score confidence."""
    computed = score_confidence(_levels(answer))

    assert computed == pytest.approx(answer["confidence"], abs=TOLERANCE), (
        f"{case}/{question}: jevper computes {computed:.4f}, the service reported "
        f"{answer['confidence']}"
    )


@pytest.mark.parametrize(
    ("case", "question", "answer"),
    [(case, question, answer) for case, question, answer in recorded() if answer["type"] == "score"],
    ids=[f"{case}-{question}" for case, question, answer in recorded() if answer["type"] == "score"],
)
def test_jevpers_score_expectation_reproduces_the_hosted_score(
    case: str, question: str, answer: dict[str, Any]
) -> None:
    """``score`` is the probability-weighted mean of the levels, and the arithmetic is exact."""
    levels = _levels(answer)
    computed = math.fsum(level * probability for level, probability in enumerate(levels))

    assert computed == pytest.approx(answer["score"]), (
        f"{case}/{question}: jevper's expectation is {computed!r}, the service reported {answer['score']}"
    )


@pytest.mark.parametrize(
    ("case", "question", "answer"),
    [(case, question, answer) for case, question, answer in recorded() if answer["type"] != "noul"],
    ids=[f"{case}-{question}" for case, question, answer in recorded() if answer["type"] != "noul"],
)
def test_a_hosted_distribution_is_already_normalized(
    case: str, question: str, answer: dict[str, Any]
) -> None:
    """The service's distributions sum to one, so jevper's rescale has nothing to do.

    This is the step that exists because a logprob readout or a sampled JSON object can arrive off
    one. On a service answer it must be a no-op: a rescale that moved a probability here would mean
    jevper was rewriting the hosted distribution rather than passing it on.
    """
    values = list(answer["probabilities"].values())
    scaled = rescale(dict(enumerate(values)))

    assert math.fsum(values) == pytest.approx(1.0, abs=1e-9), f"{case}/{question} does not sum to one"
    assert scaled == pytest.approx(dict(enumerate(values)), abs=1e-12), (
        f"{case}/{question}: rescale moved a probability the service had already normalized"
    )


@pytest.mark.parametrize("case", cases(), ids=[case["name"] for case in cases()])
def test_a_hosted_payload_is_a_system_one_response(case: dict[str, Any]) -> None:
    """Every recorded response validates as jevper's own answer envelope, unchanged.

    The service sends score legend keys as JSON strings, which is all JSON can carry; jevper declares
    them as ints and coerces them. Its ``Usage`` is the wider of the two, so the service's two token
    counts fill it and the call-shaped fields keep their defaults.
    """
    served = case["response"]

    response = SystemOneResponse.model_validate(served)

    assert response.model == served["model"]
    assert set(response.answers) == set(served["answers"])
    assert response.usage.input_tokens == served["usage"]["input_tokens"]
    assert response.usage.output_tokens == served["usage"]["output_tokens"]
    assert response.usage.n_calls == 0  # jevper counts its own calls; a hosted answer has none
    for answer in response.scores.values():
        assert isinstance(answer, ScoreAnswer)
        assert all(isinstance(level, int) for level in answer.legend)
    for answer in response.nouls.values():
        assert isinstance(answer, NoulAnswer)
        assert not hasattr(answer, "confidence")  # a noul carries none, on either side
