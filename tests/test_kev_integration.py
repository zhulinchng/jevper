"""kev, reached the way its own documentation says to reach it.

`kev <https://github.com/jaredpalmer/kev>`_ is a family of small decision models
built on Qwen3.5/Qwen3.8. Its API matches TypeSafe's System One exactly —
``POST /v1/systemone`` and ``GET /v1/models`` are the TypeSafe shapes, and its
own ``typesafe-sdk`` client is documented as working against any TypeSafe-compatible
server. That makes kev reachable through the surface jevper already has, with no
change to the library:

.. code-block:: python

    SystemOneClient(OpenAI(base_url="http://127.0.0.1:8009/v1", api_key="local"),
                    model="kev-latest", api="systemone")

What is pinned here is the part that is kev's own rather than the format's:

* kev answers ``GET /v1/models`` in the TypeSafe shape, so ``list_models()`` works;
* kev's noul answer carries only ``type`` and ``noul``, with no confidence;
* kev's choice and score answers carry ``confidence`` alongside ``probabilities``;
* kev's score ``probabilities`` and ``legend`` are keyed by ``str(level)`` (JSON
  object keys are text), read back as the indices the documented arithmetic uses;
* kev's response includes ``latency_ms`` (server-side model time), which jevper
  does not read — the field is simply ignored;
* kev accepts a bare noul (no instructions, no criteria), like CLM and Ollaya;
* kev's FastAPI error body is ``{"detail": …}``, same as CLM;
* kev's ``usage`` has ``input_tokens`` and ``output_tokens`` but no ``billing_units``
  (unlike CLM, which counts questions);
* kev's model names are ``kev-latest`` (calibrated) and ``jev-latest`` (raw).

The wire shapes below were taken from kev 0.1.0's ``kev/api.py`` and ``kev/serve.py``
on the main branch (2026-09-27). The live tests at the bottom were last run against
a real Kev-4B server on the remote box. See ``docs/local-servers.md`` for the full
integration story.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

import pytest
from fakes import async_openai_client, openai_client

from jevper import (
    AsyncSystemOneClient,
    Choice,
    ClientCapabilityError,
    InvalidQuestionError,
    JevperError,
    Noul,
    Score,
    SystemOneClient,
)

QUESTIONS = {
    "urgency": Noul(instructions="Is this urgent?"),
    "department": Choice(
        instructions="Which team should handle this?",
        criteria={"billing": "Charges, invoices, refunds", "technical": "Bugs and outages"},
    ),
    "frustration": Score(
        instructions="How frustrated is the customer?",
        criteria=["Calm", "Frustrated", "Very angry"],
    ),
}
MODEL = "kev-latest"
STATE = "Customer: my invoice was charged twice and nobody answers the phone!"


# The answer objects exactly as kev's `to_answers` builds them:
# - noul: {type, noul} — no confidence, no probabilities
# - choice: {type, choice, confidence, probabilities}
# - score: {type, score, legend, probabilities, confidence}
# Score probabilities and legend are keyed by str(level) because JSON object keys
# are text; jevper's parse_answer converts them to int indices.
NOUL = {"type": "noul", "noul": 0.93}
CHOICE = {
    "type": "choice",
    "choice": "billing",
    "confidence": 0.72,
    "probabilities": {"billing": 0.68, "technical": 0.32},
}
SCORE = {
    "type": "score",
    "score": 1.44,
    "confidence": 0.34,
    "legend": {"0": "Calm", "1": "Frustrated", "2": "Very angry"},
    "probabilities": {
        "0": 0.00,
        "1": 0.56,
        "2": 0.44,
    },
}
ANSWERS = {"urgency": NOUL, "department": CHOICE, "frustration": SCORE}
USAGE = {"input_tokens": 101, "output_tokens": 161}
# kev's response includes latency_ms, which jevper does not read
EXTRA_RESPONSE_FIELDS = {"latency_ms": 495}

MODELS = {
    "models": [
        {
            "name": "kev-latest",
            "release_date": "2026-09-24",
            "description": "Kev-4B round 10, calibrated",
        },
        {
            "name": "jev-latest",
            "release_date": "2026-09-24",
            "description": "Kev-4B round 10, raw probabilities",
        },
    ]
}


def kev_body(answers: dict[str, Any] | None = None, **overrides: Any) -> dict[str, Any]:
    """A response shaped like kev's: the System One body, with its own usage."""
    body: dict[str, Any] = {
        "model": MODEL,
        "answers": ANSWERS if answers is None else answers,
        "usage": dict(USAGE),
        **EXTRA_RESPONSE_FIELDS,
    }
    body.update(overrides)
    return body


def answering(payload: dict[str, Any]):
    return lambda _: (200, payload)


def client(stub: Any, **kwargs: Any) -> SystemOneClient:
    """A client on the /v1 base URL, which is the one kev's routes answer under."""
    return SystemOneClient(openai_client(stub), model=MODEL, api="systemone", **kwargs)


# --- the levels, which arrive as text -------------------------------------------------------------


def test_a_score_distribution_is_keyed_by_level_index_not_by_its_json_spelling(stub_server):
    """kev builds a score's distribution with ``{str(i): p}``, because a JSON object key is text.

    The documented arithmetic reaches a level by indexing the rubric's own number —
    ``answer.probabilities[0]`` — so leaving the keys as they arrive turns that into
    a ``KeyError`` on this surface alone.
    """
    stub = stub_server(systemone=answering(kev_body()))

    response = client(stub).system_one(state=STATE, questions=QUESTIONS)

    probabilities = response.answers["frustration"].probabilities
    assert sorted(probabilities) == [0, 1, 2]
    assert all(isinstance(key, int) for key in probabilities)
    assert probabilities[0] == pytest.approx(0.00)
    assert sum(probabilities.values()) == pytest.approx(1.0)


def test_a_score_legend_is_keyed_the_same_way_as_its_distribution(stub_server):
    """``legend`` sits beside the distribution and is spelled the same way, so it is read the same
    way. It is typed ``dict[int, …]`` and the prompt surfaces build these keys with
    ``int(level)``; one answer type must not mean two things."""
    stub = stub_server(systemone=answering(kev_body()))

    response = client(stub).system_one(state=STATE, questions=QUESTIONS)

    legend = response.answers["frustration"].legend
    assert sorted(legend) == [0, 1, 2]
    assert all(isinstance(key, int) for key in legend)
    assert legend[0] == "Calm"


def test_a_score_missing_a_rubric_level_is_refused(stub_server):
    """The same exact-set rule as every other System One server."""
    partial = {
        **SCORE,
        "probabilities": {"0": 0.25, "1": 0.75},
    }
    stub = stub_server(systemone=answering(kev_body({"frustration": partial})))

    with pytest.raises(JevperError):
        client(stub).system_one(state=STATE, questions=QUESTIONS)


# --- the answers as kev assembles them ------------------------------------------------------------


def test_a_noul_answer_carrying_only_its_number_is_read(stub_server):
    """kev builds a noul as ``{"type", "noul"}`` and stops — no confidence, no distribution."""
    stub = stub_server(systemone=answering(kev_body()))

    response = client(stub).system_one(state=STATE, questions=QUESTIONS)

    assert response.answers["urgency"].noul == pytest.approx(0.93)


def test_a_choice_keeps_the_option_keys_the_caller_asked_with(stub_server):
    """kev keys the distribution by the caller's own criteria."""
    stub = stub_server(systemone=answering(kev_body()))

    response = client(stub).system_one(state=STATE, questions=QUESTIONS)

    answer = response.answers["department"]
    assert answer.choice == "billing"
    assert answer.confidence == pytest.approx(0.72)
    assert set(answer.probabilities) == {"billing", "technical"}


def test_the_model_is_the_name_the_caller_asked_for(stub_server):
    """kev echoes the model it served, which is the name the caller sent."""
    stub = stub_server(systemone=answering(kev_body()))

    response = client(stub).system_one(state=STATE, questions=QUESTIONS)

    assert response.model == MODEL


def test_structured_state_is_sent_as_itself(stub_server):
    """The wire format takes structured state, and kev renders it server-side."""
    stub = stub_server(systemone=answering(kev_body()))
    state = {"messages": [{"role": "user", "content": "charged twice"}], "turn": 3}

    client(stub).system_one(state=state, questions={"urgency": Noul(instructions="Urgent?")})

    assert stub.requests[0]["state"] == state


def test_latency_ms_in_response_is_ignored(stub_server):
    """kev's response includes ``latency_ms`` (server-side model time). jevper does not
    read it — the field is extra and silently ignored by pydantic."""
    stub = stub_server(systemone=answering(kev_body()))

    response = client(stub).system_one(state=STATE, questions=QUESTIONS)

    # The response parses fine; latency_ms is not a jevper field
    assert not hasattr(response, "latency_ms")


# --- the two routes -------------------------------------------------------------------------------


def test_list_models_reads_the_typesafe_list_kev_answers(stub_server):
    """kev answers ``GET /v1/models`` in the TypeSafe shape, so ``list_models()`` works."""
    stub = stub_server(systemone=answering(kev_body()), models=(200, MODELS))

    models = client(stub).list_models()

    assert [m.name for m in models] == ["kev-latest", "jev-latest"]
    assert models[0].release_date == "2026-09-24"


def test_both_routes_are_the_ones_this_server_serves(stub_server):
    """/v1/systemone`` and ``/v1/models``, under the version prefix like every other
    System One route — the base URL is the whole of the configuration."""
    stub = stub_server(systemone=answering(kev_body()), models=(200, MODELS))
    kev = client(stub)

    kev.system_one(state=STATE, questions={"urgency": Noul(instructions="Urgent?")})
    kev.list_models()

    assert stub.paths == ["/v1/systemone", "/v1/models"]


# --- what this server does with a question --------------------------------------------------------


def test_a_bare_noul_is_refused_before_a_request_is_spent(stub_server):
    """The hosted Jev service answers 400 for a noul with neither instructions nor criteria, and
    jevper refuses it locally. kev *accepts* one, like CLM and Ollaya."""
    stub = stub_server(systemone=answering(kev_body()))

    with pytest.raises(InvalidQuestionError):
        client(stub).system_one(state=STATE, questions={"urgency": Noul()})

    assert stub.requests == []


def test_lifting_the_rule_lets_this_server_answer_it(stub_server):
    """The flag is what makes the documented kev call work."""
    bare = {"urgency": {"type": "noul", "noul": 0.5}}
    stub = stub_server(systemone=answering(kev_body(bare)))

    response = client(stub, noul_requires_question=False).system_one(
        state=STATE, questions={"urgency": Noul()}
    )

    assert response.answers["urgency"].noul == pytest.approx(0.5)


def test_temperature_is_refused_on_this_wire(stub_server):
    """kev's wire format has no temperature field. jevper refuses it by name rather than
    silently dropping it."""
    stub = stub_server(systemone=answering(kev_body()))

    with pytest.raises(ClientCapabilityError, match="this wire format takes none"):
        client(stub).system_one(state=STATE, questions=QUESTIONS, temperature=0.2)

    assert stub.requests == []


def test_a_fastapi_error_body_reaches_the_caller_as_the_providers_own_sentence(stub_server):
    """kev raises FastAPI's ``HTTPException``, so the body is ``{"detail": …}``."""
    stub = stub_server(
        systemone=lambda _: (
            422,
            {"detail": "invalid request: state exceeds max tokens"},
        )
    )

    with pytest.raises(JevperError) as caught:
        client(stub).system_one(state=STATE, questions=QUESTIONS)

    assert "state exceeds max tokens" in str(caught.value)


def test_usage_counts_requests_not_questions(stub_server):
    """kev's usage has ``input_tokens`` and ``output_tokens`` but no ``billing_units``.
    jevper's ``n_calls`` counts requests, which is 1 here."""
    stub = stub_server(systemone=answering(kev_body()))

    response = client(stub).system_one(state=STATE, questions=QUESTIONS)

    assert response.usage.n_calls == 1
    assert len(stub.requests) == 1
    assert response.usage.input_tokens == 101


# --- parity with the async client -----------------------------------------------------------------


def test_the_async_client_reaches_the_same_routes_and_reads_the_same_answers(stub_server):
    """The async twin over the real SDK, and the same string-keyed levels read back as indices."""
    stub = stub_server(systemone=answering(kev_body()), models=(200, MODELS))

    async def go() -> Any:
        kev = AsyncSystemOneClient(async_openai_client(stub), model=MODEL, api="systemone")
        return await kev.system_one(state=STATE, questions=QUESTIONS)

    response = asyncio.run(go())

    assert set(response.answers) == set(QUESTIONS)
    assert sorted(response.answers["frustration"].probabilities) == [0, 1, 2]
    assert response.answers["department"].choice == "billing"


# --- against a real kev server ---------------------------------------------------------------------
#
# Skipped unless KEV_BASE_URL is set, e.g.
#   KEV_BASE_URL=http://127.0.0.1:8009/v1 KEV_MODEL=kev-latest pytest tests/test_kev_integration.py
#
# Last run on 2026-09-27 against kev 0.1.0 serving Kev-4B on a remote box.
KEV_BASE_URL = os.environ.get("KEV_BASE_URL")
KEV_MODEL = os.environ.get("KEV_MODEL", "kev-latest")


def _live_client(**kwargs: Any) -> SystemOneClient:
    from openai import OpenAI

    return SystemOneClient(
        OpenAI(base_url=KEV_BASE_URL, api_key="local", max_retries=0, timeout=60),
        model=KEV_MODEL,
        api="systemone",
        **kwargs,
    )


@pytest.mark.skipif(not KEV_BASE_URL, reason="KEV_BASE_URL is not set")
def test_a_real_kev_answers_three_questions_and_lists_its_models() -> None:
    """The route this document is about, end to end, against a real kev server."""
    kev = _live_client()

    response = kev.system_one(state=STATE, questions=QUESTIONS)

    assert response.model == KEV_MODEL
    assert set(response.answers) == set(QUESTIONS)
    assert response.answers["urgency"].noul > 0.0
    assert response.answers["department"].choice in QUESTIONS["department"].criteria
    assert sorted(response.answers["frustration"].probabilities) == [0, 1, 2]
    assert sorted(response.answers["frustration"].legend) == [0, 1, 2]
    assert response.usage.n_calls == 1
    assert KEV_MODEL in {m.name for m in kev.list_models()}


@pytest.mark.skipif(not KEV_BASE_URL, reason="KEV_BASE_URL is not set")
def test_a_real_kev_counts_requests_the_same_way_the_questions_were_billed() -> None:
    """``usage.n_calls`` counts requests and the server's own count counts tokens."""
    response = _live_client().system_one(state=STATE, questions=QUESTIONS)

    assert response.usage.n_calls == 1
    assert response.usage.input_tokens > 0


@pytest.mark.skipif(not KEV_BASE_URL, reason="KEV_BASE_URL is not set")
def test_a_real_kev_answers_a_bare_noul_when_the_rule_is_lifted() -> None:
    """kev accepts a bare noul, like CLM and Ollaya."""
    response = _live_client(noul_requires_question=False).system_one(
        state=STATE, questions={"urgency": Noul()}
    )

    assert response.answers["urgency"].noul > 0.0
