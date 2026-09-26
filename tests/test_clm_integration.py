"""CLM, reached the way its own documentation says to reach it.

`CLM <https://github.com/Contrastive-LM/CLM>`_ serves the System One wire format
on purpose — ``POST /v1/systemone`` and ``GET /v1/models`` are the TypeSafe shapes,
and its own client is documented as accepting questions "in the wire format", so a
request written for TypeSafe replays there unchanged. That makes CLM reachable
through the surface jevper already has, with no change to the library:

.. code-block:: python

    SystemOneClient(OpenAI(base_url="http://127.0.0.1:8700/v1", api_key="clm"),
                    model="clm-latest", api="systemone")

What is pinned here is the part that is CLM's own rather than the format's, because
a second independent producer is what tells the shared rules apart from one server's
quirk:

* a score's levels arrive as **JSON object keys, which are text**, in ``probabilities``
  *and* in ``legend`` — read back as the indices the documented arithmetic uses;
* a noul answer carries **only** ``noul``, with no confidence and no distribution;
* ``GET /v1/models`` answers the TypeSafe list, so ``list_models`` parses it;
* a malformed request is a FastAPI ``{"detail": …}`` body, and the provider's own
  sentence survives into the error;
* a noul with neither instructions nor criteria is **accepted**, where the hosted
  Jev service answers 400 — so the flag that lifts jevper's own refusal is the only
  thing standing between a caller and a request that would be wasted;
* ``temperature`` is CLM's own: it divides the logits before the softmax, so it
  sharpens or flattens the distribution rather than sampling;
* ``usage.billing_units`` counts *questions*, which is not what ``usage.n_calls``
  counts;
* and the ``CLMClient`` recipe of ``examples/clm_transport.py``, including the one
  thing it has to get right for jevper's retry policy to work at all.

The wire shapes below were taken from ``clm-serve`` 0.1.0 on 2026-09-26, running its
real FastAPI app against its own mock encoder (``tools/playground_mock.py``) — real
routes, real status codes, real answer schema, and numbers that are lexical noise
rather than predictions. See ``docs/local-servers.md`` for why a real encoder was
not available and what that leaves untested.
"""

from __future__ import annotations

import asyncio
import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType
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
    RetryPolicy,
    Score,
    SystemOneClient,
)

REPO = Path(__file__).resolve().parents[1]
EXAMPLE = REPO / "examples" / "clm_transport.py"

MODEL = "clm-latest"
STATE = "Customer: my invoice was charged twice and nobody answers the phone!"

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

# The answer objects exactly as `clm.schema.answer_from_probs` builds them: the
# noul is the two fields and nothing else, and both distributions on a score are
# keyed by `str(level)`, because that is what a JSON object key is.
NOUL = {"type": "noul", "noul": 0.41074998473756347}
CHOICE = {
    "type": "choice",
    "choice": "billing",
    "confidence": 0.9863062725066321,
    "probabilities": {"billing": 0.9931531301610572, "technical": 0.006846869838942846},
}
SCORE = {
    "type": "score",
    "score": 0.9999905871640183,
    "confidence": 0.9999602569284626,
    "legend": {"0": "Calm", "1": "Frustrated", "2": "Very angry"},
    "probabilities": {
        "0": 1.7954108503222396e-05,
        "1": 0.999973504618975,
        "2": 8.541272521649937e-06,
    },
}
ANSWERS = {"urgency": NOUL, "department": CHOICE, "frustration": SCORE}
USAGE = {"billing_units": 3, "input_tokens": 97, "output_tokens": 0}
MODELS = {
    "models": [
        {
            "name": "clm-latest",
            "release_date": "2026-09-19",
            "description": "Contrastive language model: Qwen3-8B encoder + trained projection heads",
        },
        {
            "name": "clm-raw",
            "release_date": "2026-09-19",
            "description": "Ablation: cosine in the raw encoder embedding space, no projection head",
        },
    ]
}


def clm_body(answers: dict[str, Any] | None = None, **overrides: Any) -> dict[str, Any]:
    """A response shaped like CLM's: the System One body, with its own usage."""
    body: dict[str, Any] = {
        "model": MODEL,
        "answers": ANSWERS if answers is None else answers,
        "usage": dict(USAGE),
    }
    body.update(overrides)
    return body


def answering(payload: dict[str, Any]):
    return lambda _: (200, payload)


def client(stub: Any, **kwargs: Any) -> SystemOneClient:
    """A client on the ``/v1`` base URL, which is the one CLM's routes answer under."""
    return SystemOneClient(openai_client(stub), model=MODEL, api="systemone", **kwargs)


# --- the levels, which arrive as text -------------------------------------------------------------


def test_a_score_distribution_is_keyed_by_level_index_not_by_its_json_spelling(stub_server):
    """CLM builds a score's distribution with ``{str(i): p}``, because a JSON object key is text.

    The documented arithmetic reaches a level by indexing the rubric's own number —
    ``answer.probabilities[0]`` — so leaving the keys as they arrive turns that into
    a ``KeyError`` on this surface alone. The service and the model answer under one
    rule, so the keys are read back as the indices they are.
    """
    stub = stub_server(systemone=answering(clm_body()))

    response = client(stub).system_one(state=STATE, questions=QUESTIONS)

    probabilities = response.answers["frustration"].probabilities
    assert sorted(probabilities) == [0, 1, 2]
    assert all(isinstance(key, int) for key in probabilities)
    assert probabilities[0] == pytest.approx(1.7954108503222396e-05)
    assert sum(probabilities.values()) == pytest.approx(1.0)


def test_a_score_legend_is_keyed_the_same_way_as_its_distribution(stub_server):
    """``legend`` sits beside the distribution and is spelled the same way, so it is read the same
    way. It is typed ``dict[int, …]`` and the prompt surfaces build these keys with
    ``int(level)``; one answer type must not mean two things."""
    stub = stub_server(systemone=answering(clm_body()))

    response = client(stub).system_one(state=STATE, questions=QUESTIONS)

    legend = response.answers["frustration"].legend
    assert sorted(legend) == [0, 1, 2]
    assert all(isinstance(key, int) for key in legend)
    assert legend[0] == "Calm"


def test_a_score_missing_a_rubric_level_is_refused(stub_server):
    """The same exact-set rule as every other System One server, and for the same reason: a level
    the rubric does not have is an answer to a question that was not asked."""
    partial = {
        **SCORE,
        "probabilities": {"0": 0.25, "1": 0.75},
    }
    stub = stub_server(systemone=answering(clm_body({"frustration": partial})))

    with pytest.raises(JevperError):
        client(stub).system_one(state=STATE, questions=QUESTIONS)


# --- the answers as CLM assembles them ------------------------------------------------------------


def test_a_noul_answer_carrying_only_its_number_is_read(stub_server):
    """``answer_from_probs`` builds a noul as ``{"type", "noul"}`` and stops. It is the smallest
    noul any System One server sends, so it is the one that has to parse."""
    stub = stub_server(systemone=answering(clm_body()))

    response = client(stub).system_one(state=STATE, questions=QUESTIONS)

    assert response.answers["urgency"].noul == pytest.approx(0.41074998473756347)


def test_a_choice_keeps_the_option_keys_the_caller_asked_with(stub_server):
    """CLM keys the distribution by the caller's own criteria, so they are already the caller's
    strings and are read as they are."""
    stub = stub_server(systemone=answering(clm_body()))

    response = client(stub).system_one(state=STATE, questions=QUESTIONS)

    answer = response.answers["department"]
    assert answer.choice == "billing"
    assert answer.confidence == pytest.approx(0.9863062725066321)
    assert set(answer.probabilities) == {"billing", "technical"}


def test_the_model_is_the_name_the_caller_asked_for(stub_server):
    """CLM echoes the model it served, which is the name the caller sent. Nothing chooses a
    checkpoint here, so there is no second field saying which one answered."""
    stub = stub_server(systemone=answering(clm_body()))

    response = client(stub).system_one(state=STATE, questions=QUESTIONS)

    assert response.model == MODEL


def test_structured_state_is_sent_as_itself(stub_server):
    """The wire format takes structured state, and CLM renders an object as ``key: value`` prose
    server-side — so it must not be turned into a description of itself on the way out."""
    stub = stub_server(systemone=answering(clm_body()))
    state = {"messages": [{"role": "user", "content": "charged twice"}], "turn": 3}

    client(stub).system_one(state=state, questions={"urgency": Noul(instructions="Urgent?")})

    assert stub.requests[0]["state"] == state


# --- the two routes -------------------------------------------------------------------------------


def test_list_models_reads_the_typesafe_list_clm_answers(stub_server):
    """A parity gain, the same one ollaya offers: this server answers the path with the shape
    ``parse_models`` documents, where an OpenAI-style gateway on the same path does not."""
    stub = stub_server(systemone=answering(clm_body()), models=(200, MODELS))

    models = client(stub).list_models()

    assert [m.name for m in models] == ["clm-latest", "clm-raw"]
    assert models[0].release_date == "2026-09-19"


def test_both_routes_are_the_ones_this_server_serves(stub_server):
    """``/v1/systemone`` and ``/v1/models``, under the version prefix like every other
    System One route — the base URL is the whole of the configuration."""
    stub = stub_server(systemone=answering(clm_body()), models=(200, MODELS))
    clm = client(stub)

    clm.system_one(state=STATE, questions={"urgency": Noul(instructions="Urgent?")})
    clm.list_models()

    assert stub.paths == ["/v1/systemone", "/v1/models"]


# --- what this server does with a question --------------------------------------------------------


def test_a_bare_noul_is_refused_before_a_request_is_spent(stub_server):
    """The hosted Jev service answers 400 for a noul with neither instructions nor criteria, and
    jevper refuses it locally so the request is never sent. CLM *accepts* one, reading the question
    id in their place — which is exactly why the refusal is the caller's to lift rather than
    something to relax per server."""
    stub = stub_server(systemone=answering(clm_body()))

    with pytest.raises(InvalidQuestionError):
        client(stub).system_one(state=STATE, questions={"urgency": Noul()})

    assert stub.requests == []


def test_lifting_the_rule_lets_this_server_answer_it(stub_server):
    """The flag is what makes the documented CLM call work, and it is the only thing that does:
    the default above is unchanged on this server as on every other."""
    bare = {"urgency": {"type": "noul", "noul": 0.39753}}
    stub = stub_server(systemone=answering(clm_body(bare)))

    response = client(stub, noul_requires_question=False).system_one(
        state=STATE, questions={"urgency": Noul()}
    )

    assert response.answers["urgency"].noul == pytest.approx(0.39753)


def test_temperature_reaches_this_server_where_it_sharpens_rather_than_samples(stub_server):
    """CLM's own field: it divides the logits before the softmax, so it flattens above 1 and
    sharpens below. ``extra_body`` is where a caller's own body goes, and a key named there is the
    value that reaches the wire."""
    stub = stub_server(systemone=answering(clm_body()))

    client(stub, extra_body={"temperature": 0.2}).system_one(state=STATE, questions=QUESTIONS)

    assert stub.requests[0]["temperature"] == 0.2


def test_the_typed_temperature_is_refused_on_this_wire_rather_than_ignored(stub_server):
    """The same field means one thing on the prompt surfaces — which token to sample — and something
    else here, where it sets how decisive the distribution is. A typed option that silently did
    nothing would leave a caller believing they had sharpened it, so jevper refuses it by name
    instead; the test above is the way to reach the field.
    """
    stub = stub_server(systemone=answering(clm_body()))

    with pytest.raises(ClientCapabilityError, match="this wire format takes none"):
        client(stub).system_one(state=STATE, questions=QUESTIONS, temperature=0.2)

    assert stub.requests == []


def test_a_fastapi_error_body_reaches_the_caller_as_the_providers_own_sentence(stub_server):
    """This server raises FastAPI's ``HTTPException``, so the body is ``{"detail": …}`` rather than
    the ``{"error": …}`` envelope the other servers here use. The status is what makes the call
    retryable; the sentence is what makes it actionable, and it must not be lost to the shape."""
    stub = stub_server(
        systemone=lambda _: (
            422,
            {"detail": "invalid request: temperature must be in (0, 100]"},
        )
    )

    with pytest.raises(JevperError) as caught:
        client(stub).system_one(state=STATE, questions=QUESTIONS)

    assert "temperature must be in (0, 100]" in str(caught.value)


def test_usage_counts_the_requests_jevper_made_not_the_questions_the_server_served(stub_server):
    """``billing_units`` is the number of questions, and ``n_calls`` is the number of requests.
    They are different facts, and one request answered every question here, so reading the server's
    count as its own would report three calls for one."""
    stub = stub_server(systemone=answering(clm_body()))

    response = client(stub).system_one(state=STATE, questions=QUESTIONS)

    assert response.usage.n_calls == 1
    assert len(stub.requests) == 1
    assert response.usage.input_tokens == 97


# --- parity with the other client -----------------------------------------------------------------


def test_the_async_client_reaches_the_same_routes_and_reads_the_same_answers(stub_server):
    """The async twin over the real SDK, and the same string-keyed levels read back as indices."""
    stub = stub_server(systemone=answering(clm_body()), models=(200, MODELS))

    async def go() -> Any:
        clm = AsyncSystemOneClient(async_openai_client(stub), model=MODEL, api="systemone")
        return await clm.system_one(state=STATE, questions=QUESTIONS)

    response = asyncio.run(go())

    assert set(response.answers) == set(QUESTIONS)
    assert sorted(response.answers["frustration"].probabilities) == [0, 1, 2]
    assert response.answers["department"].choice == "billing"


# --- the CLMClient recipe -------------------------------------------------------------------------


def _load_example() -> ModuleType:
    """The recipe, imported as written rather than copied into the test."""
    spec = importlib.util.spec_from_file_location("clm_transport_example", EXAMPLE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _CLMError(RuntimeError):
    """``clm.CLMError``'s shape: a ``status``, and no ``status_code``."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(f"{status}: {message}")
        self.status = status


class FakeCLMClient:
    """``clm.CLMClient`` reduced to the two operations the recipe calls, with the shapes it has.

    A private ``_post(path, body)`` answering a ``(payload, response)`` tuple, a public
    ``models()`` answering the list inside the envelope, and failures raised as a ``status``
    with no ``status_code`` — the three details the recipe has to get right.
    """

    def __init__(self, body: dict[str, Any] | None = None, models: Any = None, failure: Any = None):
        self._body = clm_body() if body is None else body
        self._models = MODELS["models"] if models is None else models
        self._failure = failure
        self.posted: list[tuple[str, dict[str, Any]]] = []

    def _post(self, path: str, body: dict[str, Any]) -> tuple[dict[str, Any], Any]:
        self.posted.append((path, body))
        if self._failure is not None:
            raise self._failure
        return self._body, None

    def models(self) -> list[dict[str, Any]]:
        if self._failure is not None:
            raise self._failure
        return self._models


def test_the_recipe_runs_against_a_clm_client(capsys: pytest.CaptureFixture[str]) -> None:
    """The recipe's own program, run as written: one request, three typed answers, and the rubric's
    levels read back as the indices the documented arithmetic uses."""
    _load_example().main(FakeCLMClient())

    assert capsys.readouterr().out.split("\n")[:4] == [
        "0.41074998473756347",
        "billing",
        "[0, 1, 2]",
        "clm-latest",
    ]


def test_the_recipe_asks_for_the_route_this_server_answers() -> None:
    """A ``CLMClient``'s base URL is the server root, so the version prefix the routes sit under
    is the recipe's to add — the one thing it cannot inherit from the client object."""
    clm = FakeCLMClient()
    _load_example().main(clm)

    assert [path for path, _ in clm.posted] == ["/v1/systemone"]


def test_the_recipe_reads_the_model_list_through_the_envelope() -> None:
    """``CLMClient.models()`` answers the list *inside* the envelope, and jevper reads the envelope
    because that is what the route returns. Unwrapping it the other way answers a list where an
    object is expected."""
    recipe = _load_example()
    clm = FakeCLMClient()

    models = SystemOneClient(
        recipe.CLMTransport(clm), model=MODEL, api="systemone"
    ).list_models()

    assert [m.name for m in models] == ["clm-latest", "clm-raw"]


def test_the_recipe_gives_a_server_failure_the_status_jevper_retries_on() -> None:
    """``CLMError`` carries ``.status``; jevper reads ``.status_code``, which is what both official
    SDKs set. Left as it is, a 502 from an unreachable encoder arrives with no status — so it is
    neither retried nor reported as the transient failure it is. This is the one thing the recipe
    has to get right, and it is what this pins."""
    recipe = _load_example()
    clm = FakeCLMClient(failure=_CLMError(502, "embedder unreachable at http://127.0.0.1:8090"))

    with pytest.raises(JevperError) as caught:
        SystemOneClient(
            recipe.CLMTransport(clm),
            model=MODEL,
            api="systemone",
            retry=RetryPolicy(n_retries=1, base_delay=0, max_delay=0),
        ).system_one(state=STATE, questions={"urgency": Noul(instructions="Urgent?")})

    assert getattr(caught.value, "status_code", None) == 502
    # One initial attempt plus the one retry the policy allows: the 502 is retried as transient
    # because the status reached jevper at all.
    assert len(clm.posted) == 2


# --- against a real clm-serve ----------------------------------------------------------------------
#
# Skipped unless CLM_BASE_URL is set, e.g.
#   CLM_BASE_URL=http://127.0.0.1:8700/v1 CLM_MODEL=clm-latest pytest tests/test_clm_integration.py
#
# Last run on 2026-09-27 against clm-serve 0.1.0 with the reference head over a Qwen3-8B Q8_0 GGUF
# under llama.cpp on a 12 GB card, which is a real encoder rather than a mock one. The encoder is the
# reference head's own, not a stand-in, so a probability can be held against a published figure --
# `clm-serve` is a ranker and returns the same float for the same request every time, five calls
# measured bitwise identical. The Jev service, by contrast, wobbles about +-0.01 and gets no such
# assertion anywhere on this page.
CLM_BASE_URL = os.environ.get("CLM_BASE_URL")
CLM_MODEL = os.environ.get("CLM_MODEL", "clm-latest")
# Which encoder is behind the server decides what an over-long state does: vLLM honours the
# `truncate_prompt_tokens` the server sends and cuts the state silently, llama.cpp ignores it and
# refuses. Only the refusing behaviour can be asserted, so it is opt-in.
CLM_ENCODER = os.environ.get("CLM_ENCODER", "vllm")


def _live_client(**kwargs: Any) -> SystemOneClient:
    from openai import OpenAI

    return SystemOneClient(
        OpenAI(base_url=CLM_BASE_URL, api_key="clm"), model=CLM_MODEL, api="systemone", **kwargs
    )


@pytest.mark.skipif(not CLM_BASE_URL, reason="CLM_BASE_URL is not set")
def test_a_real_clm_answers_three_questions_and_lists_its_models() -> None:
    """The route this document is about, end to end, against a real encoder.

    ``STATE`` and ``QUESTIONS["department"]`` are the encoder's own published anchor case, so the
    distribution is checked against the figure its model card prints rather than only for shape: a
    mis-paired head, the wrong pooling, or the ``clm-raw`` ablation all land far outside this
    window, while bf16 on vLLM and Q8_0 on llama.cpp both land inside it.
    """
    clm = _live_client(noul_requires_question=False)

    response = clm.system_one(state=STATE, questions=QUESTIONS)

    assert set(response.answers) == set(QUESTIONS)
    assert 0.0 <= response.answers["urgency"].noul <= 1.0
    assert response.answers["department"].choice in QUESTIONS["department"].criteria
    assert sorted(response.answers["frustration"].probabilities) == [0, 1, 2]
    assert sorted(response.answers["frustration"].legend) == [0, 1, 2]
    assert response.usage.n_calls == 1
    assert CLM_MODEL in {m.name for m in clm.list_models()}

    # The card publishes 0.98775 for this case on bf16/vLLM; Q8_0 under llama.cpp measured 0.98390
    # in the same batch. 0.01 covers both and is an order of magnitude tighter than the gap to the
    # ablation (0.77) or to a mis-paired head.
    assert abs(response.answers["department"].probabilities["billing"] - 0.98775) < 0.01


@pytest.mark.skipif(not CLM_BASE_URL, reason="CLM_BASE_URL is not set")
def test_a_real_clm_counts_requests_the_same_way_the_questions_were_billed() -> None:
    """``usage.n_calls`` counts requests and the server's own count counts questions; both survive."""
    response = _live_client().system_one(state=STATE, questions=QUESTIONS)

    assert response.usage.n_calls == 1
    assert response.debug["llm_attempts"][0]["response"]["usage"]["billing_units"] == len(QUESTIONS)


@pytest.mark.skipif(
    not CLM_BASE_URL or CLM_ENCODER != "llamacpp",
    reason="needs CLM_BASE_URL and CLM_ENCODER=llamacpp; vLLM truncates instead of refusing",
)
def test_an_over_long_state_is_refused_rather_than_truncated() -> None:
    """What this backend does with a state past the window, and what it costs the caller.

    ``clm-serve`` asks the encoder to cut the text with ``truncate_prompt_tokens``, which is
    vLLM's parameter; llama.cpp ignores it and answers 400, which the server relays as 502. A 502
    is a 5xx, so the default policy spends every attempt on a condition that cannot improve --
    which is the part worth pinning, since the fix is the caller's window and not a retry.
    """
    # The window is the operator's setting, not this test's, so the state is grown until the
    # server refuses it. Each step that still answers is one real encoder pass, and the refusal
    # itself is immediate -- the encoder rejects on length before it computes anything.
    client = _live_client()
    long_state, refusal = "", None
    for repeats in (700, 2800, 11200, 44800):
        long_state = "Customer wrote: " + ("the invoice is wrong and nobody helps. " * repeats)
        try:
            client.system_one(state=long_state, questions={"q": Noul(instructions="Urgent?")})
        except JevperError as exc:
            refusal = exc
            break
    if refusal is None:
        pytest.skip("this server's window is wider than a state this test can build")

    assert getattr(refusal, "status_code", None) == 502
    # The default policy is two retries, so this permanent failure is attempted three times.
    assert len(getattr(refusal, "attempts", [])) == 3

    with pytest.raises(JevperError) as once:
        _live_client(retry=RetryPolicy(n_retries=0, base_delay=0, max_delay=0)).system_one(
            state=long_state, questions={"q": Noul(instructions="Urgent?")}
        )

    assert len(getattr(once.value, "attempts", [])) == 1
