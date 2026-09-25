"""The System One surface: the Jev wire format itself, reached through a client object's ``post``.

The three prompt surfaces ask a model a question in a prompt. This one posts ``{"state", "model",
"questions"}`` to ``/v1/systemone`` and reads the answer shape the service sends back, which is the
contract in ``docs/jev-comparison.md`` and the responses recorded in
``tests/fixtures/jev-1.13-free.json``. What is pinned here:

* the request body is the Jev one — questions keyed by id, ``model`` named, no prompt anywhere;
* every question travels in one request, which is the shape the service is built to be asked;
* the service's own ``score``, ``confidence``, ``choice`` and ``legend`` come back untouched, because
  it computed them from a model trained for the decision;
* usage from one request is counted once, not once per question that request answered;
* the options this wire format cannot carry are refused by name before a request is sent;
* a noul with nothing to judge is refused, because the service answers 400 for one;
* ``GET /v1/models`` is read as the service's OpenAPI declares it, and anything else is reported as
  the wrong shape rather than half-read.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

import pytest
from fakes import openai_client

from jevper import (
    AsyncSystemOneClient,
    Choice,
    ClientCapabilityError,
    Example,
    InvalidQuestionError,
    MalformedAnswerError,
    Noul,
    ProviderError,
    Score,
    SystemOneClient,
)

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "jev-1.13-free.json"
STATE = "The invoice total does not match the amount I was charged."
MODEL = "jev-1.13-free"

CHOICE = Choice(instructions="Which team?", criteria={"billing": "Payments", "technical": "Bugs"})
SCORE = Score(instructions="How bad?", criteria=["fine", "broken"])
NOUL = Noul(instructions="Do the numbers disagree?")


def recorded_answers(name: str) -> dict[str, Any]:
    document = json.loads(FIXTURE.read_text())
    case = next(case for case in document["cases"] if case["name"] == name)
    return case["response"]["answers"]


def systemone_body(answers: dict[str, Any], *, model: str = MODEL) -> dict[str, Any]:
    """A response shaped like the service's, for the stub to hand back."""
    return {"model": model, "answers": answers, "usage": {"input_tokens": 40, "output_tokens": 12}}


def answering(answers: dict[str, Any]):
    return lambda _: (200, systemone_body(answers))


def client_for(stub: Any, **kwargs: Any) -> SystemOneClient:
    return SystemOneClient(openai_client(stub), model=MODEL, api="systemone", **kwargs)


# --- the request --------------------------------------------------------------------------------


def test_the_request_is_the_jev_wire_format(stub_server):
    """``{state, model, questions}`` on ``/v1/systemone``, and nothing that belongs to a prompt."""
    stub = stub_server(systemone=answering({"which": {"type": "choice", "choice": "billing",
                                                     "confidence": 0.6,
                                                     "probabilities": {"billing": 0.6, "technical": 0.4}}}))

    client_for(stub).system_one(state=STATE, questions={"which": CHOICE})

    assert stub.paths == ["/v1/systemone"]
    sent = stub.requests[0]
    assert sent["state"] == STATE
    assert sent["model"] == MODEL
    assert sent["questions"] == {
        "which": {"type": "choice", "instructions": "Which team?",
                  "criteria": {"billing": "Payments", "technical": "Bugs"}}
    }
    assert not {"messages", "response_format", "logprobs"} & set(sent)


def test_a_null_criteria_value_stays_on_the_wire(stub_server):
    """``{"billing": null}`` is how a caller says an option needs no description.

    Dropping the key would drop the option with it, and the service would answer a question about
    fewer options than the one that was asked.
    """
    stub = stub_server(systemone=answering({"c": {"type": "choice", "choice": "billing",
                                                  "confidence": 1.0, "probabilities": {"billing": 1.0}}}))

    client_for(stub).system_one(
        state=STATE, questions={"c": Choice(instructions="Which?", criteria={"billing": None})}
    )

    assert stub.requests[0]["questions"]["c"]["criteria"] == {"billing": None}


def test_structured_state_travels_as_itself(stub_server):
    """The wire format takes an object or a list of state, and does not quote it into text."""
    stub = stub_server(systemone=answering({"n": {"type": "noul", "noul": 0.9}}))
    state = {"invoice": {"total": 100, "charged": 120}}

    client_for(stub).system_one(state=state, questions={"n": NOUL})

    assert stub.requests[0]["state"] == state


def test_examples_are_never_sent_on_this_wire(stub_server):
    """A question's few-shot examples are refused, not dropped: the request has no field for them."""
    stub = stub_server(systemone=answering({}))
    question = Choice(
        instructions="Which?",
        criteria={"a": None},
        examples=(Example(state="s", answer="a"),),
    )

    with pytest.raises(ClientCapabilityError, match="examples"):
        client_for(stub).system_one(state=STATE, questions={"c": question})

    assert stub.requests == []


# --- one request for the whole call --------------------------------------------------------------


def test_every_question_travels_in_one_request(stub_server):
    """The service evaluates a map of questions in one go, and jevper sends the whole map.

    Measured on 2026-09-26: eleven questions came back from one request in 1.08 s, where a
    per-question path spends a round trip each.
    """
    stub = stub_server(
        systemone=answering({
            "which": {"type": "choice", "choice": "billing", "confidence": 0.6,
                      "probabilities": {"billing": 0.6, "technical": 0.4}},
            "how": {"type": "score", "score": 1.0, "confidence": 1.0,
                    "legend": {"0": "fine", "1": "broken"}, "probabilities": {"0": 0.0, "1": 1.0}},
            "is_it": {"type": "noul", "noul": 0.95},
        })
    )

    response = client_for(stub).system_one(
        state=STATE, questions={"which": CHOICE, "how": SCORE, "is_it": NOUL}
    )

    assert stub.paths == ["/v1/systemone"]
    assert sorted(stub.requests[0]["questions"]) == ["how", "is_it", "which"]
    assert set(response.answers) == {"which", "how", "is_it"}


def test_the_requests_usage_is_counted_once_for_the_whole_call(stub_server):
    """One request answered three questions, so its tokens are reported once."""
    stub = stub_server(
        systemone=answering({
            "which": {"type": "choice", "choice": "billing", "confidence": 0.6,
                      "probabilities": {"billing": 0.6, "technical": 0.4}},
            "how": {"type": "score", "score": 1.0, "confidence": 1.0,
                    "legend": {"0": "fine", "1": "broken"}, "probabilities": {"0": 0.0, "1": 1.0}},
            "is_it": {"type": "noul", "noul": 0.95},
        })
    )

    response = client_for(stub).system_one(
        state=STATE, questions={"which": CHOICE, "how": SCORE, "is_it": NOUL}
    )

    assert response.usage.n_calls == 1
    assert response.usage.input_tokens == 40
    assert response.usage.output_tokens == 12


def test_each_questions_attempt_record_names_it_and_carries_the_same_request(stub_server):
    """The debug trail has to attribute one shared request to each question it answered."""
    stub = stub_server(
        systemone=answering({
            "a": {"type": "noul", "noul": 0.5},
            "b": {"type": "noul", "noul": 0.5},
        })
    )

    response = client_for(stub).system_one(state=STATE, questions={"a": NOUL, "b": NOUL})

    attempts = response.debug["llm_attempts"]
    assert [record["question_id"] for record in attempts] == ["a", "b"]
    assert all(record["surface"] == "systemone" for record in attempts)
    assert response.debug["method"] == "systemone"


# --- the answers come back as the service sent them -----------------------------------------------


def test_the_services_own_numbers_are_kept(stub_server):
    """``confidence`` and ``score`` are the service's, not jevper's recomputation of them.

    jevper's formulas agree with the service to within 0.015 on every recorded answer, which is a
    reason to trust them on a prompt surface — not a reason to overwrite a trained model's own
    arithmetic with them.
    """
    stub = stub_server(
        systemone=answering({
            "how": {"type": "score", "score": 1.7, "confidence": 0.42,
                    "legend": {"0": "fine", "1": "broken"}, "probabilities": {"0": 0.3, "1": 0.7}},
        })
    )

    answer = client_for(stub).system_one(state=STATE, questions={"how": SCORE}).answers["how"]

    assert answer.score == 1.7
    assert answer.confidence == 0.42
    assert answer.legend == {0: "fine", 1: "broken"}


def test_a_recorded_service_response_reads_back_as_jevpers_own_answers(stub_server):
    """The three real responses in the fixture, read through this surface rather than by hand."""
    document = json.loads(FIXTURE.read_text())
    case = next(item for item in document["cases"] if item["name"] == "quickstart-three-questions")
    questions = {
        question_id: _question_from_wire(wire)
        for question_id, wire in case["request"]["questions"].items()
    }
    stub = stub_server(systemone=lambda _: (200, case["response"]))

    response = client_for(stub).system_one(state=case["request"]["state"], questions=questions)

    assert response.model == case["response"]["model"]
    assert response.answers["department"].choice == "technical"
    assert response.answers["department"].confidence == case["response"]["answers"]["department"]["confidence"]
    assert response.answers["frustration"].score == 1.0
    assert response.answers["is_urgent"].noul == 0.99
    assert response.usage.input_tokens == case["response"]["usage"]["input_tokens"]


def _question_from_wire(wire: dict[str, Any]):
    """The fixture's own request questions, as jevper questions."""
    payload = {key: value for key, value in wire.items() if key != "type"}
    return {"choice": Choice, "score": Score, "noul": Noul}[wire["type"]].model_validate(payload)


def test_an_answer_of_the_wrong_type_fails_the_question(stub_server):
    """A noul asked and a choice answered is the model contradicting the schema, not an answer."""
    stub = stub_server(
        systemone=answering({"n": {"type": "choice", "choice": "yes", "confidence": 1.0,
                                    "probabilities": {"yes": 1.0}}})
    )

    with pytest.raises(MalformedAnswerError, match="asked a noul and the service answered 'choice'"):
        client_for(stub).system_one(state=STATE, questions={"n": NOUL})


def test_a_missing_answer_names_the_question_the_service_did_answer(stub_server):
    """One question of three unanswered is a failure worth naming, not an empty answer."""
    stub = stub_server(
        systemone=answering({"a": {"type": "noul", "noul": 0.5}, "c": {"type": "noul", "noul": 0.5}})
    )

    with pytest.raises(MalformedAnswerError, match="'b'.*answered \\['a', 'c'\\]"):
        client_for(stub).system_one(state=STATE, questions={"a": NOUL, "b": NOUL, "c": NOUL})


def test_a_score_answered_with_levels_outside_the_rubric_is_refused(stub_server):
    """The levels are the rubric's, so an answer naming others is not this question's answer."""
    stub = stub_server(
        systemone=answering({"s": {"type": "score", "score": 4.0, "confidence": 1.0,
                                   "legend": {"0": "fine", "1": "broken"},
                                   "probabilities": {"0": 0.0, "1": 0.0, "4": 1.0}}})
    )

    with pytest.raises(MalformedAnswerError, match="levels \\['4'\\]"):
        client_for(stub).system_one(state=STATE, questions={"s": SCORE})


def test_a_confidence_outside_zero_to_one_is_refused(stub_server):
    """``confidence`` is a share of certainty; a number outside that was not computed by anyone."""
    stub = stub_server(
        systemone=answering({"c": {"type": "choice", "choice": "billing", "confidence": 1.4,
                                    "probabilities": {"billing": 1.0, "technical": 0.0}}})
    )

    with pytest.raises(MalformedAnswerError, match="confidence"):
        client_for(stub).system_one(state=STATE, questions={"c": CHOICE})


# --- what this wire format cannot carry ------------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "offender"),
    [
        ({"method": "logprobs"}, "method='logprobs'"),
        ({"method": "grammar"}, "method='grammar'"),
        ({"method": "discrete"}, "method='discrete'"),
        ({"temperature": 0.0}, "temperature=0.0"),
        ({"prompt_cache_key": "abc"}, "prompt_cache_key="),
    ],
)
def test_an_option_this_wire_cannot_carry_is_refused_by_name(stub_server, kwargs, offender):
    """Refused rather than dropped: the service ignores what it does not know, which is the problem."""
    stub = stub_server(systemone=answering({}))

    with pytest.raises(ClientCapabilityError, match=offender):
        client_for(stub).system_one(state=STATE, questions={"n": NOUL}, **kwargs)

    assert stub.requests == []


def test_every_unexpressible_option_is_named_in_one_error(stub_server):
    """One complaint, not one per attempt: the caller fixes the call in a single go."""
    stub = stub_server(systemone=answering({}))

    with pytest.raises(ClientCapabilityError) as raised:
        client_for(stub).system_one(
            state=STATE, questions={"n": NOUL}, method="logprobs", temperature=0.0
        )

    message = str(raised.value)
    assert "method='logprobs'" in message and "temperature=0.0" in message


def test_reasoning_is_refused_and_says_what_to_do_instead(stub_server):
    """The wire format has no thinking field, and the two-step path is two requests."""
    from jevper import ReasoningConfig

    stub = stub_server(systemone=answering({}))

    with pytest.raises(ClientCapabilityError, match="reasoning="):
        client_for(stub).system_one(
            state=STATE, questions={"n": NOUL}, reasoning=ReasoningConfig(effort="medium")
        )

    assert stub.requests == []


def test_a_noul_with_nothing_to_judge_is_refused_as_the_service_refuses_it(stub_server):
    """The service answers 400 for one, with or without the key present — measured, both ways."""
    stub = stub_server(systemone=answering({}))

    for question in (Noul(), Noul(instructions=""), Noul(criteria={})):
        with pytest.raises(InvalidQuestionError, match="must carry instructions or criteria"):
            client_for(stub).system_one(state=STATE, questions={"n": question})

    assert stub.requests == []


def test_a_noul_with_either_criterion_side_is_sent(stub_server):
    """Either value alone is enough for the service, and so it is enough here."""
    stub = stub_server(systemone=answering({"n": {"type": "noul", "noul": 0.5}}))

    client_for(stub).system_one(
        state=STATE, questions={"n": Noul(criteria={"true": "it is wrong"})}
    )

    assert stub.requests[0]["questions"]["n"]["criteria"] == {"true": "it is wrong"}


def test_the_surface_is_never_chosen_by_auto():
    """``post`` is a method both official SDKs have; a call must not start posting a Jev body by
    accident just because the client object exposes one."""
    calls: list[dict[str, Any]] = []

    class OnlyPost:
        def post(self, path: str, **kwargs: Any) -> dict[str, Any]:
            calls.append({"path": path, **kwargs})
            return systemone_body({"n": {"type": "noul", "noul": 0.5}})

    client = SystemOneClient(OnlyPost(), model=MODEL)

    with pytest.raises(ClientCapabilityError, match="exposes none of"):
        client.system_one(state=STATE, questions={"n": NOUL})

    assert calls == []


def test_a_client_without_post_is_told_what_one_has():
    """A Chat-Completions-only client cannot post a Jev body, and the message says which client can.

    Not the Anthropic SDK, which does have ``post`` like every official SDK — the capability check is
    on the method, not on which vendor wrote the class.
    """

    class OnlyChat:
        class chat:
            class completions:
                @staticmethod
                def create(**kwargs: Any) -> dict[str, Any]:
                    raise AssertionError("a Jev body must never reach chat.completions.create")

    client = SystemOneClient(OnlyChat(), model=MODEL, api="systemone")

    with pytest.raises(ClientCapabilityError, match=r"OpenAI\(base_url="):
        client.system_one(state=STATE, questions={"n": NOUL})


def test_a_provider_failure_keeps_the_services_own_status_and_message(stub_server):
    """The service's 400 body reaches the caller through the error jevper raises."""
    stub = stub_server(
        systemone=lambda _: (400, {"detail": "Too many score levels. Must have at most 10 levels."})
    )

    with pytest.raises(ProviderError, match="at most 10 levels") as raised:
        client_for(stub).system_one(state=STATE, questions={"s": SCORE})

    assert raised.value.status_code == 400


def test_a_transient_failure_is_retried_by_the_policy(stub_server):
    """jevper owns the retries here as everywhere else: 429 once, then the answer."""
    attempts = {"n": 0}

    def script(_: Any) -> tuple[int, Any]:
        attempts["n"] += 1
        if attempts["n"] == 1:
            return 429, {"error": {"message": "slow down"}}
        return 200, systemone_body({"n": {"type": "noul", "noul": 0.5}})

    stub = stub_server(systemone=script)
    client = client_for(stub, retry=None)

    response = client.system_one(state=STATE, questions={"n": NOUL})

    assert response.answers["n"].noul == 0.5
    assert response.usage.n_retries == 1
    assert attempts["n"] == 2


# --- the models route ------------------------------------------------------------------------------


def test_the_models_route_is_read_as_the_openapi_declares_it(stub_server):
    """``{"models": [{"name", "description", "release_date"}]}`` — the service's own schema."""
    stub = stub_server(
        systemone=answering({}),
        models=(200, {"models": [{"name": "jev-1.13", "description": "flagship",
                                  "release_date": "2026-09-15"}]}),
    )

    models = client_for(stub).list_models()

    assert stub.paths == ["/v1/models"]
    assert [(model.name, model.release_date) for model in models] == [("jev-1.13", "2026-09-15")]


def test_a_gateway_answering_the_path_with_its_own_list_is_reported_as_the_wrong_shape(stub_server):
    """opencode Zen serves an OpenAI-style ``{"data": [{"id"}]}`` on the same URL, and that is a
    different API wearing this path — said plainly rather than half-read."""
    stub = stub_server(systemone=answering({}), models=(200, {"data": [{"id": "jev-1.13-free"}]}))

    with pytest.raises(MalformedAnswerError, match="'models' array"):
        client_for(stub).list_models()


# --- the async twin -------------------------------------------------------------------------------


def test_the_async_facade_sends_the_same_one_request(stub_server):
    """Same body, same answers, same single call: the parity the async suite holds the client to."""
    stub = stub_server(
        systemone=answering({
            "a": {"type": "noul", "noul": 0.25},
            "b": {"type": "noul", "noul": 0.75},
        })
    )
    from openai import AsyncOpenAI

    client = AsyncSystemOneClient(
        AsyncOpenAI(base_url=stub.base_url, api_key="test", max_retries=0, timeout=10),
        model=MODEL,
        api="systemone",
    )

    response = asyncio.run(client.system_one(state=STATE, questions={"a": NOUL, "b": NOUL}))

    assert stub.paths == ["/v1/systemone"]
    assert sorted(stub.requests[0]["questions"]) == ["a", "b"]
    assert response.usage.n_calls == 1
    assert response.answers["a"].noul == 0.25
    asyncio.run(client.aclose())


def test_the_async_facade_refuses_what_the_blocking_one_refuses(stub_server):
    """The refusals are the client's rules, not the facade's."""
    stub = stub_server(systemone=answering({}))
    from openai import AsyncOpenAI

    client = AsyncSystemOneClient(
        AsyncOpenAI(base_url=stub.base_url, api_key="test", max_retries=0, timeout=10),
        model=MODEL,
        api="systemone",
    )

    with pytest.raises(ClientCapabilityError, match="method='logprobs'"):
        asyncio.run(client.system_one(state=STATE, questions={"n": NOUL}, method="logprobs"))

    assert stub.requests == []


def test_the_async_facade_lists_the_models_too(stub_server):
    stub = stub_server(
        systemone=answering({}),
        models=(200, {"models": [{"name": "jev-1.13", "description": "", "release_date": None}]}),
    )
    from openai import AsyncOpenAI

    client = AsyncSystemOneClient(
        AsyncOpenAI(base_url=stub.base_url, api_key="test", max_retries=0, timeout=10),
        model=MODEL,
        api="systemone",
    )

    models = asyncio.run(client.alist_models())

    assert [model.name for model in models] == ["jev-1.13"]
    assert models[0].release_date is None


# --- the live service, when it is configured --------------------------------------------------------


@pytest.mark.skipif(
    not (os.environ.get("JEV_BASE_URL") and os.environ.get("JEV_API_KEY")),
    reason="set JEV_BASE_URL and JEV_API_KEY to exercise the real service",
)
def test_the_real_service_answers_through_this_surface():
    """One live call, skipped unless the endpoint is configured.

    The recorded fixture is the offline proof; this is the one that would notice the service changing
    its mind about a field, and it is skipped rather than required so the suite stays hermetic.
    """
    from openai import OpenAI

    from jevper import ModelMetadata  # noqa: F401 - imported to prove the export exists

    client = SystemOneClient(
        OpenAI(
            # `JEV_BASE_URL` is the whole route; a client object's base URL carries the version
            # prefix and jevper's path is relative to it, exactly as an OpenAI client expects.
            base_url=os.environ["JEV_BASE_URL"].removesuffix("/systemone"),
            api_key=os.environ["JEV_API_KEY"],
            max_retries=0,
            timeout=60.0,
        ),
        model=MODEL,
        api="systemone",
    )

    response = client.system_one(
        state="I was charged twice for the same order. Can someone look into this?",
        questions={
            "refund": Noul(instructions="Is the customer asking for a refund?"),
            "team": Choice(instructions="Which team?", criteria={"billing": None, "support": None}),
            "severity": Score(instructions="How bad?", criteria=["fine", "broken", "furious"]),
        },
    )

    assert set(response.answers) == {"refund", "team", "severity"}
    assert 0.0 <= response.answers["refund"].noul <= 1.0
    assert response.answers["team"].choice in {"billing", "support"}
    assert 0.0 <= response.answers["severity"].score <= 2.0
    assert response.usage.n_calls == 1
    assert response.usage.input_tokens and response.usage.input_tokens > 0
