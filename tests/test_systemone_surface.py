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
from fakes import async_openai_client, openai_client

from jevper import (
    AsyncSystemOneClient,
    Choice,
    ClientCapabilityError,
    Example,
    InvalidQuestionError,
    JevperError,
    MalformedAnswerError,
    Noul,
    NoulCriteria,
    ProviderError,
    ReasoningConfig,
    RetryPolicy,
    Score,
    SystemOneClient,
)
from jevper.types import parse_models

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


# --- what the caller may not put in the request --------------------------------------------------
#
# The three prompt surfaces route the caller's own request fields through ``_caller_body``, which
# refuses a ``model`` (it would put a different model on the wire than the one jevper reports) and a
# ``stream`` (it would arrive as an event stream jevper cannot read). This surface carried the merge
# without either refusal, so the whole contract the other three had was open on the fourth.


@pytest.mark.parametrize(
    "extra", [{"model": "someone-elses-model"}, {"stream": True}], ids=["model", "stream"]
)
def test_extra_body_cannot_replace_what_jevper_set(stub_server, extra):
    """The body above the merge is the call; a caller's key that contradicts it is not sent."""
    stub = stub_server(systemone=answering({"n": {"type": "noul", "noul": 0.5}}))

    with pytest.raises(JevperError, match="extra_body"):
        client_for(stub, extra_body=extra).system_one(state=STATE, questions={"n": NOUL})

    assert stub.requests == []


def test_extra_body_still_carries_a_field_of_the_callers_own(stub_server):
    """What ``extra_body`` is for: a field a deployment in front of the service understands."""
    stub = stub_server(systemone=answering({"n": {"type": "noul", "noul": 0.5}}))

    client_for(stub, extra_body={"tenant": "acme"}).system_one(state=STATE, questions={"n": NOUL})

    assert stub.requests[0]["tenant"] == "acme"
    assert stub.requests[0]["model"] == MODEL


def test_extra_headers_reach_the_system_one_request(stub_server):
    """A low-level ``post`` takes its headers in ``options``, which the builder has to say.

    Merged and then dropped is worse than not accepting them: the caller reads a header trace that
    the deployment never received, and a gateway in front of the service never sees its tenant.
    """
    stub = stub_server(systemone=answering({"n": {"type": "noul", "noul": 0.5}}))

    client_for(stub, extra_headers={"x-trace-id": "abc123"}).system_one(
        state=STATE, questions={"n": NOUL}
    )

    assert stub.headers[0].get("x-trace-id") == "abc123"


def test_no_options_key_when_there_are_no_headers():
    """A duck client taking ``(path, body, cast_to)`` is the documented shape, so the key stays out."""
    seen: list[dict[str, Any]] = []

    class Duck:
        def post(self, **kwargs: Any) -> dict[str, Any]:
            seen.append(kwargs)
            return systemone_body({"n": {"type": "noul", "noul": 0.5}})

    SystemOneClient(Duck(), model=MODEL, api="systemone").system_one(
        state=STATE, questions={"n": NOUL}
    )

    assert sorted(seen[0]) == ["body", "cast_to", "path"]


# --- a body that is not an answer ----------------------------------------------------------------


@pytest.mark.parametrize(
    "body", ["boom", b"<html>oops</html>", [1, 2], 5, None], ids=["str", "bytes", "list", "int", "null"]
)
def test_a_response_that_is_not_an_object_is_a_provider_failure(stub_server, body):
    """``cast_to=dict`` passes a JSON string, list or number through unchanged.

    Read as an empty body it reached the answer reader, which reported a malformed answer for a
    question that was never asked — blaming the question, outside the retry that the verdict the
    route actually produced belongs to.
    """
    stub = stub_server(systemone=lambda _: (200, body))

    with pytest.raises((ProviderError, ClientCapabilityError)) as raised:
        client_for(stub).system_one(state=STATE, questions={"n": NOUL})

    assert "malformed" not in str(raised.value).lower()


def test_a_transient_failure_while_reading_the_answers_is_retried(stub_server):
    """A 5xx here is the provider's own failure, so it is retried like one anywhere else."""
    calls: list[int] = []

    def handler(_: Any) -> tuple[int, Any]:
        calls.append(1)
        if len(calls) == 1:
            return (503, {"error": {"message": "overloaded"}})
        return (200, systemone_body({"n": {"type": "noul", "noul": 0.7}}))

    stub = stub_server(systemone=handler)
    client = client_for(stub, retry=RetryPolicy(n_retries=1, base_delay=0, max_delay=0))

    response = client.system_one(state=STATE, questions={"n": NOUL})

    assert response.answers["n"].noul == 0.7
    assert response.usage.n_retries == 1
    assert len(calls) == 2


def test_a_score_answer_whose_distribution_is_null_is_a_jevper_error(stub_server):
    """``probabilities: null`` reached the level check as ``None`` and raised ``TypeError``.

    Every other malformed body on this path arrives as a ``MalformedAnswerError`` naming the
    question; this one escaped the hierarchy entirely, so a caller catching ``JevperError`` for an
    unusable service body did not catch it.
    """
    stub = stub_server(
        systemone=answering(
            {
                "s": {
                    "type": "score",
                    "score": 1.0,
                    "confidence": 0.5,
                    "legend": {"0": "fine", "1": "broken"},
                    "probabilities": None,
                }
            }
        )
    )

    with pytest.raises(MalformedAnswerError, match="score distribution"):
        client_for(stub).system_one(state=STATE, questions={"s": SCORE})


def test_a_choice_answer_naming_an_option_the_question_never_offered(stub_server):
    """A decision about an option that was not asked about, which the caller then indexes with."""
    stub = stub_server(
        systemone=answering(
            {
                "c": {
                    "type": "choice",
                    "choice": "ghost",
                    "confidence": 0.9,
                    "probabilities": {"ghost": 0.9},
                }
            }
        )
    )

    with pytest.raises(MalformedAnswerError, match="not this question's"):
        client_for(stub).system_one(state=STATE, questions={"c": CHOICE})


def test_a_choice_answer_over_the_options_it_was_asked_about_is_kept(stub_server):
    """The check that refuses an invented option must not refuse the ones the caller declared."""
    stub = stub_server(
        systemone=answering(
            {
                "c": {
                    "type": "choice",
                    "choice": "billing",
                    "confidence": 0.6,
                    "probabilities": {"billing": 0.6, "technical": 0.4},
                }
            }
        )
    )

    response = client_for(stub).system_one(state=STATE, questions={"c": CHOICE})

    assert response.answers["c"].choice == "billing"


def test_a_noul_with_both_criteria_sides_unset_sends_no_criteria_object(stub_server):
    """``criteria: {}`` says no more than leaving the key off.

    The service's answer to a noul with no criteria is the same 400 either way, so the wire does not
    carry an object that says nothing.
    """
    stub = stub_server(systemone=answering({"n": {"type": "noul", "noul": 0.5}}))
    question = Noul(instructions="Do they disagree?", criteria=NoulCriteria())

    client_for(stub).system_one(state=STATE, questions={"n": question})

    assert "criteria" not in stub.requests[0]["questions"]["n"]


# --- refusals that were not refusals --------------------------------------------------------------


def test_a_method_pinned_on_the_constructor_is_refused_too(stub_server):
    """Only the per-call value was checked, so a client built with ``method="logprobs"`` made a call
    it cannot honour and reported the service's own method without a word about it."""
    stub = stub_server(systemone=answering({"n": {"type": "noul", "noul": 0.5}}))

    with pytest.raises(ClientCapabilityError, match="method='logprobs'"):
        client_for(stub, method="logprobs").system_one(state=STATE, questions={"n": NOUL})

    assert stub.requests == []


def test_examples_for_another_question_do_not_refuse_the_call(stub_server):
    """Resolved per question, the way the prompt surfaces resolve them.

    An example set keyed to a question that is not in this call never reaches this wire, and refusing
    the call over one leaves the caller no way to clear it but rebuilding the client.
    """
    stub = stub_server(systemone=answering({"n": {"type": "noul", "noul": 0.5}}))
    examples = {"other": (Example(state="s", answer="a"),)}

    response = client_for(stub, examples=examples).system_one(state=STATE, questions={"n": NOUL})

    assert response.answers["n"].noul == 0.5


def test_examples_that_would_reach_this_call_are_still_refused(stub_server):
    """The narrower test above must not have widened the refusal: these keys do match a question."""
    stub = stub_server(systemone=answering({}))

    with pytest.raises(ClientCapabilityError, match="examples"):
        client_for(stub, examples={"n": (Example(state="s", answer=True),)}).system_one(
            state=STATE, questions={"n": NOUL}
        )

    assert stub.requests == []


def test_a_failed_batch_attributes_its_trail_to_every_question(stub_server):
    """Three questions, one request, three attempts: the error named only the first.

    The request was asked for all three, so a trail that names one of them reads as though the other
    two were never asked — which is the unattributed case ``_attempts_for`` exists to prevent.
    """
    stub = stub_server(systemone=lambda _: (503, {"error": {"message": "overloaded"}}))
    client = client_for(stub, retry=RetryPolicy(n_retries=0))
    questions = {name: NOUL for name in ("a", "b", "cc")}

    with pytest.raises(ProviderError) as raised:
        client.system_one(state=STATE, questions=questions)

    assert len(raised.value.attempts) == 3
    assert [record["question_id"] for record in raised.value.attempts] == ["a", "b", "cc"]


# --- the model list ------------------------------------------------------------------------------

MODELS = {"models": [{"name": "jev-1.13-free", "description": "General-purpose.", "release_date": "2026-09-15"}]}


def test_the_model_list_error_names_what_arrived():
    """``got dict`` names the type of the thing, not the thing.

    The reader of this message is deciding whether a gateway answered a different API, a proxy
    dropped a field, or the service is not what it says — and the payload is the whole of that.
    """
    with pytest.raises(MalformedAnswerError) as raised:
        parse_models({"data": [{"id": "jev-1.13-free"}]})

    assert "data" in str(raised.value)


def test_a_provider_failure_while_listing_models_is_a_provider_error(stub_server):
    """The SDK's own exception type does not escape a method documented to return a list.

    ``system_one`` on the same client reports the identical 404 as a ``ProviderError`` with the
    status; a caller wrapping both calls in one ``except ProviderError`` caught one of them.
    """
    stub = stub_server(models=(404, {"error": {"message": "no models here"}}))

    with pytest.raises(ProviderError) as raised:
        client_for(stub).list_models()

    assert raised.value.status_code == 404


def test_a_transient_failure_while_listing_models_is_retried(stub_server):
    """It is jevper's request, so the caller's policy is the one that applies to it."""
    stub = stub_server(models=(503, {"error": {"message": "overloaded"}}))
    client = client_for(stub, retry=RetryPolicy(n_retries=1, base_delay=0, max_delay=0))

    with pytest.raises(ProviderError) as raised:
        client.list_models()

    assert len(stub.paths) == 2
    assert raised.value.status_code == 503


def test_listing_models_leaves_the_sdks_own_retry_loop_off():
    """Two SDK tries inside each of jevper's is nine requests for one call, none of them visible.

    The SDK is switched off on a copy for every request jevper makes; this one used the caller's
    client unchanged, so its retries ran on a schedule the caller could neither see nor change.
    """

    class Official:
        def __init__(self) -> None:
            self.copies: list[dict[str, Any]] = []

        def with_options(self, **kwargs: Any) -> Official:
            self.copies.append(kwargs)
            return self

        def get(self, *, path: str, cast_to: Any) -> dict[str, Any]:
            return MODELS

    client = Official()

    models = SystemOneClient(client, model=MODEL).list_models()

    assert [model.name for model in models] == ["jev-1.13-free"]
    assert client.copies == [{"max_retries": 0}]


def test_the_sync_client_refuses_an_async_client_before_the_request(stub_server):
    """The un-awaited coroutine was handed to the model-list reader.

    It reported the service's shape as wrong for a body the service never sent, and the coroutine
    was abandoned with a warning nobody asked for.
    """
    stub = stub_server(models=(200, MODELS))
    client = SystemOneClient(async_openai_client(stub), model=MODEL)

    with pytest.raises(ClientCapabilityError, match="asynchronously"):
        client.list_models()

    assert stub.paths == []


def test_the_async_twin_refuses_a_blocking_client_before_the_request(stub_server):
    """The blocking ``get`` was called, the whole request spent on the event loop, and the parsed
    answer then thrown away in favour of the error — which the check on the callable can raise first."""
    stub = stub_server(models=(200, MODELS))
    client = AsyncSystemOneClient(openai_client(stub), model=MODEL)

    with pytest.raises(ClientCapabilityError, match="synchronously"):
        asyncio.run(client.alist_models())

    assert stub.paths == []


def test_the_async_twin_retries_and_reads_the_model_list(stub_server):
    """The same policy and the same schema on the async path."""
    stub = stub_server(models=(503, {"error": {"message": "overloaded"}}))
    client = AsyncSystemOneClient(
        async_openai_client(stub), model=MODEL, retry=RetryPolicy(n_retries=1, base_delay=0, max_delay=0)
    )

    with pytest.raises(ProviderError) as raised:
        asyncio.run(client.alist_models())

    assert raised.value.status_code == 503
    assert len(stub.paths) == 2


# --- what the constructor can refuse --------------------------------------------------------------
#
# Every refused option is consulted at both levels, because a client built with one of them makes
# the same call a per-call value would. These are the constructor's four; `method` is above.


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        ({"temperature": 0.0}, "temperature="),
        ({"reasoning": ReasoningConfig(effort="high")}, "reasoning="),
        ({"prompt_cache_key": "k"}, "prompt_cache_key="),
        ({"examples": (Example(state="s", answer="a"),)}, "examples="),
    ],
    ids=["temperature", "reasoning", "prompt_cache_key", "examples"],
)
def test_an_option_the_constructor_carries_is_refused_too(stub_server, kwargs, expected):
    """A client that cannot honour it fails every call it makes, so the constructor is the right place
    to be told — before the first request rather than on the first call."""
    stub = stub_server(systemone=answering({"n": {"type": "noul", "noul": 0.5}}))

    with pytest.raises(ClientCapabilityError, match=expected):
        client_for(stub, **kwargs).system_one(state=STATE, questions={"n": NOUL})

    assert stub.requests == []


def test_every_refused_option_is_named_in_one_error(stub_server):
    """One complaint, not one per attempt: the caller fixes the call in a single edit."""
    stub = stub_server(systemone=answering({}))
    client = client_for(
        stub, temperature=0.0, prompt_cache_key="k", reasoning=ReasoningConfig(effort="high")
    )

    with pytest.raises(ClientCapabilityError) as raised:
        client.system_one(state=STATE, questions={"n": NOUL})

    message = str(raised.value)
    assert "temperature=" in message and "reasoning=" in message and "prompt_cache_key=" in message


def test_the_async_twin_refuses_what_the_blocking_one_refuses(stub_server):
    """Same refusals, same order, before any request."""
    stub = stub_server(systemone=answering({}))
    client = AsyncSystemOneClient(
        async_openai_client(stub), model=MODEL, api="systemone", temperature=0.0
    )

    with pytest.raises(ClientCapabilityError, match="temperature="):
        asyncio.run(client.system_one(state=STATE, questions={"n": NOUL}))

    assert stub.paths == []


# --- answers that do not add up -------------------------------------------------------------------


@pytest.mark.parametrize(
    "body, expected",
    [
        ({"usage": {"input_tokens": 1, "output_tokens": 1}}, "no answer for it"),
        ({"answers": {}, "usage": {}}, "answered"),
        ({"answers": {"other": {"type": "noul", "noul": 0.5}}, "usage": {}}, "no answer for it"),
    ],
    ids=["no-answers-key", "empty-answers", "wrong-key"],
)
def test_a_response_that_does_not_answer_the_question_is_reported(stub_server, body, expected):
    """A partial or miskeyed response is a failure naming the question, not a silent subset.

    Every question in a batch came from one request, so an answer the caller cannot map back to a
    question is a failed call rather than a response with fewer entries — that is the price of the
    single request, and it is paid in an error rather than in a wrong answer.
    """
    stub = stub_server(systemone=lambda _: (200, body))

    with pytest.raises(MalformedAnswerError, match=expected):
        client_for(stub).system_one(state=STATE, questions={"n": NOUL})


@pytest.mark.parametrize(
    "body, expected",
    [
        ({"model": MODEL, "answers": {"n": {"type": "noul", "noul": 0.5}}}, (None, None)),
        (
            {"model": MODEL, "answers": {"n": {"type": "noul", "noul": 0.5}},
             "usage": {"input_tokens": 7}},
            (7, None),
        ),
        (
            {
                "model": MODEL,
                "answers": {"n": {"type": "noul", "noul": 0.5}},
                "usage": {"input_tokens": 7, "output_tokens": 3, "reasoning_tokens": 2},
            },
            (7, 3),
        ),
    ],
    ids=["absent", "partial", "extra-field"],
)
def test_usage_is_read_as_it_arrives(stub_server, body, expected):
    """A server that reports less than the full pair is not a failure, and one that reports more is
    not truncated: the two counts jevper publishes are read as they arrived, and the rest is left to
    the caller. The OpenAPI requires both, so these are gateway shapes rather than the service's."""
    stub = stub_server(systemone=lambda _: (200, body))

    response = client_for(stub).system_one(state=STATE, questions={"n": NOUL})

    assert (response.usage.input_tokens, response.usage.output_tokens) == expected


def test_a_batch_that_is_retried_counts_the_request_once(stub_server):
    """A 429 then a 200: one request is two attempts, and the usage of the successful one is counted
    once for the whole batch rather than once per question it answered."""
    calls: list[int] = []

    def handler(_: Any) -> tuple[int, Any]:
        calls.append(1)
        if len(calls) == 1:
            return (429, {"error": {"message": "slow down"}})
        return (
            200,
            systemone_body(
                {
                    "a": {"type": "noul", "noul": 0.6},
                    "b": {"type": "noul", "noul": 0.4},
                    "cc": {"type": "noul", "noul": 0.2},
                }
            ),
        )

    stub = stub_server(systemone=handler)
    client = client_for(stub, retry=RetryPolicy(n_retries=1, base_delay=0, max_delay=0))
    questions = {name: NOUL for name in ("a", "b", "cc")}

    response = client.system_one(state=STATE, questions=questions)

    assert response.usage.n_calls == 1
    assert response.usage.n_retries == 1
    assert response.usage.input_tokens == 40
    assert response.usage.output_tokens == 12
    for name in questions:
        attempts = [
            record
            for record in response.debug["llm_attempts"]
            if record["question_id"] == name
        ]
        assert len(attempts) == 2, name
