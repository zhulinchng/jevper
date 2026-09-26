"""Ollaya's native decision endpoint, reached with ``native=True``.

Ollaya serves two decision routes on one port: the TypeSafe wire format at ``/v1/systemone``, which
``tests/test_systemone_surface.py`` covers, and its own at ``/api/decide``, which answers in that same
format and adds a report on the request that produced it. What is pinned here:

* ``native=True`` posts to ``/api/decide``, and the TypeSafe route is not touched;
* the request body is the TypeSafe one, plus only the two options that route adds;
* the endpoint's report — routing, truncation, its three durations — is read, not invented;
* a model named directly reports no routing, and a router reports the checkpoint that answered;
* ``extras=["laya"]`` puts a ``laya`` object on every answer, and its ``null`` act probability is
  reported as the absence it is rather than as a number;
* a reported ``0`` duration is a measurement and is kept, where an absent one is ``None``;
* neither option is sent anywhere, and the answers are unchanged, when ``native`` is off;
* both are refused by name on the TypeSafe route rather than dropped in silence;
* ``response.model`` stays the name the caller asked for, so one field means one thing on every
  surface — the checkpoint a router chose is ``routing.model`` instead.

The measured figures these fields are checked against were taken from ollaya 0.7.0 on 2026-09-26;
see ``docs/local-servers.md``.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

import pytest
from fakes import async_openai_root_client, openai_client, openai_root_client

from jevper import (
    AsyncSystemOneClient,
    Choice,
    ClientCapabilityError,
    InvalidQuestionError,
    JevperError,
    MalformedAnswerError,
    NativeSystemOneResponse,
    Noul,
    Routing,
    Score,
    SystemOneClient,
)

MODEL = "laya:en"
STATE = "I was charged twice for my subscription this month and need the second one reversed."
CREATED_AT = "2026-09-26T08:34:40.513996156Z"
TOTAL_DURATION = 47_968_610
LOAD_DURATION = 0
EVAL_DURATION = 47_888_473

QUESTIONS = {
    "refund": Noul(instructions="Does the customer ask for a refund?"),
    "team": Choice(
        instructions="Which team handles this?",
        criteria={"billing": "Payments and refunds", "support": "Everything else"},
    ),
    "severity": Score(instructions="How bad is it?", criteria=["fine", "broken"]),
}

NOUL = {"type": "noul", "noul": 0.7932}
CHOICE = {
    "type": "choice",
    "choice": "billing",
    "confidence": 0.9744,
    "probabilities": {"billing": 0.9872, "support": 0.0128},
}
SCORE = {
    "type": "score",
    "score": 1.0,
    "confidence": 0.6,
    "legend": {"0": "fine", "1": "broken"},
    "probabilities": {"0": 0.4, "1": 0.6},
}
ANSWERS = {"refund": NOUL, "team": CHOICE, "severity": SCORE}

ROUTING = {
    "router": "laya:latest",
    "model": "laya:en",
    "route": "english",
    "reason": "English Latin text",
}

# The values ollaya 0.7.0 answered with on 2026-09-26, one request, F16 on one RTX 3080.
NATIVE_FIELDS = {
    "routing": None,
    "state_truncated": False,
    "done_reason": "decide",
    "created_at": CREATED_AT,
    "total_duration": TOTAL_DURATION,
    "load_duration": LOAD_DURATION,
    "eval_duration": EVAL_DURATION,
}


def native_body(answers: dict[str, Any] | None = None, **overrides: Any) -> dict[str, Any]:
    """A response shaped like ollaya's native one: the TypeSafe body plus its own report."""
    body: dict[str, Any] = {
        "model": "laya:en",
        "answers": ANSWERS if answers is None else answers,
        "usage": {"input_tokens": 118, "output_tokens": 0},
        **NATIVE_FIELDS,
    }
    body.update(overrides)
    return body


def answering(payload: dict[str, Any]):
    return lambda _: (200, payload)


def native_client(stub: Any, **kwargs: Any) -> SystemOneClient:
    """A native client on the server root, which is the only base URL that reaches ``/api/decide``."""
    return SystemOneClient(
        openai_root_client(stub), model=MODEL, api="systemone", native=True, **kwargs
    )


def typesafe_client(stub: Any, **kwargs: Any) -> SystemOneClient:
    return SystemOneClient(openai_client(stub), model=MODEL, api="systemone", **kwargs)


# --- the route ---------------------------------------------------------------------------------


def test_the_native_option_reaches_the_native_route(stub_server):
    """/api/decide, and nothing on the TypeSafe one. The two are one server and are not both reachable
    from one base URL, so which one was asked is the whole of what ``native`` decides."""
    stub = stub_server(decide=answering(native_body()))

    with native_client(stub) as client:
        client.system_one(state=STATE, questions=QUESTIONS)

    assert stub.paths == ["/api/decide"]


def test_the_request_is_the_typesafe_body_and_nothing_else(stub_server):
    """The same ``{state, model, questions}`` the other route takes. An option left unset is absent
    rather than null, so the service's own default stands."""
    stub = stub_server(decide=answering(native_body()))

    with native_client(stub) as client:
        client.system_one(state=STATE, questions=QUESTIONS)

    sent = stub.requests[0]
    assert set(sent) == {"state", "model", "questions"}
    assert set(sent["questions"]) == set(QUESTIONS)


def test_the_typesafe_route_is_untouched_without_the_option(stub_server):
    """The default is the route every other caller has always used, byte for byte."""
    stub = stub_server(
        systemone=lambda _: (200, {"model": MODEL, "answers": ANSWERS, "usage": {}}),
        decide=answering(native_body()),
    )

    with typesafe_client(stub) as client:
        response = client.system_one(state=STATE, questions=QUESTIONS)

    assert stub.paths == ["/v1/systemone"]
    assert type(response) is not NativeSystemOneResponse


# --- the endpoint's own report -------------------------------------------------------------------


def test_the_endpoints_report_is_read(stub_server):
    """Truncation, the reason the call finished, when it was produced and how long it took."""
    stub = stub_server(
        decide=answering(native_body(state_truncated=True, done_reason="decide", total_duration=1))
    )

    with native_client(stub) as client:
        response = client.system_one(state=STATE, questions=QUESTIONS)

    assert isinstance(response, NativeSystemOneResponse)
    assert response.state_truncated is True
    assert response.done_reason == "decide"
    assert response.created_at == CREATED_AT
    assert response.total_duration == 1
    assert response.load_duration == LOAD_DURATION
    assert response.eval_duration == EVAL_DURATION


def test_a_reported_zero_duration_is_a_measurement(stub_server):
    """``load_duration=0`` means the model was warm, which is not the same as unreported — so it is
    kept as the 0 the service sent, and only an absent field reads as ``None``."""
    stub = stub_server(decide=answering(native_body()))

    with native_client(stub) as client:
        response = client.system_one(state=STATE, questions=QUESTIONS)

    assert response.load_duration == 0
    assert response.eval_duration == EVAL_DURATION


def test_an_absent_duration_reads_as_none(stub_server):
    """A body that omits the report has not made the claim, and ``None`` is how jevper says that
    rather than a zero that would read as a measurement."""
    payload = {"model": MODEL, "answers": ANSWERS, "usage": {"input_tokens": 10}}
    stub = stub_server(decide=answering(payload))

    with native_client(stub) as client:
        response = client.system_one(state=STATE, questions=QUESTIONS)

    assert response.total_duration is None
    assert response.load_duration is None
    assert response.eval_duration is None
    assert response.routing is None
    assert response.state_truncated is False


def test_a_model_named_directly_reports_no_routing(stub_server):
    """``laya:en`` is a checkpoint, not a router, and says so by reporting nothing."""
    stub = stub_server(decide=answering(native_body()))

    with native_client(stub) as client:
        response = client.system_one(state=STATE, questions=QUESTIONS)

    assert response.routing is None


def test_a_router_reports_the_checkpoint_that_answered(stub_server):
    """``routing`` is the report, and ``model`` stays the name the caller asked for.

    On the wire ollaya's ``model`` is the checkpoint that answered, so reading that instead would
    make one field mean two things across jevper's surfaces. The checkpoint is here, in
    ``routing.model``, and the route is the stable key to branch on.
    """
    stub = stub_server(decide=answering(native_body(routing=ROUTING)))

    with native_client(stub) as client:
        response = client.system_one(state="I was charged twice.", questions=QUESTIONS, model="laya")

    assert response.model == "laya"
    assert isinstance(response.routing, Routing)
    assert response.routing.router == "laya:latest"
    assert response.routing.model == "laya:en"
    assert response.routing.route == "english"
    assert response.routing.reason == "English Latin text"


def test_a_routing_object_missing_its_reason_is_read(stub_server):
    """``reason`` is the router's prose, which the service documents as free to change or drop, so
    it defaults rather than failing a call whose numbers are all present."""
    stub = stub_server(decide=answering(native_body(routing={"router": "laya", "model": "laya:en",
                                                             "route": "english"})))

    with native_client(stub) as client:
        response = client.system_one(state=STATE, questions=QUESTIONS)

    assert response.routing is not None
    assert response.routing.reason == ""


# --- the two options that route adds ----------------------------------------------------------------


def test_extras_reaches_the_body_and_its_answers_come_back(stub_server):
    """``extras=["laya"]`` is the endpoint's own field, carrying the model's own confidence beside
    TypeSafe's. Namespaced so the two cannot be read as one number."""
    answers = {**ANSWERS, "team": {**CHOICE, "laya": {"confidence": 0.901, "act_probability": 1.0}}}
    stub = stub_server(decide=answering(native_body(answers)))

    with native_client(stub) as client:
        response = client.system_one(state=STATE, questions=QUESTIONS, extras=["laya"])

    assert stub.requests[0]["extras"] == ["laya"]
    answer = response.answers["team"]
    assert answer.laya is not None
    assert answer.laya.confidence == 0.901
    assert answer.laya.act_probability == 1.0
    # TypeSafe's own confidence beside it is untouched.
    assert answer.confidence == 0.9744


def test_a_null_act_probability_reads_as_none(stub_server):
    """A model with no act head reports null, and a number standing in for that would be invented."""
    answers = {
        **ANSWERS,
        "team": {**CHOICE, "laya": {"confidence": 0.5, "act_probability": None}},
    }
    stub = stub_server(decide=answering(native_body(answers)))

    with native_client(stub) as client:
        response = client.system_one(state=STATE, questions=QUESTIONS, extras=["laya"])

    assert response.answers["team"].laya.act_probability is None


def test_an_answer_without_the_object_leaves_it_none(stub_server):
    """The object is per answer and only arrives when asked for; ``None`` everywhere else is what
    every other surface sees, so nothing that reads a confidence changes meaning."""
    stub = stub_server(decide=answering(native_body()))

    with native_client(stub) as client:
        response = client.system_one(state=STATE, questions=QUESTIONS, extras=["laya"])

    assert all(answer.laya is None for answer in response.answers.values())


def test_a_typesafe_answer_carrying_a_laya_object_is_not_an_error(stub_server):
    """A gateway that sends the object on the TypeSafe route is adding a field, not breaking the
    contract; the answer is read as it arrived and the object is simply not read there."""
    answers = {**ANSWERS, "team": {**CHOICE, "laya": {"confidence": 0.4, "act_probability": 0.2}}}
    stub = stub_server(
        systemone=lambda _: (200, {"model": MODEL, "answers": answers, "usage": {}}),
    )

    with typesafe_client(stub) as client:
        response = client.system_one(state=STATE, questions=QUESTIONS)

    assert response.answers["team"].choice == "billing"


def test_keep_alive_reaches_the_body(stub_server):
    """Ollama's own lifecycle control, in that server's units, sent as given."""
    stub = stub_server(decide=answering(native_body()))

    with native_client(stub) as client:
        client.system_one(state=STATE, questions=QUESTIONS, keep_alive="10m")

    assert stub.requests[0]["keep_alive"] == "10m"


def test_a_zero_keep_alive_is_sent_and_not_treated_as_unset(stub_server):
    """``0`` unloads the model, which is a request like any other. A check written as "is it truthy"
    would drop it, and the model would stay loaded against the caller's word."""
    stub = stub_server(decide=answering(native_body()))

    with native_client(stub) as client:
        client.system_one(state=STATE, questions=QUESTIONS, keep_alive=0)

    assert stub.requests[0]["keep_alive"] == 0


def test_neither_option_is_sent_when_the_caller_asks_for_nothing(stub_server):
    """Absent, not null: the service reads the two the same way, and an empty array is its default."""
    stub = stub_server(decide=answering(native_body()))

    with native_client(stub) as client:
        client.system_one(state=STATE, questions=QUESTIONS, extras=(), keep_alive=None)

    assert not {"extras", "keep_alive"} & set(stub.requests[0])


def test_a_routing_report_that_cannot_be_read_is_a_jevper_error(stub_server):
    """The documented ``except JevperError`` has to catch it. A body reporting a router without the
    fields a router reports is a response that could not be read, which is this library's error and
    not a pydantic one escaping from inside assembly."""
    stub = stub_server(decide=answering(native_body(routing={"route": "english"})))

    client = native_client(stub)
    with pytest.raises(MalformedAnswerError, match="routing report could not be read"):
        client.system_one(state=STATE, questions=QUESTIONS)


@pytest.mark.parametrize("routing", ["english", ["english"], 3], ids=["string", "list", "number"])
def test_a_routing_field_that_is_not_an_object_is_unreadable(stub_server, routing):
    """``null`` is how the endpoint says a model named directly answered, so it is the one value that
    reads as no router. Anything else that is not an object is a report that could not be read, and
    reading it as "no router" would assert something the service never said."""
    stub = stub_server(decide=answering(native_body(routing=routing)))

    client = native_client(stub)
    with pytest.raises(MalformedAnswerError, match="must be an object or null"):
        client.system_one(state=STATE, questions=QUESTIONS)


@pytest.mark.parametrize(
    ("kwargs", "named"),
    [({"extras": ["laya"]}, "extras"), ({"keep_alive": "10m"}, "keep_alive")],
    ids=["extras", "keep_alive"],
)
def test_the_native_options_are_refused_on_the_typesafe_route(stub_server, kwargs, named):
    """The TypeSafe route has neither field and ignores what it does not know, so a caller who set
    one there is told by name — before a request is paid for."""
    stub = stub_server(systemone=lambda _: (200, {"model": MODEL, "answers": ANSWERS, "usage": {}}))

    client = typesafe_client(stub)
    with pytest.raises(ClientCapabilityError, match=named):
        client.system_one(state=STATE, questions=QUESTIONS, **kwargs)

    assert stub.requests == []


def test_the_refusal_names_the_option_that_reaches_them(stub_server):
    """The fix is the opposite of dropping them, so the message says it."""
    stub = stub_server(systemone=lambda _: (200, {"model": MODEL, "answers": ANSWERS, "usage": {}}))

    client = typesafe_client(stub)
    with pytest.raises(ClientCapabilityError, match="native=True"):
        client.system_one(state=STATE, questions=QUESTIONS, extras=["laya"])


# --- parity with the other client -----------------------------------------------------------------


def test_the_async_client_reaches_the_same_route_and_reads_the_same_report(stub_server):
    """The async twin answers identically, down to the endpoint's own report."""
    stub = stub_server(decide=answering(native_body(routing=ROUTING)))

    async def run() -> NativeSystemOneResponse:
        client = AsyncSystemOneClient(
            async_openai_root_client(stub), model=MODEL, api="systemone", native=True
        )
        async with client:
            return await client.system_one(state=STATE, questions=QUESTIONS)

    response = asyncio.run(run())

    assert stub.paths == ["/api/decide"]
    assert isinstance(response, NativeSystemOneResponse)
    assert response.routing is not None
    assert response.routing.route == "english"
    assert response.total_duration == TOTAL_DURATION
    assert set(response.answers) == set(QUESTIONS)


def test_usage_is_counted_once_for_the_shared_request(stub_server):
    """One request answered every question, so the tokens are counted once and not per question."""
    stub = stub_server(decide=answering(native_body()))

    with native_client(stub) as client:
        response = client.system_one(state=STATE, questions=QUESTIONS)

    assert len(stub.requests) == 1
    assert response.usage.n_calls == 1
    assert response.usage.input_tokens == 118


# --- against a real ollaya --------------------------------------------------------------------------
#
# Skipped unless OLLAYA_BASE_URL is set, e.g.
#   OLLAYA_BASE_URL=http://127.0.0.1:11435/v1 OLLAYA_MODEL=laya:en pytest tests/test_ollaya_native.py
OLLAYA_BASE_URL = os.environ.get("OLLAYA_BASE_URL")
OLLAYA_ROOT_URL = os.environ.get("OLLAYA_ROOT_URL") or (
    OLLAYA_BASE_URL[: -len("/v1")] if OLLAYA_BASE_URL and OLLAYA_BASE_URL.endswith("/v1") else None
)
"""Where ``/api/decide`` lives, which is the server root rather than the ``/v1`` prefix the TypeSafe
routes answer under. Derived from ``OLLAYA_BASE_URL`` unless given, because the two routes sit at
different depths and one base URL reaches only one of them."""


@pytest.mark.skipif(not OLLAYA_BASE_URL, reason="OLLAYA_BASE_URL is not set")
def test_ollaya_answers_three_questions_in_one_request_and_lists_its_models() -> None:
    """The TypeSafe route, and the parity gain with it: this server answers ``GET /v1/models`` in the
    shape ``list_models()`` reads, which a gateway answering that path OpenAI-style is reported as
    getting wrong."""
    from openai import OpenAI

    model = os.environ.get("OLLAYA_MODEL", "laya:en")
    with SystemOneClient(
        OpenAI(base_url=OLLAYA_BASE_URL, api_key="ollaya", max_retries=0),
        model=model,
        api="systemone",
    ) as client:
        response = client.system_one(state=STATE, questions=QUESTIONS)
        models = [entry.name for entry in client.list_models()]

    assert set(response.answers) == {"refund", "team", "severity"}
    assert response.usage.n_calls == 1
    assert model in models


@pytest.mark.skipif(not OLLAYA_ROOT_URL, reason="OLLAYA_ROOT_URL is not set")
def test_ollaya_answers_on_its_native_route_and_reports_the_request() -> None:
    """``/api/decide`` at the server root: the route the native report exists for, and the only base
    URL that reaches it."""
    from openai import OpenAI

    model = os.environ.get("OLLAYA_MODEL", "laya:en")
    with SystemOneClient(
        OpenAI(base_url=OLLAYA_ROOT_URL, api_key="ollaya", max_retries=0),
        model=model,
        api="systemone",
        native=True,
    ) as client:
        response = client.system_one(state=STATE, questions=QUESTIONS, extras=["laya"])

    assert isinstance(response, NativeSystemOneResponse)
    assert set(response.answers) == {"refund", "team", "severity"}
    assert response.usage.n_calls == 1
    assert isinstance(response.created_at, str) and response.created_at
    assert isinstance(response.done_reason, str) and response.done_reason
    assert isinstance(response.state_truncated, bool)
    assert all(
        isinstance(getattr(response, name), int | None)
        for name in ("total_duration", "load_duration", "eval_duration")
    )
    assert all(answer.laya is not None for answer in response.answers.values())


# --- what the report says when the endpoint says nothing, or the wrong thing ---------------------


def test_a_null_reason_reads_as_no_prose(stub_server):
    """``reason`` is the one routing field with a default, and ``null`` is how this endpoint spells an
    absent value everywhere else — the same spelling ``act_probability`` uses. Reading it as unreadable
    would throw away a call whose answers all parsed."""
    stub = stub_server(
        decide=answering(
            native_body(
                routing={"router": "laya:latest", "model": "laya:en", "route": "english",
                         "reason": None}
            )
        )
    )

    response = native_client(stub).system_one(state=STATE, questions=QUESTIONS)

    assert response.routing.reason == ""
    assert response.routing.route == "english"


@pytest.mark.parametrize(
    "laya",
    [{"confidence": 0.5, "act_probability": 7.5}, {"confidence": 0.5, "act_probability": float("nan")}],
    ids=["out-of-range", "nan"],
)
def test_an_act_probability_outside_zero_to_one_is_refused(stub_server, laya):
    """It is declared a probability and sits beside a ``confidence`` that is bounded, so an unbounded
    one would let a number that cannot be a probability through as one."""
    answers = {"refund": {**NOUL, "laya": laya}}
    stub = stub_server(decide=answering(native_body(answers)))

    client = native_client(stub)
    with pytest.raises(MalformedAnswerError, match="act_probability"):
        client.system_one(state=STATE, questions=QUESTIONS, extras=["laya"])


def test_a_null_native_scalar_reads_as_the_default_and_a_string_flag_is_not_true(stub_server):
    """``str(None)`` is the four-character string ``"None"`` and ``bool("false")`` is ``True``, so
    neither can be how these are read: ``created_at`` is documented as a timestamp to compare against
    the service's own log, and ``state_truncated`` is a flag a caller branches on."""
    stub = stub_server(
        decide=answering(
            native_body(created_at=None, done_reason=None, state_truncated="false")
        )
    )

    response = native_client(stub).system_one(state=STATE, questions=QUESTIONS)

    assert response.created_at == ""
    assert response.done_reason == "decide"
    assert response.state_truncated is False


# --- the two flags the documented ollaya call uses together --------------------------------------


def test_a_bare_noul_is_answered_on_the_native_route_when_the_rule_is_lifted(stub_server):
    """The combination ``docs/local-servers.md`` tells a reader to copy: ollaya reads the question id
    in place of a missing instruction, and the native endpoint is the route that carries the report."""
    stub = stub_server(decide=answering(native_body({"bare": {"type": "noul", "noul": 0.428}})))

    with native_client(stub, noul_requires_question=False) as client:
        response = client.system_one(state=STATE, questions={"bare": Noul()})

    assert response.answers["bare"].noul == 0.428
    assert stub.paths == ["/api/decide"]


def test_the_default_still_refuses_a_bare_noul_on_the_native_route(stub_server):
    """Also the pair: ``native=True`` does not lift the rule, only the caller's own flag does."""
    stub = stub_server(decide=answering(native_body()))

    client = native_client(stub)
    with pytest.raises(InvalidQuestionError, match="'bare'"):
        client.system_one(state=STATE, questions={"bare": Noul()})

    assert stub.requests == []


def test_native_with_an_auto_surface_is_refused_rather_than_ignored(stub_server):
    """``api="auto"`` never selects the systemone surface, so ``native`` could not take effect on any
    call this client makes. Left unsaid it posts to a prompt route instead, which a server serving
    only the decision routes answers with a 404 for a path the caller never named."""
    stub = stub_server()

    client = SystemOneClient(openai_root_client(stub), model=MODEL, native=True)
    with pytest.raises(ClientCapabilityError, match="api='auto' never selects"):
        client.system_one(state=STATE, questions=QUESTIONS)

    assert stub.requests == []


@pytest.mark.parametrize(
    "extras", ["laya", None, 5], ids=["a-bare-string", "None", "a-number"]
)
def test_a_malformed_extras_argument_is_refused_before_any_request(stub_server, extras):
    """A ``str`` is a ``Sequence[str]`` by Python's rules, so ``extras="laya"`` type-checks and then
    means the opposite of what it looks like — four names, one per character — and a non-iterable
    escaped as a raw ``TypeError`` from the public method. Both are named here instead."""
    stub = stub_server(decide=answering(native_body()))

    client = native_client(stub)
    with pytest.raises(JevperError, match="extras must be a sequence of names"):
        client.system_one(state=STATE, questions=QUESTIONS, extras=extras)

    assert stub.requests == []
