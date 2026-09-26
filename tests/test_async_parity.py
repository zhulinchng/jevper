"""The async facade against the same boundaries the sync one is held to.

``AsyncSystemOneClient`` has its own driver, its own awaitable call path and its own batch
collection, so every rule the blocking client keeps — jevper owns the retries, a caller's header
replaces the SDK's credential rather than joining it, a caller's own body field is the value on the
wire, a truncation is a truncation, a cancellation is not a provider failure — needs proving on that
path too, or it is a rule with one implementation and two claims.
"""

from __future__ import annotations

import asyncio
import json

import pytest
from fakes import (
    anthropic_client,
    async_anthropic_client,
    async_openai_client,
    async_openai_root_client,
    chat_body,
    messages_body,
    openai_client,
    responses_body,
)

from jevper import (
    AsyncSystemOneClient,
    Choice,
    Example,
    IncompleteAnswerError,
    InvalidQuestionError,
    LabelReadoutError,
    ModelMetadata,
    Noul,
    ProviderError,
    ReasoningConfig,
    RetryPolicy,
    SystemOneClient,
)

CRITERIA = {"billing": None, "technical": None, "sales": None}
STRUCTURED = json.dumps({"probabilities": {"billing": 0.7, "technical": 0.2, "sales": 0.1}})


def async_structured(stub, **kwargs) -> AsyncSystemOneClient:
    return AsyncSystemOneClient(async_openai_client(stub), model="stub", method="structured", **kwargs)


def run(coroutine):
    return asyncio.run(coroutine)


def test_the_async_client_makes_one_request_per_attempt(stub_server):
    """A client built with the SDK's own ``max_retries=2`` must not retry behind jevper's back."""
    from openai import AsyncOpenAI

    server = stub_server(chat=lambda _: (503, {"error": {"message": "unavailable"}}))
    client = AsyncOpenAI(base_url=server.base_url, api_key="test", max_retries=2, timeout=10)
    configured = AsyncSystemOneClient(
        client, model="stub", api="chat_completions", method="structured",
        retry=RetryPolicy(n_retries=0),
    )

    async def call():
        async with configured:
            await configured.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

    with pytest.raises(ProviderError):
        run(call())
    assert len(server.bodies("/chat/completions")) == 1


def test_the_async_anthropic_client_makes_one_request_per_attempt(stub_server):
    """The same rule on the other SDK, whose retry default is the same two extra attempts."""
    from anthropic import AsyncAnthropic

    server = stub_server(messages=lambda _: (529, {"error": {"message": "overloaded"}}))
    client = AsyncAnthropic(base_url=server.base_url, api_key="test", max_retries=2, timeout=10)
    configured = AsyncSystemOneClient(
        client, model="stub", api="messages", method="structured", retry=RetryPolicy(n_retries=0)
    )

    async def call():
        async with configured:
            await configured.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

    with pytest.raises(ProviderError):
        run(call())
    assert len(server.bodies("/messages")) == 1


def test_async_extra_headers_replace_the_anthropic_credential(stub_server):
    """``x-api-key`` under the caller's spelling is the SDK's spelling: one key, not two."""
    server = stub_server(messages=lambda _: (200, messages_body(text=STRUCTURED)))

    async def call():
        configured = AsyncSystemOneClient(
            async_anthropic_client(server),
            model="stub",
            api="messages",
            method="structured",
            extra_headers={"x-api-key": "replacement"},
        )
        async with configured:
            return await configured.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

    response = run(call())
    assert response.answers["q"].choice == "billing"
    keys = [name.lower() for name, _ in server.header_pairs[0]]
    assert keys.count("x-api-key") == 1
    sent = {name.lower(): value for name, value in server.header_pairs[0]}
    assert sent["x-api-key"] == "replacement"


def test_async_extra_headers_replace_the_openai_credential(stub_server):
    server = stub_server(responses=lambda _: (200, responses_body(text=STRUCTURED)))

    async def call():
        configured = async_structured(
            server, api="responses", extra_headers={"authorization": "Bearer replacement"}
        )
        async with configured:
            return await configured.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

    run(call())
    credentials = [value for name, value in server.header_pairs[0] if name.lower() == "authorization"]
    assert credentials == ["Bearer replacement"]


def test_a_child_cancellation_is_raised_not_counted_as_a_failure(stub_server):
    """A client that cancels one question's call is not that question's provider failing."""

    class CancellingCompletions:
        def create(self, **kwargs):
            raise asyncio.CancelledError

    class CancellingChat:
        completions = CancellingCompletions()

    class CancellingClient:
        chat = CancellingChat()

    configured = AsyncSystemOneClient(
        CancellingClient(), model="stub", api="chat_completions", method="structured"
    )

    async def call():
        async with configured:
            await configured.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

    with pytest.raises(asyncio.CancelledError):
        run(call())


def test_a_cancelled_call_leaves_no_partial_response(stub_server):
    """One question answering, one cancelled: the caller is told, not handed half an answer."""

    class HalfCompletions:
        def __init__(self):
            self.calls = 0

        async def create(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise asyncio.CancelledError
            return chat_body(content=STRUCTURED)

    class HalfChat:
        completions = HalfCompletions()

    class HalfClient:
        chat = HalfChat()

    configured = AsyncSystemOneClient(
        HalfClient(), model="stub", api="chat_completions", method="structured"
    )

    async def call():
        async with configured:
            await configured.system_one(
                state="s",
                questions={"first": Choice(criteria=CRITERIA), "second": Choice(criteria=CRITERIA)},
            )

    with pytest.raises(asyncio.CancelledError):
        run(call())


def test_cancelling_the_caller_still_cancels_the_questions(stub_server):
    """The interrupt a caller sends is the interrupt they get, with the request abandoned."""
    started = asyncio.Event()

    class SlowCompletions:
        async def create(self, **kwargs):
            started.set()
            await asyncio.sleep(30)
            return chat_body(content=STRUCTURED)

    class SlowChat:
        completions = SlowCompletions()

    class SlowClient:
        chat = SlowChat()

    configured = AsyncSystemOneClient(
        SlowClient(), model="stub", api="chat_completions", method="structured"
    )

    async def main():
        async with configured:
            task = asyncio.create_task(
                configured.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})
            )
            await asyncio.wait_for(started.wait(), timeout=5)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

    run(main())


def test_a_keyboard_interrupt_from_a_worker_is_raised():
    """An interrupt that reaches a question is not held until a sibling settles, nor dropped."""

    class InterruptingCompletions:
        def create(self, **kwargs):
            raise KeyboardInterrupt

    class InterruptingChat:
        completions = InterruptingCompletions()

    class InterruptingClient:
        chat = InterruptingChat()

    configured = SystemOneClient(
        InterruptingClient(), model="stub", api="chat_completions", method="structured"
    )
    with pytest.raises(KeyboardInterrupt):
        configured.system_one(
            state="s",
            questions={"first": Choice(criteria=CRITERIA), "second": Choice(criteria=CRITERIA)},
        )


def test_one_failing_question_still_raises_in_question_order(stub_server):
    """The ordinary-failure rule is unchanged: every question runs, the first failure is raised."""

    def script(body):
        text = json.dumps(body["messages"])
        if "first" in text:
            return 500, {"error": {"message": "first failed"}}
        return 200, chat_body(content=STRUCTURED)

    server = stub_server(chat=script)
    configured = SystemOneClient(
        openai_client(server),
        model="stub",
        api="chat_completions",
        method="structured",
        retry=RetryPolicy(n_retries=0),
    )
    with pytest.raises(ProviderError, match="first failed"):
        configured.system_one(
            state="first second",
            questions={"q1": Choice(criteria=CRITERIA), "q2": Choice(criteria=CRITERIA)},
        )
    assert len(server.bodies("/chat/completions")) == 2


def test_the_async_client_reads_an_incomplete_responses_body(stub_server):
    """The Responses statuses the sync client honours are honoured on the async path as well."""
    server = stub_server(
        responses=lambda _: (
            200,
            {
                "id": "resp_1",
                "object": "response",
                "status": "incomplete",
                "incomplete_details": {"reason": "max_output_tokens"},
                "output": [
                    {
                        "id": "msg_1",
                        "type": "message",
                        "role": "assistant",
                        "status": "incomplete",
                        "content": [{"type": "output_text", "text": STRUCTURED[:12], "annotations": []}],
                    }
                ],
            },
        )
    )

    async def call():
        configured = async_structured(server, api="responses", n_retry_malformed=0)
        async with configured:
            await configured.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

    with pytest.raises(IncompleteAnswerError, match="max_output_tokens"):
        run(call())


def test_the_public_client_owns_the_responses_logprob_switches(stub_server):
    """``extra_body={"logprobs": false}`` reaches the builder through the public client, not a test."""
    server = stub_server(responses=lambda _: (200, responses_body(text=STRUCTURED)))

    configured = SystemOneClient(
        openai_client(server),
        model="stub",
        api="responses",
        method="structured",
        extra_body={"logprobs": False},
    )
    configured.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

    body = server.bodies("/responses")[0]
    assert "top_logprobs" not in body
    assert "include" not in body
    assert body["logprobs"] is False


def test_a_caller_owned_top_logprobs_is_the_value_on_the_wire(stub_server):
    server = stub_server(responses=lambda _: (200, responses_body(text=STRUCTURED)))

    configured = SystemOneClient(
        openai_client(server),
        model="stub",
        api="responses",
        method="logprobs",
        top_logprobs=5,
        extra_body={"top_logprobs": 3},
        n_retry_malformed=0,
    )
    with pytest.raises(LabelReadoutError):
        # The logprob readout cannot read a text-only body; the request shape is what is under test.
        configured.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

    body = server.bodies("/responses")[0]
    assert body["top_logprobs"] == 3


def test_a_caller_owned_text_format_is_the_value_on_the_wire(stub_server):
    """``extra_body={"text": ...}`` wins, and jevper's schema moves into the prompt."""
    server = stub_server(responses=lambda _: (200, responses_body(text=STRUCTURED)))

    configured = SystemOneClient(
        openai_client(server),
        model="stub",
        api="responses",
        method="structured",
        extra_body={"text": {"format": {"type": "json_object"}}},
    )
    response = configured.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

    body = server.bodies("/responses")[0]
    assert body["text"] == {"format": {"type": "json_object"}}
    prompt = json.dumps(body["input"])
    assert "probabilities" in prompt
    assert response.answers["q"].choice == "billing"


def test_a_caller_owned_output_config_is_the_value_on_the_wire(stub_server):
    """The same rule on the Messages schema carrier."""
    server = stub_server(messages=lambda _: (200, messages_body(text=STRUCTURED)))

    configured = SystemOneClient(
        anthropic_client(server),
        model="stub",
        api="messages",
        method="structured",
        extra_body={"output_config": {"format": {"type": "json_object"}}},
    )
    response = configured.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

    body = server.bodies("/messages")[0]
    assert body["output_config"] == {"format": {"type": "json_object"}}
    # The schema reaches the model in the system prompt on this surface, not in the turns.
    assert "probabilities" in json.dumps(body["system"])
    assert response.answers["q"].choice == "billing"


def test_the_async_client_answers_a_batch_in_order(stub_server):
    """Three question types, one async call, every answer keyed where the caller put it."""
    from jevper import Noul, Score

    def script(body):
        text = json.dumps(body.get("input") or body.get("messages") or "")
        if "severity question" in text:
            return 200, responses_body(
                text=json.dumps({"probabilities": {"0": 0.1, "1": 0.2, "2": 0.7}})
            )
        if "reality question" in text:
            return 200, responses_body(text=json.dumps({"noul": 0.9}))
        return 200, responses_body(text=STRUCTURED)

    server = stub_server(responses=script)
    questions = {
        "intent": Choice(instructions="intent question", criteria=CRITERIA),
        "how_bad": Score(instructions="severity question", criteria=["mild", "moderate", "severe"]),
        "is_it_real": Noul(instructions="reality question"),
    }

    async def call():
        configured = async_structured(server, api="responses")
        async with configured:
            return await configured.system_one(state="s", questions=questions)

    response = run(call())
    assert list(response.answers) == ["intent", "how_bad", "is_it_real"]
    assert response.answers["intent"].choice == "billing"
    assert response.answers["how_bad"].score == pytest.approx(1.6)
    assert response.answers["is_it_real"].noul == pytest.approx(0.9)
    assert response.usage.n_calls == 3


# --- per-call overrides --------------------------------------------------------------------------
#
# ``docs/api.md`` promises "per-call values win over constructor defaults" without naming a facade.
# The async driver resolves those values in its own ``system_one``, so each is shown on the wire from
# the async call — the constructor's value on one call and the per-call value on the next, in one
# test, so a value that was taken from the constructor is not mistaken for one that was forwarded.
# Both calls share one event loop because the SDK's transport is bound to the loop it is first used
# on, so a second ``asyncio.run`` against the same client would fail for a reason the test is not
# about.

CHOICE_LOGS = [("A", -0.12), ("B", -2.47), ("C", -3.48)]


def chat_client(stub, *, model="stub", **kwargs) -> AsyncSystemOneClient:
    return AsyncSystemOneClient(
        async_openai_client(stub), model=model, api="chat_completions", method="structured", **kwargs
    )


def ask_pair(client: AsyncSystemOneClient, **overrides):
    """The constructor's call and the per-call one, awaited in one loop."""

    async def call():
        questions = {"q": Choice(criteria=CRITERIA)}
        first = await client.system_one(state="s", questions=questions)
        second = await client.system_one(state="s", questions=questions, **overrides)
        return first, second

    return run(call())


def test_a_per_call_model_reaches_the_async_wire_and_beats_the_constructor(stub_server):
    stub = stub_server(chat=lambda _: (200, chat_body(content=STRUCTURED)))
    client = chat_client(stub, model="constructor-model")

    ask_pair(client, model="per-call-model")

    assert [body["model"] for body in stub.bodies("/chat/completions")] == [
        "constructor-model",
        "per-call-model",
    ]


def test_a_per_call_method_reaches_the_async_wire_and_beats_the_constructor(stub_server):
    def script(body):
        if body.get("logprobs"):
            return 200, chat_body(content="A", logprobs=CHOICE_LOGS)
        return 200, chat_body(content=STRUCTURED)

    stub = stub_server(chat=script)
    client = chat_client(stub)  # the constructor says "structured"

    first, second = ask_pair(client, method="logprobs")

    assert first.answers["q"].choice == "billing"
    assert second.answers["q"].choice == "billing"
    bodies = stub.bodies("/chat/completions")
    assert bodies[0]["response_format"]["type"] == "json_schema"
    assert bodies[0].get("logprobs") is None
    assert bodies[1]["logprobs"] is True


def test_a_per_call_api_reaches_the_async_wire_and_beats_the_constructor(stub_server):
    stub = stub_server(
        chat=lambda _: (200, chat_body(content=STRUCTURED)),
        responses=lambda _: (200, responses_body(text=STRUCTURED)),
    )
    client = chat_client(stub)  # the constructor says "chat_completions"

    ask_pair(client, api="responses")

    assert stub.paths == ["/v1/chat/completions", "/v1/responses"]


def test_a_per_call_reasoning_reaches_the_async_wire_and_beats_the_constructor(stub_server):
    stub = stub_server(chat=lambda _: (200, chat_body(content=STRUCTURED)))
    client = chat_client(stub, reasoning=ReasoningConfig(mode="native", effort="high"))

    ask_pair(client, reasoning=ReasoningConfig(mode="native", effort="low"))

    assert [body.get("reasoning_effort") for body in stub.bodies("/chat/completions")] == [
        "high",
        "low",
    ]


def test_a_per_call_temperature_reaches_the_async_wire_and_beats_the_constructor(stub_server):
    stub = stub_server(chat=lambda _: (200, chat_body(content=STRUCTURED)))
    client = chat_client(stub, temperature=0.0)

    ask_pair(client, temperature=0.5)

    assert [body.get("temperature") for body in stub.bodies("/chat/completions")] == [0.0, 0.5]


def test_a_per_call_cache_key_reaches_the_async_wire_and_beats_the_constructor(stub_server):
    stub = stub_server(chat=lambda _: (200, chat_body(content=STRUCTURED)))
    client = chat_client(stub, prompt_cache_key="constructor-key")

    ask_pair(client, prompt_cache_key="per-call-key")

    assert [body.get("prompt_cache_key") for body in stub.bodies("/chat/completions")] == [
        "constructor-key",
        "per-call-key",
    ]


def test_per_call_examples_reach_the_async_wire_and_beat_the_constructor(stub_server):
    stub = stub_server(chat=lambda _: (200, chat_body(content=STRUCTURED)))
    client = chat_client(stub, examples=(Example(state="constructor example", answer="billing"),))

    ask_pair(client, examples=(Example(state="per-call example", answer="billing"),))

    prompts = [json.dumps(body["messages"]) for body in stub.bodies("/chat/completions")]
    assert "constructor example" in prompts[0] and "per-call example" not in prompts[0]
    assert "per-call example" in prompts[1] and "constructor example" not in prompts[1]


# --- the model list, on both sides of the documented base-URL split ------------------------------

MODELS = {
    "models": [
        {"name": "jev-1.13-free", "description": "General-purpose.", "release_date": "2026-09-15"}
    ]
}


def test_the_async_model_list_is_read_from_the_v1_client(stub_server):
    """``GET /v1/models``, the one route that lives under the version prefix."""
    stub = stub_server(models=(200, MODELS))
    client = AsyncSystemOneClient(async_openai_client(stub), model="jev-1.13-free")

    models = run(client.alist_models())

    assert [model.name for model in models] == ["jev-1.13-free"]
    assert isinstance(models[0], ModelMetadata)
    assert models[0].description == "General-purpose."
    assert stub.paths == ["/v1/models"]


def test_the_async_model_list_on_a_root_base_url_reports_the_missing_route(stub_server):
    """A root base URL reaches ``/api/decide`` but not the model list, which is ``/v1``'s.

    The root client asks ``/models``, not ``/v1/models``, so a server that serves the list under the
    version prefix answers a 404 there — and that has to arrive as the ``ProviderError`` every other
    provider failure arrives as, not as the SDK's own exception escaping a documented list method.
    """
    stub = stub_server(models=(404, {"error": {"message": "no route at /models"}}))
    client = AsyncSystemOneClient(async_openai_root_client(stub), model="jev-1.13-free")

    with pytest.raises(ProviderError) as raised:
        run(client.alist_models())

    assert raised.value.status_code == 404
    assert "404" in str(raised.value)
    assert stub.paths == ["/models"]


# --- the async lifecycle -------------------------------------------------------------------------


def test_the_async_context_manager_yields_the_client_and_keeps_the_callers_client_open(stub_server):
    """``async with`` hands back the configured client; ``aclose`` touches nothing the caller owns.

    The async facade holds no pool of its own, so ``aclose`` is safe to call any number of times and
    the caller's provider client — which jevper never closes — is the one that must still be open.
    """
    stub = stub_server(chat=lambda _: (200, chat_body(content=STRUCTURED)))
    provider = async_openai_client(stub)
    configured = AsyncSystemOneClient(
        provider, model="stub", api="chat_completions", method="structured"
    )

    async def call():
        async with configured as entered:
            assert entered is configured
            await configured.aclose()
            await configured.aclose()
        # Still usable after ``aclose``: nothing was torn down that a call needs.
        return await configured.system_one(
            state="s",
            questions={"a": Choice(criteria=CRITERIA), "b": Choice(criteria=CRITERIA)},
        )

    response = run(call())
    assert set(response.answers) == {"a", "b"}
    assert provider.is_closed() is False


def test_the_async_exit_closes_on_the_exception_path(stub_server):
    """A call that raises inside ``async with`` still runs ``__aexit__``'s release, not a bare return."""
    released: list[int] = []

    class Spy(AsyncSystemOneClient):
        async def aclose(self) -> None:
            released.append(1)
            await super().aclose()

    configured = Spy(
        async_openai_client(stub_server(chat=lambda _: (200, chat_body(content=STRUCTURED)))),
        model="stub",
        api="chat_completions",
        method="structured",
    )

    async def call():
        async with configured:
            raise RuntimeError("the caller's own failure")

    with pytest.raises(RuntimeError, match="the caller's own failure"):
        run(call())
    assert released == [1]


# --- the bare noul, on the async facade ----------------------------------------------------------

NOUL_ANSWER = {"model": "stub", "answers": {"n": {"type": "noul", "noul": 0.1249}}, "usage": {}}


def test_the_async_client_refuses_a_bare_noul_before_any_request(stub_server):
    stub = stub_server(systemone=lambda _: (200, NOUL_ANSWER))
    client = AsyncSystemOneClient(async_openai_client(stub), model="stub", api="systemone")

    with pytest.raises(InvalidQuestionError, match="must carry instructions or criteria"):
        run(client.system_one(state="s", questions={"n": Noul()}))

    assert stub.requests == []


def test_the_async_client_sends_a_bare_noul_when_the_caller_allows_it(stub_server):
    """``noul_requires_question=False`` on the async constructor reaches the same relaxed rule."""
    stub = stub_server(systemone=lambda _: (200, NOUL_ANSWER))
    client = AsyncSystemOneClient(
        async_openai_client(stub), model="stub", api="systemone", noul_requires_question=False
    )

    response = run(client.system_one(state="s", questions={"n": Noul()}))

    assert stub.requests[0]["questions"]["n"] == {"type": "noul"}
    assert response.answers["n"].noul == 0.1249


# --- ollaya's native options, on the async facade ------------------------------------------------

NATIVE_MODEL = "laya:en"
NATIVE_ANSWERS = {
    "team": {
        "type": "choice",
        "choice": "billing",
        "confidence": 0.9744,
        "probabilities": {"billing": 0.9872, "support": 0.0128},
        "laya": {"confidence": 0.901, "act_probability": 1.0},
    }
}


def native_body() -> dict:
    return {
        "model": NATIVE_MODEL,
        "answers": NATIVE_ANSWERS,
        "usage": {"input_tokens": 118, "output_tokens": 0},
    }


def native_client(stub) -> AsyncSystemOneClient:
    """The native route needs the server root: ``/api/decide`` is not under ``/v1``."""
    return AsyncSystemOneClient(
        async_openai_root_client(stub), model=NATIVE_MODEL, api="systemone", native=True
    )


def test_async_extras_reach_the_native_body_and_its_laya_object_is_read(stub_server):
    stub = stub_server(decide=lambda _: (200, native_body()))

    async def call():
        async with native_client(stub) as client:
            return await client.system_one(
                state="s",
                questions={
                    "team": Choice(
                        instructions="Which team?",
                        criteria={"billing": "Payments", "support": "Other"},
                    )
                },
                extras=["laya"],
            )

    response = run(call())

    assert stub.paths == ["/api/decide"]
    assert stub.requests[0]["extras"] == ["laya"]
    answer = response.answers["team"]
    assert answer.laya is not None
    assert answer.laya.confidence == 0.901
    assert answer.laya.act_probability == 1.0
    assert answer.confidence == 0.9744


def test_async_keep_alive_reaches_the_native_body(stub_server):
    stub = stub_server(decide=lambda _: (200, native_body()))

    async def call():
        async with native_client(stub) as client:
            return await client.system_one(
                state="s",
                questions={"team": Choice(criteria={"billing": None, "support": None})},
                keep_alive="10m",
            )

    run(call())

    assert stub.requests[0]["keep_alive"] == "10m"
