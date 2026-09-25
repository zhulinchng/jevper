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
    chat_body,
    messages_body,
    openai_client,
    responses_body,
)

from jevper import (
    AsyncSystemOneClient,
    Choice,
    IncompleteAnswerError,
    LabelReadoutError,
    ProviderError,
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
