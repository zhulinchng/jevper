"""Prompt caching: the key jevper sends, and the cached-token counts providers report.

The key is a routing hint, so what matters is that it is *stable* for the calls that can reuse a prefix
and different for the calls that cannot — and that a server which refuses it costs one request, not the
answer. The counts are what tells a caller caching is working, so an unreported count has to stay
distinguishable from a reported zero. The message order that makes a prefix reusable at all is pinned by
the state-rendering tests in ``test_client_behaviour.py``.
"""

from __future__ import annotations

import asyncio

import pytest
from fakes import async_openai_client, chat_body, openai_client, responses_body

from jevper import (
    AsyncSystemOneClient,
    Choice,
    Example,
    JevperError,
    ReasoningConfig,
    RetryPolicy,
    SystemOneClient,
)

CHOICE_LOGS = [("A", -0.12), ("B", -2.47), ("C", -3.48)]
CRITERIA = {"billing": None, "technical": None, "sales": None}
OTHER = {"billing": None, "refunds": None, "sales": None}


def client_for(stub, **kwargs):
    kwargs.setdefault("api", "chat_completions")
    return SystemOneClient(openai_client(stub), model="stub", **kwargs)


def ask(client, *, state="s", question=None, **kwargs):
    return client.system_one(
        state=state, questions={"q": question or Choice(criteria=CRITERIA)}, **kwargs
    )


def answer(body):
    return 200, chat_body(content="A", logprobs=CHOICE_LOGS, **body)


def keys_sent(stub):
    return [body.get("prompt_cache_key") for body in stub.bodies("/chat/completions")]


# -- the key jevper sends --------------------------------------------------------------


def test_a_derived_key_is_sent_and_does_not_depend_on_the_state(stub_server):
    """The whole point: one rubric, many states, one key — so the provider routes them together."""
    stub = stub_server(chat=lambda _: answer({}))
    client = client_for(stub)

    ask(client, state="the first record")
    ask(client, state="an entirely different record")

    first, second = keys_sent(stub)
    assert first == second
    assert first.startswith("jevper-")


def test_a_derived_key_changes_with_the_rubric(stub_server):
    stub = stub_server(chat=lambda _: answer({}))
    client = client_for(stub)

    ask(client, question=Choice(criteria=CRITERIA))
    ask(client, question=Choice(criteria=OTHER))

    first, second = keys_sent(stub)
    assert first != second


def test_a_derived_key_changes_with_the_demonstrations(stub_server):
    """Another demonstration set is another prefix, so it must not share the key of the first."""
    stub = stub_server(chat=lambda _: answer({}))
    client = client_for(stub)
    example = Example(state="an example", answer="billing")

    ask(client)
    ask(client, examples=[example])

    first, second = keys_sent(stub)
    assert first != second


def test_the_constructor_key_is_sent_verbatim(stub_server):
    stub = stub_server(chat=lambda _: answer({}))
    client = client_for(stub, prompt_cache_key="tenant-42")

    ask(client)

    assert keys_sent(stub) == ["tenant-42"]


def test_a_per_call_key_overrides_the_constructor_key(stub_server):
    stub = stub_server(chat=lambda _: answer({}))
    client = client_for(stub, prompt_cache_key="tenant-42")

    ask(client, prompt_cache_key="conversation-7")
    ask(client)

    assert keys_sent(stub) == ["conversation-7", "tenant-42"]


def test_the_key_is_sent_on_the_responses_surface(stub_server):
    stub = stub_server(responses=lambda _: (200, responses_body(text="A", logprobs=CHOICE_LOGS)))
    client = SystemOneClient(openai_client(stub), model="stub", api="responses")

    ask(client)

    assert stub.bodies("/responses")[0]["prompt_cache_key"].startswith("jevper-")


def test_both_passes_of_a_two_step_call_carry_the_same_key(stub_server):
    """The analysis and answer passes have different prefixes but belong to one conversation."""
    stub = stub_server(chat=lambda _: answer({}))
    client = client_for(stub, reasoning=ReasoningConfig(mode="two_step"))

    ask(client)

    sent = keys_sent(stub)
    assert len(sent) == 2
    assert sent[0] == sent[1]


def test_a_corrective_retry_carries_the_same_key(stub_server):
    calls: list[dict] = []

    def script(body):
        calls.append(body)
        if len(calls) == 1:
            return 200, chat_body(content="not a JSON object")
        return 200, chat_body(content='{"probabilities": {"billing": 1.0, "technical": 0.0, "sales": 0.0}}')

    stub = stub_server(chat=script)
    client = client_for(stub, method="structured")

    response = ask(client)

    assert response.usage.n_calls == 2
    assert response.debug["retry_reasons"]
    assert keys_sent(stub)[0] == keys_sent(stub)[1]


def test_extra_body_can_override_the_derived_key(stub_server):
    """``extra_body`` is merged over the typed parameters, so a caller keeps the last word."""
    stub = stub_server(chat=lambda _: answer({}))
    client = client_for(stub, extra_body={"prompt_cache_key": "hand-rolled"})

    ask(client)

    assert keys_sent(stub) == ["hand-rolled"]


@pytest.mark.parametrize("bad", [42, "", "   ", "x" * 257])
def test_a_bad_key_is_rejected_before_any_request(stub_server, bad):
    stub = stub_server(chat=lambda _: answer({}))
    client = client_for(stub)

    with pytest.raises(JevperError):
        ask(client, prompt_cache_key=bad)

    assert stub.requests == []


@pytest.mark.parametrize("bad", [42, "", "   ", "x" * 257])
def test_a_bad_constructor_key_is_rejected(stub_server, bad):
    stub = stub_server(chat=lambda _: answer({}))

    with pytest.raises(JevperError):
        client_for(stub, prompt_cache_key=bad)


def test_a_256_character_key_is_accepted(stub_server):
    stub = stub_server(chat=lambda _: answer({}))
    client = client_for(stub, prompt_cache_key="k" * 256)

    ask(client)

    assert keys_sent(stub) == ["k" * 256]


# -- a server that refuses the key -----------------------------------------------------


def refusing_key(evidence: str):
    calls: list[dict] = []

    def script(body):
        calls.append(body)
        if "prompt_cache_key" in body:
            return 400, {"error": {"message": evidence, "type": "invalid_request_error"}}
        return answer({})

    return calls, script


def test_a_refused_key_is_dropped_and_remembered(stub_server):
    calls, script = refusing_key("Unknown parameter: 'prompt_cache_key'.")
    stub = stub_server(chat=script)
    client = client_for(stub)

    first = ask(client)
    second = ask(client, state="another record")

    assert first.answers["q"].choice == "billing"
    assert first.debug["server_limits"]["cache_key"] is False
    assert "prompt_cache_key" in calls[0] and "prompt_cache_key" not in calls[1]
    # Remembered: the second call does not pay for the same discovery again.
    assert len(calls) == 3
    assert "prompt_cache_key" not in calls[2]
    assert second.debug["server_limits"]["cache_key"] is False


def test_a_stricter_server_cap_also_drops_the_key(stub_server):
    """A value complaint about an optional routing hint is not worth failing the answer over."""
    calls, script = refusing_key("prompt_cache_key must be at most 64 characters")
    stub = stub_server(chat=script)
    client = client_for(stub, prompt_cache_key="k" * 80)

    response = ask(client)

    assert response.answers["q"].choice == "billing"
    assert "prompt_cache_key" in calls[0] and "prompt_cache_key" not in calls[1]


def test_a_refusal_on_one_surface_does_not_silence_the_other(stub_server):
    """Limits are per surface, so a chat-only refusal must not cost the key on responses."""
    stub = stub_server(
        chat=lambda body: (
            (400, {"error": {"message": "prompt_cache_key is not supported"}})
            if "prompt_cache_key" in body
            else answer({})
        ),
        responses=lambda _: (200, responses_body(text="A", logprobs=CHOICE_LOGS)),
    )
    chat = client_for(stub)
    responses = SystemOneClient(openai_client(stub), model="stub", api="responses")

    assert ask(chat).answers["q"].choice == "billing"
    assert ask(responses).answers["q"].choice == "billing"

    assert stub.bodies("/chat/completions")[1].get("prompt_cache_key") is None
    assert stub.bodies("/responses")[0]["prompt_cache_key"].startswith("jevper-")


def test_a_schema_downgrade_keeps_the_key(stub_server):
    """The ladder drops one field at a time; the key is not collateral."""
    calls: list[dict] = []

    def script(body):
        calls.append(body)
        if "response_format" in body:
            return 400, {"error": {"message": "response_format is not supported"}}
        return 200, chat_body(
            content='{"probabilities": {"billing": 1.0, "technical": 0.0, "sales": 0.0}}'
        )

    stub = stub_server(chat=script)
    client = client_for(stub, method="structured")

    response = ask(client)

    assert response.answers["q"].choice == "billing"
    assert calls[0]["prompt_cache_key"] == calls[1]["prompt_cache_key"]
    assert calls[1]["response_format"] == {"type": "json_object"}


# -- what the provider reports back ----------------------------------------------------


def test_cached_tokens_are_reported_from_chat_usage(stub_server):
    stub = stub_server(chat=lambda _: answer({"cached_tokens": 512}))
    client = client_for(stub)

    response = ask(client)

    assert response.usage.cached_tokens == 512
    assert response.usage.input_tokens == 10


def test_cached_tokens_are_reported_from_responses_usage(stub_server):
    stub = stub_server(
        responses=lambda _: (200, responses_body(text="A", logprobs=CHOICE_LOGS, cached_tokens=384))
    )
    client = SystemOneClient(openai_client(stub), model="stub", api="responses")

    response = ask(client)

    assert response.usage.cached_tokens == 384


def test_a_reported_zero_is_not_the_same_as_silence(stub_server):
    """A cold cache reports ``0``; a provider that does not implement caching says nothing."""
    silent = stub_server(chat=lambda _: answer({}))
    reporting = stub_server(chat=lambda _: answer({"cached_tokens": 0}))

    assert ask(client_for(silent)).usage.cached_tokens is None
    assert ask(client_for(reporting)).usage.cached_tokens == 0


def test_a_null_token_details_object_is_read_as_silence(stub_server):
    """vLLM has shipped ``prompt_tokens_details: null``; it must not become a crash or a zero."""

    def script(_):
        body = chat_body(content="A", logprobs=CHOICE_LOGS)
        body["usage"]["prompt_tokens_details"] = None
        return 200, body

    stub = stub_server(chat=script)

    assert ask(client_for(stub)).usage.cached_tokens is None


def test_cached_tokens_aggregate_over_questions(stub_server):
    stub = stub_server(chat=lambda _: answer({"cached_tokens": 256}))
    client = client_for(stub)

    response = client.system_one(
        state="s",
        questions={"one": Choice(criteria=CRITERIA), "two": Choice(criteria=CRITERIA)},
    )

    assert response.usage.n_calls == 2
    assert response.usage.cached_tokens == 512


def test_one_silent_call_makes_the_total_unknown(stub_server):
    """The existing rule for token counts: a total is only reported if every call reported it."""
    calls: list[int] = []

    def script(_):
        calls.append(1)
        return answer({"cached_tokens": 128} if len(calls) == 1 else {})

    stub = stub_server(chat=script)
    client = client_for(stub)

    response = client.system_one(
        state="s",
        questions={"one": Choice(criteria=CRITERIA), "two": Choice(criteria=CRITERIA)},
    )

    assert response.usage.cached_tokens is None


def test_cached_tokens_survive_a_retry_and_are_not_double_counted(stub_server):
    """Only the calls that produced a result are counted: a failed attempt reports nothing."""
    calls: list[int] = []

    def script(_):
        calls.append(1)
        if len(calls) == 1:
            return 429, {"error": {"message": "slow down", "type": "rate_limit_error"}}
        return answer({"cached_tokens": 640})

    stub = stub_server(chat=script)
    client = client_for(stub, retry=RetryPolicy(n_retries=1, base_delay=0.0))

    response = ask(client)

    assert response.usage.n_calls == 1
    assert response.usage.cached_tokens == 640


@pytest.mark.parametrize(
    ("value", "expected"),
    [("512", 512), (512.7, 512), ("many", None), ({"nested": 1}, None), ([512], None)],
)
def test_a_count_nobody_can_read_is_unreported(stub_server, value, expected):
    """A token count is whatever the provider put in its JSON; garbage must not cost the answer.

    A numeric string and a float are read; a word, an object and a list are not. The openai SDK's own
    model coerces a boolean to ``1`` before jevper sees it, so that one case cannot be told apart here.
    """

    def script(_):
        body = chat_body(content="A", logprobs=CHOICE_LOGS)
        body["usage"]["prompt_tokens_details"] = {"cached_tokens": value}
        return 200, body

    stub = stub_server(chat=script)

    response = ask(client_for(stub))

    assert response.answers["q"].choice == "billing"
    assert response.usage.cached_tokens == expected


def test_a_prompt_tokens_details_that_is_not_an_object_is_silence(stub_server):
    """``"nope"`` and a list have no ``cached_tokens`` to read, and neither may raise."""

    def script(_):
        body = chat_body(content="A", logprobs=CHOICE_LOGS)
        body["usage"]["prompt_tokens_details"] = "nope"
        return 200, body

    stub = stub_server(chat=script)

    assert ask(client_for(stub)).usage.cached_tokens is None


# -- async parity ----------------------------------------------------------------------


def test_the_async_client_sends_the_same_key_and_reads_the_same_count(stub_server):
    stub = stub_server(chat=lambda _: answer({"cached_tokens": 128}))
    client = AsyncSystemOneClient(async_openai_client(stub), model="stub", api="chat_completions")

    response = asyncio.run(client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)}))

    assert response.answers["q"].choice == "billing"
    assert response.usage.cached_tokens == 128
    assert keys_sent(stub)[0].startswith("jevper-")


def test_the_async_client_sends_a_caller_key(stub_server):
    stub = stub_server(chat=lambda _: answer({}))
    client = AsyncSystemOneClient(
        async_openai_client(stub), model="stub", api="chat_completions", prompt_cache_key="tenant-9"
    )

    asyncio.run(client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)}))

    assert keys_sent(stub) == ["tenant-9"]


def test_the_key_differs_per_method_and_is_shared_by_both_reasoning_passes(stub_server):
    """A method is a different prefix: another system prompt, another answer shape, another key.

    The two passes of a two-step call are the deliberate exception. They share everything except the
    system prompt, and the point of the key is to route the answer pass at the analysis pass's cache,
    so both passes of one call key alike.
    """
    structured_answer = '{"probabilities": {"billing": 0.8, "technical": 0.1, "sales": 0.1}}'

    def script(body):
        if body.get("response_format") is not None:
            return 200, chat_body(content=structured_answer)
        return 200, chat_body(content="A", logprobs=CHOICE_LOGS)

    stub = stub_server(chat=script)
    question = Choice(criteria=CRITERIA)
    client_for(stub, method="logprobs").system_one(state="s", questions={"q": question})
    client_for(stub, method="structured").system_one(state="s", questions={"q": question})
    client_for(stub, method="structured", reasoning=ReasoningConfig(mode="two_step")).system_one(
        state="s", questions={"q": question}
    )

    keys = keys_sent(stub)
    assert all(isinstance(key, str) for key in keys)
    assert keys[0] != keys[1]  # the same question, the same state, another method
    assert keys[2] == keys[3]  # analysis pass, answer pass — the routing the design wants
    # Two-step is a way of *asking*, not another prefix: it keys like the method it answers with, so
    # the analysis pass and a plain call with the same rubric still land in one bucket.
    assert keys[2] == keys[1]
