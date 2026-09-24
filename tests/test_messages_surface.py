"""The Messages surface: the Anthropic-compatible API that every server in the fleet now exposes.

Three things about it change what jevper has to do. ``system`` is a top-level field rather than a
turn, so the prompt has to be split apart and put back together. ``max_tokens`` has no server-side
default anywhere that implements it, so a request without one is refused. And there are no logprobs
in the API at all — not withheld by some servers, absent from the protocol — so a label readout there
can never succeed and is refused before a request is spent discovering that.

What is pinned here is that the prompt still arrives whole, that the budget is always sent, that
thinking blocks and cache counts come back in the same shapes the other two surfaces use, and that a
server which refuses the ``thinking`` field costs one request rather than the answer.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from fakes import (
    StubServer,
    anthropic_client,
    async_anthropic_client,
    chat_body,
    messages_body,
    openai_client,
)

from jevper import (
    AsyncSystemOneClient,
    Choice,
    JevperError,
    MalformedAnswerError,
    ProviderError,
    ReasoningConfig,
    SystemOneClient,
    UnsupportedMethodError,
    reasoning_text,
)
from jevper.transport import DEFAULT_MAX_TOKENS, SURFACES

CRITERIA = {"billing": None, "technical": None, "sales": None}
JSON_ANSWER = '{"probabilities": {"billing": 0.8, "technical": 0.1, "sales": 0.1}}'


def client_for(stub: StubServer, **kwargs: object) -> SystemOneClient:
    kwargs.setdefault("api", "messages")
    kwargs.setdefault("method", "structured")
    return SystemOneClient(anthropic_client(stub), model="stub", **kwargs)


def ask(client: SystemOneClient, *, state: object = "charged twice", **kwargs: object):
    return client.system_one(state=state, questions={"q": Choice(criteria=CRITERIA)}, **kwargs)


def answer(**body: object):
    return lambda _: (200, messages_body(text=JSON_ANSWER, **body))


def body_sent(stub: StubServer) -> dict:
    assert len(stub.requests) == 1, stub.requests
    return stub.requests[0]


# -- the prompt arrives whole -----------------------------------------------------------


def test_the_system_prompt_is_a_top_level_field_and_not_a_turn(stub_server):
    """Anthropic has no ``system`` role inside ``messages``; a server that sees one may 400."""
    stub = stub_server(messages=answer())
    ask(client_for(stub))

    body = body_sent(stub)
    assert isinstance(body["system"], str)
    assert body["system"].strip()
    # The question block and the state, and no system turn among them.
    assert [message["role"] for message in body["messages"]] == ["user", "user"]


def test_the_question_block_comes_before_the_state(stub_server):
    """The order the cache depends on, carried through the split unchanged."""
    stub = stub_server(messages=answer())
    ask(client_for(stub), state="charged twice for one order")

    question, state = body_sent(stub)["messages"]
    assert question["role"] == "user" and state["role"] == "user"
    assert "Options:" in question["content"]
    assert "charged twice for one order" in state["content"]


def test_a_state_instruction_turn_joins_the_system_prompt(stub_server):
    """``hoist_instructions`` already folds it in; the split must not undo that or drop it."""
    stub = stub_server(messages=answer())
    ask(
        client_for(stub),
        state=[
            {"role": "system", "content": "Answer as a support triage assistant."},
            {"role": "user", "content": "I was charged twice."},
        ],
    )

    body = body_sent(stub)
    # The caller's instruction is in the system prompt, ahead of the schema jevper appends to it.
    assert "Answer as a support triage assistant." in body["system"]
    assert [message["role"] for message in body["messages"]] == ["user", "user"]


def test_max_tokens_is_always_sent(stub_server):
    """Every implementation of this API refuses a request without one."""
    stub = stub_server(messages=answer())
    ask(client_for(stub))

    assert body_sent(stub)["max_tokens"] == DEFAULT_MAX_TOKENS


def test_max_tokens_from_extra_body_wins_and_is_not_sent_twice(stub_server):
    stub = stub_server(messages=answer())
    ask(client_for(stub, extra_body={"max_tokens": 64}))

    body = body_sent(stub)
    assert body["max_tokens"] == 64
    assert "max_tokens" not in body.get("extra_body", {})


def test_the_schema_travels_in_the_system_prompt(stub_server):
    """This API has no schema field, so the answer's shape has to be stated in the prompt itself."""
    stub = stub_server(messages=answer())
    ask(client_for(stub))

    system = body_sent(stub)["system"]
    assert '"probabilities"' in system
    assert '"billing"' in system and '"technical"' in system and '"sales"' in system


def test_temperature_and_extra_body_travel(stub_server):
    stub = stub_server(messages=answer())
    ask(client_for(stub, extra_body={"top_k": 20}), temperature=0.6)

    body = body_sent(stub)
    assert body["temperature"] == 0.6
    # The SDK merges ``extra_body`` into the JSON body rather than nesting it.
    assert body["top_k"] == 20


def test_the_cache_key_is_never_sent(stub_server):
    """The Messages API has no such field; sending one would be a 400 on a strict server."""
    stub = stub_server(messages=answer())
    ask(client_for(stub))

    body = body_sent(stub)
    assert "prompt_cache_key" not in body
    assert "prompt_cache_key" not in body.get("extra_body", {})


# -- the answer is read ----------------------------------------------------------------


def test_the_answer_is_read_from_the_text_blocks(stub_server):
    stub = stub_server(messages=answer())
    response = ask(client_for(stub))

    assert response.answers["q"].choice == "billing"
    assert response.debug["api"] == "messages"
    assert response.debug["method"] == "structured"


def test_thinking_blocks_become_reasoning_parts_with_their_signature(stub_server):
    stub = stub_server(messages=answer(thinking="Two charges, one subscription.", signature="sig-1"))
    response = ask(client_for(stub))

    assert reasoning_text(response.reasoning) == "Two charges, one subscription."
    assert response.reasoning[0].signature == "sig-1"


def test_an_empty_thinking_block_is_not_a_trace(stub_server):
    """``display: "omitted"`` answers with an empty block; that is not something to report."""
    stub = stub_server(messages=answer(thinking=""))
    response = ask(client_for(stub))

    assert response.reasoning == ()


def test_the_cache_and_thinking_counts_are_read(stub_server):
    stub = stub_server(messages=answer(cached_tokens=1989, thinking_tokens=42))
    response = ask(client_for(stub))

    assert response.usage.cached_tokens == 1989
    assert response.usage.reasoning_tokens == 42


def test_an_unreported_count_stays_none(stub_server):
    """A server without prompt caching says nothing; that is not a reported zero."""
    stub = stub_server(messages=answer())
    response = ask(client_for(stub))

    assert response.usage.cached_tokens is None
    assert response.usage.reasoning_tokens is None


def test_a_reasoning_only_answer_says_so(stub_server):
    """vLLM and SGLang with thinking on answer with a thinking block and no text block at all."""
    stub = stub_server(messages=lambda _: (200, messages_body(text="", thinking="Two charges.")))

    with pytest.raises(MalformedAnswerError, match="reasoning only"):
        ask(client_for(stub))


def test_an_error_carried_in_a_200_is_raised(stub_server):
    normalize = SURFACES["messages"][1]
    with pytest.raises(JevperError, match="overloaded"):
        normalize({"type": "error", "error": {"type": "overloaded_error", "message": "overloaded"}}, {})


# -- no logprobs exist here ------------------------------------------------------------


def test_auto_answers_in_json_and_never_asks_for_logprobs(stub_server):
    stub = stub_server(messages=answer())
    response = ask(client_for(stub, method="auto"))

    assert response.debug["method"] == "structured"
    body = body_sent(stub)
    assert "logprobs" not in body and "top_logprobs" not in body


def test_an_explicit_logprobs_method_is_refused_before_any_request(stub_server):
    stub = stub_server(messages=answer())
    with pytest.raises(UnsupportedMethodError, match="no logprobs"):
        ask(client_for(stub, method="logprobs"))

    assert stub.requests == []


def test_grammar_is_refused_for_this_surface_too(stub_server):
    """Grammar needs a Chat Completions ``grammar`` field, which this surface has no place for."""
    stub = stub_server(messages=answer())
    with pytest.raises(UnsupportedMethodError, match="grammar requires"):
        ask(client_for(stub, method="grammar"))

    assert stub.requests == []


def test_a_client_without_the_messages_route_says_so(stub_server):
    stub = stub_server(chat=answer())
    with pytest.raises(JevperError, match="messages.create"):
        ask(SystemOneClient(openai_client(stub), model="stub", api="messages", method="structured"))


# -- surface switching -----------------------------------------------------------------


class HybridClient:
    """A client that answers both the Messages and the Chat Completions routes."""

    def __init__(self, stub: StubServer) -> None:
        self.messages = anthropic_client(stub).messages
        self.chat = openai_client(stub).chat
        self.responses = None


def test_auto_prefers_a_surface_that_can_carry_logprobs(stub_server):
    """``messages`` is the last choice for ``auto``: no label readout can succeed on it."""
    stub = stub_server(
        chat=lambda _: (
            200,
            chat_body(content="A", logprobs=[("A", -0.1), ("B", -2.3), ("C", -3.1)]),
        ),
        messages=answer(),
    )
    client = SystemOneClient(HybridClient(stub), model="stub", api="auto", method="auto")

    response = ask(client)

    assert response.answers["q"].choice == "billing"
    assert response.debug["api"] == "chat_completions"
    assert stub.paths == ["/v1/chat/completions"]


def test_a_client_with_no_other_surface_reports_the_missing_route(stub_server):
    stub = stub_server(messages=None)
    client = client_for(stub, api="messages")

    with pytest.raises(JevperError):
        ask(client)


def test_a_thinking_budget_is_sent_when_one_is_asked_for(stub_server):
    stub = stub_server(messages=answer())
    ask(client_for(stub, reasoning=ReasoningConfig(mode="native", budget_tokens=2048)))

    assert body_sent(stub)["thinking"] == {"type": "enabled", "budget_tokens": 2048}


def test_no_thinking_field_without_a_budget(stub_server):
    """``effort`` is not translated into a budget: the mapping is the caller's, not jevper's."""
    stub = stub_server(messages=answer())
    ask(client_for(stub, reasoning=ReasoningConfig(mode="native", effort="high")))

    assert "thinking" not in body_sent(stub)


def test_a_budget_below_one_is_refused_locally():
    with pytest.raises(ValueError, match="positive"):
        ReasoningConfig(mode="native", budget_tokens=0)


def test_a_refused_thinking_field_is_dropped_and_the_call_re_asked(stub_server):
    """vLLM's Messages protocol has no ``thinking`` field, so the ladder drops it and re-asks."""
    calls: list[dict] = []

    def script(body: dict):
        calls.append(body)
        if "thinking" in body:
            return 400, {"error": {"type": "invalid_request_error", "message": "unknown field: thinking"}}
        return 200, messages_body(text=JSON_ANSWER)

    stub = stub_server(messages=script)
    response = ask(client_for(stub, reasoning=ReasoningConfig(mode="native", budget_tokens=2048)))

    assert response.answers["q"].choice == "billing"
    assert "thinking" in calls[0] and "thinking" not in calls[1]
    assert response.debug["server_limits"]["thinking"] is False


# -- async parity ----------------------------------------------------------------------


def test_the_async_client_behaves_identically(stub_server):
    stub = stub_server(messages=answer(thinking="because", cached_tokens=7))
    client = AsyncSystemOneClient(
        async_anthropic_client(stub), model="stub", api="messages", method="structured"
    )

    response = asyncio.run(
        client.system_one(state="charged twice", questions={"q": Choice(criteria=CRITERIA)})
    )

    assert response.answers["q"].choice == "billing"
    assert response.usage.cached_tokens == 7
    assert reasoning_text(response.reasoning) == "because"
    assert [message["role"] for message in body_sent(stub)["messages"]] == ["user", "user"]


def test_the_async_client_refuses_a_label_readout_too(stub_server):
    stub = stub_server(messages=answer())
    client = AsyncSystemOneClient(
        async_anthropic_client(stub), model="stub", api="messages", method="logprobs"
    )

    with pytest.raises(UnsupportedMethodError):
        asyncio.run(client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)}))
    assert stub.requests == []


def test_the_messages_surface_is_registered_once(stub_server):
    """The registry entry is what ``api="auto"`` and the downgrade ladder look things up in."""
    builder, normalizer, path = SURFACES["messages"]
    assert path == "messages"
    assert callable(builder) and callable(normalizer)


def test_a_mapping_shaped_response_is_read_too():
    """``_get`` reads mappings as well as objects, so a duck-typed client works unchanged."""
    normalize = SURFACES["messages"][1]
    result = normalize(
        SimpleNamespace(
            content=[{"type": "text", "text": JSON_ANSWER}],
            stop_reason="max_tokens",
            usage={"input_tokens": 5, "output_tokens": 6},
        ),
        {},
    )

    assert result.text == JSON_ANSWER
    assert result.stop == "max_tokens"
    assert result.input_tokens == 5 and result.output_tokens == 6
    assert result.token_logprobs == ()


# -- the budget, the stop reason, and a refusal -----------------------------------------


def test_the_default_budget_grows_by_the_thinking_budget(stub_server):
    """Anthropic requires the budget to be strictly below ``max_tokens`` and answers 400 otherwise.

    jevper owns the default, so the pair the docs recommend — the 1024 they call the floor — would be
    refused by jevper's own 1024. The answer keeps the whole default and the thinking is paid for out
    of the extra.
    """
    stub = stub_server(messages=answer())
    client = client_for(stub, reasoning=ReasoningConfig(mode="native", budget_tokens=2048))

    ask(client)

    sent = body_sent(stub)
    assert sent["thinking"] == {"type": "enabled", "budget_tokens": 2048}
    assert sent["max_tokens"] == DEFAULT_MAX_TOKENS + 2048
    assert sent["max_tokens"] > sent["thinking"]["budget_tokens"]


def test_the_smallest_budget_the_docs_allow_still_leaves_an_answer_budget(stub_server):
    stub = stub_server(messages=answer())
    client = client_for(stub, reasoning=ReasoningConfig(mode="native", budget_tokens=1024))

    ask(client)

    sent = body_sent(stub)
    assert sent["max_tokens"] == DEFAULT_MAX_TOKENS + 1024


def test_a_caller_max_tokens_still_wins_over_the_grown_default(stub_server):
    stub = stub_server(messages=answer())
    client = client_for(
        stub, reasoning=ReasoningConfig(mode="native", budget_tokens=2048), extra_body={"max_tokens": 4096}
    )

    ask(client)

    assert body_sent(stub)["max_tokens"] == 4096


def test_the_budget_does_not_grow_once_the_server_refused_the_thinking_field(stub_server):
    """The growth exists to make room for a thinking block the server is going to produce."""
    def script(body):
        if "thinking" in body:
            return 400, {"error": {"message": "thinking: Extra inputs are not permitted"}}
        return 200, messages_body(text=JSON_ANSWER)

    stub = stub_server(messages=script)
    client = client_for(stub, reasoning=ReasoningConfig(mode="native", budget_tokens=2048))

    response = ask(client)

    assert response.answers["q"].choice == "billing"
    assert response.debug["server_limits"]["thinking"] is False
    assert [body["max_tokens"] for body in stub.requests] == [
        DEFAULT_MAX_TOKENS + 2048,
        DEFAULT_MAX_TOKENS,
    ]


def test_a_thinking_budget_refused_for_its_value_is_not_dropped(stub_server):
    """A budget the server will not take is not a missing capability: the field exists.

    Dropping it would answer the question with the caller's reasoning quietly switched off — and
    remember that as this server's limit, for a configuration the caller never repeated.
    """
    def script(body):
        if "thinking" in body:
            return 400, {
                "error": {
                    "type": "invalid_request_error",
                    "message": "thinking.budget_tokens: must be at least 1024",
                }
            }
        return 200, messages_body(text=JSON_ANSWER)

    stub = stub_server(messages=script)
    client = client_for(stub, reasoning=ReasoningConfig(mode="native", budget_tokens=512))

    with pytest.raises(ProviderError) as error:
        ask(client)

    assert "must be at least 1024" in str(error.value)
    assert len(stub.requests) == 1  # no re-ask without the field
    assert "thinking" in stub.requests[0]

    # A later call with a budget the server does take still sends it: nothing was written off.
    stub.messages = lambda _: (200, messages_body(text=JSON_ANSWER))
    ask(client, reasoning=ReasoningConfig(mode="native", budget_tokens=2048))
    assert stub.requests[-1]["thinking"]["budget_tokens"] == 2048


def test_a_truncated_messages_answer_names_the_budget(stub_server):
    """This API reports a spent output budget as ``max_tokens``, not as Chat Completions' ``length``."""
    stub = stub_server(messages=lambda _: (200, messages_body(text="", stop_reason="max_tokens")))
    client = client_for(stub, n_retry_malformed=0)

    with pytest.raises(MalformedAnswerError) as error:
        ask(client)

    assert "ran out of output tokens" in str(error.value)
    assert "max_tokens" in str(error.value)


def test_a_refused_messages_answer_says_the_model_refused(stub_server):
    """``stop_reason: "refusal"`` is a 200 with no answer, and reads as a parsing bug without this."""
    stub = stub_server(messages=lambda _: (200, messages_body(text="", stop_reason="refusal")))
    client = client_for(stub, n_retry_malformed=0)

    with pytest.raises(MalformedAnswerError) as error:
        ask(client)

    assert "refused to answer" in str(error.value)


def test_a_messages_discrete_answer_is_read(stub_server):
    """The one-label method needs no distribution, which is the only thing this surface cannot carry."""
    stub = stub_server(messages=lambda _: (200, messages_body(text='{"choice": "B"}')))
    client = client_for(stub, method="discrete")

    response = ask(client)

    assert response.answers["q"].choice == "technical"
    assert response.answers["q"].probabilities == {"billing": 0.0, "technical": 1.0, "sales": 0.0}
    assert response.debug["api"] == "messages"


def test_a_budget_resolves_to_the_thinking_field_under_mode_auto(stub_server):
    """The budget is the only reason to ask for this surface's own thinking, so it selects it.

    Under ``mode="auto"`` the Messages surface would otherwise resolve to the two-step path, which sends
    no ``thinking`` field at all — leaving the documented budget silently unused.
    """
    stub = stub_server(messages=answer())
    client = client_for(stub, reasoning=ReasoningConfig(budget_tokens=1024))

    response = ask(client)

    sent = body_sent(stub)
    assert sent["thinking"] == {"type": "enabled", "budget_tokens": 1024}
    assert sent["max_tokens"] == DEFAULT_MAX_TOKENS + 1024
    assert response.debug["reasoning_mode"] == "native"
    assert response.answers["q"].choice == "billing"
