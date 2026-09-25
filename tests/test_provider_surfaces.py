"""What four real servers answered, replayed.

The bodies in ``tests/fixtures/providers/`` were recorded with raw HTTP from ollama 0.34.3, llama.cpp
b11139, vLLM 0.30.1 and SGLang 0.5.20, each serving ``Qwen3.5-9B`` at 4-bit (the recipes are in
``docs/local-servers.md``); the ``messages-*`` cases come from the same servers serving the Qwen3-4B
2507 pair, through the Anthropic-compatible route. Replaying them keeps the shapes those servers really send in front of every
future change — an empty logprob array, ``reasoning`` versus ``reasoning_content``, ``{"detail": "Not
Found"}``, a 404 that names the model, ``status: "incomplete"`` — without needing a GPU.

Each case is pushed through a real ``openai`` client, so the SDK's parsing is exercised too.
"""

from __future__ import annotations

import asyncio
import json
from functools import cache
from pathlib import Path
from typing import Any

import pytest
from fakes import (
    StubServer,
    anthropic_client,
    async_anthropic_client,
    chat_body,
    openai_client,
)

from jevper import (
    AsyncSystemOneClient,
    Choice,
    ClientCapabilityError,
    IncompleteAnswerError,
    JevperError,
    LabelReadoutError,
    MalformedAnswerError,
    ProviderError,
    RetryPolicy,
    SystemOneClient,
    reasoning_text,
)
from jevper.transport import SURFACES

FIXTURES = Path(__file__).parent / "fixtures" / "providers"
SERVERS = ("ollama", "llamacpp", "vllm", "sglang")
NO_RETRIES = RetryPolicy(n_retries=0)

STATE = "My invoice shows a charge I do not recognize and I need it explained."
CRITERIA = {
    "billing": "asks about an invoice",
    "technical": "asks about the API",
    "sales": "asks about pricing",
}
QUESTIONS = {"intent": Choice(criteria=CRITERIA)}


@cache
def recorded(server: str) -> dict[str, dict[str, Any]]:
    document = json.loads((FIXTURES / f"{server}.json").read_text())
    return {case["name"]: case for case in document["cases"]}


def replay(server: str, case: str) -> tuple[int, Any]:
    """The recorded status and payload of one case, ready to be served.

    A dict payload is JSON-encoded by the stub. A string payload goes out verbatim, which is how the
    plain-text and bare-JSON-string bodies these servers really send are replayed: ollama answers
    ``404 page not found`` and SGLang answers some 400s with a JSON string rather than an object.
    """
    record = recorded(server)[case]
    if "body" in record:
        body = record["body"]
        return record["status"], body if isinstance(body, dict) else json.dumps(body)
    return record["status"], record["raw"]


def served(stub_server: Any, chat: tuple[str, str], responses: tuple[str, str] | None = None) -> StubServer:
    """A stub serving the recorded ``chat`` case, and optionally a different ``responses`` case."""
    stub = stub_server(
        chat=lambda _: replay(*chat),
        responses=(lambda _: replay(*responses)) if responses else None,
    )
    return stub


def client_for(stub: StubServer, **kwargs: Any) -> SystemOneClient:
    return SystemOneClient(openai_client(stub), model="qwen3.5-9b", **kwargs)


# --- reading a real distribution -------------------------------------------------------------


@pytest.mark.parametrize("server", SERVERS)
def test_a_recorded_logprob_stream_reads_the_answer_token(stub_server, server):
    """Every server's Chat Completions carries the OpenAI logprob shape; the answer token is 'A'."""
    stub = served(stub_server, (server, "chat-thinking-off-logprobs"))
    client = client_for(stub, api="chat_completions", method="logprobs")

    response = client.system_one(state=STATE, questions=QUESTIONS)
    answer = response.answers["intent"]

    assert answer.choice == "billing"  # 'A' is the first option
    assert set(answer.probabilities) == set(CRITERIA)
    assert sum(answer.probabilities.values()) == pytest.approx(1.0)
    assert response.usage.n_calls == 1


@pytest.mark.parametrize("server", SERVERS)
def test_a_reasoning_only_answer_names_the_token_budget(stub_server, server):
    """Thinking-on, a small budget: the stream is the thinking span and nothing else came back."""
    stub = served(stub_server, (server, "chat-logprobs"))
    client = client_for(stub, api="chat_completions", method="logprobs", n_retry_malformed=0)

    with pytest.raises(IncompleteAnswerError) as error:
        client.system_one(state=STATE, questions=QUESTIONS)

    message = str(error.value)
    assert "ran out of output tokens" in message  # from the recorded finish_reason
    assert len(stub.requests) == 1


@pytest.mark.parametrize("server", ("vllm", "sglang"))
def test_a_truncated_responses_answer_names_the_budget(stub_server, server):
    """vLLM and SGLang answer an exhausted Responses call with status 'incomplete' and no message."""
    stub = served(
        stub_server,
        (server, "chat-thinking-off-logprobs"),
        (server, "responses-logprobs"),
    )
    client = client_for(stub, api="responses", method="logprobs", n_retry_malformed=0)

    with pytest.raises(IncompleteAnswerError) as error:
        client.system_one(state=STATE, questions=QUESTIONS)

    assert "max_output_tokens" in str(error.value)
    assert "ran out of output tokens" in str(error.value)


def test_an_empty_logprob_array_is_read_as_no_logprobs(stub_server):
    """ollama's Responses surface returns the message with an empty ``logprobs`` list."""
    stub = served(
        stub_server,
        ("ollama", "chat-thinking-off-logprobs"),
        ("ollama", "responses-logprobs"),
    )
    client = client_for(stub, api="responses", method="logprobs", n_retry_malformed=0)

    with pytest.raises(LabelReadoutError) as error:
        client.system_one(state=STATE, questions=QUESTIONS)

    message = str(error.value)
    assert "does not report them" in message
    assert "output tokens" not in message  # a completed response, not a truncated one


# --- leaving a surface that cannot carry the distribution -------------------------------------


def test_auto_leaves_a_surface_whose_logprobs_are_empty(stub_server):
    """The ollama shape: Responses answers 200 with an empty logprob array, Chat carries the stream."""
    stub = served(
        stub_server,
        ("ollama", "chat-thinking-off-logprobs"),
        ("ollama", "responses-logprobs"),
    )
    client = client_for(stub, api="auto", method="auto")

    response = client.system_one(state=STATE, questions=QUESTIONS)

    assert response.debug["api"] == "chat_completions"
    assert response.debug["methods"] == {"intent": "logprobs"}
    assert response.answers["intent"].choice == "billing"
    assert any("api='chat_completions'" in reason for reason in response.debug["retry_reasons"])
    assert stub.paths == ["/v1/responses", "/v1/chat/completions"]


def test_auto_leaves_a_route_that_refuses_the_logprob_fields(stub_server):
    """The llama.cpp shape: its Responses shim answers 400 for any request carrying top_logprobs."""
    stub = served(
        stub_server,
        ("llamacpp", "chat-thinking-off-logprobs"),
        ("llamacpp", "responses-logprobs"),
    )
    client = client_for(stub, api="auto", method="auto")

    response = client.system_one(state=STATE, questions=QUESTIONS)

    assert response.debug["api"] == "chat_completions"
    assert response.answers["intent"].choice == "billing"
    assert any("top_logprobs requires logprobs" in reason for reason in response.debug["retry_reasons"])


@pytest.mark.parametrize("server", SERVERS)
def test_a_route_404_that_does_not_name_the_model_falls_back(stub_server, server):
    """Every server's route 404 — plain text, ``{"detail"}``, ``{"error"}`` — is a missing route."""
    stub = served(stub_server, (server, "chat-thinking-off-logprobs"), (server, "chat-bad-path"))
    client = client_for(stub, api="auto", method="auto")

    response = client.system_one(state=STATE, questions=QUESTIONS)

    assert response.debug["api"] == "chat_completions"
    assert response.answers["intent"].choice == "billing"
    assert any("no 'responses' route" in reason for reason in response.debug["retry_reasons"])


@pytest.mark.parametrize("server", ("ollama", "vllm", "sglang"))
class MessagesOnly:
    """A client that can speak nothing but the Messages API: no ``responses``, no ``chat.completions``."""

    def __init__(self, client: Any) -> None:
        self.messages = client.messages


def test_a_messages_only_client_keeps_reporting_the_missing_route(stub_server):
    """When the only surface the client can speak has no route, every call must say so.

    The first call learns the route is missing and raises the provider's 404. The second must not flip
    to a surface the client cannot speak — that answered with an ``AttributeError`` for ``responses``
    instead of the same 404.
    """
    stub = stub_server()  # no messages script: the stub answers 404 for /v1/messages
    client = SystemOneClient(MessagesOnly(anthropic_client(stub)), model="stub", api="auto")

    for attempt in (1, 2):
        with pytest.raises(ProviderError) as caught:
            client.system_one(state=STATE, questions=QUESTIONS)
        assert caught.value.status_code == 404, f"attempt {attempt}: {caught.value}"
        assert "no 'messages' route" in str(caught.value), f"attempt {attempt}: {caught.value}"


def test_an_explicit_messages_surface_reports_the_underlying_404(stub_server):
    """The explicit surface is the reference: an explicit choice gets the provider's own error."""
    stub = stub_server()
    client = SystemOneClient(MessagesOnly(anthropic_client(stub)), model="stub", api="messages")

    with pytest.raises(ProviderError) as caught:
        client.system_one(state=STATE, questions=QUESTIONS)

    assert caught.value.status_code == 404
    assert "404" in str(caught.value)


class ChatOnly:
    """A client that can speak only chat completions: no ``responses``, no ``messages``."""

    def __init__(self, client: Any) -> None:
        self.chat = client.chat


def test_a_chat_only_client_keeps_reporting_the_missing_route(stub_server):
    """The same rule on the OpenAI side: no chat route, and nothing to flip to."""
    stub = stub_server()  # no chat script: the stub answers 404 for /chat/completions
    client = SystemOneClient(ChatOnly(openai_client(stub)), model="stub", api="auto")

    for attempt in (1, 2):
        with pytest.raises(ProviderError) as caught:
            client.system_one(state=STATE, questions=QUESTIONS)
        assert caught.value.status_code == 404, f"attempt {attempt}: {caught.value}"
        assert "no 'chat_completions' route" in str(caught.value), f"attempt {attempt}: {caught.value}"


def test_a_messages_only_client_reports_the_missing_route_when_async(stub_server):
    """The async driver shares the same surface selection, so it shares the same rule."""
    stub = stub_server()
    client = AsyncSystemOneClient(MessagesOnly(async_anthropic_client(stub)), model="stub", api="auto")

    for attempt in (1, 2):
        with pytest.raises(ProviderError) as caught:
            asyncio.run(client.system_one(state=STATE, questions=QUESTIONS))
        assert caught.value.status_code == 404, f"attempt {attempt}: {caught.value}"
        assert "no 'messages' route" in str(caught.value), f"attempt {attempt}: {caught.value}"


def test_the_flip_still_happens_when_the_client_can_speak_the_other_surface(stub_server):
    """The remembered 404 is still an optimisation: a client with both surfaces skips the dead route."""
    distribution = [("A", -0.12), ("B", -2.47), ("C", -3.48)]
    stub = stub_server(chat=lambda _: (200, chat_body(content="A", logprobs=distribution)))
    client = SystemOneClient(
        openai_client(stub), model="stub", api="auto", method="logprobs", retry=NO_RETRIES
    )

    first = client.system_one(state=STATE, questions=QUESTIONS)
    assert first.debug["api"] == "chat_completions"
    responses_calls = len([path for path in stub.paths if path.endswith("/responses")])
    assert responses_calls == 1, "the first call is the one that discovers the missing route"

    second = client.system_one(state=STATE, questions=QUESTIONS)

    assert second.debug["api"] == "chat_completions"
    assert second.answers["intent"].choice == "billing"
    assert len([path for path in stub.paths if path.endswith("/responses")]) == responses_calls


@pytest.mark.parametrize("server", ("ollama", "vllm", "sglang"))
def test_a_404_that_names_the_model_is_not_a_missing_route(stub_server, server):
    """A bad model id 404s on every surface; switching would only hide it behind the same error."""
    stub = served(
        stub_server,
        (server, "chat-thinking-off-logprobs"),
        (server, "responses-bad-model"),
    )
    # The recorded 404 names the model it was asked for, which is what a real one does.
    client = SystemOneClient(
        openai_client(stub),
        model="definitely-not-a-model",
        api="auto",
        method="auto",
        retry=NO_RETRIES,
    )

    with pytest.raises(ProviderError) as error:
        client.system_one(state=STATE, questions=QUESTIONS)

    assert error.value.status_code == 404
    assert "definitely-not-a-model" in str(error.value)
    assert stub.bodies("/chat/completions") == []


# --- remembering a refusal, and not remembering a cap ------------------------------------------


def structured_answer() -> tuple[int, dict[str, Any]]:
    return 200, chat_body(
        content='{"probabilities": {"billing": 0.8, "technical": 0.1, "sales": 0.1}}'
    )


def test_a_structural_refusal_is_remembered_even_with_no_other_surface(stub_server):
    """llama.cpp's real refusal names a condition, not a bad value: the next call skips logprobs.

    Pinned to Chat Completions there is nowhere to move the readout, so remembering is the only thing
    that stops every later call from paying for the same rejected request. Questions asked together
    cannot benefit — they all start before any of them learns — so this asks twice, in sequence.
    """
    rejection = replay("llamacpp", "chat-tops-without-logprobs")

    def script(body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        return rejection if body.get("logprobs") else structured_answer()

    stub = stub_server(chat=script)
    client = client_for(stub, api="chat_completions", method="auto", retry=NO_RETRIES)

    first = client.system_one(state=STATE, questions=QUESTIONS)
    second = client.system_one(state=STATE, questions=QUESTIONS)

    assert first.answers["intent"].choice == "billing"
    assert second.debug["method"] == "structured"
    logprob_requests = [body for body in stub.bodies("/chat/completions") if body.get("logprobs")]
    assert len(logprob_requests) == 1  # paid once, then remembered


def test_a_top_logprobs_cap_is_not_remembered(stub_server):
    """ollama's real cap message names a bad value, so logprobs are tried again on the next call."""
    rejection = replay("ollama", "chat-logprobs-21-tops")

    def script(body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        return rejection if body.get("logprobs") else structured_answer()

    stub = stub_server(chat=script)
    client = client_for(stub, api="chat_completions", method="auto", retry=NO_RETRIES)

    client.system_one(state=STATE, questions=QUESTIONS)
    client.system_one(state=STATE, questions=QUESTIONS)

    logprob_requests = [body for body in stub.bodies("/chat/completions") if body.get("logprobs")]
    assert len(logprob_requests) == 2  # a bad value says nothing about the next call


# --- what the provider says when it says no ---------------------------------------------------


@pytest.mark.parametrize("server", SERVERS)
def test_a_recorded_rejection_reaches_the_caller_with_its_own_words(stub_server, server):
    stub = served(stub_server, (server, "chat-missing-messages"))
    client = client_for(stub, api="chat_completions", method="structured", retry=NO_RETRIES)

    with pytest.raises(ProviderError) as error:
        client.system_one(state=STATE, questions=QUESTIONS)

    assert error.value.status_code == 400
    assert "messages" in str(error.value)


@pytest.mark.parametrize("server", SERVERS)
def test_every_recorded_error_body_is_reported_without_a_crash(stub_server, server):
    """Four different error envelopes, including SGLang's bare JSON string body."""
    stub = served(stub_server, (server, "chat-bad-json-body"))
    client = client_for(stub, api="chat_completions", method="structured", retry=NO_RETRIES)

    with pytest.raises(ProviderError) as error:
        client.system_one(state=STATE, questions=QUESTIONS)

    assert error.value.status_code in (400, 500)
    assert error.value.attempts  # the failure is recorded for debug


# --- the reasoning field each server uses -----------------------------------------------------


@pytest.mark.parametrize("server", SERVERS)
def test_the_reasoning_text_each_server_reports_is_read(stub_server, server):
    """``reasoning`` (ollama, vLLM) and ``reasoning_content`` (llama.cpp, SGLang), with a real answer.

    The recorded thinking span is spliced onto the recorded thinking-off answer, which is the shape a
    server returns when it finishes thinking inside the budget: reasoning, then the answer token.
    """
    status, body = replay(server, "chat-thinking-off-logprobs")
    thinking = replay(server, "chat-logprobs")[1]["choices"][0]["message"]
    key = next(name for name in ("reasoning", "reasoning_content", "thinking") if name in thinking)
    body["choices"][0]["message"][key] = thinking[key]
    stub = stub_server(chat=lambda _: (status, body))
    client = client_for(stub, api="chat_completions", method="logprobs")

    response = client.system_one(state=STATE, questions=QUESTIONS)

    assert response.answers["intent"].choice == "billing"
    assert thinking[key].strip() in reasoning_text(response.reasoning)


@pytest.mark.parametrize("server", SERVERS)
def test_jevper_only_sends_roles_every_server_accepts(stub_server, server):
    """SGLang answers 400 for a ``developer`` message, so jevper's own turns must stay conventional."""
    stub = served(stub_server, (server, "chat-thinking-off-logprobs"))
    client = client_for(stub, api="chat_completions", method="logprobs")

    client.system_one(state=STATE, questions=QUESTIONS)
    sent = stub.bodies("/chat/completions")[0]["messages"]

    assert {message["role"] for message in sent} <= {"system", "user", "assistant"}
    assert sent[0]["role"] == "system"


@pytest.mark.parametrize("server", ("ollama", "llamacpp", "vllm"))
def test_a_long_thinking_run_still_yields_the_answer_token(stub_server, server):
    """The real thing: 97-140 tokens of thinking, the answer, and on vLLM a trailing ``<|im_end|>``.

    Recorded with a 160-token budget so the model finishes. vLLM's stream ends ``['A', '<|im_end|>']``
    while its content is ``'\\n\\nA'``, which is the shape the end-of-turn tolerance exists for.
    """
    stub = served(stub_server, (server, "chat-logprobs-long"))
    client = client_for(stub, api="chat_completions", method="logprobs")

    response = client.system_one(state=STATE, questions=QUESTIONS)

    assert response.answers["intent"].choice == "billing"  # the model answered 'A'
    assert response.usage.n_calls == 1
    assert response.debug["retry_reasons"] == []
    assert reasoning_text(response.reasoning).strip()  # the trace came along


def test_a_long_run_that_hits_the_budget_says_so(stub_server):
    """SGLang spent the whole 160-token budget thinking and never wrote an answer."""
    stub = served(stub_server, ("sglang", "chat-logprobs-long"))
    client = client_for(
        stub, api="chat_completions", method="logprobs", n_retry_malformed=0
    )

    with pytest.raises(IncompleteAnswerError) as error:
        client.system_one(state=STATE, questions=QUESTIONS)

    message = str(error.value)
    assert "ran out of output tokens" in message
    assert "'length'" in message

@pytest.mark.parametrize("server", SERVERS)
def test_a_recorded_cache_hit_is_read(server):
    """The same prompt twice, and whatever the server said about reuse.

    These captures are plain requests — no logprobs, so no readout can consume the answer text — which
    is why this asserts where the count is read: the normalizer ``system_one`` itself calls. ollama,
    llama.cpp, vLLM and SGLang all reported a real reuse here (1985, 1985, 1584 and 1984 of ~1989
    prompt tokens), and a server that reported nothing would leave the count ``None``.
    """
    usage = (recorded(server)["chat-cache-warm"]["body"] or {}).get("usage") or {}
    reported = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    normalize = SURFACES["chat_completions"][1]

    result = normalize(recorded(server)["chat-cache-warm"]["body"], {})

    assert result.cached_tokens == reported
    assert reported is not None and reported > 0


@pytest.mark.parametrize("server", SERVERS)
def test_a_recorded_responses_cache_is_read(server):
    """The Responses surface reports reuse under ``usage.input_tokens_details``, not the chat path.

    llama.cpp's Responses shim answers this request with a 400, so that capture has nothing to read.
    """
    record = recorded(server)["responses-cache-warm"]
    if record["status"] != 200:
        pytest.skip(f"{server} answered {record['status']} on the Responses surface for this capture")
    usage = (record["body"] or {}).get("usage") or {}
    reported = (usage.get("input_tokens_details") or {}).get("cached_tokens")
    normalize = SURFACES["responses"][1]

    result = normalize(record["body"], {})

    assert result.cached_tokens == reported
    assert reported is not None and reported > 0


def test_a_server_that_reports_nothing_is_not_a_reported_zero():
    """SGLang sends ``prompt_tokens_details: null`` when nothing was cached, and a count when it was.

    Both bodies are from the same server and the same prompt, so the pair is what proves the
    distinction: silence has to stay ``None``, because a reported ``0`` means a cold or disabled cache.
    """
    normalize = SURFACES["chat_completions"][1]

    cold = normalize(recorded("sglang")["chat-cache-key"]["body"], {})
    warm = normalize(recorded("sglang")["chat-cache-warm"]["body"], {})

    assert "prompt_tokens_details" in cold.response["usage"]
    assert cold.response["usage"]["prompt_tokens_details"] is None
    assert cold.cached_tokens is None
    assert warm.cached_tokens and warm.cached_tokens > 0


def messages_servers() -> list[str]:
    """Every server whose fixtures carry the Messages route.

    LM Studio is the one server recorded for that route alone — its OpenAI-route cases were not captured —
    so it takes part in these tests and not the others.
    """
    return [
        server
        for server in (*SERVERS, "lmstudio")
        if "messages-plain" in recorded(server)
    ]


@pytest.mark.parametrize("server", messages_servers())
def test_a_recorded_message_answer_is_read(server):
    """The Messages route is a different wire shape: a content-block list, not a choices list.

    ollama's and llama.cpp's cases come from a non-thinking model and carry a text block. vLLM's and
    SGLang's come from the thinking model with thinking on, which their reasoning parsers answer with a
    thinking block and no text block at all — the shape that leaves nothing to parse. Pinned here so the
    normalizer keeps the two apart: reasoning is never passed off as the answer.
    """
    record = recorded(server)["messages-plain"]
    blocks = [block["type"] for block in record["body"]["content"]]
    normalize = SURFACES["messages"][1]

    result = normalize(record["body"], {})

    assert record["status"] == 200
    assert result.stop in ("end_turn", "max_tokens")
    assert result.input_tokens == record["body"]["usage"]["input_tokens"]
    assert result.token_logprobs == ()
    if "text" in blocks:
        assert result.text.strip()
    else:
        assert result.text.strip() == ""
        assert reasoning_text(result.reasoning).strip()


@pytest.mark.parametrize("server", messages_servers())
def test_a_recorded_message_cache_hit_is_read(server):
    """``cache_read_input_tokens`` is where this API reports reuse, and both servers reported one."""
    record = recorded(server)["messages-warm"]
    reported = record["body"]["usage"].get("cache_read_input_tokens")
    normalize = SURFACES["messages"][1]

    result = normalize(record["body"], {})

    assert result.cached_tokens == reported
    assert reported is not None and reported > 0


# --- a 200 that is not simply an answer ---------------------------------------------------------


def _answers() -> dict[str, Any]:
    return {"probabilities": {"billing": 0.8, "technical": 0.1, "sales": 0.1}}


@pytest.mark.parametrize("surface", ["chat_completions", "responses", "messages"])
def test_an_error_carried_in_a_200_beats_a_readable_answer(stub_server, surface):
    """A body that says both "here is your answer" and "the upstream failed" is not an answer.

    OpenRouter reports an overloaded upstream in a 200 with no choices at all; a gateway that adds a
    stale body beside the error is the same failure wearing a usable mask. Reading the answer would
    report a decision the provider never made.
    """
    key = {"chat_completions": "chat", "responses": "responses", "messages": "messages"}[surface]
    body: dict[str, Any] = {"error": {"message": "upstream failed", "code": 503}}
    if surface == "chat_completions":
        body.update(chat_body(content="A", logprobs=[("A", -0.1), ("B", -2.0), ("C", -3.0)]))
    elif surface == "responses":
        from fakes import responses_body

        body.update(responses_body(text=json.dumps(_answers())))
    else:
        from fakes import messages_body

        body.update(messages_body(text=json.dumps(_answers())))
    stub = stub_server(**{key: lambda _b: (200, body, {})})
    sdk = anthropic_client(stub) if surface == "messages" else openai_client(stub)
    client = SystemOneClient(
        sdk, model="stub", api=surface, method="structured", retry=NO_RETRIES
    )

    with pytest.raises(ProviderError) as raised:
        client.system_one(state="s", questions={"q": Choice(criteria=QUESTIONS["intent"].criteria)})

    assert "upstream failed" in str(raised.value)
    assert len(stub.requests) == 1


def test_a_numeric_error_code_sent_as_a_string_is_still_transient(stub_server):
    """OpenRouter sends ``"code": "503"`` as often as a number, and a transient it stays."""
    calls: list[int] = []

    def script(_body):
        calls.append(1)
        if len(calls) == 1:
            return 200, {"error": {"message": "temporarily overloaded", "code": "503"}}, {}
        return 200, chat_body(content="A", logprobs=[("A", -0.1), ("B", -2.0), ("C", -3.0)])

    stub = stub_server(chat=script)
    client = SystemOneClient(
        openai_client(stub),
        model="stub",
        api="chat_completions",
        retry=RetryPolicy(n_retries=2, base_delay=0.0),
    )

    response = client.system_one(
        state="s", questions={"q": Choice(criteria=QUESTIONS["intent"].criteria)}
    )

    assert response.usage.n_retries == 1
    assert response.answers["q"].choice == "billing"


@pytest.mark.parametrize("stop", [[], {}, 3.5, True])
def test_a_stop_reason_that_is_not_a_name_is_an_incomplete_answer(stub_server, stop):
    """The field is documented as a string; a list or a dict there must not be a ``TypeError``."""
    body = chat_body(content="A", finish_reason=stop)
    stub = stub_server(chat=lambda _b: (200, body, {}))
    client = SystemOneClient(
        openai_client(stub), model="stub", api="chat_completions", method="logprobs"
    )

    with pytest.raises(IncompleteAnswerError) as raised:
        client.system_one(
            state="s", questions={"q": Choice(criteria=QUESTIONS["intent"].criteria)}
        )

    assert "stopped before the answer was complete" in str(raised.value)


def test_a_choice_that_carries_only_the_legacy_text_is_read(stub_server):
    """A proxy answering a chat request with the completions shape puts the answer in ``text``."""
    body = {
        "id": "x",
        "object": "text_completion",
        "choices": [{"index": 0, "text": json.dumps(_answers()), "finish_reason": "stop"}],
    }
    stub = stub_server(chat=lambda _b: (200, body, {}))
    client = SystemOneClient(
        openai_client(stub), model="stub", api="chat_completions", method="structured"
    )

    response = client.system_one(
        state="s", questions={"q": Choice(criteria=QUESTIONS["intent"].criteria)}
    )

    assert response.answers["q"].choice == "billing"


def test_a_partial_model_dump_falls_back_to_the_attributes():
    """A duck client whose ``model_dump`` leaves fields out has not lost the answer."""

    class Message:
        def __init__(self):
            self.content = "A"
            self.refusal = None

    class ChoiceModel:
        def __init__(self):
            self.message = Message()
            self.finish_reason = "stop"
            self.logprobs = None

        def model_dump(self, **kwargs):
            return {}

    normalize = SURFACES["chat_completions"][1]
    result = normalize({"id": "x", "choices": [ChoiceModel()]}, {})

    assert result.text == "A"
    assert result.stop == "stop"


@pytest.mark.parametrize("finish", [None, "absent"])
def test_a_chat_answer_without_a_finish_reason_is_not_an_answer(stub_server, finish):
    """Chat Completions types ``finish_reason`` as a required, non-null literal.

    A body without one is a generation whose completion the provider never reported — a snapshot, not
    an answer — and reading its text anyway would report a decision from a body that says it stopped
    for an unknown reason.
    """
    body = chat_body(content=json.dumps(_answers()))
    if finish == "absent":
        body["choices"][0].pop("finish_reason")
    else:
        body["choices"][0]["finish_reason"] = None
    stub = stub_server(chat=lambda _b: (200, body))
    client = SystemOneClient(
        openai_client(stub), model="stub", api="chat_completions", method="structured",
        n_retry_malformed=0,
    )

    with pytest.raises(IncompleteAnswerError) as raised:
        client.system_one(
            state="s", questions={"q": Choice(criteria=QUESTIONS["intent"].criteria)}
        )

    assert "finish_reason_missing" in str(raised.value)


def test_a_null_message_is_no_answer_rather_than_a_malformed_one(stub_server):
    """``message: null`` is a choice with no carrier, which the capability verdict already names."""
    body = {"id": "x", "object": "chat.completion", "choices": [{"index": 0, "finish_reason": "stop", "message": None}]}
    stub = stub_server(chat=lambda _b: (200, body))
    client = SystemOneClient(
        openai_client(stub), model="stub", api="chat_completions", method="structured",
        n_retry_malformed=0,
    )

    with pytest.raises(ClientCapabilityError) as raised:
        client.system_one(
            state="s", questions={"q": Choice(criteria=QUESTIONS["intent"].criteria)}
        )

    assert "no choices" in str(raised.value)


def test_a_streaming_chunk_where_a_response_belongs_is_named(stub_server):
    """A server that streams an answer nobody asked for gets a named error, not a malformed answer."""
    body = {
        "id": "x",
        "object": "chat.completion.chunk",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "delta": {"role": "assistant", "content": json.dumps(_answers())},
            }
        ],
    }
    stub = stub_server(chat=lambda _b: (200, body))
    client = SystemOneClient(
        openai_client(stub), model="stub", api="chat_completions", method="structured",
        n_retry_malformed=0,
    )

    with pytest.raises(ProviderError) as raised:
        client.system_one(
            state="s", questions={"q": Choice(criteria=QUESTIONS["intent"].criteria)}
        )

    assert "streaming chunk" in str(raised.value)
    assert len(stub.requests) == 1


# --- the OpenResponses bodies these servers answered ---------------------------------------------


@pytest.mark.parametrize("server", SERVERS)
def test_the_typed_input_body_each_server_answered_is_read(stub_server, server):
    """The item shape jevper sends — every input turn carrying its ``type`` — is answered by all five.

    What the server then does with it is its own business: a 4B model on a thinking run can spend the
    budget on the trace, and what the caller gets then is a named failure, not a crash and not a
    decision read from a body that carries no answer.
    """
    stub = served(
        stub_server,
        (server, "chat-thinking-off-logprobs"),
        (server, "responses-openresponses-items"),
    )
    client = client_for(stub, api="responses", method="structured", n_retry_malformed=0)

    try:
        response = client.system_one(state=STATE, questions=QUESTIONS)
    except JevperError as exc:
        assert "no JSON object" in str(exc) or "output tokens" in str(exc), str(exc)
        return

    assert response.answers["intent"].choice in CRITERIA


@pytest.mark.parametrize("server", ("vllm", "sglang"))
def test_a_recorded_spent_budget_names_the_responses_budget(stub_server, server):
    """vLLM and SGLang report an exhausted Responses call as ``incomplete`` with a reason."""
    stub = served(
        stub_server,
        (server, "chat-thinking-off-logprobs"),
        (server, "responses-budget-truncated"),
    )
    client = client_for(stub, api="responses", method="structured", n_retry_malformed=0)

    with pytest.raises(IncompleteAnswerError) as raised:
        client.system_one(state=STATE, questions=QUESTIONS)

    assert "max_output_tokens" in str(raised.value)


@pytest.mark.parametrize("server", ("ollama", "llamacpp", "lmstudio"))
def test_a_server_that_reports_a_spent_budget_as_completed_is_taken_at_its_word(stub_server, server):
    """These three answer a caller-set budget of eight tokens with ``status: "completed"``.

    ollama and llama.cpp return the reasoning trace and no message; LM Studio returns an answer cut off
    mid-string. There is no signal to read in any of them, so what the caller is told is what the body
    says happened: a malformed answer, named as one, with the reasoning-only note when the answer is
    empty.
    """
    stub = served(
        stub_server,
        (server, "chat-thinking-off-logprobs"),
        (server, "responses-budget-truncated"),
    )
    client = client_for(stub, api="responses", method="structured", n_retry_malformed=0)

    with pytest.raises(MalformedAnswerError) as raised:
        client.system_one(state=STATE, questions=QUESTIONS)

    assert "JSON object" in str(raised.value)
    assert len(stub.requests) == 1

