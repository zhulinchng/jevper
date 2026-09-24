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

import json
from functools import cache
from pathlib import Path
from typing import Any

import pytest
from fakes import StubServer, chat_body, openai_client

from jevper import (
    Choice,
    LabelReadoutError,
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

    with pytest.raises(LabelReadoutError) as error:
        client.system_one(state=STATE, questions=QUESTIONS)

    message = str(error.value)
    assert "'Thinking'" in message
    assert "output tokens" in message  # the budget note, from the recorded finish_reason


@pytest.mark.parametrize("server", ("vllm", "sglang"))
def test_a_truncated_responses_answer_names_the_budget(stub_server, server):
    """vLLM and SGLang answer an exhausted Responses call with status 'incomplete' and no message."""
    stub = served(
        stub_server,
        (server, "chat-thinking-off-logprobs"),
        (server, "responses-logprobs"),
    )
    client = client_for(stub, api="responses", method="logprobs", n_retry_malformed=0)

    with pytest.raises(LabelReadoutError) as error:
        client.system_one(state=STATE, questions=QUESTIONS)

    assert "max_output_tokens" in str(error.value)
    assert "does not report them" in str(error.value)


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

    with pytest.raises(LabelReadoutError) as error:
        client.system_one(state=STATE, questions=QUESTIONS)

    message = str(error.value)
    assert "'Thinking'" in message
    assert "output tokens" in message


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
