"""A server that streams where a whole response belongs: what the caller is told.

Every non-streaming surface can be answered with ``text/event-stream`` by a gateway that misroutes
the request or fails mid-flight. Two different things can arrive that way, and they are not the same
verdict: a stream that carries an ``error`` frame is the provider's own failure — with the status
that decides whether retrying is worth anything — while a stream that carries only events is a
protocol mismatch jevper cannot read at all.
"""

from __future__ import annotations

import json

import pytest
from fakes import (
    anthropic_client,
    async_openai_client,
    chat_body,
    openai_client,
    responses_body,
)

from jevper import (
    Choice,
    ClientCapabilityError,
    IncompleteAnswerError,
    ProviderError,
    RetryPolicy,
    SystemOneClient,
)

CRITERIA = {"billing": "money", "technical": "errors", "sales": "pricing"}
STREAM_HEADERS = {"Content-Type": "text/event-stream"}


def event_frame(event: str, payload: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(payload)}\n\n"


def error_frame(status: int, message: str, *, openresponses: bool = False) -> str:
    """The two spellings an error event takes: the OpenResponses one and the plain proxy one."""
    if openresponses:
        return event_frame(
            "error", {"type": "error", "status": status, "error": {"code": "overloaded", "message": message}}
        )
    return event_frame("error", {"error": {"message": message, "code": status}})


def structured_text() -> str:
    return json.dumps({"probabilities": {"billing": 0.8, "technical": 0.1, "sales": 0.1}})


def test_a_responses_error_frame_is_the_providers_own_failure(stub_server):
    """``event: error`` with a status is a provider failure, and the status travels with it."""
    server = stub_server(
        responses=lambda _: (200, error_frame(429, "quota exhausted", openresponses=True), STREAM_HEADERS)
    )

    client = SystemOneClient(
        openai_client(server), model="stub", api="responses", method="structured",
        n_retry_malformed=0,
    )
    with pytest.raises(ProviderError) as caught:
        client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

    assert caught.value.status_code == 429
    assert "quota exhausted" in str(caught.value)


def test_a_responses_error_frame_is_retried_and_then_raised(stub_server):
    """A transient status in an event frame is retried like any other transient failure."""
    server = stub_server(
        responses=lambda _: (200, error_frame(503, "upstream busy", openresponses=True), STREAM_HEADERS)
    )

    client = SystemOneClient(
        openai_client(server), model="stub", api="responses", method="structured",
        retry=RetryPolicy(n_retries=2, base_delay=0.0, max_delay=0.0),
    )
    with pytest.raises(ProviderError) as caught:
        client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

    assert caught.value.status_code == 503
    assert len(server.bodies("/responses")) == 3


def test_a_chat_error_frame_names_the_provider_failure(stub_server):
    """The same frame on Chat Completions, where the SDK had no response object to complain about."""
    server = stub_server(chat=lambda _: (200, error_frame(500, "internal error"), STREAM_HEADERS))

    client = SystemOneClient(
        openai_client(server), model="stub", api="chat_completions", method="structured",
        retry=RetryPolicy(n_retries=0),
    )
    with pytest.raises(ProviderError) as caught:
        client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

    assert caught.value.status_code == 500
    assert "internal error" in str(caught.value)
    assert len(server.bodies("/chat/completions")) == 1


def test_a_chat_stream_with_no_error_frame_is_a_mismatch(stub_server):
    """A stream that carries only events is the protocol problem, and it is named as one."""
    server = stub_server(
        chat=lambda _: (
            200,
            event_frame("message", {"choices": [{"delta": {"content": "A"}}]}),
            STREAM_HEADERS,
        )
    )

    client = SystemOneClient(
        openai_client(server), model="stub", api="chat_completions", method="structured",
        n_retry_malformed=0,
    )
    with pytest.raises(ProviderError, match="event stream"):
        client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})


def test_a_messages_event_stream_is_not_an_empty_message(stub_server):
    """The Messages surface had no guard at all: the stream was read as an empty answer."""
    server = stub_server(
        messages=lambda _: (200, event_frame("error", {"type": "error", "error": {"message": "overloaded"}}), STREAM_HEADERS)
    )

    client = SystemOneClient(
        anthropic_client(server), model="stub", api="messages", method="structured",
        n_retry_malformed=0,
    )
    with pytest.raises(ProviderError, match="overloaded"):
        client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})


def test_a_messages_stream_with_no_error_frame_is_a_mismatch(stub_server):
    server = stub_server(
        messages=lambda _: (
            200,
            event_frame("content_block_delta", {"delta": {"text": "A"}}),
            STREAM_HEADERS,
        )
    )

    client = SystemOneClient(
        anthropic_client(server), model="stub", api="messages", method="structured",
        n_retry_malformed=0,
    )
    with pytest.raises(ProviderError, match="event stream"):
        client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})


def test_a_completed_stream_on_responses_is_still_a_mismatch(stub_server):
    """The 0.7.0 verdict for a completed-event stream, read through the same path as an error one."""
    server = stub_server(
        responses=lambda _: (
            200,
            event_frame("response.completed", {"response": responses_body(text=structured_text())}),
            STREAM_HEADERS,
        )
    )

    client = SystemOneClient(
        openai_client(server), model="stub", api="responses", method="structured",
        n_retry_malformed=0,
    )
    with pytest.raises(ProviderError, match="event stream"):
        client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})


def test_an_event_frame_cannot_turn_a_good_call_into_a_bad_one(stub_server):
    """A stream that completes normally still ends the same way: the mismatch, not an answer."""
    server = stub_server(
        chat=lambda _: (
            200,
            event_frame("message", {"choices": [{"delta": {"content": structured_text()}}]}),
            STREAM_HEADERS,
        )
    )

    client = SystemOneClient(
        openai_client(server), model="stub", api="chat_completions", method="structured",
        n_retry_malformed=0,
    )
    with pytest.raises(ProviderError) as caught:
        client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

    assert "chat_body" not in str(caught.value)
    assert "not one of the labels" not in str(caught.value)


def test_an_async_error_frame_takes_the_same_path(stub_server):
    """The async driver has its own call path; the reader it reaches is the same one."""
    import asyncio

    from jevper import AsyncSystemOneClient

    server = stub_server(
        responses=lambda _: (200, error_frame(429, "async quota", openresponses=True), STREAM_HEADERS)
    )

    async def run():
        client = AsyncSystemOneClient(
            async_openai_client(server), model="stub", api="responses", method="structured",
            n_retry_malformed=0,
        )
        async with client:
            await client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

    with pytest.raises(ProviderError) as caught:
        asyncio.run(run())
    assert caught.value.status_code == 429
    assert "async quota" in str(caught.value)


def test_an_incomplete_body_is_still_read_as_a_truncation(stub_server):
    """The guard is only for event streams: a whole body that says it ran out of room is unchanged."""
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
                        "content": [
                            {
                                "type": "output_text",
                                "text": json.dumps(
                                    {"probabilities": {"billing": 0.8, "technical": 0.1, "sales": 0.1}}
                                )[:20],
                                "annotations": [],
                            }
                        ],
                    }
                ],
                "usage": {"input_tokens": 10, "output_tokens": 8, "total_tokens": 18},
            },
        )
    )

    client = SystemOneClient(
        openai_client(server), model="stub", api="responses", method="structured",
        n_retry_malformed=0,
    )
    with pytest.raises(IncompleteAnswerError, match="max_output_tokens"):
        client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})


def test_a_plain_json_body_on_responses_is_unaffected(stub_server):
    """The guard keys on the body being text, not on a content type: real bodies still answer."""
    server = stub_server(responses=lambda _: (200, responses_body(text=structured_text())))

    client = SystemOneClient(
        openai_client(server), model="stub", api="responses", method="structured"
    )
    response = client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})
    assert response.answers["q"].choice == "billing"


def test_a_chat_body_with_no_choices_is_still_a_capability_error(stub_server):
    """A JSON body with no choices is the old verdict; only a stream is the new one."""
    server = stub_server(chat=lambda _: (200, {"id": "c", "object": "chat.completion", "choices": []}))

    client = SystemOneClient(
        openai_client(server), model="stub", api="chat_completions", method="structured",
        n_retry_malformed=0,
    )
    with pytest.raises(ClientCapabilityError, match="no choices"):
        client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})


def test_the_message_from_a_chat_body_is_not_read_as_a_stream(stub_server):
    """The body is a dict here, so the normalizer reads it exactly as before."""
    server = stub_server(chat=lambda _: (200, chat_body(content=structured_text())))

    client = SystemOneClient(
        openai_client(server), model="stub", api="chat_completions", method="structured"
    )
    response = client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})
    assert response.answers["q"].choice == "billing"


def test_a_response_failed_frame_is_the_providers_own_failure(stub_server):
    """OpenAI's typed ``response.failed`` event: the error lives inside the response object."""
    frame = event_frame(
        "response.failed",
        {
            "type": "response.failed",
            "response": {
                "id": "resp_1",
                "object": "response",
                "status": "failed",
                "error": {"code": "server_error", "message": "The model failed to generate."},
                "output": [],
            },
        },
    )
    server = stub_server(responses=lambda _: (200, frame, STREAM_HEADERS))

    client = SystemOneClient(
        openai_client(server), model="stub", api="responses", method="structured",
        retry=RetryPolicy(n_retries=0),
    )
    with pytest.raises(ProviderError) as caught:
        client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

    assert "The model failed to generate." in str(caught.value)
    assert caught.value.status_code is None, "the frame named no HTTP status, so none is invented"


def test_a_response_failed_frame_with_a_status_is_retried(stub_server):
    """A gateway that puts the status in the frame keeps it retryable, as an event-level error does."""
    frame = event_frame(
        "response.failed",
        {
            "type": "response.failed",
            "status": 503,
            "response": {
                "id": "resp_1",
                "status": "failed",
                "error": {"code": 503, "message": "upstream busy"},
                "output": [],
            },
        },
    )
    server = stub_server(responses=lambda _: (200, frame, STREAM_HEADERS))

    client = SystemOneClient(
        openai_client(server), model="stub", api="responses", method="structured",
        retry=RetryPolicy(n_retries=1, base_delay=0.0, max_delay=0.0),
    )
    with pytest.raises(ProviderError) as caught:
        client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

    assert caught.value.status_code == 503
    assert len(server.bodies("/responses")) == 2


def test_a_completed_event_with_no_failure_is_still_a_mismatch(stub_server):
    """Only a failure is read out of a stream; a completed one has no error to report."""
    frame = event_frame(
        "response.completed",
        {"type": "response.completed", "response": responses_body(text=structured_text())},
    )
    server = stub_server(responses=lambda _: (200, frame, STREAM_HEADERS))

    client = SystemOneClient(
        openai_client(server), model="stub", api="responses", method="structured",
        n_retry_malformed=0,
    )
    with pytest.raises(ProviderError, match="event stream"):
        client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})


def test_a_multiline_data_frame_is_one_event(stub_server):
    """SSE joins the data lines of an event with newlines; a JSON body split across lines is one frame."""
    frame = (
        "event: error\n"
        "data: {\n"
        'data:   "type": "error",\n'
        'data:   "status": 503,\n'
        'data:   "error": {"code": "overloaded", "message": "upstream busy"}\n'
        "data: }\n"
        "\n"
    )
    server = stub_server(responses=lambda _: (200, frame, STREAM_HEADERS))

    client = SystemOneClient(
        openai_client(server), model="stub", api="responses", method="structured",
        retry=RetryPolicy(n_retries=1, base_delay=0.0, max_delay=0.0),
    )
    with pytest.raises(ProviderError) as caught:
        client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

    assert caught.value.status_code == 503
    assert "upstream busy" in str(caught.value)
    assert len(server.bodies("/responses")) == 2, "a transient frame in two lines is still retried"


def test_a_strict_client_reads_the_same_frames(stub_server):
    """A client with strict response validation refuses the stream before jevper sees it, as a body."""
    from openai import OpenAI

    frame = event_frame(
        "error", {"type": "error", "status": 429, "error": {"code": "rate_limit", "message": "slow down"}}
    )
    server = stub_server(responses=lambda _: (200, frame, STREAM_HEADERS))
    client = OpenAI(
        base_url=server.base_url,
        api_key="test",
        max_retries=0,
        timeout=10,
        _strict_response_validation=True,
    )
    configured = SystemOneClient(
        client, model="stub", api="responses", method="structured", retry=RetryPolicy(n_retries=0)
    )

    with pytest.raises(ProviderError) as caught:
        configured.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

    assert caught.value.status_code == 429
    assert "slow down" in str(caught.value)


def crlf(text: str) -> str:
    """The same stream with the other line ending SSE allows."""
    return text.replace("\n", "\r\n")


def test_a_crlf_framed_stream_still_yields_the_providers_status(stub_server):
    """SSE permits CRLF, and a body framed with it contains no ``"\\n\\n"`` at all.

    Splitting on the LF pair alone reads the whole stream as one frame, joins its data lines into
    invalid JSON, and reports a provider's ``503`` as the protocol mismatch — a verdict that is not
    retryable, so a transient failure upstream is written off for the life of the call.
    """
    stream = crlf(
        event_frame(
            "response.created", {"type": "response.created", "response": {"status": "in_progress"}}
        )
        + error_frame(503, "upstream busy")
    )
    server = stub_server(responses=lambda _: (200, stream, STREAM_HEADERS))

    client = SystemOneClient(
        openai_client(server), model="stub", api="responses", method="structured",
        retry=RetryPolicy(n_retries=1, base_delay=0.0, max_delay=0.0),
    )
    with pytest.raises(ProviderError) as caught:
        client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

    assert caught.value.status_code == 503
    assert "upstream busy" in str(caught.value)
    assert len(server.bodies("/responses")) == 2, "the status is read, so the frame is retried"
