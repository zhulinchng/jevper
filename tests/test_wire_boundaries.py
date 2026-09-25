"""The boundaries between jevper, the official SDK clients and what a caller hands in.

Everything here is about the seam: a body the SDK hands over unchanged, a client passed with a retry
loop of its own, a question id or state that no request could ever carry, a caller field that
desynchronises the request from the wire. Each test drives the real ``openai`` client (or the real
``anthropic`` one) against the stub server, so what is asserted is what a caller would see.
"""

from __future__ import annotations

import json
import warnings
from typing import Any

import pytest
from fakes import (
    async_openai_client,
    chat_body,
    messages_body,
    openai_client,
)

from jevper import (
    AsyncSystemOneClient,
    Choice,
    ClientCapabilityError,
    InvalidQuestionError,
    JevperError,
    ProviderError,
    RetryPolicy,
    SystemOneClient,
)

CRITERIA = {"billing": None, "sales": None}
STRUCTURED = json.dumps({"probabilities": {"billing": 0.7, "sales": 0.3}})
NO_RETRY = RetryPolicy(n_retries=0, base_delay=0.0)


def question() -> Choice:
    return Choice(instructions="Pick the intent.", criteria=CRITERIA)


def chat_client(stub, **kwargs) -> SystemOneClient:
    kwargs.setdefault("retry", NO_RETRY)
    return SystemOneClient(
        openai_client(stub), model="stub", api="chat_completions", method="structured",
        n_retry_malformed=0, **kwargs
    )


# ------------------------------------------------------------------ the SDK's own retry loop


def test_the_sdk_does_not_retry_behind_jevpers_back(stub_server):
    """The official clients retry twice by default; jevper's budget is the one the caller chose.

    Both loops running multiplies every attempt — nine requests for one call at the defaults — and
    the SDK's tries never appear in ``usage.n_retries``. The caller's own client keeps its setting;
    jevper's calls go through a copy with the SDK loop off.
    """
    stub = stub_server(chat=lambda _: (503, {"error": {"message": "busy"}}))
    sdk = openai_client(stub)
    sdk.max_retries = 2  # the official default, which the repository helper turns off
    client = SystemOneClient(
        sdk, model="stub", api="chat_completions", method="structured", retry=NO_RETRY
    )

    with pytest.raises(ProviderError) as error:
        client.system_one(state="s", questions={"q": question()})

    assert error.value.status_code == 503
    assert len(stub.requests) == 1, "the SDK retried inside jevper's single attempt"
    assert sdk.max_retries == 2, "jevper changed the caller's client"


def test_one_jevper_retry_is_one_more_request(stub_server):
    stub = stub_server(chat=lambda _: (503, {"error": {"message": "busy"}}))
    client = SystemOneClient(
        openai_client(stub),
        model="stub",
        api="chat_completions",
        method="structured",
        retry=RetryPolicy(n_retries=1, base_delay=0.0),
    )

    with pytest.raises(ProviderError):
        client.system_one(state="s", questions={"q": question()})

    assert len(stub.requests) == 2


def test_a_duck_clients_own_retries_are_the_callers_business(stub_server):
    """A client without ``with_options`` is used as it is — the check is on the type, not the name."""
    calls: list[int] = []

    class Completions:
        def create(self, **kwargs):
            calls.append(1)
            return chat_body(content=STRUCTURED)

    class Chat:
        completions = Completions()

    class Client:
        def __init__(self) -> None:
            self.chat = Chat()

    response = SystemOneClient(Client(), model="stub", method="structured").system_one(
        state="s", questions={"q": question()}
    )

    assert response.answers["q"].choice == "billing"
    assert len(calls) == 1


# ------------------------------------------------------------------ a 200 that carries a failure


def test_a_body_404_is_not_a_missing_route(stub_server):
    """A body's code is the provider's report inside a successful response, not the status line.

    Reading it as a route 404 would answer the question on the other surface and hide the error the
    provider actually sent.
    """
    stub = stub_server(
        responses=lambda _: (200, {"error": {"message": "Not Found", "code": 404}}),
        chat=lambda _: (200, chat_body(content=STRUCTURED)),
    )
    client = SystemOneClient(
        openai_client(stub), model="stub", api="auto", method="structured", retry=NO_RETRY
    )

    with pytest.raises(ProviderError) as error:
        client.system_one(state="s", questions={"q": question()})

    assert error.value.embedded is True
    assert error.value.status_code == 404
    assert stub.paths == ["/v1/responses"]


def test_a_strict_validation_client_still_reads_the_embedded_error(stub_server):
    """With strict validation the SDK refuses the body before jevper sees it; the error is on it.

    An overloaded upstream carried in a ``200`` is transient, so it is retried like any other, and
    the caller's message and status survive the exhaustion.
    """
    attempts = {"n": 0}

    def script(_body):
        attempts["n"] += 1
        if attempts["n"] == 1:
            return 200, {"error": {"message": "overloaded", "code": 503}}
        return 200, chat_body(content=STRUCTURED)

    stub = stub_server(chat=script)
    from openai import OpenAI

    sdk = OpenAI(
        base_url=stub.base_url, api_key="test", max_retries=0, timeout=10,
        _strict_response_validation=True,
    )
    client = SystemOneClient(
        sdk, model="stub", api="chat_completions", method="structured",
        retry=RetryPolicy(n_retries=1, base_delay=0.0),
    )

    response = client.system_one(state="s", questions={"q": question()})

    assert response.usage.n_retries == 1
    assert response.answers["q"].choice == "billing"


def test_a_strict_validation_client_reports_an_overload_after_the_budget(stub_server):
    stub = stub_server(chat=lambda _: (200, {"error": {"message": "overloaded", "code": 503}}))
    from openai import OpenAI

    sdk = OpenAI(
        base_url=stub.base_url, api_key="test", max_retries=0, timeout=10,
        _strict_response_validation=True,
    )
    client = SystemOneClient(
        sdk, model="stub", api="chat_completions", method="structured", retry=NO_RETRY
    )

    with pytest.raises(ProviderError) as error:
        client.system_one(state="s", questions={"q": question()})

    assert "overloaded" in str(error.value)
    assert error.value.status_code == 503


# ------------------------------------------------------------------ sync and async clients


def test_the_blocking_facade_refuses_an_async_client_without_a_request(stub_server):
    stub = stub_server(chat=lambda _: (200, chat_body(content=STRUCTURED)))
    client = SystemOneClient(
        async_openai_client(stub), model="stub", api="chat_completions", method="structured"
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(ClientCapabilityError) as error:
            client.system_one(state="s", questions={"q": question()})

    assert "AsyncSystemOneClient" in str(error.value)
    assert stub.requests == []
    assert not [item for item in caught if "never awaited" in str(item.message)]


def test_the_async_facade_refuses_a_blocking_client_without_a_request(stub_server):
    import asyncio

    stub = stub_server(chat=lambda _: (200, chat_body(content=STRUCTURED)))
    client = AsyncSystemOneClient(
        openai_client(stub), model="stub", api="chat_completions", method="structured"
    )

    with pytest.raises(ClientCapabilityError) as error:
        asyncio.run(client.system_one(state="s", questions={"q": question()}))

    assert "SystemOneClient" in str(error.value)
    assert stub.requests == []


def test_the_async_facade_still_reads_a_duck_client_that_returns_an_awaitable(stub_server):
    """A duck ``create`` may be an ordinary function returning an awaitable; it is awaited."""
    import asyncio

    class Completions:
        async def create(self, **kwargs):
            return chat_body(content=STRUCTURED)

    class Chat:
        completions = Completions()

    class Client:
        def __init__(self) -> None:
            self.chat = Chat()

    client = AsyncSystemOneClient(Client(), model="stub", method="structured")

    response = asyncio.run(client.system_one(state="s", questions={"q": question()}))

    assert response.answers["q"].choice == "billing"


# ------------------------------------------------------------------ surface order under api="auto"


class _ResponsesAndMessages:
    """A hybrid client: Responses and Messages, no Chat Completions."""

    def __init__(self, responses_status: int = 404) -> None:
        self.responses_status = responses_status
        self.paths: list[str] = []

        outer = self

        class Responses:
            def create(self, **kwargs):
                outer.paths.append("responses")
                raise _Status(outer.responses_status, "the server has no 'responses' route")

        class Messages:
            def create(self, **kwargs):
                outer.paths.append("messages")
                return messages_body(text=STRUCTURED)

        self.responses = Responses()
        self.messages = Messages()


class _Status(Exception):
    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.response = type("R", (), {"status_code": status_code})()


def test_a_hybrid_client_falls_back_to_messages_after_a_404(stub_server):
    """The documented order is Responses, then Chat Completions, then Messages — and the third
    one is tried when the client has no Chat Completions at all."""

    class Client(_ResponsesAndMessages):
        def __init__(self) -> None:
            super().__init__()
            outer = self

            class Messages:
                def create(self, **kwargs):
                    outer.paths.append("messages")
                    return messages_body(text=STRUCTURED)

            self.messages = Messages()

    client = SystemOneClient(Client(), model="stub", api="auto", method="structured", retry=NO_RETRY)

    response = client.system_one(state="s", questions={"q": question()})

    assert response.answers["q"].choice == "billing"
    assert response.debug["api"] == "messages"


def test_a_pinned_label_readout_does_not_move_to_messages(stub_server):
    """Messages carries no logprobs, so a pinned ``logprobs`` has nowhere to go and says so."""

    class Client(_ResponsesAndMessages):
        pass

    client = SystemOneClient(Client(), model="stub", api="auto", method="logprobs", retry=NO_RETRY)

    with pytest.raises(ProviderError) as error:
        client.system_one(
            state="s", questions={"q": Choice(criteria=CRITERIA)}, 
        )

    assert error.value.status_code == 404


def test_a_pinned_label_readout_keeps_a_server_failure_a_provider_error(stub_server):
    """``api="auto"`` is a route decision; a server that failed every attempt is not a logprob verdict.

    Turning the outage into a logprob absence would move the call to the other surface and report a
    capability the provider never claimed to lack.
    """
    stub = stub_server(
        responses=lambda _: (503, {"error": {"message": "busy"}}),
        chat=lambda _: (200, chat_body(content=STRUCTURED)),
    )
    client = SystemOneClient(
        openai_client(stub), model="stub", api="auto", method="logprobs", retry=NO_RETRY
    )

    with pytest.raises(ProviderError) as error:
        client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

    assert error.value.status_code == 503
    assert stub.paths == ["/v1/responses"]


# ------------------------------------------------------------------ caller fields


def test_extra_body_cannot_ask_for_a_stream(stub_server):
    stub = stub_server(chat=lambda _: (200, chat_body(content=STRUCTURED)))
    client = chat_client(stub, extra_body={"stream": True})

    with pytest.raises(JevperError) as error:
        client.system_one(state="s", questions={"q": question()})

    assert "stream" in str(error.value)
    assert stub.requests == []


def test_extra_body_cannot_carry_the_model(stub_server):
    stub = stub_server(chat=lambda _: (200, chat_body(content=STRUCTURED)))
    client = chat_client(stub, extra_body={"model": "other"})

    with pytest.raises(JevperError) as error:
        client.system_one(state="s", questions={"q": question()})

    assert "model" in str(error.value)
    assert stub.requests == []


def test_a_header_spelled_differently_replaces_rather_than_duplicates(stub_server):
    """The SDK merges default and request headers case-sensitively, then sends them case-insensitively.

    A caller who spells ``authorization`` where the client sends ``Authorization`` would otherwise put
    two credentials on one request.
    """
    stub = stub_server(chat=lambda _: (200, chat_body(content=STRUCTURED)))
    client = chat_client(stub, extra_headers={"authorization": "Bearer replacement"})

    client.system_one(state="s", questions={"q": question()})

    values = [value for name, value in stub.header_pairs[0] if name.lower() == "authorization"]
    assert values == ["Bearer replacement"]


def test_ordinary_extra_headers_still_reach_the_wire(stub_server):
    stub = stub_server(chat=lambda _: (200, chat_body(content=STRUCTURED)))
    client = chat_client(stub, extra_headers={"X-Jevper-Probe": "1"})

    client.system_one(state="s", questions={"q": question()})

    values = [value for name, value in stub.header_pairs[0] if name.lower() == "x-jevper-probe"]
    assert values == ["1"]


# ------------------------------------------------------------------ text no request could carry


@pytest.mark.parametrize(
    "state,extra",
    [
        ("\ud800", {}),
        ([{"role": "user", "content": "fine \ud800"}], {}),
        ({"note": "x \ud800"}, {}),
    ],
)
def test_a_state_that_cannot_be_encoded_fails_before_any_request(stub_server, state, extra):
    """A lone surrogate reaches jevper from JSON data; inside the SDK it would read as a provider
    failure after a request that was never made."""

    stub = stub_server(chat=lambda _: (200, chat_body(content=STRUCTURED)))
    client = chat_client(stub, **extra)

    with pytest.raises(JevperError) as error:
        client.system_one(state=state, questions={"q": question()})

    assert "UTF-8" in str(error.value)
    assert stub.requests == []


def test_a_question_key_that_cannot_be_encoded_fails_before_any_request(stub_server):
    stub = stub_server(chat=lambda _: (200, chat_body(content=STRUCTURED)))
    client = chat_client(stub)

    with pytest.raises(InvalidQuestionError) as error:
        client.system_one(
            state="s", questions={"q": Choice(criteria={"\ud800": None, "sales": None})}
        )

    assert "UTF-8" in str(error.value)
    assert stub.requests == []


def test_a_cache_key_that_cannot_be_encoded_fails_before_any_request(stub_server):
    stub = stub_server(chat=lambda _: (200, chat_body(content=STRUCTURED)))
    client = chat_client(stub)

    with pytest.raises(JevperError) as error:
        client.system_one(state="s", questions={"q": question()}, prompt_cache_key="key \ud800")

    assert "UTF-8" in str(error.value)
    assert stub.requests == []


def test_extra_body_that_cannot_be_encoded_fails_before_any_request(stub_server):
    stub = stub_server(chat=lambda _: (200, chat_body(content=STRUCTURED)))
    client = chat_client(stub, extra_body={"metadata": {"note": "x \ud800"}})

    with pytest.raises(JevperError) as error:
        client.system_one(state="s", questions={"q": question()})

    assert "UTF-8" in str(error.value)
    assert stub.requests == []


def test_ordinary_non_ascii_text_still_reaches_the_wire(stub_server):
    stub = stub_server(chat=lambda _: (200, chat_body(content=STRUCTURED)))
    client = chat_client(stub)

    client.system_one(state="café \U0001f600 中文", questions={"q": question()})

    sent = stub.bodies("/chat/completions")[0]
    assert "café \U0001f600 中文" in json.dumps(sent, ensure_ascii=False)


def test_a_header_value_that_cannot_be_encoded_fails_before_any_request(stub_server):
    """A surrogate in a header is the caller's mistake, and httpx's encoder would say so late."""
    stub = stub_server(chat=lambda _: (200, chat_body(content=STRUCTURED)))

    with pytest.raises(JevperError) as error:
        chat_client(stub, extra_headers={"X-Tenant": "tenant \ud800"})

    assert "X-Tenant" in str(error.value)
    assert "ASCII" in str(error.value)
    assert stub.requests == []


def test_a_header_that_could_inject_another_is_refused(stub_server):
    """CRLF in a value is a second header to any transport that does not check; httpx raises instead."""
    stub = stub_server(chat=lambda _: (200, chat_body(content=STRUCTURED)))

    with pytest.raises(JevperError, match="X-Inject"):
        chat_client(stub, extra_headers={"X-Inject": "one\r\nX-Admin: true"})

    assert stub.requests == []


def test_a_header_name_that_is_not_a_header_name_is_refused(stub_server):
    stub = stub_server(chat=lambda _: (200, chat_body(content=STRUCTURED)))

    with pytest.raises(JevperError, match="not a valid HTTP header name"):
        chat_client(stub, extra_headers={"X Probe": "1"})

    assert stub.requests == []


def test_a_non_string_header_value_is_refused(stub_server):
    """httpx takes str or bytes; jevper takes the one a header can be written with, and says so."""
    stub = stub_server(chat=lambda _: (200, chat_body(content=STRUCTURED)))

    with pytest.raises(JevperError, match="must be strings"):
        chat_client(stub, extra_headers={"X-Count": 3})

    assert stub.requests == []


def test_an_ordinary_header_still_reaches_the_wire(stub_server):
    """Validation is not a filter: a legal header is sent exactly as written."""
    stub = stub_server(chat=lambda _: (200, chat_body(content=STRUCTURED)))
    client = chat_client(stub, extra_headers={"X-Trace-Id": "abc-123", "X-Tenant": "eu-1"})

    client.system_one(state="s", questions={"q": question()})

    sent = {name.lower(): value for name, value in stub.header_pairs[0]}
    assert sent["x-trace-id"] == "abc-123"
    assert sent["x-tenant"] == "eu-1"


def test_a_header_with_non_ascii_text_is_refused(stub_server):
    """httpx encodes a header as ASCII, so a non-ASCII value cannot be sent at all."""
    stub = stub_server(chat=lambda _: (200, chat_body(content=STRUCTURED)))

    with pytest.raises(JevperError, match="X-Label"):
        chat_client(stub, extra_headers={"X-Label": "café"})

    assert stub.requests == []


def strict_client(allowed: set[str], answered: Any):
    """A client that types its keyword arguments the way an older SDK release did.

    ``openai`` grew ``prompt_cache_key`` on Chat Completions and Responses after 1.92, which is the
    floor this package declares. A field the SDK has no parameter for cannot be sent as one: the call
    raises ``TypeError`` from inside the SDK, which the caller sees as a provider failure. The body
    channel — ``extra_body``, merged into the request by every version — is where a field of the API
    rather than of the SDK has to go.
    """
    calls: list[dict[str, Any]] = []

    def create(**kwargs: Any) -> Any:
        for name in kwargs:
            if name not in allowed:
                raise TypeError(f"create() got an unexpected keyword argument {name!r}")
        calls.append(kwargs)
        return answered

    class Resource:
        pass

    resource = Resource()
    resource.create = create  # type: ignore[attr-defined]

    class Client:
        pass

    client = Client()
    client.chat = type("Chat", (), {"completions": resource})()  # type: ignore[attr-defined]
    client.responses = resource  # type: ignore[attr-defined]
    return client, calls


CHAT_KEYWORDS = {
    "model", "messages", "response_format", "temperature", "logprobs", "top_logprobs", "grammar",
    "extra_body", "extra_headers",
}
RESPONSES_KEYWORDS = {
    "model", "input", "store", "text", "include", "top_logprobs", "temperature",
    "extra_body", "extra_headers",
}


def test_a_cache_key_reaches_a_chat_sdk_that_has_no_parameter_for_it():
    """The regression this pins: a derived key sent as a typed keyword failed every call on the floor."""
    from fakes import chat_body

    client, calls = strict_client(CHAT_KEYWORDS, chat_body(content=STRUCTURED))

    SystemOneClient(client, model="stub", api="chat_completions", method="structured").system_one(
        state="s", questions={"q": question()}, prompt_cache_key="jevper-test-key"
    )

    assert calls[0]["extra_body"]["prompt_cache_key"] == "jevper-test-key"
    assert "prompt_cache_key" not in calls[0]


def test_a_cache_key_reaches_a_responses_sdk_that_has_no_parameter_for_it():
    """The same field, the same channel, on the surface where the key is a spec'd API field too."""
    from fakes import responses_body

    client, calls = strict_client(RESPONSES_KEYWORDS, responses_body(text=STRUCTURED))

    SystemOneClient(client, model="stub", api="responses", method="structured").system_one(
        state="s", questions={"q": question()}, prompt_cache_key="jevper-test-key"
    )

    assert calls[0]["extra_body"]["prompt_cache_key"] == "jevper-test-key"
    assert "prompt_cache_key" not in calls[0]


def test_a_derived_cache_key_also_travels_the_body_channel():
    """No key of the caller's own: the one jevper derives must reach the wire the same way."""
    from fakes import chat_body

    client, calls = strict_client(CHAT_KEYWORDS, chat_body(content=STRUCTURED))

    SystemOneClient(client, model="stub", api="chat_completions", method="structured").system_one(
        state="s", questions={"q": question()}
    )

    assert calls[0]["extra_body"]["prompt_cache_key"].startswith("jevper-")


def test_a_self_referential_body_is_refused_rather_than_followed(stub_server):
    """A structure that contains itself has no encoding; the walk must end, not loop."""
    stub = stub_server(chat=lambda _: (200, chat_body(content=STRUCTURED)))
    body: dict = {}
    body["self"] = body

    with pytest.raises(JevperError, match="refers to itself"):
        chat_client(stub, extra_body=body).system_one(state="s", questions={"q": question()})

    assert stub.requests == []


def test_a_self_referential_state_is_refused_too(stub_server):
    """The state is rendered before anything else, so its own cycle check answers first — locally."""
    stub = stub_server(chat=lambda _: (200, chat_body(content=STRUCTURED)))
    state: list = []
    state.append(state)
    client = chat_client(stub)

    with pytest.raises(JevperError, match="[Cc]ircular reference"):
        client.system_one(state=state, questions={"q": question()})

    assert stub.requests == []


def test_a_state_nested_past_the_encoder_limit_is_reported_not_crashed(stub_server):
    """CPython's encoder nests as it writes, and where it gives out differs by runtime.

    The same 2000-level state is refused on 3.12 and encoded on 3.14, so a test that asserted an
    answer would be testing the interpreter. What must hold everywhere is the contract: a state
    that cannot be encoded is the caller's problem, reported as the library's own error, before a
    request — never a ``RecursionError`` out of a public call.
    """
    stub = stub_server(chat=lambda _: (200, chat_body(content=STRUCTURED)))
    state: object = "leaf"
    for _ in range(2000):
        state = {"nested": state}
    client = chat_client(stub)

    try:
        response = client.system_one(state=state, questions={"q": question()})
    except JevperError as exc:
        assert "too deeply" in str(exc)
        assert stub.requests == []
        return

    assert response.answers["q"].type == "choice"


def test_a_header_mapping_edited_after_construction_is_not_what_gets_sent(stub_server):
    """What was validated is what is sent: the client keeps a copy, so a later edit cannot slip past."""
    stub = stub_server(chat=lambda _: (200, chat_body(content=STRUCTURED)))
    headers = {"X-Tenant": "ok"}
    client = chat_client(stub, extra_headers=headers)

    headers["X-Tenant"] = "one\r\nX-Admin: true"
    client.system_one(state="s", questions={"q": question()})

    sent = {name.lower(): value for name, value in stub.header_pairs[0]}
    assert sent["x-tenant"] == "ok"
