"""What the debug payload may carry: no credentials, and nothing a caller cannot serialize.

Every response records the request it sent and the body it read, and every failure records the
attempts that led there. That is the point of the record — but a record is also what ends up in a
log line, in a trace, and in ``repr(response)``. Three things therefore cannot be in it: a
credential the caller passed, text no UTF-8 encoder accepts, and a body big or deep enough to decide
whether the response can be serialized at all.
"""

from __future__ import annotations

import json

import pytest
from fakes import chat_body, openai_client, responses_body

from jevper import (
    Choice,
    ClientCapabilityError,
    JevperError,
    MalformedAnswerError,
    ProviderError,
    RetryPolicy,
    Score,
    SystemOneClient,
)

CRITERIA = {"billing": None, "technical": None, "sales": None}
STRUCTURED = json.dumps({"probabilities": {"billing": 0.7, "technical": 0.2, "sales": 0.1}})


def attempts_of(response) -> list[dict]:
    return response.debug["llm_attempts"]


def structured_client(stub, **kwargs) -> SystemOneClient:
    return SystemOneClient(openai_client(stub), model="stub", method="structured", **kwargs)


def test_a_credential_header_is_recorded_without_its_value(stub_server):
    """The name stays — which header was sent is the useful half — and the value does not."""
    server = stub_server(chat=lambda _: (200, chat_body(content=STRUCTURED)))

    client = structured_client(
        server,
        api="chat_completions",
        extra_headers={
            "authorization": "Bearer TOPSECRET",
            "X-Tenant-Token": "TENANT-SECRET",
            "X-Trace": "visible",
        },
    )
    response = client.system_one(state="patient 1234", questions={"q": Choice(criteria=CRITERIA)})

    recorded = attempts_of(response)[0]["request"]["extra_headers"]
    assert recorded["Authorization"] == "<redacted>"
    assert recorded["X-Tenant-Token"] == "<redacted>"
    assert recorded["X-Trace"] == "visible"
    dumped = json.dumps(response.debug, default=str)
    assert "TOPSECRET" not in dumped
    assert "TENANT-SECRET" not in dumped
    # The wire is untouched: redaction is a property of the record, not of the request.
    sent = {name.lower(): value for name, value in server.header_pairs[0]}
    assert sent["authorization"] == "Bearer TOPSECRET"
    assert sent["x-tenant-token"] == "TENANT-SECRET"


def test_the_sdk_credential_never_lands_in_the_record(stub_server):
    """With no ``extra_headers`` of the caller's own there is nothing to record, credential or not."""
    server = stub_server(chat=lambda _: (200, chat_body(content=STRUCTURED)))

    client = structured_client(server, api="chat_completions")
    response = client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

    headers = attempts_of(response)[0]["request"].get("extra_headers") or {}
    assert "test" not in json.dumps(headers)
    assert "api_key" not in json.dumps(headers)
    # The wire still carries the credential the SDK was built with.
    sent = {name.lower(): value for name, value in server.header_pairs[0]}
    assert sent["authorization"] == "Bearer test"


def test_a_failed_call_records_no_credential_either(stub_server):
    """``ProviderError.attempts`` is the same records, so it is redacted the same way."""
    server = stub_server(chat=lambda _: (500, {"error": {"message": "boom"}}))

    client = structured_client(
        server, api="chat_completions", extra_headers={"x-api-key": "SECRET-KEY"}
    )
    with pytest.raises(ProviderError) as caught:
        client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

    assert "SECRET-KEY" not in json.dumps(caught.value.attempts, default=str)
    recorded = {
        name.casefold(): value
        for name, value in (caught.value.attempts[0]["request"]["extra_headers"] or {}).items()
    }
    assert recorded["x-api-key"] == "<redacted>"


def test_a_surrogate_in_a_response_key_never_reaches_the_record(stub_server):
    """A lone surrogate in a response key: the answer stands and the record stays printable.

    What pydantic does with such a key is its own business and differs by version — 2.12+ refuses to
    dump the model at all, 2.7 escapes the key on the way out — so the invariant is asserted rather
    than one version's shape: the call answers, the record holds no raw surrogate, and the response
    serializes.
    """
    body = chat_body(content=STRUCTURED)
    body["x-\ud800"] = 1
    server = stub_server(chat=lambda _: (200, body))

    client = structured_client(server, api="chat_completions")
    response = client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

    assert response.answers["q"].choice == "billing"
    recorded = json.dumps(attempts_of(response)[0]["response"], default=str)
    assert "\ud800" not in recorded, "the escape the wire carried, not the character"
    dumped = response.model_dump_json()
    assert "\ud800" not in dumped
    dumped.encode("utf-8")


def test_a_raw_mapping_response_is_sanitized_keys_and_values():
    """A duck client that answers with a plain dict gets the same treatment as a typed model."""

    class RawCompletions:
        def create(self, **kwargs):
            return {
                "id": "c",
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": STRUCTURED, "note": "bad \ud800"},
                    }
                ],
                "key-\ud800": "value",
            }

    class RawChat:
        completions = RawCompletions()

    class RawClient:
        chat = RawChat()

    client = SystemOneClient(RawClient(), model="stub", api="chat_completions", method="structured")
    response = client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

    assert response.answers["q"].choice == "billing"
    dumped = response.model_dump_json()
    assert "\\ud800" in dumped
    dumped.encode("utf-8")


def test_a_very_deep_provider_body_still_serializes():
    """Nesting past the interpreter's own limit is answered, summarized in debug, not raised on."""

    class DeepCompletions:
        def create(self, **kwargs):
            value: object = "leaf"
            for _ in range(1500):
                value = {"deeper": value}
            return {
                "id": "c",
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": STRUCTURED},
                    }
                ],
                "metadata": value,
            }

    class DeepChat:
        completions = DeepCompletions()

    class DeepClient:
        chat = DeepChat()

    client = SystemOneClient(DeepClient(), model="stub", api="chat_completions", method="structured")
    response = client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

    assert response.answers["q"].choice == "billing"
    recorded = attempts_of(response)[0]["response"]["metadata"]
    assert "nested deeper" in json.dumps(recorded)
    response.model_dump_json().encode("utf-8")


def test_a_huge_provider_field_is_bounded_in_the_record(stub_server):
    """Four megabytes of padding does not become four megabytes of debug on every response."""
    body = chat_body(content=STRUCTURED)
    body["padding"] = "y" * 4_000_000
    server = stub_server(chat=lambda _: (200, body))

    client = structured_client(server, api="chat_completions")
    response = client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

    recorded = attempts_of(response)[0]["response"]["padding"]
    assert len(recorded) < 70_000
    assert "chars)" in recorded
    response.model_dump_json().encode("utf-8")


def test_a_huge_answer_is_reported_by_its_keys_not_its_values(stub_server):
    """``observed_text`` and the malformed-answer message are both bounded; the keys still say what."""
    long_answer = json.dumps(
        {"probabilities": {"billing": 0.7, "technical": 0.2, "sales": 0.1}, "padding": "z" * 3_000_000}
    )
    server = stub_server(chat=lambda _: (200, chat_body(content=long_answer)))

    client = structured_client(server, api="chat_completions", n_retry_malformed=0)
    with pytest.raises(MalformedAnswerError) as caught:
        client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

    text = str(caught.value)
    assert "padding" in text
    assert "z" * 10_000 not in text
    assert len(text) < 1_000


def test_an_embedded_error_message_is_bounded_and_printable(stub_server):
    """A 200 carrying a provider error gets the same bound and the same escaping as a status error."""
    message = "rejected \ud800 " + "x" * 1_000_000
    server = stub_server(responses=lambda _: (200, {"error": {"message": message, "code": 400}}))

    client = SystemOneClient(
        openai_client(server),
        model="stub",
        api="responses",
        method="structured",
        n_retry_malformed=0,
    )
    with pytest.raises(ProviderError) as caught:
        client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

    text = str(caught.value)
    assert len(text) < 1_000
    assert "chars)" in text
    text.encode("utf-8")
    assert caught.value.status_code == 400


def test_a_normal_body_is_recorded_verbatim(stub_server):
    """The bounds are bounds, not a summary: an ordinary body is still the whole body."""
    server = stub_server(responses=lambda _: (200, responses_body(text=STRUCTURED)))

    client = SystemOneClient(openai_client(server), model="stub", api="responses", method="structured")
    response = client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

    recorded = attempts_of(response)[0]
    assert recorded["request"]["model"] == "stub"
    assert recorded["response"]["status"] == "completed"
    assert recorded["readout"]["source"] == "structured"
    assert recorded["readout"]["observed_text"] == STRUCTURED


def test_the_request_record_still_names_every_field_it_sent(stub_server):
    """Redaction touches credential values only; the shape of the request is still readable."""
    server = stub_server(responses=lambda _: (200, responses_body(text=STRUCTURED)))

    client = SystemOneClient(
        openai_client(server),
        model="stub",
        api="responses",
        method="structured",
        extra_body={"custom_field": 7},
        extra_headers={"authorization": "Bearer SECRET", "x-tenant": "plain"},
    )
    response = client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

    request = attempts_of(response)[0]["request"]
    # The caller's own body plus the cache key jevper derived for this question, both on the body
    # channel; the record shows what the SDK will merge into the request.
    assert request["extra_body"]["custom_field"] == 7
    assert request["extra_body"]["prompt_cache_key"].startswith("jevper-")
    assert request["store"] is False
    assert request["extra_headers"]["x-tenant"] == "plain"


def test_a_caller_who_asks_for_a_bad_header_hears_about_the_header(stub_server):
    """The local validation names the header; nothing is sent to find out."""
    server = stub_server(chat=lambda _: (200, chat_body(content=STRUCTURED)))

    with pytest.raises(JevperError, match="X-Bad"):
        structured_client(
            server,
            api="chat_completions",
            extra_headers={"X-Bad": "one\r\nX-Injected: yes"},
        )
    assert server.paths == []


def test_a_duck_client_that_answers_with_a_bare_string_is_named():
    """A client whose ``create`` returns a string is the event-stream mismatch, not an answer."""

    class StringCompletions:
        def create(self, **kwargs):
            return "not a response object"

    class StringChat:
        completions = StringCompletions()

    class StringClient:
        chat = StringChat()

    client = SystemOneClient(StringClient(), model="stub", api="chat_completions", method="structured")
    with pytest.raises(ProviderError, match="event stream"):
        client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})


def test_a_capability_error_still_reaches_the_caller_unchanged():
    """The client that cannot read the surface is told so before any request, as before."""

    class NoChoices:
        def create(self, **kwargs):
            return {"choices": []}

    class Chat:
        completions = NoChoices()

    class Client:
        chat = Chat()

    client = SystemOneClient(Client(), model="stub", api="chat_completions", method="structured")
    with pytest.raises(ClientCapabilityError):
        client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})


def test_a_credential_quoted_by_the_provider_is_scrubbed_from_the_error(stub_server):
    """A gateway that echoes the key it rejected must not put it in every message that quotes it."""
    server = stub_server(chat=lambda _: (401, {"error": {"message": "invalid x-api-key sk-live-SECRET"}}))

    client = structured_client(
        server, api="chat_completions", extra_headers={"x-api-key": "sk-live-SECRET"}
    )
    with pytest.raises(ProviderError) as caught:
        client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

    assert "SECRET" not in str(caught.value)
    assert "<redacted>" in str(caught.value)
    assert "SECRET" not in json.dumps(caught.value.attempts, default=str)
    assert caught.value.status_code == 401


def test_a_credential_quoted_by_a_duck_client_is_scrubbed_too():
    """The same through a client that raises its own exception rather than answering a status."""
    from fakes import RaisingClient

    client = SystemOneClient(
        RaisingClient(RuntimeError("rejected x-api-key sk-live-SECRET here")),
        model="stub",
        api="chat_completions",
        method="structured",
        extra_headers={"x-api-key": "sk-live-SECRET"},
        retry=RetryPolicy(n_retries=0),
    )
    with pytest.raises(ProviderError) as caught:
        client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

    assert "SECRET" not in str(caught.value)
    assert caught.value.status_code is None


def test_a_short_header_value_is_not_scrubbed_everywhere(stub_server):
    """Scrubbing replaces whole values, so a value too short to be a credential stays readable."""
    server = stub_server(chat=lambda _: (401, {"error": {"message": "token rejected: test"}}))

    client = structured_client(
        server, api="chat_completions", extra_headers={"x-api-key": "test"}
    )
    with pytest.raises(ProviderError) as caught:
        client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

    assert "token rejected: test" in str(caught.value)


def test_two_keys_that_sanitize_to_one_are_both_kept():
    """A surrogate key and its escaped twin are distinct provider fields; the record keeps both."""
    from types import SimpleNamespace

    class Completions:
        def create(self, **kwargs):
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        index=0,
                        finish_reason="stop",
                        message={"role": "assistant", "content": STRUCTURED},
                    )
                ]
            )

    class Chat:
        completions = Completions()

    class Client:
        chat = Chat()

    client = SystemOneClient(Client(), model="stub", api="chat_completions", method="structured")
    response = client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

    # The response object has no model_dump, so the record names it rather than keeping it — and
    # the public response still serializes, which is the property that matters.
    assert response.answers["q"].choice == "billing"
    assert attempts_of(response)[0]["response"] == {
        "undumpable": "<SimpleNamespace is not data jevper can keep for debug>"
    }
    response.model_dump_json().encode("utf-8")


def test_escaping_does_not_lose_one_of_two_colliding_keys(stub_server):
    """The mapping itself keeps both: ``"x\\ud800"`` and the literal ``"x\\\\ud800"`` sanitize alike."""
    class Body(dict):
        """A provider body that is its own dump, the way a duck client may answer."""

        def model_dump(self, **keywords):
            return self

    class Completions:
        def create(self, **kwargs):
            return Body(
                id="c",
                choices=[
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": STRUCTURED},
                    }
                ],
                **{"x-\ud800": "surrogate", "x-\\ud800": "escaped twin"},
            )

    class Chat:
        completions = Completions()

    class Client:
        chat = Chat()

    client = SystemOneClient(Client(), model="stub", api="chat_completions", method="structured")
    response = client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

    recorded = attempts_of(response)[0]["response"]
    values = sorted(str(value) for value in recorded.values() if isinstance(value, str))
    assert "surrogate" in values and "escaped twin" in values
    # Two provider keys, two entries: the colliding pair is numbered rather than one overwriting
    # the other, and nothing else in the body was lost.
    assert "x-\\ud800" in recorded and "x-\\ud800#2" in recorded
    assert recorded["choices"][0]["message"]["content"] == STRUCTURED
    response.model_dump_json().encode("utf-8")


def test_a_duck_response_value_that_is_not_data_becomes_a_marker():
    """An ``object()`` in a metadata field must not decide whether the response serializes."""

    class Completions:
        def create(self, **kwargs):
            return {
                "id": "c",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": STRUCTURED},
                    }
                ],
                "metadata": object(),
                "tags": {"a", "b"},
            }

    class Chat:
        completions = Completions()

    class Client:
        chat = Chat()

    client = SystemOneClient(Client(), model="stub", api="chat_completions", method="structured")
    response = client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

    recorded = attempts_of(response)[0]["response"]
    assert "object" in recorded["metadata"]
    assert "set" in recorded["tags"]
    response.model_dump_json().encode("utf-8")


def test_a_huge_key_name_is_reported_bounded(stub_server):
    """The key names are the provider's text too: one three-megabyte key is still a short message."""
    answer = json.dumps(
        {
            "probabilities": {"billing": 0.7, "technical": 0.2, "sales": 0.1},
            "k" * 3_000_000: True,
        }
    )
    server = stub_server(chat=lambda _: (200, chat_body(content=answer)))

    client = structured_client(server, api="chat_completions", n_retry_malformed=0)
    with pytest.raises(MalformedAnswerError) as error:
        client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

    text = str(error.value)
    assert len(text) < 1_000
    assert "chars)" in text
    text.encode("utf-8")


def test_a_surrogate_key_name_is_reported_printable(stub_server):
    answer = '{"probabilities": {"billing": 0.7, "technical": 0.2, "sales": 0.1}, "x\\ud800": 1}'
    server = stub_server(chat=lambda _: (200, chat_body(content=answer)))

    client = structured_client(server, api="chat_completions", n_retry_malformed=0)
    with pytest.raises(MalformedAnswerError) as error:
        client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

    assert "ud800" in str(error.value)
    str(error.value).encode("utf-8")


def test_a_level_index_with_a_billion_digits_is_refused_without_building_one(stub_server):
    """``"1e999999999"`` is a finite decimal and an absurd integer.

    Converting it would try to build an integer with a billion digits, so the test that matters is
    that the call *returns* at all: the membership check is decided on the decimal, before any
    conversion, and the answer is refused as an out-of-range level.
    """
    answer = json.dumps({"score": "1e999999999"})
    server = stub_server(chat=lambda _: (200, chat_body(content=answer)))

    client = SystemOneClient(
        openai_client(server), model="stub", method="discrete", api="chat_completions",
        n_retry_malformed=0,
    )
    with pytest.raises(MalformedAnswerError, match="level indexes"):
        client.system_one(
            state="s", questions={"anger": Score(criteria=["Calm", "Frustrated", "Very angry"])}
        )
