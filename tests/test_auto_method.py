"""``method="auto"``: picking a method, falling back when logprobs are unavailable, remembering it."""

from __future__ import annotations

import asyncio
import json

import pytest
from fakes import async_openai_client, chat_body, openai_client, responses_body

from jevper import (
    AsyncSystemOneClient,
    Choice,
    JevperError,
    LabelReadoutError,
    ProviderError,
    ReasoningConfig,
    RetryPolicy,
    SystemOneClient,
)

CRITERIA = {"billing": None, "technical": None, "sales": None}
CHOICE_LOGS = [("A", -0.12), ("B", -2.47), ("C", -3.48)]
STRUCTURED = json.dumps({"probabilities": {"billing": 0.6, "technical": 0.3, "sales": 0.1}})
REJECTION = {
    "error": {
        "message": "logprobs are not supported with reasoning models.",
        "type": "invalid_request_error",
        "param": "include",
        "code": "unsupported_parameter",
    }
}
NO_RETRY = RetryPolicy(n_retries=1, base_delay=0.0)
# OpenRouter's Responses API, verbatim: the refusal names the `include` path and its allowed values,
# and never the word "logprob".
INCLUDE_REJECTION = {
    "error": {"code": "invalid_prompt", "message": "Invalid Responses API request"},
    "metadata": {
        "raw": '[{"code": "invalid_value", "values": ["file_search_call.results", '
        '"reasoning.encrypted_content"], "path": ["include", 0], "message": "Invalid option: expected '
        'one of \\"file_search_call.results\\"|\\"reasoning.encrypted_content\\""}]'
    },
}


def hostile(body):
    """A provider that refuses the logprob fields and answers in JSON instead."""
    if body.get("logprobs"):
        return 400, REJECTION
    return 200, chat_body(content=STRUCTURED)


def one_candidate(body):
    """A provider that returns the sampled token's logprob with no alternatives."""
    if body.get("logprobs"):
        return 200, chat_body(content="A", logprobs=[("A", -0.12)])
    return 200, chat_body(content=STRUCTURED)


def test_auto_uses_logprobs_when_the_provider_has_them(stub_server):
    stub = stub_server(chat=lambda _: (200, chat_body(content="A", logprobs=CHOICE_LOGS)))
    client = SystemOneClient(openai_client(stub), model="stub", api="chat_completions")

    response = client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

    assert response.answers["q"].choice == "billing"
    assert response.answers["q"].probabilities["billing"] == pytest.approx(0.884873983, abs=1e-9)
    assert response.debug["method"] == "logprobs"
    assert response.debug["methods"] == {"q": "logprobs"}
    assert len(stub.bodies("/chat/completions")) == 1


def test_auto_falls_back_when_the_provider_rejects_logprobs(stub_server):
    stub = stub_server(chat=hostile)
    client = SystemOneClient(openai_client(stub), model="stub", api="chat_completions")

    response = client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

    assert response.answers["q"].choice == "billing"
    assert response.answers["q"].probabilities == {"billing": 0.6, "technical": 0.3, "sales": 0.1}
    assert response.debug["methods"] == {"q": "structured"}
    assert any("rejected the logprob request" in reason for reason in response.debug["retry_reasons"])

    sent = stub.bodies("/chat/completions")
    assert len(sent) == 2  # the rejected attempt is recorded in llm_attempts, not in n_calls
    assert sent[0]["logprobs"] is True
    assert "logprobs" not in sent[1] and "response_format" in sent[1]
    assert response.usage.n_calls == 1


def test_auto_remembers_the_fallback_for_the_next_call(stub_server):
    stub = stub_server(chat=hostile)
    client = SystemOneClient(openai_client(stub), model="stub", api="chat_completions")

    client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})
    assert len(stub.bodies("/chat/completions")) == 2

    response = client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

    # The verdict is remembered: the second call pays nothing to rediscover it.
    assert len(stub.bodies("/chat/completions")) == 3
    assert "logprobs" not in stub.bodies("/chat/completions")[-1]
    assert response.debug["methods"] == {"q": "structured"}


def test_auto_resolves_per_model(stub_server):
    def script(body):
        if body["model"] == "hostile":
            return hostile(body)
        return 200, chat_body(content="A", logprobs=CHOICE_LOGS)

    stub = stub_server(chat=script)
    client = SystemOneClient(openai_client(stub), model="stub", api="chat_completions")

    client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)}, model="hostile")
    response = client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)}, model="stub")

    assert response.debug["methods"] == {"q": "logprobs"}


def test_auto_falls_back_when_the_provider_drops_logprobs(stub_server):
    stub = stub_server(chat=lambda _: (200, chat_body(content=STRUCTURED)))
    client = SystemOneClient(openai_client(stub), model="stub", api="chat_completions")

    response = client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

    assert response.debug["methods"] == {"q": "structured"}
    # The answer and the fallback: a corrective retry cannot conjure logprobs the provider lacks.
    assert len(stub.bodies("/chat/completions")) == 2


def test_auto_falls_back_when_only_the_answer_token_has_a_logprob(stub_server):
    stub = stub_server(chat=one_candidate)
    client = SystemOneClient(openai_client(stub), model="stub", api="chat_completions")

    response = client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

    assert response.debug["methods"] == {"q": "structured"}
    assert response.answers["q"].probabilities == {"billing": 0.6, "technical": 0.3, "sales": 0.1}
    assert len(stub.bodies("/chat/completions")) == 2


def test_auto_answers_a_wide_choice_in_json(stub_server):
    wide = {f"k{index}": None for index in range(27)}
    body = chat_body(content=json.dumps({"probabilities": {key: 1 / 27 for key in wide}}))
    stub = stub_server(chat=lambda _: (200, body))
    client = SystemOneClient(openai_client(stub), model="stub", api="chat_completions")

    response = client.system_one(state="s", questions={"q": Choice(criteria=wide)})

    # 27 options cannot be told apart by one label token, so auto never asks for one.
    assert response.debug["methods"] == {"q": "structured"}
    sent = stub.bodies("/chat/completions")[0]
    assert "logprobs" not in sent and "response_format" in sent


def test_auto_keeps_logprobs_when_a_transient_failure_recovers(stub_server):
    calls: list[dict] = []

    def script(body):
        calls.append(body)
        if len(calls) == 1:
            return 500, {"error": {"message": "temporarily unavailable"}}
        return 200, chat_body(content="A", logprobs=CHOICE_LOGS)

    stub = stub_server(chat=script)
    client = SystemOneClient(
        openai_client(stub), model="stub", api="chat_completions", retry=NO_RETRY
    )

    response = client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

    assert response.debug["methods"] == {"q": "logprobs"}
    assert response.usage.n_retries == 1


def test_auto_falls_back_after_server_errors_without_remembering(stub_server):
    def script(body):
        if body.get("logprobs"):
            return 500, {"error": {"message": "temporarily unavailable"}}
        return 200, chat_body(content=STRUCTURED)

    stub = stub_server(chat=script)
    client = SystemOneClient(
        openai_client(stub), model="stub", api="chat_completions", retry=NO_RETRY
    )

    response = client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

    assert response.debug["methods"] == {"q": "structured"}
    assert response.usage.n_retries == 1

    # A server error is not a verdict on the provider: the next call tries logprobs again.
    client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})
    assert stub.bodies("/chat/completions")[3]["logprobs"] is True


def test_auto_falls_back_inside_two_step_reasoning(stub_server):
    def script(body):
        if body.get("logprobs"):
            return 400, REJECTION
        if "response_format" in body:
            return 200, chat_body(content=STRUCTURED)
        return 200, chat_body(content="The customer is upset about a duplicate charge.")

    stub = stub_server(chat=script)
    client = SystemOneClient(
        openai_client(stub),
        model="stub",
        api="chat_completions",
        reasoning=ReasoningConfig(effort="low"),
    )

    response = client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

    assert response.debug["reasoning_mode"] == "two_step"
    assert response.debug["methods"] == {"q": "structured"}
    # Analysis, rejected answer, analysis again, structured answer. The rejected attempt is not in
    # n_calls (it produced no usage), so the request count is what shows the cost.
    assert len(stub.bodies("/chat/completions")) == 4
    assert response.usage.n_calls == 3


def test_auto_falls_back_on_the_responses_surface(stub_server):
    """The default surface for an OpenAI client: the rejected field there is `include`, not `logprobs`."""

    def script(body):
        if body.get("include") or body.get("top_logprobs"):
            return 400, REJECTION
        return 200, responses_body(text=STRUCTURED)

    stub = stub_server(responses=script)
    client = SystemOneClient(openai_client(stub), model="stub")  # api="auto" prefers responses

    response = client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

    assert response.debug["api"] == "responses"
    assert response.debug["methods"] == {"q": "structured"}
    assert response.answers["q"].choice == "billing"

    sent = stub.bodies("/responses")
    assert len(sent) == 2
    assert sent[0]["include"] == ["message.output_text.logprobs"]
    assert "include" not in sent[1] and "top_logprobs" not in sent[1]


def test_auto_falls_back_when_a_provider_refuses_the_include_path(stub_server):
    """OpenRouter's Responses API refuses the logprob includable without ever saying "logprob"."""

    def script(body):
        if body.get("include") or body.get("top_logprobs"):
            return 400, INCLUDE_REJECTION
        return 200, responses_body(text=STRUCTURED)

    stub = stub_server(responses=script)
    client = SystemOneClient(openai_client(stub), model="stub")  # api="auto" prefers responses

    response = client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

    assert response.debug["api"] == "responses"
    assert response.debug["methods"] == {"q": "structured"}
    assert response.answers["q"].choice == "billing"

    # Refusing the carrier is a fact about the surface, so the next call does not pay for it again.
    client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})
    assert stub.bodies("/responses")[2].get("include") is None


def test_serial_questions_share_the_discovery(stub_server):
    """A question that has not started yet takes the verdict: one probe, not one per question."""

    stub = stub_server(chat=hostile)
    client = SystemOneClient(
        openai_client(stub), model="stub", api="chat_completions", max_concurrency=1
    )

    response = client.system_one(
        state="s", questions={name: Choice(criteria=CRITERIA) for name in ("a", "b", "c")}
    )

    assert response.debug["methods"] == {"a": "structured", "b": "structured", "c": "structured"}
    bodies = stub.bodies("/chat/completions")
    assert sum(1 for body in bodies if body.get("logprobs")) == 1
    assert len(bodies) == 4


def test_async_client_auto_falls_back(stub_server):
    stub = stub_server(chat=hostile)
    client = AsyncSystemOneClient(async_openai_client(stub), model="stub", api="chat_completions")

    response = asyncio.run(
        client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})
    )

    assert response.answers["q"].choice == "billing"
    assert response.debug["methods"] == {"q": "structured"}
    assert len(stub.bodies("/chat/completions")) == 2


def test_pinned_method_reports_no_methods_key(stub_server):
    stub = stub_server(chat=lambda _: (200, chat_body(content="A", logprobs=CHOICE_LOGS)))
    client = SystemOneClient(
        openai_client(stub), model="stub", method="logprobs", api="chat_completions"
    )

    response = client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

    assert response.debug["method"] == "logprobs"
    assert "methods" not in response.debug


def test_pinned_logprobs_without_alternatives_is_an_error_not_certainty(stub_server):
    stub = stub_server(chat=lambda _: (200, chat_body(content="A", logprobs=[("A", -0.12)])))
    client = SystemOneClient(
        openai_client(stub), model="stub", method="logprobs", api="chat_completions"
    )

    with pytest.raises(LabelReadoutError) as error:
        client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

    assert "not a distribution over the options" in str(error.value)
    assert "method='structured'" in str(error.value)
    assert len(stub.bodies("/chat/completions")) == 1


def test_pinned_logprobs_absent_is_an_error_without_a_corrective_retry(stub_server):
    stub = stub_server(chat=lambda _: (200, chat_body(content="A")))
    client = SystemOneClient(
        openai_client(stub), model="stub", method="logprobs", api="chat_completions"
    )

    with pytest.raises(LabelReadoutError) as error:
        client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

    assert "no logprobs returned for the answer token" in str(error.value)
    assert "method='structured'" in str(error.value)
    assert len(stub.bodies("/chat/completions")) == 1


def test_pinned_logprobs_rejection_is_still_a_provider_error(stub_server):
    stub = stub_server(chat=lambda _: (400, REJECTION))
    client = SystemOneClient(
        openai_client(stub),
        model="stub",
        method="logprobs",
        api="chat_completions",
        retry=RetryPolicy(n_retries=0),
    )

    with pytest.raises(ProviderError) as error:
        client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

    assert "logprobs are not supported with reasoning models" in str(error.value)
    assert len(stub.bodies("/chat/completions")) == 1


def test_method_selection_is_validated():
    with pytest.raises(JevperError) as error:
        SystemOneClient(object(), model="m", method="bogus")

    assert "must be one of" in str(error.value)
    assert "'auto'" in str(error.value)


RANGE_REJECTION = {
    "error": {
        "message": "Invalid 'top_logprobs': integer must be between 0 and 5, but got 20.",
        "type": "invalid_request_error",
        "param": "top_logprobs",
        "code": "invalid_value",
    }
}


class AttributeRejection(Exception):
    """A 400 that names the logprob field in its attributes, not in its message."""

    status_code = 400
    param = "logprobs"
    code = "unsupported_parameter"


class AttrRejectingClient:
    """A duck-typed provider that refuses logprobs and answers in JSON otherwise."""

    def __init__(self) -> None:
        self.requests: list[dict] = []
        outer = self

        class Completions:
            def create(self, **kwargs):
                outer.requests.append(kwargs)
                if kwargs.get("logprobs"):
                    raise AttributeRejection("Request rejected.")
                return chat_body(content=STRUCTURED)

        class Chat:
            completions = Completions()

        self.chat = Chat()


def range_hostile(body):
    """A server whose top_logprobs cap is lower than the default: it refuses the value, not the field."""
    if body.get("logprobs"):
        return 400, RANGE_REJECTION
    return 200, chat_body(content=STRUCTURED)


def test_auto_does_not_remember_a_value_rejection(stub_server):
    stub = stub_server(chat=range_hostile)
    client = SystemOneClient(openai_client(stub), model="stub", api="chat_completions", retry=NO_RETRY)

    first = client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})
    second = client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

    assert first.answers["q"].choice == "billing"
    assert second.answers["q"].choice == "billing"
    assert first.debug["methods"] == {"q": "structured"}
    assert second.debug["methods"] == {"q": "structured"}
    assert "not remembered" in first.debug["retry_reasons"][0]

    bodies = stub.bodies("/chat/completions")
    assert len(bodies) == 4
    assert bodies[2]["top_logprobs"] == 20  # the second call tried logprobs again


def test_auto_remembers_a_rejection_that_names_the_field_only_in_param():
    stub = AttrRejectingClient()
    client = SystemOneClient(stub, model="stub", api="chat_completions", retry=NO_RETRY)

    first = client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})
    second = client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

    assert first.answers["q"].choice == "billing"
    assert second.answers["q"].choice == "billing"
    assert first.debug["methods"] == {"q": "structured"}
    assert len(stub.requests) == 3
    assert "logprobs" not in stub.requests[2]  # the verdict was remembered
