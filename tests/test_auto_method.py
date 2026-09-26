"""``method="auto"``: picking a method, falling back when logprobs are unavailable, remembering it."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

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
    # Analysis, rejected answer, structured answer. The rejected attempt is not in n_calls (it
    # produced no usage), and the analysis is bought once: the fallback re-asks how to read the
    # answer, not what the question is about.
    assert len(stub.bodies("/chat/completions")) == 3
    assert response.usage.n_calls == 2


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


def test_a_rejection_moves_the_readout_to_the_other_surface(stub_server):
    """llama.cpp's Responses shim refuses the logprob fields; Chat Completions carries them.

    A refusal is about the surface that made it — the same question on the other surface is worth one
    request, because that is where the distribution is. llama.cpp answers
    ``400 top_logprobs requires logprobs to be set to true`` on ``/v1/responses`` and a full
    distribution on ``/v1/chat/completions``.
    """
    stub = stub_server(
        chat=lambda _: (200, chat_body(content="A", logprobs=CHOICE_LOGS)),
        responses=lambda body: (400, INCLUDE_REJECTION)
        if body.get("include")
        else (200, responses_body(text=STRUCTURED)),
    )
    client = SystemOneClient(openai_client(stub), model="stub")  # api="auto" prefers responses

    response = client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

    assert response.debug["api"] == "chat_completions"
    assert response.debug["methods"] == {"q": "logprobs"}
    assert stub.paths == ["/v1/responses", "/v1/chat/completions"]
    assert response.answers["q"].probabilities["billing"] == pytest.approx(0.884873983, abs=1e-9)


def test_a_surface_that_delivers_again_is_not_written_off(stub_server):
    """The verdict moves the readout away; it does not condemn the surface. A distribution clears it."""
    calls = {"probes": 0}

    def responses_script(body):
        if body.get("include") or body.get("top_logprobs"):
            calls["probes"] += 1
            if calls["probes"] == 1:
                return 200, responses_body(text="A")  # answered, carrying no logprobs
            return 200, responses_body(text="A", logprobs=CHOICE_LOGS)  # this time it does
        return 200, responses_body(text=STRUCTURED)

    stub = stub_server(
        chat=lambda _: (200, chat_body(content="A", logprobs=CHOICE_LOGS)),
        responses=responses_script,
    )
    client = SystemOneClient(openai_client(stub), model="stub")
    question = {"q": Choice(criteria=CRITERIA)}

    client.system_one(state="s", questions=question)  # responses withholds: the readout moves to chat
    assert stub.paths == ["/v1/responses", "/v1/chat/completions"]

    # A pinned call proves the surface can carry a distribution after all, which clears the verdict.
    client.system_one(state="s", questions=question, api="responses", method="logprobs")
    assert stub.paths[-1] == "/v1/responses"

    # So the next auto call opens there again, rather than staying away for the client's life — and
    # it needs only that one request, because the surface now carries the distribution.
    client.system_one(state="s", questions=question)
    assert stub.paths[-1] == "/v1/responses"
    assert stub.paths.count("/v1/responses") == 3
    assert stub.paths.count("/v1/chat/completions") == 1


def test_a_refusal_is_remembered_even_when_the_surface_cannot_move(stub_server):
    """Native reasoning keeps the surface, so the only verdict left is the provider's own."""

    def script(body):
        # Native reasoning also sends an `include` (for encrypted content); only the logprob
        # includable is refused here, which is what the real rejection names.
        if "message.output_text.logprobs" in (body.get("include") or []):
            return 400, INCLUDE_REJECTION
        return 200, responses_body(text=STRUCTURED)

    stub = stub_server(responses=script)
    client = SystemOneClient(
        openai_client(stub),
        model="stub",
        reasoning=ReasoningConfig(mode="native", effort="low"),
    )
    question = {"q": Choice(criteria=CRITERIA)}

    client.system_one(state="s", questions=question)
    client.system_one(state="s", questions=question)

    # The refusal names the surface's carrier, so it is a capability verdict: no second probe.
    assert "message.output_text.logprobs" not in (
        stub.bodies("/responses")[2].get("include") or []
    )
    assert stub.paths.count("/v1/responses") == 3  # probe, structured, structured


def test_coming_back_to_a_marked_surface_does_not_ask_for_logprobs_again(stub_server):
    """The method verdict is keyed by surface: returning to one that withheld logprobs stays in JSON."""

    def responses_script(body):
        if body.get("include") or body.get("top_logprobs"):
            return 200, responses_body(text="A")  # answered, carrying no logprobs
        return 200, responses_body(text=STRUCTURED)

    stub = stub_server(responses=responses_script)  # no chat route: the move cannot land
    client = SystemOneClient(openai_client(stub), model="stub")

    response = client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

    # responses withholds → chat is tried and 404s → back on responses, answered in JSON, not probed.
    assert stub.paths == ["/v1/responses", "/v1/chat/completions", "/v1/responses"]
    assert response.debug["api"] == "responses"
    assert response.answers["q"].choice == "billing"


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


STRUCTURAL_REJECTION = {
    "error": {
        "message": "top_logprobs requires logprobs to be set to true",
        "type": "invalid_request_error",
        "code": 400,
    }
}

CONDITIONAL_VALUE_REJECTION = {
    "error": {
        "message": "top_logprobs requires a value between 0 and 20",
        "type": "invalid_request_error",
    }
}


def structural_hostile(body):
    """llama.cpp's shape: the field is refused outright, in the words its Responses shim uses."""
    if body.get("logprobs"):
        return 400, STRUCTURAL_REJECTION
    return 200, chat_body(content=STRUCTURED)


def conditionally_hostile(body):
    """A server that states its cap as a requirement: the field is fine, the value is not."""
    if body.get("logprobs"):
        return 400, CONDITIONAL_VALUE_REJECTION
    return 200, chat_body(content=STRUCTURED)


def test_auto_remembers_a_refusal_that_states_a_condition(stub_server):
    stub = stub_server(chat=structural_hostile)
    client = SystemOneClient(openai_client(stub), model="stub", api="chat_completions", retry=NO_RETRY)

    first = client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})
    second = client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

    assert first.answers["q"].choice == "billing"
    assert second.debug["methods"] == {"q": "structured"}

    bodies = stub.bodies("/chat/completions")
    assert len(bodies) == 3
    assert "logprobs" not in bodies[2]  # the verdict was remembered


def test_a_value_bound_stated_as_a_requirement_is_not_remembered(stub_server):
    stub = stub_server(chat=conditionally_hostile)
    client = SystemOneClient(openai_client(stub), model="stub", api="chat_completions", retry=NO_RETRY)

    first = client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})
    client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

    assert first.answers["q"].choice == "billing"
    assert "not remembered" in first.debug["retry_reasons"][0]

    bodies = stub.bodies("/chat/completions")
    assert len(bodies) == 4
    assert bodies[2]["top_logprobs"] == 20  # a bounded value is not a missing capability


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


def drops_logprobs(body):
    """A provider that answers every request but never returns logprobs for the answer token."""
    if body.get("logprobs"):
        return 200, chat_body(content="A")
    return 200, chat_body(content=STRUCTURED)


def test_one_absent_logprob_response_does_not_pin_the_method(stub_server):
    """One response without readable logprobs is a bad minute, not a verdict on the provider."""
    stub = stub_server(chat=drops_logprobs)
    client = SystemOneClient(openai_client(stub), model="stub", api="chat_completions")

    client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})
    assert len(stub.bodies("/chat/completions")) == 2  # the probe, then the JSON fallback

    response = client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

    assert stub.bodies("/chat/completions")[2]["logprobs"] is True  # logprobs are tried again
    assert response.debug["methods"] == {"q": "structured"}


def test_two_absent_logprob_responses_pin_the_method(stub_server):
    stub = stub_server(chat=drops_logprobs)
    client = SystemOneClient(openai_client(stub), model="stub", api="chat_completions")

    client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})
    client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})
    assert len(stub.bodies("/chat/completions")) == 4

    response = client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

    assert len(stub.bodies("/chat/completions")) == 5  # no probe this time
    assert "logprobs" not in stub.bodies("/chat/completions")[-1]
    assert response.debug["methods"] == {"q": "structured"}


def test_a_readable_distribution_retires_earlier_absences(stub_server):
    """A success in between proves the provider can do it, so the count starts over."""
    probes = 0

    def script(body):
        nonlocal probes
        if not body.get("logprobs"):
            return 200, chat_body(content=STRUCTURED)
        probes += 1
        if probes == 2:
            return 200, chat_body(content="A", logprobs=CHOICE_LOGS)
        return 200, chat_body(content="A")

    stub = stub_server(chat=script)
    client = SystemOneClient(openai_client(stub), model="stub", api="chat_completions")

    client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})  # absent
    client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})  # readable: resets
    client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})  # absent again
    client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

    # Two absences separated by a readable distribution are still only one absence in a row.
    assert stub.bodies("/chat/completions")[5]["logprobs"] is True


def test_auto_switches_to_chat_when_the_server_has_no_responses_route(stub_server):
    """A server with no ``/v1/responses`` route: the SDK object exposes ``responses.create`` regardless.

    The stub answers 404 for a path it has no script for, exactly as such a server does, so the default
    ``api="auto"`` must notice and answer on the surface the server actually implements.
    """
    stub = stub_server(chat=lambda _: (200, chat_body(content="A", logprobs=CHOICE_LOGS)))
    client = SystemOneClient(openai_client(stub), model="stub")  # api="auto" prefers responses

    response = client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

    assert stub.paths == ["/v1/responses", "/v1/chat/completions"]
    assert response.debug["api"] == "chat_completions"
    assert response.answers["q"].choice == "billing"
    assert response.usage.n_retries == 0  # a surface switch is not a transient retry


def test_the_missing_responses_surface_is_remembered(stub_server):
    """One 404 is enough: later calls on the same client go straight to the surface that works."""
    stub = stub_server(chat=lambda _: (200, chat_body(content="A", logprobs=CHOICE_LOGS)))
    client = SystemOneClient(openai_client(stub), model="stub")

    client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})
    client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

    assert stub.paths.count("/v1/responses") == 1
    assert stub.paths.count("/v1/chat/completions") == 2


def test_a_404_that_names_the_model_does_not_switch_surface(stub_server):
    """The model is missing on either surface, so the provider's own error must reach the caller."""

    def script(_):
        return 404, {"error": {"message": "The model 'stub' does not exist", "code": "model_not_found"}}

    stub = stub_server(responses=script)
    client = SystemOneClient(openai_client(stub), model="stub")

    with pytest.raises(ProviderError) as error:
        client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

    assert "does not exist" in str(error.value)
    assert stub.paths == ["/v1/responses"]


def test_an_explicit_responses_surface_reports_the_404(stub_server):
    """``api="responses"`` is a decision, not a preference: it is never silently overridden."""
    stub = stub_server(chat=lambda _: (200, chat_body(content="A", logprobs=CHOICE_LOGS)))
    client = SystemOneClient(openai_client(stub), model="stub", api="responses")

    with pytest.raises(ProviderError):
        client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

    assert stub.paths == ["/v1/responses"]


def test_switching_surface_re_derives_the_reasoning_mode(stub_server):
    """``mode="auto"`` is native on Responses and two-step on Chat: the switch must follow the surface."""
    stub = stub_server(chat=lambda _: (200, chat_body(content="A", logprobs=CHOICE_LOGS)))
    client = SystemOneClient(
        openai_client(stub),
        model="stub",
        reasoning=ReasoningConfig(effort="medium", mode="auto"),
    )

    response = client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

    assert stub.paths[0] == "/v1/responses"  # the 404 that teaches the verdict
    assert response.debug["api"] == "chat_completions"
    assert response.debug["reasoning_mode"] == "two_step"
    sent = stub.bodies("/chat/completions")
    assert len(sent) == 2  # analysis pass, then the answer pass
    assert all("reasoning_effort" not in body for body in sent)  # two-step asks for no native reasoning
    assert response.answers["q"].choice == "billing"


def test_auto_answers_from_chat_when_the_responses_surface_carries_no_logprobs(stub_server):
    """ollama and llama.cpp implement ``/v1/responses`` with an empty logprob list; chat has them.

    The surface preference exists for native reasoning, but a label readout with nothing to read is
    worth one request on the other surface — which on the same server carries a full distribution.
    """
    stub = stub_server(
        chat=lambda _: (200, chat_body(content="A", logprobs=CHOICE_LOGS)),
        responses=lambda _: (200, responses_body(text="A")),
    )
    client = SystemOneClient(openai_client(stub), model="stub")

    response = client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

    assert response.debug["api"] == "chat_completions"
    assert response.debug["methods"] == {"q": "logprobs"}
    assert response.answers["q"].probabilities["billing"] == pytest.approx(0.884873983, abs=1e-9)
    assert stub.paths == ["/v1/responses", "/v1/chat/completions"]


def test_native_reasoning_keeps_the_responses_surface(stub_server):
    """Switching surfaces would silently turn native reasoning into a two-step pass: it stays put."""
    stub = stub_server(
        chat=lambda _: (200, chat_body(content="A", logprobs=CHOICE_LOGS)),
        responses=lambda _: (200, responses_body(text=STRUCTURED)),
    )
    client = SystemOneClient(
        openai_client(stub), model="stub", reasoning=ReasoningConfig(mode="native", effort="low")
    )

    response = client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

    assert response.debug["api"] == "responses"
    assert response.debug["reasoning_mode"] == "native"
    assert response.debug["methods"] == {"q": "structured"}
    assert stub.paths == ["/v1/responses", "/v1/responses"]
    assert response.answers["q"].choice == "billing"


def test_the_surface_verdict_is_remembered_from_the_first_call(stub_server):
    """One surface without logprobs is enough to stop opening there: the next call starts elsewhere."""
    stub = stub_server(
        chat=lambda _: (200, chat_body(content="A", logprobs=CHOICE_LOGS)),
        responses=lambda _: (200, responses_body(text="A")),
    )
    client = SystemOneClient(openai_client(stub), model="stub")

    for _ in range(3):
        client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

    assert stub.paths.count("/v1/responses") == 1  # the discovery, paid once
    assert stub.paths.count("/v1/chat/completions") == 3


def test_async_auto_switches_surface_too(stub_server):
    stub = stub_server(chat=lambda _: (200, chat_body(content="A", logprobs=CHOICE_LOGS)))
    client = AsyncSystemOneClient(async_openai_client(stub), model="stub")

    response = asyncio.run(client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)}))

    assert stub.paths == ["/v1/responses", "/v1/chat/completions"]
    assert response.debug["api"] == "chat_completions"
    assert response.answers["q"].choice == "billing"


def test_an_explicit_logprobs_moves_to_the_surface_that_carries_it(stub_server):
    """OpenRouter: its Responses API refuses the logprob includable; Chat Completions carries them.

    An explicit ``method="logprobs"`` asks for a distribution, not for one particular surface to
    produce it, so the readout moves there and the caller's method is kept.
    """
    stub = stub_server(
        chat=lambda _: (200, chat_body(content="A", logprobs=CHOICE_LOGS)),
        responses=lambda body: (400, INCLUDE_REJECTION) if body.get("include") else (200, responses_body(text="A")),
    )
    client = SystemOneClient(openai_client(stub), model="stub", method="logprobs")

    response = client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

    assert response.debug["api"] == "chat_completions"
    assert response.debug["method"] == "logprobs"
    assert response.answers["q"].choice == "billing"
    assert response.answers["q"].probabilities["billing"] == pytest.approx(0.884873983, abs=1e-9)
    assert stub.paths == ["/v1/responses", "/v1/chat/completions"]


def test_an_explicit_logprobs_starts_where_the_distribution_is_next_time(stub_server):
    """The move is remembered, so the refusal is paid for once rather than on every call."""
    stub = stub_server(
        chat=lambda _: (200, chat_body(content="A", logprobs=CHOICE_LOGS)),
        responses=lambda body: (400, INCLUDE_REJECTION) if body.get("include") else (200, responses_body(text="A")),
    )
    client = SystemOneClient(openai_client(stub), model="stub", method="logprobs")

    for _ in range(2):
        response = client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

    assert response.debug["api"] == "chat_completions"
    assert stub.paths.count("/v1/responses") == 1  # the discovery, paid once
    assert stub.paths.count("/v1/chat/completions") == 2


def test_an_explicit_logprobs_is_not_swapped_for_another_readout(stub_server):
    """With nowhere to move, the refusal is reported — the readout is not silently replaced."""
    stub = stub_server(responses=lambda body: (400, INCLUDE_REJECTION) if body.get("include") else (200, responses_body(text=STRUCTURED)))
    client = SystemOneClient(SimpleNamespace(responses=openai_client(stub).responses), model="stub", method="logprobs")

    with pytest.raises(LabelReadoutError) as caught:
        client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

    assert "rejected the logprob request" in str(caught.value)
    assert stub.paths == ["/v1/responses"]  # never re-asked in JSON
    assert stub.bodies("/responses")[0]["include"] == ["message.output_text.logprobs"]


def test_an_explicit_surface_keeps_the_providers_refusal(stub_server):
    """`api="responses"` is a decision: its refusal reaches the caller, not a fallback."""
    stub = stub_server(
        chat=lambda _: (200, chat_body(content="A", logprobs=CHOICE_LOGS)),
        responses=lambda body: (400, INCLUDE_REJECTION) if body.get("include") else (200, responses_body(text="A")),
    )
    client = SystemOneClient(openai_client(stub), model="stub", api="responses", method="logprobs")

    with pytest.raises(ProviderError) as caught:
        client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

    assert caught.value.status_code == 400
    assert stub.paths == ["/v1/responses"]


def test_async_explicit_logprobs_moves_surface_too(stub_server):
    stub = stub_server(
        chat=lambda _: (200, chat_body(content="A", logprobs=CHOICE_LOGS)),
        responses=lambda body: (400, INCLUDE_REJECTION) if body.get("include") else (200, responses_body(text="A")),
    )
    client = AsyncSystemOneClient(async_openai_client(stub), model="stub", method="logprobs")

    response = asyncio.run(client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)}))

    assert response.debug["api"] == "chat_completions"
    assert response.debug["method"] == "logprobs"
    assert stub.paths == ["/v1/responses", "/v1/chat/completions"]




UNSUPPORTED_INCLUDE = {
    "error": {
        "message": "Unsupported parameter: 'include' is not supported with this model.",
        "type": "invalid_request_error",
        "param": "include",
        "code": "unsupported_parameter",
    }
}


def unsupported_include_host(body):
    """OpenAI's own wording for a model that offers no includable at all."""
    if any("logprobs" in str(entry) for entry in body.get("include", ())):
        return 400, UNSUPPORTED_INCLUDE
    return 200, responses_body(text=STRUCTURED)


def test_an_unsupported_include_is_a_logprob_verdict_not_a_provider_error(stub_server):
    """A Responses server that refuses ``include`` by name has refused the logprob carrier.

    The word it uses is "unsupported", not "invalid option" — the phrasing this case was first seen
    in. Either way the request asked for a distribution through the only field that can carry one,
    and there is no other field to drop, so the verdict belongs to the label readout: auto answers
    the question in JSON, and an explicit logprobs call reports the refusal rather than a raw 400.
    """
    stub = stub_server(responses=unsupported_include_host)
    client = SystemOneClient(openai_client(stub), model="stub", api="responses", retry=NO_RETRY)

    response = client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

    assert response.answers["q"].choice == "billing"
    assert response.debug["methods"] == {"q": "structured"}
    assert "logprob" in response.debug["retry_reasons"][0].lower()
    assert len(stub.requests) == 2


def test_an_explicit_logprobs_call_moves_surface_on_an_unsupported_include(stub_server):
    """The move is a verdict about the surface, so it is made however the method was chosen.

    OpenAI's own wording for a model that offers no includable names the field and says the field is
    unsupported — no "invalid option" anywhere. Read as a plain provider error it would end the call;
    read as the carrier refusal it is, the distribution is one surface away.
    """
    stub = stub_server(
        responses=unsupported_include_host,
        chat=lambda _: (200, chat_body(content="A", logprobs=CHOICE_LOGS)),
    )
    client = SystemOneClient(openai_client(stub), model="stub", method="logprobs", retry=NO_RETRY)

    response = client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

    assert response.answers["q"].choice == "billing"
    assert response.debug["api"] == "chat_completions"
    assert "retrying the label readout on api='chat_completions'" in response.debug["retry_reasons"][0]


def test_a_stale_downgrade_cannot_move_a_remembered_limit_back_up(stub_server):
    """Two questions on one call run concurrently, each holding the snapshot its transport was built
    with, so a downgrade computed from the older one can land after a sibling has already walked
    further down. Writing it wholesale would put that rung back, and the next call would re-send the
    very field the server had refused twice."""
    from jevper.transport import Limits

    client = SystemOneClient(openai_client(stub_server()), model="stub")
    base = Limits()
    key = ("stub", "chat_completions")

    client._remember_limits("stub", "chat_completions", base, Limits(structured="object"))
    client._remember_limits("stub", "chat_completions", base, Limits(structured="none"))
    # The stale question now reports the rung it discovered, from before either of those.
    client._remember_limits("stub", "chat_completions", base, Limits(structured="object"))

    assert client._limits[key].structured == "none"
