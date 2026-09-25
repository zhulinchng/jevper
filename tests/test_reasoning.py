"""Reasoning: native forwarding and the two-step think-then-classify fallback."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest
from fakes import chat_body, openai_client, reasoning_item, responses_body

from jevper import Choice, ReasoningConfig, SystemOneClient, reasoning_text
from jevper.prompts import ANALYSIS_SYSTEM_PROMPT, ANSWER_CUE

CHOICE_LOGS = [("A", -0.12), ("B", -2.47), ("C", -3.48)]
CRITERIA = {"billing": None, "technical": None, "sales": None}
TRACE = "The customer describes a duplicate charge, so billing fits best."


def test_two_step_on_the_chat_surface(stub_server):
    calls: list[dict] = []

    def script(body):
        calls.append(body)
        if len(calls) == 1:
            return 200, chat_body(content=TRACE)
        return 200, chat_body(content="A", logprobs=CHOICE_LOGS)

    stub = stub_server(chat=script)
    client = SystemOneClient(
        openai_client(stub),
        model="stub",
        api="chat_completions",
        reasoning=ReasoningConfig(effort="medium"),
    )

    response = client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

    requests = stub.bodies("/chat/completions")
    assert len(requests) == 2
    analysis, answer = requests
    assert analysis["messages"][0]["content"] == ANALYSIS_SYSTEM_PROMPT
    assert "logprobs" not in analysis
    assert "response_format" not in analysis
    assert "reasoning_effort" not in analysis
    assert answer["logprobs"] is True
    assert "reasoning_effort" not in answer
    assert answer["messages"][-2] == {"role": "assistant", "content": TRACE}
    assert answer["messages"][-1] == {"role": "user", "content": ANSWER_CUE}
    assert response.answers["q"].choice == "billing"
    assert response.reasoning[0].summary[0].text == TRACE
    assert reasoning_text(response.reasoning) == TRACE
    assert response.usage.n_calls == 2
    assert response.debug["reasoning_mode"] == "two_step"


def test_two_step_forwards_reasoning_on_the_responses_surface(stub_server):
    calls: list[dict] = []

    def script(body):
        calls.append(body)
        if len(calls) == 1:
            return 200, responses_body(text=TRACE, reasoning=[reasoning_item("native analysis")])
        return 200, responses_body(text="A", logprobs=CHOICE_LOGS)

    stub = stub_server(responses=script)
    client = SystemOneClient(
        openai_client(stub), model="stub", reasoning=ReasoningConfig(effort="medium", mode="two_step")
    )

    response = client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

    analysis, answer = stub.bodies("/responses")
    assert analysis["reasoning"] == {"effort": "medium"}
    assert analysis["store"] is False
    assert "text" not in analysis
    assert "reasoning" not in answer
    assert answer["include"] == ["message.output_text.logprobs"]
    assert response.reasoning[0].summary[0].text == TRACE
    assert response.reasoning[1].model_extra["encrypted_content"] == "encrypted"
    assert reasoning_text(response.reasoning) == f"{TRACE}\n\nnative analysis"


def test_native_reasoning_on_the_responses_surface(stub_server):
    body = responses_body(text="A", logprobs=CHOICE_LOGS, reasoning=[reasoning_item("step one")])
    stub = stub_server(responses=lambda _: (200, body))
    client = SystemOneClient(
        openai_client(stub), model="stub", reasoning=ReasoningConfig(effort="medium", summary="auto")
    )

    response = client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

    assert len(stub.requests) == 1
    sent = stub.bodies("/responses")[0]
    assert sent["reasoning"] == {"effort": "medium", "summary": "auto"}
    assert sent["store"] is False
    assert set(sent["include"]) == {"message.output_text.logprobs", "reasoning.encrypted_content"}
    assert response.debug["reasoning_mode"] == "native"
    assert response.answers["q"].choice == "billing"
    assert response.reasoning[0].summary[0].text == "step one"
    assert response.reasoning[0].model_extra["id"] == "rs_stub"
    assert response.reasoning[0].model_extra["status"] == "completed"
    assert response.usage.reasoning_tokens == 0


def test_native_reasoning_on_the_chat_surface_surfaces_reasoning_content(stub_server):
    body = chat_body(content="A", logprobs=CHOICE_LOGS, reasoning="billing because of the duplicate")
    stub = stub_server(chat=lambda _: (200, body))
    client = SystemOneClient(
        openai_client(stub),
        model="stub",
        api="chat_completions",
        reasoning=ReasoningConfig(effort="low", mode="native"),
    )

    response = client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

    assert stub.bodies("/chat/completions")[0]["reasoning_effort"] == "low"
    assert response.debug["reasoning_mode"] == "native"
    assert response.reasoning[0].content[0].text == "billing because of the duplicate"
    assert reasoning_text(response.reasoning) == "billing because of the duplicate"


def test_reasoning_absent_by_default(stub_server):
    stub = stub_server(chat=lambda _: (200, chat_body(content="A", logprobs=CHOICE_LOGS)))
    client = SystemOneClient(openai_client(stub), model="stub", api="chat_completions")

    response = client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

    assert len(stub.requests) == 1
    assert "reasoning_effort" not in stub.bodies("/chat/completions")[0]
    assert response.reasoning == ()
    assert response.debug["reasoning_mode"] == "off"


def test_reasoning_mode_must_be_a_config(stub_server):
    stub = stub_server(chat=lambda _: (200, chat_body(content="A")))
    with pytest.raises(Exception) as error:
        SystemOneClient(openai_client(stub), model="stub", reasoning={"effort": "medium"})  # type: ignore[arg-type]
    assert "ReasoningConfig" in str(error.value)


def test_two_step_without_analysis_text_sends_no_empty_assistant_turn(stub_server):
    calls: list[dict] = []

    def script(body):
        calls.append(body)
        if len(calls) == 1:
            return 200, chat_body(content="")
        return 200, chat_body(content="A", logprobs=CHOICE_LOGS)

    stub = stub_server(chat=script)
    client = SystemOneClient(
        openai_client(stub), model="stub", api="chat_completions", reasoning=ReasoningConfig(mode="two_step")
    )

    response = client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

    answer = stub.bodies("/chat/completions")[1]
    assert [message["role"] for message in answer["messages"]] == ["system", "user", "user", "user"]
    assert answer["messages"][-1]["content"] == ANSWER_CUE
    assert response.answers["q"].choice == "billing"
    assert response.reasoning == ()


def test_two_step_falls_back_to_reasoning_text_when_the_analysis_is_silent(stub_server):
    calls: list[dict] = []

    def script(body):
        calls.append(body)
        if len(calls) == 1:
            return 200, responses_body(text="", reasoning=[reasoning_item(TRACE)])
        return 200, responses_body(text="A", logprobs=CHOICE_LOGS)

    stub = stub_server(responses=script)
    client = SystemOneClient(
        openai_client(stub), model="stub", reasoning=ReasoningConfig(effort="medium", mode="two_step")
    )

    response = client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

    answer = stub.bodies("/responses")[1]
    assert answer["input"][-2] == {"type": "message", "role": "assistant", "content": TRACE}
    assert response.answers["q"].choice == "billing"
    # the provider's own item carries the trace: no synthetic part, so no duplicated text
    assert len(response.reasoning) == 1
    assert reasoning_text(response.reasoning) == TRACE


def test_a_null_reasoning_payload_does_not_cost_the_answer(stub_server):
    """Null reasoning fields are the provider's business; the answer is still right there."""
    empty = reasoning_item("dropped")
    empty["summary"] = None
    empty["content"] = None
    partial = reasoning_item("kept summary")
    partial["content"] = None
    stub = stub_server(
        responses=lambda _: (
            200,
            responses_body(text="A", logprobs=CHOICE_LOGS, reasoning=[empty, partial]),
        )
    )
    client = SystemOneClient(openai_client(stub), model="stub", api="responses")

    response = client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

    assert response.answers["q"].choice == "billing"
    # A null field is emptied, not fatal: the summary that came with it is still read.
    assert reasoning_text(response.reasoning) == "kept summary"


def test_an_unreadable_reasoning_part_is_skipped(stub_server):
    """One malformed part among readable ones is dropped, not fatal."""
    body = chat_body(content="A", logprobs=CHOICE_LOGS)
    body["choices"][0]["message"]["reasoning"] = [
        {"type": "reasoning", "summary": [{"type": "summary_text"}]},  # a summary_text without its text
        {"type": "reasoning", "summary": None, "content": None},  # null fields
        {"type": "reasoning", "summary": [{"type": "summary_text", "text": "kept"}]},
    ]
    stub = stub_server(chat=lambda _: (200, body))
    client = SystemOneClient(openai_client(stub), model="stub", api="chat_completions")

    response = client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

    assert response.answers["q"].choice == "billing"
    assert reasoning_text(response.reasoning) == "kept"


def test_a_reasoning_part_without_a_dict_is_skipped():
    """A duck-typed client can return parts that are neither mappings nor pydantic models."""

    class SlotsPart:
        __slots__ = ("summary", "type")

        def __init__(self) -> None:
            self.type = "reasoning"
            self.summary = None

    class ObjectClient:
        def __init__(self) -> None:
            part = SlotsPart()

            class Completions:
                def create(self, **kwargs: Any) -> Any:
                    message = SimpleNamespace(content='{"choice": "A"}', reasoning=[part])
                    choice = SimpleNamespace(message=message, finish_reason="stop")
                    return SimpleNamespace(choices=[choice], usage=None)

            class Chat:
                completions = Completions()

            self.chat = Chat()

    client = SystemOneClient(ObjectClient(), model="m", method="discrete", api="chat_completions")

    response = client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

    assert response.answers["q"].choice == "billing"


def test_a_budget_edited_after_construction_is_still_validated():
    """The constructor is not the only way in, and the invalid number would reach the provider.

    Without assignment validation, ``config.budget_tokens = 0`` skips the validator and jevper
    serializes ``thinking={"type": "enabled", "budget_tokens": 0}`` — a value the same config refuses
    when it is passed to the constructor.
    """
    config = ReasoningConfig(mode="native", budget_tokens=2048)

    with pytest.raises(ValueError):
        config.budget_tokens = 0
    assert config.budget_tokens == 2048


@pytest.mark.parametrize("budget", [1024, 2048])
def test_a_thinking_budget_is_not_sent_on_the_openai_surfaces(stub_server, budget):
    """Only the Messages API has a budget field; the OpenAI surfaces carry an effort and an object."""
    stub = stub_server(
        chat=lambda _: (200, chat_body(content="A", logprobs=CHOICE_LOGS)),
        responses=lambda _: (200, responses_body(text="A", logprobs=CHOICE_LOGS)),
    )
    config = ReasoningConfig(mode="native", effort="low", budget_tokens=budget)

    SystemOneClient(
        openai_client(stub), model="stub", api="chat_completions", reasoning=config
    ).system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})
    SystemOneClient(
        openai_client(stub), model="stub", api="responses", reasoning=config
    ).system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

    chat = stub.bodies("/chat/completions")[0]
    responses = stub.bodies("/responses")[0]
    assert chat["reasoning_effort"] == "low"
    assert "budget_tokens" not in json.dumps(chat) and "thinking" not in json.dumps(chat)
    assert responses["reasoning"] == {"effort": "low"}
    assert "budget_tokens" not in json.dumps(responses)


@pytest.mark.parametrize(
    ("field", "value"),
    [("effort", "enormous"), ("summary", "verbose"), ("context", "history"), ("mode", "sometimes")],
)
def test_an_unknown_reasoning_value_is_refused_locally(field, value):
    with pytest.raises(ValueError):
        ReasoningConfig(**{field: value})
