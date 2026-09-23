"""Reasoning: native forwarding and the two-step think-then-classify fallback."""

from __future__ import annotations

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
    assert answer["input"][-2] == {"role": "assistant", "content": TRACE}
    assert response.answers["q"].choice == "billing"
    # the provider's own item carries the trace: no synthetic part, so no duplicated text
    assert len(response.reasoning) == 1
    assert reasoning_text(response.reasoning) == TRACE
