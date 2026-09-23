"""Few-shot examples: rendering, precedence, validation and wire compatibility."""

from __future__ import annotations

import json

import pytest
from fakes import chat_body, openai_client

from jevper import (
    Choice,
    Example,
    InvalidQuestionError,
    Noul,
    ReasoningConfig,
    SystemOneClient,
)
from jevper.prompts import ANALYSIS_SYSTEM_PROMPT

CHOICE_LOGS = [("A", -0.12), ("B", -2.47), ("C", -3.48)]
CRITERIA = {"billing": None, "technical": None, "sales": None}


def test_examples_render_as_question_turns_then_label_turns(stub_server):
    stub = stub_server(chat=lambda _: (200, chat_body(content="A", logprobs=CHOICE_LOGS)))
    client = SystemOneClient(openai_client(stub), model="stub", api="chat_completions")
    question = Choice(
        criteria=CRITERIA,
        examples=[
            Example(state="Charge duplicated on the invoice", answer="billing"),
            Example(state="Login fails after the update", answer="technical"),
        ],
    )

    client.system_one(state="The receipt shows two identical charges", questions={"q": question})

    messages = stub.bodies("/chat/completions")[0]["messages"]
    assert [message["role"] for message in messages] == [
        "system",
        "user",
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
    ]
    assert "Charge duplicated on the invoice" in messages[2]["content"]
    assert "Options:" in messages[2]["content"]
    assert messages[3]["content"] == "A"
    assert "Login fails after the update" in messages[4]["content"]
    assert messages[5]["content"] == "B"
    assert "The receipt shows two identical charges" in messages[1]["content"]
    assert "The receipt shows two identical charges" not in messages[6]["content"]
    assert messages[6]["content"].startswith("Options:")


def test_question_level_examples_win_over_call_and_constructor(stub_server):
    stub = stub_server(chat=lambda _: (200, chat_body(content="A", logprobs=CHOICE_LOGS)))
    client = SystemOneClient(
        openai_client(stub),
        model="stub",
        api="chat_completions",
        examples=[Example(state="from-constructor", answer="sales")],
    )
    question = Choice(criteria=CRITERIA, examples=[Example(state="from-question", answer="billing")])

    client.system_one(
        state="s",
        questions={"q": question},
        examples={"q": [Example(state="from-call", answer="technical")]},
    )

    prompt = json.dumps(stub.bodies("/chat/completions")[0]["messages"])
    assert "from-question" in prompt
    assert "from-call" not in prompt and "from-constructor" not in prompt


def test_call_examples_win_over_constructor_defaults(stub_server):
    stub = stub_server(chat=lambda _: (200, chat_body(content="A", logprobs=CHOICE_LOGS)))
    client = SystemOneClient(
        openai_client(stub),
        model="stub",
        api="chat_completions",
        examples=[Example(state="from-constructor", answer="sales")],
    )

    client.system_one(
        state="s",
        questions={"q": Choice(criteria=CRITERIA)},
        examples=[Example(state="from-call", answer="technical")],
    )

    prompt = json.dumps(stub.bodies("/chat/completions")[0]["messages"])
    assert "from-call" in prompt and "from-constructor" not in prompt


def test_constructor_examples_apply_as_the_last_resort(stub_server):
    stub = stub_server(chat=lambda _: (200, chat_body(content="A", logprobs=CHOICE_LOGS)))
    client = SystemOneClient(
        openai_client(stub),
        model="stub",
        api="chat_completions",
        examples={"q": [Example(state="from-constructor", answer="sales")]},
    )

    client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

    messages = stub.bodies("/chat/completions")[0]["messages"]
    assert "from-constructor" in json.dumps(messages)
    assert messages[3]["content"] == "C"


def test_structured_examples_render_json_answers(stub_server):
    payload = {"probabilities": {"billing": 0.8, "technical": 0.1, "sales": 0.1}}

    def script(body):
        name = body["response_format"]["json_schema"]["name"]
        if name == "jevper_noul":
            return 200, chat_body(content=json.dumps({"noul": 0.7}))
        return 200, chat_body(content=json.dumps(payload))

    stub = stub_server(chat=script)
    client = SystemOneClient(
        openai_client(stub), model="stub", method="structured", api="chat_completions"
    )
    question = Choice(
        criteria=CRITERIA,
        examples=[
            Example(state="weighted", answer="billing", probabilities={"billing": 0.6, "technical": 0.3, "sales": 0.1}),
            Example(state="certain", answer="technical"),
        ],
    )

    client.system_one(
        state="s",
        questions={
            "q": question,
            "verdict": Noul(
                examples=[Example(state="calm", answer=True, probabilities={True: 0.9}), Example(state="angry", answer=False)]
            ),
        },
    )

    choice_body, noul_body = stub.bodies("/chat/completions")
    messages = choice_body["messages"]
    assert messages[0]["content"].startswith("You are a precise classification engine. Answer the question by returning a JSON object")
    assert json.loads(messages[3]["content"]) == {
        "probabilities": {"billing": 0.6, "technical": 0.3, "sales": 0.1}
    }
    assert json.loads(messages[5]["content"]) == {
        "probabilities": {"billing": 0.0, "technical": 1.0, "sales": 0.0}
    }
    noul_messages = noul_body["messages"]
    assert json.loads(noul_messages[3]["content"]) == {"noul": 0.9}
    assert json.loads(noul_messages[5]["content"]) == {"noul": 0.0}


def test_examples_are_not_part_of_the_wire_question(stub_server):
    question = Choice(criteria=CRITERIA, examples=[Example(state="s", answer="billing")])

    assert question.model_dump() == {"type": "choice", "instructions": None, "criteria": CRITERIA}
    assert "examples" not in json.loads(question.model_dump_json())


def test_example_answers_are_validated_against_the_question(stub_server):
    stub = stub_server(chat=lambda _: (200, chat_body(content="A", logprobs=CHOICE_LOGS)))
    client = SystemOneClient(openai_client(stub), model="stub", api="chat_completions")

    with pytest.raises(InvalidQuestionError) as error:
        client.system_one(
            state="s",
            questions={"q": Choice(criteria=CRITERIA, examples=[Example(state="s", answer="nope")])},
        )

    assert "question 'q'" in str(error.value)
    assert "example 0" in str(error.value)
    assert "answer 'nope'" in str(error.value)
    assert stub.requests == []


def test_example_answer_forms_resolve_to_labels(stub_server):
    stub = stub_server(chat=lambda _: (200, chat_body(content="A", logprobs=CHOICE_LOGS)))
    client = SystemOneClient(openai_client(stub), model="stub", api="chat_completions")

    client.system_one(
        state="s",
        questions={
            "verdict": Noul(examples=[Example(state="clear yes", answer=True), Example(state="clear no", answer="B")]),
            "intent": Choice(
                criteria=CRITERIA,
                examples=[Example(state="by key", answer="sales"), Example(state="by label", answer="c")],
            ),
        },
    )

    bodies = stub.bodies("/chat/completions")
    verdict = next(body for body in bodies if "A: Yes" in body["messages"][-1]["content"])
    intent = next(body for body in bodies if "A: billing" in body["messages"][-1]["content"])
    labels = lambda body: [message["content"] for message in body["messages"] if message["role"] == "assistant"]
    assert labels(verdict) == ["A", "B"]
    assert labels(intent) == ["C", "C"]


def test_examples_reach_both_two_step_passes(stub_server):
    calls: list[dict] = []

    def script(body):
        calls.append(body)
        if len(calls) == 1:
            return 200, chat_body(content="analysis")
        return 200, chat_body(content="A", logprobs=CHOICE_LOGS)

    stub = stub_server(chat=script)
    client = SystemOneClient(
        openai_client(stub),
        model="stub",
        api="chat_completions",
        reasoning=ReasoningConfig(mode="two_step"),
    )
    question = Choice(criteria=CRITERIA, examples=[Example(state="duplicate charge", answer="billing")])

    client.system_one(state="s", questions={"q": question})

    analysis, answer = stub.bodies("/chat/completions")
    assert analysis["messages"][0]["content"] == ANALYSIS_SYSTEM_PROMPT
    assert "duplicate charge" in json.dumps(analysis["messages"])
    assert "duplicate charge" in json.dumps(answer["messages"])
    assert answer["messages"][-2]["content"] == "analysis"
