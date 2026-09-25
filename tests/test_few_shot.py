"""Few-shot examples: rendering, precedence, validation and wire compatibility."""

from __future__ import annotations

import json

import pytest
from fakes import chat_body, openai_client

from jevper import (
    Choice,
    Example,
    InvalidQuestionError,
    JevperError,
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
        "assistant",
        "user",
        "assistant",
        "user",
        "user",
    ]
    assert "Charge duplicated on the invoice" in messages[1]["content"]
    assert "Options:" in messages[1]["content"]
    assert messages[2]["content"] == "A"
    assert "Login fails after the update" in messages[3]["content"]
    assert messages[4]["content"] == "B"
    assert messages[5]["content"].startswith("Options:")
    assert "The receipt shows two identical charges" not in messages[5]["content"]
    assert "The receipt shows two identical charges" in messages[6]["content"]


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
    assert messages[2]["content"] == "C"


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

    bodies = stub.bodies("/chat/completions")
    assert len(bodies) == 2
    # The two questions are answered concurrently, so arrival order is not guaranteed.
    choice_body = next(body for body in bodies if "A: billing" in body["messages"][-2]["content"])
    noul_body = next(body for body in bodies if "A: Yes" in body["messages"][-2]["content"])
    messages = choice_body["messages"]
    assert messages[0]["content"].startswith("You are a precise classification engine. Answer the question by returning a JSON object")
    assert json.loads(messages[2]["content"]) == {
        "probabilities": {"billing": 0.6, "technical": 0.3, "sales": 0.1}
    }
    assert json.loads(messages[4]["content"]) == {
        "probabilities": {"billing": 0.0, "technical": 1.0, "sales": 0.0}
    }
    noul_messages = noul_body["messages"]
    assert json.loads(noul_messages[2]["content"]) == {"noul": 0.9}
    assert json.loads(noul_messages[4]["content"]) == {"noul": 0.0}


def test_examples_are_not_part_of_the_wire_question(stub_server):
    question = Choice(criteria=CRITERIA, examples=[Example(state="s", answer="billing")])

    assert question.model_dump() == {"type": "choice", "instructions": None, "criteria": CRITERIA}
    assert "examples" not in json.loads(question.model_dump_json())


def test_example_answers_are_validated_against_the_question(stub_server):
    stub = stub_server(chat=lambda _: (200, chat_body(content="A", logprobs=CHOICE_LOGS)))
    client = SystemOneClient(openai_client(stub), model="stub", api="chat_completions")

    # A question built in Python is checked where it is built, and the same question handed
    # over as data is checked on the way in: both refuse it, both name the example.
    with pytest.raises(InvalidQuestionError) as error:
        Choice(criteria=CRITERIA, examples=[Example(state="s", answer="nope")])
    assert "example 0" in str(error.value)
    assert "answer 'nope'" in str(error.value)

    with pytest.raises(InvalidQuestionError) as error:
        client.system_one(
            state="s",
            questions={"q": {"type": "choice", "criteria": CRITERIA, "examples": [{"state": "s", "answer": "nope"}]}},
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
    verdict = next(body for body in bodies if "A: Yes" in body["messages"][-2]["content"])
    intent = next(body for body in bodies if "A: billing" in body["messages"][-2]["content"])
    labels = lambda body: [message["content"] for message in body["messages"] if message["role"] == "assistant"]
    assert labels(verdict) == ["A", "B"]
    assert labels(intent) == ["C", "C"]


def test_example_probabilities_must_match_the_question(stub_server):
    stub = stub_server(chat=lambda _: (200, chat_body(content="A", logprobs=CHOICE_LOGS)))

    wrong_keys = Example(state="s", answer="billing", probabilities={"billing": 0.6, "sales": 0.4})
    with pytest.raises(InvalidQuestionError) as error:
        Choice(criteria=CRITERIA, examples=[wrong_keys])
    assert "example 0" in str(error.value) and "must have exactly the keys" in str(error.value)

    negative = Example(
        state="s", answer="billing", probabilities={"billing": 1.2, "technical": -0.1, "sales": -0.1}
    )
    with pytest.raises(InvalidQuestionError) as error:
        Choice(criteria=CRITERIA, examples=[negative])
    assert "must be >= 0" in str(error.value)

    with pytest.raises(InvalidQuestionError) as error:
        Noul(examples=[Example(state="s", answer=True, probabilities={"yes": 0.9})])
    assert "True or False key" in str(error.value)

    assert stub.requests == []


def test_example_probabilities_are_refused_by_a_question_handed_over_as_data(stub_server):
    """The mapping path checks the same numbers, and names the question it arrived with."""
    stub = stub_server(chat=lambda _: (200, chat_body(content="A", logprobs=CHOICE_LOGS)))
    client = SystemOneClient(
        openai_client(stub), model="stub", method="structured", api="chat_completions"
    )

    with pytest.raises(InvalidQuestionError) as error:
        client.system_one(
            state="s",
            questions={
                "verdict": {
                    "type": "noul",
                    "examples": [{"state": "s", "answer": True, "probabilities": {"yes": 0.9}}],
                }
            },
        )

    assert "question 'verdict'" in str(error.value)
    assert "True or False key" in str(error.value)
    assert stub.requests == []


def test_non_finite_example_probabilities_are_refused_as_an_invalid_example():
    with pytest.raises(InvalidQuestionError) as error:
        Example(state="s", answer="billing", probabilities={"billing": float("nan")})
    assert "finite" in str(error.value)
    with pytest.raises(InvalidQuestionError):
        Example(state="s", answer=True, probabilities={True: float("inf")})


def test_unserializable_example_state_names_the_example(stub_server):
    stub = stub_server(chat=lambda _: (200, chat_body(content="A", logprobs=CHOICE_LOGS)))
    client = SystemOneClient(openai_client(stub), model="stub", api="chat_completions")
    question = Choice(criteria=CRITERIA, examples=[Example(state={1, 2}, answer="billing")])

    with pytest.raises(JevperError) as error:
        client.system_one(state="s", questions={"q": question})

    assert "example 0" in str(error.value) and "JSON-serializable" in str(error.value)
    assert stub.requests == []


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


def test_example_probabilities_are_checked_whatever_the_method():
    """Only ``structured`` renders the numbers, and the check does not wait to find that out.

    The question is refused where it is built — before a method is resolved, before a prompt is
    rendered, and before any client could spend a request on it.
    """
    wrong_keys = Example(state="s", answer="billing", probabilities={"billing": 0.6, "sales": 0.4})
    with pytest.raises(InvalidQuestionError) as error:
        Choice(criteria=CRITERIA, examples=[wrong_keys])
    assert "must have exactly the keys" in str(error.value)

    negative = Example(
        state="s", answer="billing", probabilities={"billing": 1.2, "technical": -0.1, "sales": -0.1}
    )
    with pytest.raises(InvalidQuestionError) as error:
        Choice(criteria=CRITERIA, examples=[negative])
    assert "must be >= 0" in str(error.value)

    with pytest.raises(InvalidQuestionError) as error:
        Noul(examples=[Example(state="s", answer=True, probabilities={True: 1.4})])
    assert "noul probability must be in [0, 1]" in str(error.value)


def test_example_answer_prefers_an_exact_key_over_a_label(stub_server):
    """Criteria keys that look like labels: the exact key wins, so the demonstration matches the answer."""
    stub = stub_server(chat=lambda _: (200, chat_body(content="A", logprobs=CHOICE_LOGS)))
    client = SystemOneClient(openai_client(stub), model="stub", api="chat_completions")
    question = Choice(criteria={"b": None, "a": None}, examples=[Example(state="s", answer="a")])

    client.system_one(state="s", questions={"q": question})

    sent = stub.bodies("/chat/completions")[0]
    demonstrated = [m["content"] for m in sent["messages"] if m["role"] == "assistant"]
    assert demonstrated == ["B"]  # the label of the option keyed "a", not the first label "A"
