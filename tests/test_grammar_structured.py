"""Grammar, structured and discrete methods, plus the Responses surface."""

from __future__ import annotations

import json
import math

import pytest
from fakes import chat_body, openai_client, responses_body

from jevper import (
    Choice,
    Example,
    LabelReadoutError,
    MalformedAnswerError,
    Noul,
    ReasoningConfig,
    Score,
    SystemOneClient,
    UnsupportedMethodError,
)
from jevper.prompts import STRUCTURED_ANSWER_CUE, STRUCTURED_SYSTEM_PROMPT

CHOICE_LOGS = [("A", -0.12), ("B", -2.47), ("C", -3.48)]
CRITERIA = {"billing": None, "technical": None, "sales": None}


def test_grammar_sends_gbnf_on_the_chat_surface(stub_server):
    stub = stub_server(chat=lambda _: (200, chat_body(content="A", logprobs=CHOICE_LOGS)))
    client = SystemOneClient(openai_client(stub), model="stub", method="grammar")

    response = client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

    assert stub.paths == ["/v1/chat/completions"]
    sent = stub.bodies("/chat/completions")[0]
    assert sent["grammar"] == 'root ::= "A" | "B" | "C"\n'
    assert sent["logprobs"] is True
    assert sent["top_logprobs"] == 20
    assert response.answers["q"].choice == "billing"
    assert response.debug["method"] == "grammar"


def test_grammar_on_the_responses_surface_fails_before_any_request(stub_server):
    stub = stub_server(
        chat=lambda _: (200, chat_body(content="A")),
        responses=lambda _: (200, responses_body(text="A")),
    )
    client = SystemOneClient(openai_client(stub), model="stub", method="grammar", api="responses")

    with pytest.raises(UnsupportedMethodError) as error:
        client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

    assert "chat_completions" in str(error.value)
    assert stub.requests == []


def test_grammar_without_logprobs_points_at_discrete(stub_server):
    stub = stub_server(chat=lambda _: (200, chat_body(content="A")))
    client = SystemOneClient(openai_client(stub), model="stub", method="grammar")

    from jevper import LabelReadoutError

    with pytest.raises(LabelReadoutError) as error:
        client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

    assert "method='discrete'" in str(error.value)


def test_structured_reads_probabilities_verbatim(stub_server):
    payload = {"probabilities": {"billing": 0.8, "technical": 0.1, "sales": 0.1}}
    stub = stub_server(chat=lambda _: (200, chat_body(content=json.dumps(payload))))
    client = SystemOneClient(
        openai_client(stub), model="stub", method="structured", api="chat_completions"
    )

    response = client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

    answer = response.answers["q"]
    assert answer.probabilities == {"billing": 0.8, "technical": 0.1, "sales": 0.1}
    assert answer.choice == "billing"
    assert answer.confidence == pytest.approx(0.7, abs=1e-12)
    assert response.debug["probability_errors"] == {}
    assert response.debug["original_probabilities"] == {}
    sent = stub.bodies("/chat/completions")[0]
    response_format = sent["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["name"] == "jevper_choice"
    assert response_format["json_schema"]["strict"] is True
    schema = response_format["json_schema"]["schema"]
    assert schema["properties"]["probabilities"]["additionalProperties"] is False
    assert schema["required"] == ["probabilities"]


def test_structured_rescales_off_distribution_and_keeps_the_original(stub_server):
    payload = {"probabilities": {"billing": 0.5, "technical": 0.1, "sales": 0.1}}
    stub = stub_server(chat=lambda _: (200, chat_body(content=json.dumps(payload))))
    client = SystemOneClient(
        openai_client(stub), model="stub", method="structured", api="chat_completions"
    )

    response = client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

    answer = response.answers["q"]
    assert answer.probabilities["billing"] == pytest.approx(0.5 / 0.7, abs=1e-12)
    assert answer.probabilities["technical"] == pytest.approx(0.1 / 0.7, abs=1e-12)
    assert response.debug["probability_errors"]["q"] == pytest.approx(0.3, abs=1e-12)
    assert response.debug["original_probabilities"]["q"] == {
        "billing": 0.5,
        "technical": 0.1,
        "sales": 0.1,
    }


def test_structured_rescales_a_distribution_that_overflows_a_float(stub_server):
    """Three values near the float limit: the sum overflows, and a bare OverflowError is not a jevper error."""
    payload = {"probabilities": {"billing": 1e308, "technical": 1e308, "sales": 1e308}}
    stub = stub_server(chat=lambda _: (200, chat_body(content=json.dumps(payload))))
    client = SystemOneClient(
        openai_client(stub), model="stub", method="structured", api="chat_completions"
    )

    response = client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

    answer = response.answers["q"]
    assert answer.probabilities == {
        "billing": pytest.approx(1 / 3),
        "technical": pytest.approx(1 / 3),
        "sales": pytest.approx(1 / 3),
    }
    assert answer.confidence == pytest.approx(0.0, abs=1e-12)
    assert response.debug["probability_errors"]["q"] == math.inf
    assert response.debug["original_probabilities"]["q"] == {
        "billing": 1e308,
        "technical": 1e308,
        "sales": 1e308,
    }
    # The infinite error still has to serialize: pydantic writes a non-finite float as null.
    assert json.loads(response.model_dump_json())["debug"]["probability_errors"]["q"] is None


def test_structured_normalization_can_be_disabled(stub_server):
    payload = {"probabilities": {"billing": 0.5, "technical": 0.1, "sales": 0.1}}
    stub = stub_server(chat=lambda _: (200, chat_body(content=json.dumps(payload))))
    client = SystemOneClient(
        openai_client(stub),
        model="stub",
        method="structured",
        api="chat_completions",
        normalize_probabilities=False,
    )

    response = client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

    assert response.answers["q"].probabilities == {"billing": 0.5, "technical": 0.1, "sales": 0.1}
    assert response.debug["probability_errors"]["q"] == pytest.approx(0.3, abs=1e-12)
    assert response.debug["original_probabilities"] == {}


def test_structured_without_strict_outputs_falls_back_to_json_object(stub_server):
    payload = {"probabilities": {"billing": 0.5, "sales": 0.5}}
    stub = stub_server(chat=lambda _: (200, chat_body(content=json.dumps(payload))))
    client = SystemOneClient(
        openai_client(stub),
        model="stub",
        method="structured",
        api="chat_completions",
        structured_outputs=False,
    )

    client.system_one(state="s", questions={"q": Choice(criteria={"billing": None, "sales": None})})

    assert stub.bodies("/chat/completions")[0]["response_format"] == {"type": "json_object"}


def test_structured_noul_and_score_shapes(stub_server):
    bodies = {
        "jevper_noul": {"noul": 0.9},
        "jevper_score": {"probabilities": {"0": 0.1, "1": 0.2, "2": 0.7}},
    }

    def script(body):
        name = body["response_format"]["json_schema"]["name"]
        return 200, chat_body(content=json.dumps(bodies[name]))

    stub = stub_server(chat=script)
    client = SystemOneClient(
        openai_client(stub), model="stub", method="structured", api="chat_completions"
    )

    response = client.system_one(
        state="s",
        questions={"verdict": Noul(), "anger": Score(criteria=["Calm", "Frustrated", "Very angry"])},
    )

    assert response.answers["verdict"].noul == pytest.approx(0.9, abs=1e-12)
    score = response.answers["anger"]
    assert score.probabilities == {0: 0.1, 1: 0.2, 2: 0.7}
    assert score.score == pytest.approx(1.6, abs=1e-12)
    assert score.legend == {0: "Calm", 1: "Frustrated", 2: "Very angry"}


def test_discrete_reports_one_hot_distributions(stub_server):
    bodies = {
        "jevper_choice": {"choice": "technical"},
        "jevper_noul": {"noul": True},
        "jevper_score": {"score": 2},
    }

    def script(body):
        name = body["response_format"]["json_schema"]["name"]
        return 200, chat_body(content=json.dumps(bodies[name]))

    stub = stub_server(chat=script)
    client = SystemOneClient(openai_client(stub), model="stub", method="discrete", api="chat_completions")

    response = client.system_one(
        state="s",
        questions={
            "q": Choice(criteria=CRITERIA),
            "verdict": Noul(),
            "anger": Score(criteria=["Calm", "Frustrated", "Very angry"]),
        },
    )

    choice = response.answers["q"]
    assert choice.probabilities == {"billing": 0.0, "technical": 1.0, "sales": 0.0}
    assert choice.choice == "technical"
    assert choice.confidence == pytest.approx(1.0, abs=1e-12)
    assert response.answers["verdict"].noul == 1.0
    assert response.answers["anger"].probabilities == {0: 0.0, 1: 0.0, 2: 1.0}
    assert response.answers["anger"].score == 2.0
    assert response.debug["method"] == "discrete"


def test_structured_handles_more_than_26_options(stub_server):
    options = {f"option_{index}": None for index in range(30)}
    payload = {"probabilities": {name: (1.0 if name == "option_29" else 0.0) for name in options}}
    stub = stub_server(chat=lambda _: (200, chat_body(content=json.dumps(payload))))
    client = SystemOneClient(
        openai_client(stub), model="stub", method="structured", api="chat_completions"
    )

    response = client.system_one(state="s", questions={"q": Choice(criteria=options)})

    answer = response.answers["q"]
    assert answer.choice == "option_29"
    assert len(answer.probabilities) == 30
    assert sum(answer.probabilities.values()) == pytest.approx(1.0, abs=1e-12)
    assert answer.confidence == pytest.approx(1.0, abs=1e-12)
    schema = stub.bodies("/chat/completions")[0]["response_format"]["json_schema"]["schema"]
    properties = schema["properties"]["probabilities"]
    assert set(properties["properties"]) == set(options)
    assert len(properties["required"]) == 30
    assert properties["additionalProperties"] is False


def test_discrete_labels_grow_to_two_letters_beyond_26_options(stub_server):
    options = {f"option_{index}": None for index in range(30)}
    stub = stub_server(chat=lambda _: (200, chat_body(content=json.dumps({"choice": "BD"}))))
    client = SystemOneClient(openai_client(stub), model="stub", method="discrete", api="chat_completions")

    response = client.system_one(state="s", questions={"q": Choice(criteria=options)})

    # 30 options run AA..AZ then BA..BD, so BD is the thirtieth option
    assert response.answers["q"].choice == "option_29"
    assert response.answers["q"].probabilities["option_29"] == 1.0
    sent = stub.bodies("/chat/completions")[0]
    enum = sent["response_format"]["json_schema"]["schema"]["properties"]["choice"]["enum"]
    assert (enum[:3], enum[25:27], enum[-1]) == (["AA", "AB", "AC"], ["AZ", "BA"], "BD")
    assert len(enum) == 30
    assert "BD: option_29" in sent["messages"][-2]["content"]


def test_discrete_accepts_equivalent_score_index_forms(stub_server):
    answers = [{"score": 2.0}, {"score": "2"}, {"score": "2.0"}]
    stub = stub_server(chat=lambda _: (200, chat_body(content=json.dumps(answers.pop(0)))))
    client = SystemOneClient(openai_client(stub), model="stub", method="discrete", api="chat_completions")

    for _ in range(3):
        response = client.system_one(
            state="s", questions={"anger": Score(criteria=["Calm", "Frustrated", "Very angry"])}
        )
        assert response.answers["anger"].probabilities == {0: 0.0, 1: 0.0, 2: 1.0}
        assert response.answers["anger"].score == 2.0


def test_discrete_accepts_the_string_form_of_a_boolean(stub_server):
    answers = [{"noul": "false"}, {"noul": "TRUE"}]
    stub = stub_server(chat=lambda _: (200, chat_body(content=json.dumps(answers.pop(0)))))
    client = SystemOneClient(openai_client(stub), model="stub", method="discrete", api="chat_completions")

    assert client.system_one(state="s", questions={"v": Noul()}).answers["v"].noul == 0.0
    assert client.system_one(state="s", questions={"v": Noul()}).answers["v"].noul == 1.0


def test_discrete_rejects_a_score_that_is_not_a_level_index(stub_server):
    stub = stub_server(chat=lambda _: (200, chat_body(content=json.dumps({"score": "very angry"}))))
    client = SystemOneClient(openai_client(stub), model="stub", method="discrete", api="chat_completions")

    with pytest.raises(MalformedAnswerError) as error:
        client.system_one(state="s", questions={"anger": Score(criteria=["Calm", "Frustrated", "Very angry"])})

    assert "must be one of the level indexes" in str(error.value)
    assert len(stub.bodies("/chat/completions")) == 2  # one corrective retry, then the error


def test_responses_surface_logprobs(stub_server):
    stub = stub_server(responses=lambda _: (200, responses_body(text="A", logprobs=CHOICE_LOGS)))
    client = SystemOneClient(openai_client(stub), model="stub")

    response = client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

    assert stub.paths == ["/v1/responses"]
    sent = stub.bodies("/responses")[0]
    assert sent["top_logprobs"] == 20
    assert sent["store"] is False
    assert sent["include"] == ["message.output_text.logprobs"]
    assert "logprobs" not in sent
    answer = response.answers["q"]
    assert answer.choice == "billing"
    assert answer.probabilities["billing"] == pytest.approx(0.884873983, abs=1e-9)
    assert response.debug["api"] == "responses"


def test_grammar_without_logprobs_spends_no_corrective_retry(stub_server):
    """The server cannot be talked into reporting logprobs, so no correction turn is spent."""
    stub = stub_server(chat=lambda _: (200, chat_body(content="A")))
    client = SystemOneClient(openai_client(stub), model="stub", method="grammar")

    with pytest.raises(LabelReadoutError) as error:
        client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

    assert "grammar mode needs logprobs in the response" in str(error.value)
    assert len(stub.bodies("/chat/completions")) == 1


def test_discrete_asks_for_the_json_object_it_reads(stub_server):
    """The discrete schema is a JSON object, so the prompt, the demonstrations and the cue all say JSON."""

    def script(body):
        if "response_format" not in body:
            return 200, chat_body(content="The options differ in what the customer is asking about.")
        payload = {
            "jevper_choice": {"choice": "A"},
            "jevper_noul": {"noul": True},
            "jevper_score": {"score": 0},
        }[body["response_format"]["json_schema"]["name"]]
        return 200, chat_body(content=json.dumps(payload))

    stub = stub_server(chat=script)
    client = SystemOneClient(
        openai_client(stub),
        model="stub",
        method="discrete",
        api="chat_completions",
        reasoning=ReasoningConfig(mode="two_step"),
    )

    response = client.system_one(
        state="s",
        questions={
            "intent": Choice(
                criteria=CRITERIA, examples=[Example(state="duplicate charge", answer="billing")]
            ),
            "verdict": Noul(examples=[Example(state="clear yes", answer=True)]),
            "anger": Score(criteria=["Calm", "Frustrated"], examples=[Example(state="very calm", answer=0)]),
        },
    )

    answers = [body for body in stub.bodies("/chat/completions") if "response_format" in body]
    assert len(answers) == 3
    for body in answers:
        assert body["messages"][0]["content"] == STRUCTURED_SYSTEM_PROMPT
        assert body["messages"][-1] == {"role": "user", "content": STRUCTURED_ANSWER_CUE}
    demonstrated = {
        message["content"]
        for body in answers
        for message in body["messages"]
        if message["role"] == "assistant" and message["content"].startswith("{")
    }
    assert demonstrated == {'{"choice": "A"}', '{"noul": true}', '{"score": 0}'}
    assert response.answers["intent"].choice == "billing"
    assert response.answers["verdict"].noul == 1.0
    assert response.answers["anger"].score == 0.0


def test_schemas_bound_probabilities(stub_server):
    """A schema-valid answer is always a readable one, so a bad number costs no corrective retry."""
    stub = stub_server(
        chat=lambda _: (
            200,
            chat_body(content=json.dumps({"probabilities": {"billing": 1.0, "technical": 0.0, "sales": 0.0}})),
        )
    )
    client = SystemOneClient(
        openai_client(stub), model="stub", method="structured", api="chat_completions"
    )

    client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

    schema = stub.bodies("/chat/completions")[0]["response_format"]["json_schema"]["schema"]
    assert schema["properties"]["probabilities"]["properties"] == {
        key: {"type": "number", "minimum": 0} for key in CRITERIA
    }

    noul_stub = stub_server(chat=lambda _: (200, chat_body(content=json.dumps({"noul": 1.0}))))
    noul = SystemOneClient(
        openai_client(noul_stub), model="stub", method="structured", api="chat_completions"
    )

    noul.system_one(state="s", questions={"q": Noul()})

    noul_schema = noul_stub.bodies("/chat/completions")[0]["response_format"]["json_schema"]["schema"]
    assert noul_schema["properties"]["noul"] == {"type": "number", "minimum": 0, "maximum": 1}
