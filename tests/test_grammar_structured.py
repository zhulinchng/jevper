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
    ProviderError,
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

    body = stub.bodies("/chat/completions")[0]
    assert body["response_format"] == {"type": "json_object"}
    # ``json_object`` constrains the answer to be *an* object, not to be *this* object, so the schema
    # itself has to travel in the prompt — otherwise the structured system prompt's "matches the
    # provided schema exactly" refers to nothing that was provided.
    assert '"probabilities"' in body["messages"][0]["content"]


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


# --- boundaries the Jev API sets ------------------------------------------------------------------


def test_the_widest_choice_the_api_allows_is_answered(stub_server):
    """255 options is the documented maximum, and its last label is two letters long.

    The suite already proves 256 is refused; the widest legal question is the one where a label
    readout stops being single-letter and the mapping from label to key has to hold at the far end.
    """
    criteria = {f"option_{index}": None for index in range(255)}
    last = list(criteria)[-1]
    stub = stub_server(chat=lambda _: (200, chat_body(content='{"choice": "JU"}')))
    client = SystemOneClient(
        openai_client(stub), model="stub", method="discrete", api="chat_completions"
    )

    response = client.system_one(
        state="s", questions={"q": Choice(criteria=criteria, instructions="Pick one")}
    )

    answer = response.answers["q"]
    assert answer.choice == last
    assert answer.probabilities == {key: (1.0 if key == last else 0.0) for key in criteria}
    assert answer.confidence == 1.0


def test_the_widest_score_the_api_allows_is_answered(stub_server):
    """Ten levels is the documented maximum, and level 9 is the last index the legend carries."""
    levels = [str(index) for index in range(10)]
    stub = stub_server(chat=lambda _: (200, chat_body(content='{"score": 9}')))
    client = SystemOneClient(
        openai_client(stub), model="stub", method="discrete", api="chat_completions"
    )

    response = client.system_one(
        state="s", questions={"q": Score(criteria=levels, instructions="How bad?")}
    )

    answer = response.answers["q"]
    assert answer.score == 9.0
    assert answer.probabilities == {index: (1.0 if index == 9 else 0.0) for index in range(10)}
    assert answer.legend == {index: str(index) for index in range(10)}


@pytest.mark.parametrize(("value", "expected"), [("0", 0.0), ("1", 1.0)])
def test_a_noul_reads_both_ends_of_its_interval(stub_server, value, expected):
    """0 and 1 are inside ``[0, 1]``, and neither is normalized away from what the model said."""
    stub = stub_server(chat=lambda _: (200, chat_body(content=json.dumps({"noul": expected}))))
    client = SystemOneClient(
        openai_client(stub), model="stub", method="structured", api="chat_completions"
    )

    response = client.system_one(state="s", questions={"q": Noul()})

    assert response.answers["q"].noul == expected
    assert response.usage.n_calls == 1


def test_a_uniform_distribution_is_no_confidence_at_all(stub_server):
    """A flat distribution is the honest answer to a question the model cannot decide."""
    answer = '{"probabilities": {"billing": 0.5, "technical": 0.5}}'
    stub = stub_server(chat=lambda _: (200, chat_body(content=answer)))
    client = SystemOneClient(
        openai_client(stub), model="stub", method="structured", api="chat_completions"
    )

    response = client.system_one(
        state="s", questions={"q": Choice(criteria={"billing": None, "technical": None})}
    )

    result = response.answers["q"]
    assert result.confidence == 0.0
    assert result.probabilities == {"billing": 0.5, "technical": 0.5}


def test_a_tie_is_broken_by_criteria_order_not_alphabet(stub_server):
    """Two equal maxima resolve to the first option the caller wrote, which is a choice they made."""
    answer = '{"probabilities": {"zeta": 0.5, "alpha": 0.5}}'
    stub = stub_server(chat=lambda _: (200, chat_body(content=answer)))
    client = SystemOneClient(
        openai_client(stub), model="stub", method="structured", api="chat_completions"
    )

    response = client.system_one(
        state="s",
        questions={"q": Choice(criteria={"zeta": "Z", "alpha": "A"})},
    )

    result = response.answers["q"]
    assert result.choice == "zeta"
    assert result.confidence == 0.0
    assert list(result.probabilities) == ["zeta", "alpha"]
    assert response.debug["probability_errors"] == {}


# --- a distribution the provider did not actually report -------------------------------------------


@pytest.mark.parametrize("bad", [0.9, 1.0, 2.5])
def test_a_positive_logprob_is_not_a_distribution(stub_server, bad):
    """A log probability is never above zero, so one that is says the field carries something else.

    A gateway that passes raw probabilities through as logprobs would otherwise be exponentiated into
    a confident-looking split no model ever reported.
    """
    body = chat_body(content="A", logprobs=[("A", bad)], alternatives=[("A", bad), ("B", -0.1)])
    stub = stub_server(chat=lambda _: (200, body))
    client = SystemOneClient(
        openai_client(stub), model="stub", method="logprobs", api="chat_completions"
    )

    with pytest.raises(LabelReadoutError) as raised:
        client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

    assert "positive" in str(raised.value)


def test_alternatives_that_carry_no_logprob_are_not_a_distribution(stub_server):
    """Two null alternatives are two missing labels, and with no rival at all there is nothing to read."""
    body = chat_body(content="A")
    body["choices"][0]["logprobs"] = {
        "content": [
            {
                "token": "A",
                "logprob": 0.0,
                "bytes": [65],
                "top_logprobs": [
                    {"token": "B", "logprob": None, "bytes": [66]},
                    {"token": "C", "logprob": None, "bytes": [67]},
                ],
            }
        ]
    }
    stub = stub_server(chat=lambda _: (200, body))
    client = SystemOneClient(
        openai_client(stub),
        model="stub",
        method="logprobs",
        api="chat_completions",
        n_retry_malformed=0,
    )

    with pytest.raises(LabelReadoutError) as raised:
        client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

    assert "none of which carried a logprob" in str(raised.value)


def test_a_sampled_token_that_contradicts_the_answer_text_is_refused(stub_server):
    """The logprobs and the text are two views of one generation; when they disagree, neither is trusted."""
    body = chat_body(content="B", logprobs=[("A", -0.12), ("B", -2.47), ("C", -3.48)])
    stub = stub_server(chat=lambda _: (200, body))
    client = SystemOneClient(
        openai_client(stub),
        model="stub",
        method="logprobs",
        api="chat_completions",
        n_retry_malformed=0,
    )

    with pytest.raises(LabelReadoutError) as raised:
        client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

    assert "contradicts the answer text" in str(raised.value)


def test_a_sampled_token_that_contradicts_a_punctuated_label_is_refused(stub_server):
    """``A.`` names label ``A``; punctuation after the letter is not prose that hides the claim."""
    body = chat_body(content="A.", logprobs=[("B", -0.12), ("A", -2.47), ("C", -3.48)])
    stub = stub_server(chat=lambda _: (200, body))
    client = SystemOneClient(
        openai_client(stub),
        model="stub",
        method="logprobs",
        api="chat_completions",
        n_retry_malformed=0,
    )

    with pytest.raises(LabelReadoutError) as raised:
        client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

    assert "contradicts the answer text" in str(raised.value)


def test_a_label_with_punctuation_is_read_as_that_label(stub_server):
    """The same matcher on both sides: ``A.`` in the text and ``A`` sampled agree, and answer."""
    body = chat_body(content="A.", logprobs=[("A", -0.12), ("B", -2.47)])
    stub = stub_server(chat=lambda _: (200, body))
    client = SystemOneClient(
        openai_client(stub), model="stub", method="logprobs", api="chat_completions"
    )

    response = client.system_one(
        state="s", questions={"q": Choice(criteria={"billing": None, "sales": None})}
    )

    assert response.answers["q"].choice == "billing"


def test_a_positive_alternative_beside_a_non_option_token_is_refused(stub_server):
    """A log probability is never positive, whatever token it belongs to.

    The value used to be filtered by label first, so a broken number on a token outside the option
    set was dropped quietly and the remaining one-hot stood.
    """
    body = chat_body(content="A", logprobs=[("A", -0.1)])
    body["choices"][0]["logprobs"]["content"][0]["top_logprobs"].append(
        {"token": "X", "logprob": 0.9, "bytes": [88]}
    )
    stub = stub_server(chat=lambda _: (200, body))
    client = SystemOneClient(
        openai_client(stub),
        model="stub",
        method="logprobs",
        api="chat_completions",
        n_retry_malformed=0,
    )

    with pytest.raises(LabelReadoutError) as raised:
        client.system_one(
            state="s", questions={"q": Choice(criteria={"billing": None, "sales": None})}
        )

    assert "positive" in str(raised.value)


def test_an_answer_text_that_is_not_a_label_does_not_contradict_anything(stub_server):
    """A JSON answer on the logprob channel is not a rival claim: the text is prose, the token is the label."""
    body = chat_body(
        content='{"probabilities": {"billing": 0.8, "technical": 0.1, "sales": 0.1}}',
        logprobs=CHOICE_LOGS,
    )
    stub = stub_server(chat=lambda _: (200, body))
    client = SystemOneClient(
        openai_client(stub), model="stub", method="logprobs", api="chat_completions"
    )

    response = client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

    assert response.answers["q"].choice == "billing"


# --- the answer's own JSON -------------------------------------------------------------------------


@pytest.mark.parametrize(
    "second",
    [
        '{"probabilities": {"billing": 0.1, "technical": 0.1, "sales": 0.8}}',
        ' and then {"probabilities": {"billing": 0.1, "technical": 0.1, "sales": 0.8}}',
        # A brace in prose is not an answer, and must not hide the second one behind it.
        ' example {not-json}; correction: {"probabilities": {"billing": 0.1, "technical": 0.1, "sales": 0.8}}',
    ],
)
def test_two_json_objects_in_one_answer_are_malformed(stub_server, second):
    """A generation that answered twice contradicts itself, and reading the first hides that."""
    first = '{"probabilities": {"billing": 0.8, "technical": 0.1, "sales": 0.1}}'
    stub = stub_server(chat=lambda _: (200, chat_body(content=first + second)))
    client = SystemOneClient(
        openai_client(stub),
        model="stub",
        method="structured",
        api="chat_completions",
        n_retry_malformed=0,
    )

    with pytest.raises(MalformedAnswerError) as raised:
        client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

    assert "more than one JSON object" in str(raised.value)


def test_a_brace_in_prose_before_the_answer_is_stepped_over(stub_server):
    """A model that shows the shape it was asked for has still answered; the object after it is read."""
    answer = '{"probabilities": {"billing": 0.8, "technical": 0.1, "sales": 0.1}}'
    stub = stub_server(
        chat=lambda _: (200, chat_body(content="I will use this shape: {example}; answer: " + answer))
    )
    client = SystemOneClient(
        openai_client(stub),
        model="stub",
        method="structured",
        api="chat_completions",
        n_retry_malformed=0,
    )

    response = client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

    assert response.answers["q"].choice == "billing"


@pytest.mark.parametrize("digits", [400, 5000])
def test_a_number_too_large_to_be_one_is_malformed(stub_server, digits):
    """A 400-digit integer parses and overflows a float; a 5000-digit one does not even parse.

    Neither may leave as a raw ``OverflowError``/``ValueError`` from the middle of a readout.
    """
    stub = stub_server(
        chat=lambda _: (200, chat_body(content='{"probabilities": {"billing": ' + "9" * digits + "}}"))
    )
    client = SystemOneClient(
        openai_client(stub),
        model="stub",
        method="structured",
        api="chat_completions",
        n_retry_malformed=0,
    )

    with pytest.raises(MalformedAnswerError):
        client.system_one(state="s", questions={"q": Choice(criteria={"billing": None})})


# --- a caller who takes over a field --------------------------------------------------------------


def test_a_caller_who_names_logprobs_still_gets_the_alternatives(stub_server):
    """``logprobs`` and ``top_logprobs`` are two fields; naming the first is not naming the second.

    Sending ``logprobs: true`` alone returns the sampled token's own logprob and no rivals, which the
    label readout cannot read — a caller who asked for the distribution gets a failure instead.
    """
    stub = stub_server(chat=lambda _: (200, chat_body(content="A", logprobs=CHOICE_LOGS)))
    client = SystemOneClient(
        openai_client(stub),
        model="stub",
        method="logprobs",
        api="chat_completions",
        extra_body={"logprobs": True},
    )

    response = client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

    body = stub.bodies("/chat/completions")[0]
    assert body["logprobs"] is True
    assert body["top_logprobs"] == 20
    assert response.answers["q"].choice == "billing"


def test_a_caller_who_turns_logprobs_off_sends_neither_field(stub_server):
    """``top_logprobs`` without ``logprobs`` is a 400 on OpenAI, so the caller's off is respected whole."""
    def script(body):
        # A server that was not asked for logprobs does not send any.
        if body.get("logprobs"):
            return 200, chat_body(content="A", logprobs=CHOICE_LOGS)
        return 200, chat_body(content="A")

    stub = stub_server(chat=script)
    client = SystemOneClient(
        openai_client(stub),
        model="stub",
        method="logprobs",
        api="chat_completions",
        extra_body={"logprobs": False},
        n_retry_malformed=0,
    )

    with pytest.raises(LabelReadoutError):
        client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

    body = stub.bodies("/chat/completions")[0]
    assert "top_logprobs" not in body
    assert body["logprobs"] is False


def test_a_structured_answer_with_an_extra_key_is_malformed(stub_server):
    """A model that says both "billing" and "technical" has contradicted itself; the field jevper
    happened to read is not the one it can vouch for, and the schema says no extra keys."""
    answer = json.dumps(
        {"probabilities": {"billing": 0.7, "technical": 0.2, "sales": 0.1}, "choice": "technical"}
    )
    stub = stub_server(chat=lambda _: (200, chat_body(content=answer)))
    client = SystemOneClient(
        openai_client(stub), model="stub", method="structured", api="chat_completions",
        n_retry_malformed=0,
    )

    with pytest.raises(MalformedAnswerError) as error:
        client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

    assert "'choice'" in str(error.value)
    assert "exactly 'probabilities'" in str(error.value)


def test_a_structured_answer_missing_its_key_is_malformed(stub_server):
    stub = stub_server(chat=lambda _: (200, chat_body(content=json.dumps({"noul": 0.5}))))
    client = SystemOneClient(
        openai_client(stub), model="stub", method="structured", api="chat_completions",
        n_retry_malformed=0,
    )

    with pytest.raises(MalformedAnswerError, match="exactly 'probabilities'"):
        client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})


def test_a_discrete_answer_with_an_extra_key_is_malformed(stub_server):
    """``{"choice": "A", "probabilities": {...}}`` is the same contradiction in the other shape."""
    answer = json.dumps({"choice": "A", "probabilities": {"billing": 0.1, "technical": 0.1, "sales": 0.8}})
    stub = stub_server(chat=lambda _: (200, chat_body(content=answer)))
    client = SystemOneClient(
        openai_client(stub), model="stub", method="discrete", api="chat_completions",
        n_retry_malformed=0,
    )

    with pytest.raises(MalformedAnswerError, match="exactly 'choice'"):
        client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})


def test_a_corrective_retry_gets_a_turn_at_an_extra_key(stub_server):
    """The bounded correction is the documented answer to a malformed one: one more request."""
    answers = [
        json.dumps({"probabilities": {"billing": 0.7, "technical": 0.2, "sales": 0.1}, "choice": "sales"}),
        json.dumps({"probabilities": {"billing": 0.7, "technical": 0.2, "sales": 0.1}}),
    ]
    stub = stub_server(chat=lambda _: (200, chat_body(content=answers.pop(0))))
    client = SystemOneClient(
        openai_client(stub), model="stub", method="structured", api="chat_completions"
    )

    response = client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

    assert response.answers["q"].choice == "billing"
    assert len(stub.bodies("/chat/completions")) == 2
    assert any("choice" in str(reason) for reason in response.debug["retry_reasons"])


def test_an_option_key_with_spaces_is_the_option_the_model_named(stub_server):
    """A key spelled with surrounding whitespace is that key; only an unmatched answer is stripped."""
    options = {" billing ": None, "other": None}
    answer = json.dumps({"choice": " billing "})
    stub = stub_server(chat=lambda _: (200, chat_body(content=answer)))
    client = SystemOneClient(
        openai_client(stub), model="stub", method="discrete", api="chat_completions",
        n_retry_malformed=0,
    )

    response = client.system_one(state="s", questions={"q": Choice(criteria=options)})

    assert response.answers["q"].choice == " billing "


def test_a_label_with_surrounding_whitespace_is_still_a_label(stub_server):
    """Stripping is for the label form, which is where a model adds punctuation."""
    answer = json.dumps({"choice": " A "})
    stub = stub_server(chat=lambda _: (200, chat_body(content=answer)))
    client = SystemOneClient(
        openai_client(stub), model="stub", method="discrete", api="chat_completions",
        n_retry_malformed=0,
    )

    response = client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

    assert response.answers["q"].choice == "billing"


def test_a_level_index_that_is_not_integral_is_malformed(stub_server):
    """``"2.0000000000000000000001"`` is not level 2, even though a float would round it there."""
    answer = json.dumps({"score": "2.0000000000000000000001"})
    stub = stub_server(chat=lambda _: (200, chat_body(content=answer)))
    client = SystemOneClient(
        openai_client(stub), model="stub", method="discrete", api="chat_completions",
        n_retry_malformed=0,
    )

    with pytest.raises(MalformedAnswerError, match="level indexes"):
        client.system_one(
            state="s", questions={"anger": Score(criteria=["Calm", "Frustrated", "Very angry"])}
        )


def test_a_written_integral_level_is_still_accepted(stub_server):
    """The rule is integrality, not a spelling: trailing zeros are what a model writes."""
    answers = [{"score": "2.000"}, {"score": 2.0}, {"score": "2"}]
    stub = stub_server(chat=lambda _: (200, chat_body(content=json.dumps(answers.pop(0)))))
    client = SystemOneClient(
        openai_client(stub), model="stub", method="discrete", api="chat_completions",
        n_retry_malformed=0,
    )

    for _ in range(3):
        response = client.system_one(
            state="s", questions={"anger": Score(criteria=["Calm", "Frustrated", "Very angry"])}
        )
        assert response.answers["anger"].score == 2.0


def test_a_malformed_answer_message_does_not_quote_the_whole_payload(stub_server):
    """A provider that pads its answer by a megabyte does not put that megabyte in every error."""
    answer = json.dumps(
        {"probabilities": {"billing": "not a number", "technical": 0.2, "sales": 0.1}, "pad": "x" * 900_000}
    )
    stub = stub_server(chat=lambda _: (200, chat_body(content=answer)))
    client = SystemOneClient(
        openai_client(stub), model="stub", method="structured", api="chat_completions",
        n_retry_malformed=0,
    )

    with pytest.raises(MalformedAnswerError) as error:
        client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

    assert len(str(error.value)) < 1_000
    assert "x" * 10_000 not in str(error.value)


def test_deeply_nested_json_is_a_malformed_answer_not_a_recursion_error(stub_server):
    """CPython's decoder refuses to nest past its limit; that is the answer's problem, not the caller's."""
    answer = '{"a":' * 2000 + "1" + "}" * 2000
    stub = stub_server(chat=lambda _: (200, chat_body(content=answer)))
    client = SystemOneClient(
        openai_client(stub), model="stub", method="structured", api="chat_completions",
        n_retry_malformed=0,
    )

    with pytest.raises(MalformedAnswerError):
        client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})


def test_a_nested_object_beside_the_answer_is_still_malformed_by_its_keys(stub_server):
    """A nested value inside an otherwise valid shape fails on the key set, not on the parser.

    Sixty levels, not two thousand: deep enough that the record of the answer has to be summarized
    rather than walked whole, and shallow enough that the *test's own* ``json.dumps`` succeeds on
    every runtime the package supports — 3.10's encoder refuses far sooner than 3.12's, and a test
    that only passes on the newest one is testing the interpreter, not jevper.
    """
    deep: object = 1
    for _ in range(60):
        deep = {"deeper": deep}
    answer = json.dumps({"probabilities": {"billing": 0.7, "technical": 0.2, "sales": 0.1}, "extra": deep})
    stub = stub_server(chat=lambda _: (200, chat_body(content=answer)))
    client = SystemOneClient(
        openai_client(stub), model="stub", method="structured", api="chat_completions",
        n_retry_malformed=0,
    )

    with pytest.raises(MalformedAnswerError, match="exactly 'probabilities'"):
        client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})


def test_grammar_is_not_rotated_onto_a_surface_that_cannot_carry_it(stub_server):
    """A grammar is a Chat Completions convention with no counterpart on either other surface, so when
    that route is gone there is nowhere to go. Rotating anyway spends a request whose constraint the
    other surface drops from the body in silence, and only then reports the same verdict the caller
    gets without it — so the 404 is the answer, and the prompt surface is never asked."""
    stub = stub_server(responses=lambda _: (200, responses_body(text="A")))
    client = SystemOneClient(openai_client(stub), model="stub", method="grammar")

    with pytest.raises(ProviderError) as error:
        client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

    assert error.value.status_code == 404
    assert stub.paths == ["/v1/chat/completions"]
