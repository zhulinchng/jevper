"""Client-side validation of structured and discrete answers.

Every case goes through the real client and the real ``openai`` SDK talking to ``StubServer``, so what
is pinned is the caller's outcome — the answer, or the exception type and the message it carries —
rather than the parse path behind it. A malformed answer costs one corrective retry (the default
``n_retry_malformed=1``), so a rejected answer is asked for twice before the error reaches the caller.
"""

from __future__ import annotations

import json

import pytest
from fakes import (
    StubServer,
    anthropic_client,
    chat_body,
    messages_body,
    openai_client,
    responses_body,
)

from jevper import (
    Choice,
    Example,
    IncompleteAnswerError,
    InvalidQuestionError,
    MalformedAnswerError,
    ModelRefusalError,
    Noul,
    Score,
    SystemOneClient,
)

STATE = "My invoice shows a charge I do not recognize and I need it explained."
CRITERIA = {"billing": None, "technical": None, "sales": None}
LEVELS = ["Calm", "Frustrated", "Very angry"]


def structured_client(stub_server, content, **kwargs):
    """A structured call whose provider answer is ``content`` verbatim."""
    stub = stub_server(chat=lambda _: (200, chat_body(content=content)))
    return SystemOneClient(
        openai_client(stub), model="stub", method="structured", api="chat_completions", **kwargs
    )


def discrete_client(stub_server, content, **kwargs):
    """A discrete call whose provider answer is ``content`` verbatim."""
    stub = stub_server(chat=lambda _: (200, chat_body(content=content)))
    return SystemOneClient(
        openai_client(stub), model="stub", method="discrete", api="chat_completions", **kwargs
    )


# --- finding the object in the answer ----------------------------------------------------------


def test_structured_reads_the_object_inside_prose(stub_server):
    content = (
        "The customer is asking about an invoice.\n"
        '{"probabilities": {"billing": 0.8, "technical": 0.1, "sales": 0.1}}\n'
        "Let me know if you need anything else."
    )
    client = structured_client(stub_server, content)

    response = client.system_one(state=STATE, questions={"q": Choice(criteria=CRITERIA)})

    answer = response.answers["q"]
    assert answer.choice == "billing"
    assert answer.probabilities == {"billing": 0.8, "technical": 0.1, "sales": 0.1}


def test_structured_reads_the_object_inside_a_markdown_fence(stub_server):
    content = '```json\n{"probabilities": {"billing": 0.2, "technical": 0.3, "sales": 0.5}}\n```'
    client = structured_client(stub_server, content)

    response = client.system_one(state=STATE, questions={"q": Choice(criteria=CRITERIA)})

    answer = response.answers["q"]
    assert answer.choice == "sales"
    assert answer.probabilities == {"billing": 0.2, "technical": 0.3, "sales": 0.5}


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ("[1, 2, 3]", "expected a JSON object, got list"),
        ('"billing"', "expected a JSON object, got str"),
        ("42", "expected a JSON object, got int"),
        ("null", "expected a JSON object, got NoneType"),
        ("true", "expected a JSON object, got bool"),
    ],
)
def test_structured_rejects_anything_that_is_not_a_json_object(stub_server, content, expected):
    client = structured_client(stub_server, content)

    with pytest.raises(MalformedAnswerError) as error:
        client.system_one(state=STATE, questions={"q": Choice(criteria=CRITERIA)})

    assert expected in str(error.value)


def test_structured_rejects_a_trailing_comma(stub_server):
    content = '{"probabilities": {"billing": 0.5, "technical": 0.3, "sales": 0.2,}}'
    client = structured_client(stub_server, content)

    with pytest.raises(MalformedAnswerError) as error:
        client.system_one(state=STATE, questions={"q": Choice(criteria=CRITERIA)})

    assert "could not parse a JSON object" in str(error.value)


# --- the numbers inside a structured answer ----------------------------------------------------


@pytest.mark.parametrize(
    ("probabilities", "expected"),
    [
        ({"billing": 0.5, "technical": 0.5}, ["must have exactly the option keys", "'sales'"]),
        (
            {"billing": 0.5, "technical": 0.3, "sales": 0.1, "other": 0.1},
            ["must have exactly the option keys", "'other'"],
        ),
        (
            {"billing": "high", "technical": 0.3, "sales": 0.1},
            ["probability for 'billing' must be a finite number", "'high'"],
        ),
        (
            {"billing": -0.5, "technical": 0.3, "sales": 0.1},
            ["probability for 'billing' must be >= 0", "-0.5"],
        ),
        (
            {"billing": float("nan"), "technical": 0.3, "sales": 0.1},
            ["probability for 'billing' must be a finite number"],
        ),
        (
            {"billing": float("inf"), "technical": 0.3, "sales": 0.1},
            ["probability for 'billing' must be a finite number"],
        ),
        (
            {"billing": True, "technical": 0.3, "sales": 0.1},
            ["probability for 'billing' must be a finite number", "True"],
        ),
    ],
)
def test_structured_rejects_a_probability_it_cannot_read(stub_server, probabilities, expected):
    client = structured_client(stub_server, json.dumps({"probabilities": probabilities}))

    with pytest.raises(MalformedAnswerError) as error:
        client.system_one(state=STATE, questions={"q": Choice(criteria=CRITERIA)})

    message = str(error.value)
    for fragment in expected:
        assert fragment in message


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (1.5, "'noul' must be in [0, 1.0], got 1.5"),
        (-0.1, "'noul' must be in [0, 1.0], got -0.1"),
    ],
)
def test_structured_rejects_a_noul_outside_the_unit_interval(stub_server, value, expected):
    client = structured_client(stub_server, json.dumps({"noul": value}))

    with pytest.raises(MalformedAnswerError) as error:
        client.system_one(state=STATE, questions={"q": Noul()})

    assert expected in str(error.value)


# --- discrete answers --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("answer", "expected_choice"),
    [
        ("B", "technical"),
        ("sales", "sales"),
        ("b", "technical"),
    ],
)
def test_discrete_reads_a_choice_given_as_a_label_or_an_option_key(
    stub_server, answer, expected_choice
):
    client = discrete_client(stub_server, json.dumps({"choice": answer}))

    response = client.system_one(state=STATE, questions={"q": Choice(criteria=CRITERIA)})

    answer_obj = response.answers["q"]
    assert answer_obj.choice == expected_choice
    assert answer_obj.probabilities == {
        key: (1.0 if key == expected_choice else 0.0) for key in CRITERIA
    }


def test_discrete_prefers_an_exact_option_key_over_a_label(stub_server):
    """An option key that is also a label must not be read as the first option by accident."""
    criteria = {"b": None, "A": None, "C": None}
    client = discrete_client(stub_server, json.dumps({"choice": "A"}))

    response = client.system_one(state=STATE, questions={"q": Choice(criteria=criteria)})

    answer = response.answers["q"]
    # "A" is the second option's key; the label "A" names the first option and must lose to it.
    assert answer.choice == "A"
    assert answer.probabilities == {"b": 0.0, "A": 1.0, "C": 0.0}


def test_discrete_rejects_a_choice_that_is_neither_a_label_nor_a_key(stub_server):
    client = discrete_client(stub_server, json.dumps({"choice": "zzz"}))

    with pytest.raises(MalformedAnswerError) as error:
        client.system_one(state=STATE, questions={"q": Choice(criteria=CRITERIA)})

    message = str(error.value)
    assert "'choice' must be one of the labels" in message
    assert "'zzz'" in message


@pytest.mark.parametrize("value", [2, 2.0, "2"])
def test_discrete_reads_a_score_index_in_any_integral_form(stub_server, value):
    client = discrete_client(stub_server, json.dumps({"score": value}))

    response = client.system_one(state=STATE, questions={"anger": Score(criteria=LEVELS)})

    answer = response.answers["anger"]
    assert answer.score == 2.0
    assert answer.probabilities == {0: 0.0, 1: 0.0, 2: 1.0}


@pytest.mark.parametrize("value", [1.5, 5, -1])
def test_discrete_rejects_a_score_that_is_not_a_level_index(stub_server, value):
    client = discrete_client(stub_server, json.dumps({"score": value}))

    with pytest.raises(MalformedAnswerError) as error:
        client.system_one(state=STATE, questions={"anger": Score(criteria=LEVELS)})

    message = str(error.value)
    assert "must be one of the level indexes" in message
    assert repr(value) in message


@pytest.mark.parametrize(
    ("answer", "expected_noul"),
    [
        (True, 1.0),
        ("true", 1.0),
        (False, 0.0),
    ],
)
def test_discrete_reads_a_noul_given_as_a_boolean_or_its_string(stub_server, answer, expected_noul):
    client = discrete_client(stub_server, json.dumps({"noul": answer}))

    response = client.system_one(state=STATE, questions={"verdict": Noul()})

    assert response.answers["verdict"].noul == expected_noul


# --- distributions that do not sum to one ------------------------------------------------------


def test_structured_all_zero_distribution_falls_back_to_uniform(stub_server):
    content = json.dumps({"probabilities": {"billing": 0, "technical": 0, "sales": 0}})
    client = structured_client(stub_server, content)

    response = client.system_one(state=STATE, questions={"q": Choice(criteria=CRITERIA)})

    answer = response.answers["q"]
    assert answer.probabilities == {key: pytest.approx(1 / 3) for key in CRITERIA}
    assert answer.confidence == pytest.approx(0.0, abs=1e-12)
    assert response.debug["probability_errors"]["q"] == pytest.approx(1.0, abs=1e-12)
    # The zeros the model sent are what the uniform fallback replaced, so they are kept.
    assert response.debug["original_probabilities"]["q"] == {
        "billing": 0.0,
        "technical": 0.0,
        "sales": 0.0,
    }


def test_structured_without_normalization_reports_the_models_own_numbers(stub_server):
    payload = {"probabilities": {"0": 0.5, "1": 0.1, "2": 0.1}}
    client = structured_client(stub_server, json.dumps(payload), normalize_probabilities=False)

    response = client.system_one(state=STATE, questions={"anger": Score(criteria=LEVELS)})

    answer = response.answers["anger"]
    assert answer.probabilities == {0: 0.5, 1: 0.1, 2: 0.1}
    # The score is an expected value, so it is read off the distribution rescaled to sum to one —
    # (0.5·0 + 0.1·1 + 0.1·2) / 0.7 — even though the reported probabilities stay the model's own.
    # Carrying the raw total into the score would put it at 0.3, below the 0 the model gave no mass
    # to, and the TypeSafe reference adapter rescales for exactly this.
    assert answer.score == pytest.approx(3 / 7, abs=1e-12)
    assert response.debug["probability_errors"]["anger"] == pytest.approx(0.3, abs=1e-12)
    # Nothing was rewritten, so there is no pre-normalization copy of the distribution.
    assert "original_probabilities" in response.debug
    assert response.debug["original_probabilities"] == {}


# --- the corrective retry ----------------------------------------------------------------------


def test_a_malformed_answer_is_retried_and_the_reason_names_the_failure(stub_server):
    answers = [json.dumps({"choice": "zzz"}), json.dumps({"choice": "sales"})]
    stub = stub_server(chat=lambda _: (200, chat_body(content=answers.pop(0))))
    client = SystemOneClient(
        openai_client(stub),
        model="stub",
        method="discrete",
        api="chat_completions",
        n_retry_malformed=1,
    )

    response = client.system_one(state=STATE, questions={"q": Choice(criteria=CRITERIA)})

    assert response.answers["q"].choice == "sales"
    assert len(stub.bodies("/chat/completions")) == 2
    reasons = response.debug["retry_reasons"]
    assert len(reasons) == 1
    assert "'choice' must be one of the labels" in reasons[0]
    correction = stub.bodies("/chat/completions")[1]["messages"][-1]
    assert correction["role"] == "user"
    assert "'choice' must be one of the labels" in correction["content"]


# --- a generation the provider withheld ----------------------------------------------------------


@pytest.mark.parametrize(
    ("body", "reported"),
    [
        (chat_body(content=None, finish_reason="content_filter"), "content_filter"),
        (chat_body(content='{"choice": "billing"}', finish_reason="content_filter"), "content_filter"),
    ],
)
def test_a_safety_filter_is_a_refusal_not_an_incomplete_answer(stub_server, body, reported):
    """A filtered generation was withheld on purpose; that is a refusal, not a cut-off answer.

    OpenAI reports it as ``finish_reason: "content_filter"`` and the Responses surface as an
    ``incomplete`` whose reason is the same word. Reading either as "stopped before the answer was
    complete" sends the caller looking for an output budget that was never the problem.
    """
    stub = stub_server(chat=lambda _: (200, body))
    client = SystemOneClient(
        openai_client(stub),
        model="stub",
        method="structured",
        api="chat_completions",
        n_retry_malformed=2,
    )

    with pytest.raises(ModelRefusalError) as raised:
        client.system_one(state=STATE, questions={"q": Choice(criteria=CRITERIA)})

    message = str(raised.value)
    assert "filtered the content for safety" in message
    assert reported in message
    assert len(stub.bodies("/chat/completions")) == 1


def test_a_responses_incomplete_reason_of_content_filter_is_a_refusal(stub_server):
    """The Responses surface says the same thing in ``incomplete_details.reason``."""
    body = {**responses_body(text=""), "status": "incomplete"}
    body["incomplete_details"] = {"reason": "content_filter"}
    stub = StubServer(responses=lambda _: (200, body, {}))
    client = SystemOneClient(
        openai_client(stub), model="stub", method="structured", api="responses", n_retry_malformed=2
    )

    with pytest.raises(ModelRefusalError) as raised:
        client.system_one(state=STATE, questions={"q": Choice(criteria=CRITERIA)})

    assert "content_filter" in str(raised.value)
    assert len(stub.bodies("/responses")) == 1


# --- the advice in a truncation error -------------------------------------------------------------


@pytest.mark.parametrize(
    ("surface", "reason", "knob", "beside"),
    [
        ("chat_completions", "length", "max_completion_tokens", "max_tokens"),
        ("messages", "max_tokens", "max_tokens", None),
        ("responses", "max_output_tokens", "max_output_tokens", None),
    ],
)
def test_a_truncation_error_names_the_surfaces_own_budget_field(
    stub_server, surface, reason, knob, beside
):
    """The three surfaces do not share a name for the output budget, and the advice has to follow.

    The Responses surface calls it ``max_output_tokens`` and refuses a ``max_tokens`` it does not
    know, so advice that names the Chat spelling is advice the caller cannot act on. On Chat the
    name has itself moved — OpenAI's route now takes ``max_completion_tokens`` and refuses
    ``max_tokens`` for its reasoning models — while most local servers still take only
    ``max_tokens``, so that spelling is named beside it.
    """
    if surface == "chat_completions":
        stub = StubServer(chat=lambda _: (200, chat_body(content="", finish_reason=reason), {}))
        sdk = openai_client(stub)
    elif surface == "messages":
        stub = StubServer(messages=lambda _: (200, messages_body(text="", stop_reason=reason), {}))
        sdk = anthropic_client(stub)
    else:
        body = {**responses_body(text=""), "status": "incomplete"}
        body["incomplete_details"] = {"reason": reason}
        stub = StubServer(responses=lambda _: (200, body, {}))
        sdk = openai_client(stub)
    client = SystemOneClient(
        sdk, model="stub", method="structured", api=surface, n_retry_malformed=1
    )

    with pytest.raises(IncompleteAnswerError) as raised:
        client.system_one(state=STATE, questions={"q": Choice(criteria=CRITERIA)})

    message = str(raised.value)
    assert f"extra_body={{{knob!r}: 2048}}" in message
    if beside is None:
        assert f"'{knob}'" in message and message.count("extra_body=") == 1
    else:
        assert f"'{beside}'" in message


# --- an example's own distribution -----------------------------------------------------------------


def test_probability_keys_that_collide_after_normalization_are_refused():
    """``{1: 0.9, "1": 0.1}`` are two keys in Python and one in JSON, so the demonstration would lie.

    The rendered assistant turn can only carry one number per option; picking either silently drops
    the other, and the model is then shown a distribution its caller never wrote. The question is
    refused where it is built — before a prompt is rendered, and long before a request.
    """
    with pytest.raises(InvalidQuestionError) as raised:
        Choice(
            criteria={"1": None, "2": None},
            examples=[Example(state="s", answer="1", probabilities={1: 0.9, "1": 0.1, 2: 0.8})],
        )

    assert "use one spelling" in str(raised.value)


def test_an_empty_noul_probability_mapping_is_refused():
    """A Noul example that carries no distribution renders a demonstration the renderer cannot read."""
    with pytest.raises(InvalidQuestionError) as raised:
        Noul(examples=[Example(state="s", answer=True, probabilities={})])

    assert "True or False key" in str(raised.value)


def test_noul_keys_that_collide_after_normalization_are_refused():
    """``{True: 0.2, "true": 0.8}`` are two keys in Python and one answer, so one number would vanish."""
    with pytest.raises(InvalidQuestionError) as raised:
        Noul(examples=[Example(state="s", answer=True, probabilities={True: 0.2, "true": 0.8})])

    assert "one spelling" in str(raised.value)


@pytest.mark.parametrize("key", ["True", "FALSE", "yes", 2])
def test_a_noul_key_the_renderer_cannot_find_is_refused(key):
    """Only the spellings the demonstration renderer looks up are answers: booleans, 0/1, 'true'/'false'.

    ``{"True": 0.9}`` used to pass validation and then raise a ``KeyError`` while the prompt was
    being rendered — after the questions ahead of it had already spent provider calls.
    """
    with pytest.raises(InvalidQuestionError) as raised:
        Noul(examples=[Example(state="s", answer=True, probabilities={key: 0.9})])

    assert "True or False key" in str(raised.value)


@pytest.mark.parametrize(
    "key,expected", [(True, 0.9), ("true", 0.9), (1, 0.9), (False, 0.1), ("false", 0.1), (0, 0.1)]
)
def test_a_noul_key_the_renderer_can_find_is_rendered(stub_server, key, expected):
    """Every accepted spelling reaches the demonstration with the caller's number, under its answer."""
    import json as _json

    answer = _json.dumps({"noul": 0.1})
    stub = stub_server(chat=lambda _: (200, chat_body(content=answer)))
    client = SystemOneClient(openai_client(stub), model="stub", method="structured")
    question = Noul(examples=[Example(state="s", answer=True, probabilities={key: 0.9})])

    client.system_one(state=STATE, questions={"q": question})

    sent = stub.bodies("/chat/completions")[0]
    assert _json.loads(sent["messages"][2]["content"])["noul"] == pytest.approx(expected)
