"""``logprobs`` readout: distribution, confidence, label coverage and corrective retries."""

from __future__ import annotations

import math

import pytest
from fakes import chat_body, openai_client

from jevper import Choice, LabelReadoutError, Noul, Score, SystemOneClient

CHOICE_LOGS = [("A", -0.12), ("B", -2.47), ("C", -3.48)]


def test_choice_logprobs_distribution_and_confidence(stub_server):
    stub = stub_server(chat=lambda _: (200, chat_body(content="A", logprobs=CHOICE_LOGS)))
    client = SystemOneClient(openai_client(stub), model="stub", api="chat_completions")

    response = client.system_one(
        state="I was charged twice this month",
        questions={"intent": Choice(criteria={"billing": None, "technical": None, "sales": None})},
    )

    answer = response.answers["intent"]
    assert answer.choice == "billing"
    assert list(answer.probabilities) == ["billing", "technical", "sales"]
    assert answer.probabilities["billing"] == pytest.approx(0.884873983, abs=1e-9)
    assert answer.probabilities["technical"] == pytest.approx(0.084389690, abs=1e-9)
    assert answer.probabilities["sales"] == pytest.approx(0.030736327, abs=1e-9)
    assert math.fsum(answer.probabilities.values()) == pytest.approx(1.0, abs=1e-12)
    assert answer.confidence == pytest.approx(0.827310974, abs=1e-9)
    assert response.usage.n_calls == 1
    assert response.usage.input_tokens == 10
    assert response.usage.output_tokens == 3
    assert response.debug["method"] == "logprobs"
    assert response.debug["api"] == "chat_completions"
    assert response.debug["llm_attempts"][0]["readout"]["source"] == "logprobs"

    sent = stub.bodies("/chat/completions")[0]
    assert sent["logprobs"] is True
    assert sent["top_logprobs"] == 20
    assert "max_tokens" not in sent and "max_completion_tokens" not in sent
    assert "stop" not in sent
    assert sent["messages"][0]["role"] == "system"
    assert "I was charged twice this month" in sent["messages"][-1]["content"]
    assert "A: billing" in sent["messages"][-2]["content"]


def test_noul_logprobs_has_no_confidence(stub_server):
    stub = stub_server(chat=lambda _: (200, chat_body(content="A", logprobs=[("A", -0.05), ("B", -3.0)])))
    client = SystemOneClient(openai_client(stub), model="stub", api="chat_completions")

    response = client.system_one(state="The sky is blue", questions={"yes": Noul()})

    answer = response.answers["yes"]
    assert answer.noul == pytest.approx(0.950263488, abs=1e-9)
    assert "confidence" not in answer.model_dump()
    question = stub.bodies("/chat/completions")[0]["messages"][-2]["content"]
    assert "A: Yes" in question and "B: No" in question


def test_score_logprobs_weighted_score_and_confidence(stub_server):
    stub = stub_server(chat=lambda _: (200, chat_body(content="A", logprobs=[("A", -8.0), ("B", -0.05), ("C", -3.0)])))
    client = SystemOneClient(openai_client(stub), model="stub", api="chat_completions")

    response = client.system_one(
        state="Customer is furious about the delay",
        questions={"anger": Score(criteria=["Calm", "Frustrated", "Very angry"])},
    )

    answer = response.answers["anger"]
    assert answer.probabilities[0] == pytest.approx(0.000335010, abs=1e-9)
    assert answer.probabilities[1] == pytest.approx(0.949945141, abs=1e-9)
    assert answer.probabilities[2] == pytest.approx(0.049719849, abs=1e-9)
    assert answer.score == pytest.approx(1.049384840, abs=1e-9)
    assert answer.legend == {0: "Calm", 1: "Frustrated", 2: "Very angry"}
    assert answer.confidence == pytest.approx(0.924917711, abs=1e-9)


def test_label_readout_skips_leading_whitespace_tokens(stub_server):
    stub = stub_server(
        chat=lambda _: (200, chat_body(content="A", logprobs=[("\n", -0.01), ("A", -0.12), ("B", -2.47)]))
    )
    client = SystemOneClient(openai_client(stub), model="stub", api="chat_completions")

    response = client.system_one(state="s", questions={"q": Choice(criteria={"billing": None, "sales": None})})

    assert response.answers["q"].choice == "billing"


def test_top_logprobs_omitting_a_label_reports_zero_and_debug(stub_server):
    stub = stub_server(chat=lambda _: (200, chat_body(content="A", logprobs=[("A", -0.12), ("B", -2.47)])))
    client = SystemOneClient(openai_client(stub), model="stub", api="chat_completions")

    response = client.system_one(
        state="s", questions={"q": Choice(criteria={"billing": None, "technical": None, "sales": None})}
    )

    assert response.answers["q"].probabilities["sales"] == 0.0
    assert response.debug["labels_missing"] == {"q": ["C"]}


def test_non_label_answer_is_retried_then_raises(stub_server):
    body = chat_body(
        content="The answer is A",
        logprobs=[("The", -0.2), (" answer", -0.3), (" is", -0.4), (" A", -0.5)],
    )
    stub = stub_server(chat=lambda _: (200, body))
    client = SystemOneClient(openai_client(stub), model="stub", api="chat_completions")

    with pytest.raises(LabelReadoutError) as error:
        client.system_one(state="s", questions={"q": Choice(criteria={"billing": None, "sales": None})})

    assert "not one of the labels" in str(error.value)
    assert "first non-whitespace token 'The'" in str(error.value)
    requests = stub.bodies("/chat/completions")
    assert len(requests) == 2
    assert "Your previous reply was invalid" in requests[1]["messages"][-1]["content"]
    assert "A, B" in requests[1]["messages"][-1]["content"]


def test_a_separated_reasoning_trace_does_not_hide_the_answer_token(stub_server):
    """A reasoning server reports logprobs for every generated token, the thinking span included.

    vLLM, SGLang and ollama all do this while separating the trace into ``reasoning_content``, so the
    first token of the stream is the first token of the *thinking*. The answer text is the anchor: the
    token that starts its exact tail is the answer's first token, and the distribution is read there.
    """
    stream = [
        ("The", -0.1),
        (" user", -0.2),
        (" was", -0.3),
        (" charged", -0.4),
        (" twice", -0.5),
        (".", -0.6),
        ("A", -0.12),
    ]
    body = chat_body(
        content="A",
        logprobs=stream,
        alternatives=[("A", -0.12), ("B", -2.47), ("C", -3.48)],
        reasoning="The user was charged twice, so this is about money.",
    )
    stub = stub_server(chat=lambda _: (200, body))
    client = SystemOneClient(openai_client(stub), model="stub", api="chat_completions", method="logprobs")

    response = client.system_one(state="s", questions={"q": Choice(criteria={"billing": None, "technical": None, "sales": None})})

    answer = response.answers["q"]
    assert answer.choice == "billing"
    assert answer.probabilities["billing"] == pytest.approx(0.884873983, abs=1e-9)
    assert response.usage.n_calls == 1  # the readout needed no corrective retry
    assert response.debug["retry_reasons"] == []


def test_a_trailing_end_of_turn_token_does_not_hide_the_answer(stub_server):
    """vLLM and SGLang append their own ``<|im_end|>`` to the stream after the answer text.

    Without dropping it the tail test fails, and a thinking-on label readout reports the first token of
    the *thinking* as the answer — which is exactly what both servers did.
    """
    stream = [
        ("The", -0.1),
        (" user", -0.2),
        (" was", -0.3),
        (" charged", -0.4),
        (" twice", -0.5),
        (".", -0.6),
        ("A", -0.12),
        ("<|im_end|>", -0.01),
    ]
    body = chat_body(
        content="A",
        logprobs=stream,
        alternatives=[("A", -0.12), ("B", -2.47), ("C", -3.48)],
        reasoning="The user was charged twice, so this is about money.",
    )
    stub = stub_server(chat=lambda _: (200, body))
    client = SystemOneClient(
        openai_client(stub), model="stub", api="chat_completions", method="logprobs"
    )

    response = client.system_one(
        state="s", questions={"q": Choice(criteria={"billing": None, "technical": None, "sales": None})}
    )

    assert response.answers["q"].choice == "billing"
    assert response.usage.n_calls == 1
    assert response.debug["retry_reasons"] == []


def test_the_trailing_token_tolerance_is_bounded(stub_server):
    """Three trailing tokens are not an end-of-turn marker, they are the answer: the anchor refuses."""
    stream = [
        ("The", -0.1),
        (" user", -0.2),
        (" was", -0.3),
        (" charged", -0.4),
        ("A", -0.12),
        ("x", -1.0),
        ("y", -1.0),
        ("z", -1.0),
    ]
    body = chat_body(
        content="A",
        logprobs=stream,
        alternatives=[("A", -0.12), ("B", -2.47), ("C", -3.48)],
        reasoning="The user was charged twice, so this is about money.",
    )
    stub = stub_server(chat=lambda _: (200, body))
    client = SystemOneClient(
        openai_client(stub), model="stub", api="chat_completions", method="logprobs", n_retry_malformed=0
    )

    with pytest.raises(LabelReadoutError) as error:
        client.system_one(
            state="s",
            questions={"q": Choice(criteria={"billing": None, "technical": None, "sales": None})},
        )

    assert "'The'" in str(error.value)


def test_the_answer_is_the_tail_of_the_stream_not_its_first_mention(stub_server):
    """The trace weighs the options by name; only the last occurrence of the answer text is the answer."""
    body = chat_body(
        content="B",
        logprobs=[
            ("The", -0.2),
            (" user", -0.3),
            (" paid", -0.4),
            (" twice", -0.5),
            (",", -0.6),
            (" so", -0.7),
            ("A", -0.8),
            (" is", -0.9),
            (" wrong", -1.0),
            (" and", -1.1),
            (" B", -1.2),
            (" fits", -1.3),
            (".", -1.4),
            ("B", -0.05),
        ],
        alternatives=[("B", -0.05), ("A", -3.0), ("C", -4.0)],
        reasoning="The user paid twice, so A is wrong and B fits.",
    )
    stub = stub_server(chat=lambda _: (200, body))
    client = SystemOneClient(openai_client(stub), model="stub", api="chat_completions", method="logprobs")

    response = client.system_one(state="s", questions={"q": Choice(criteria={"billing": None, "technical": None, "sales": None})})

    answer = response.answers["q"]
    assert answer.choice == "technical"  # B, not the A the trace rules out
    # The distribution is the final token's: the trace's mention of B carries logprob -1.2 there.
    assert answer.probabilities["technical"] == pytest.approx(0.933188894, abs=1e-9)


def test_a_trailing_token_the_answer_contains_is_not_dropped(stub_server):
    """The end-of-turn tolerance is only for tokens that cannot be part of the answer.

    A token whose text occurs in the answer could be part of it, so nothing is skipped and the stream is
    read as it arrives — reporting the first token as it is, rather than guessing where the answer
    starts. A server's own ``<|im_end|>`` cannot be part of the answer and is dropped instead; that is
    the difference between removing what the provider excluded from ``content`` and guessing.
    """
    body = chat_body(
        content="AB",
        logprobs=[
            ("The", -0.2),
            (" user", -0.3),
            (" paid", -0.4),
            (" twice", -0.5),
            (".", -0.6),
            ("A", -0.12),
            ("A", -0.9),
        ],
        alternatives=[("A", -0.12), ("B", -2.47), ("C", -3.48)],
        reasoning="The user paid twice.",
    )
    stub = stub_server(chat=lambda _: (200, body))
    client = SystemOneClient(openai_client(stub), model="stub", api="chat_completions", method="logprobs")

    with pytest.raises(LabelReadoutError) as error:
        client.system_one(state="s", questions={"q": Choice(criteria={"billing": None, "technical": None, "sales": None})})

    assert "first non-whitespace token 'The'" in str(error.value)


def test_corrective_retry_succeeds_and_records_reason(stub_server):
    good = chat_body(content="B", logprobs=[("B", -0.05), ("A", -3.0)])
    bad = chat_body(content="maybe", logprobs=[("maybe", -0.1), ("A", -2.0), ("B", -2.5)])
    calls: list[dict] = []

    def script(body):
        calls.append(body)
        return (200, bad) if len(calls) == 1 else (200, good)

    stub = stub_server(chat=script)
    client = SystemOneClient(openai_client(stub), model="stub", api="chat_completions")

    response = client.system_one(state="s", questions={"q": Choice(criteria={"billing": None, "sales": None})})

    assert response.answers["q"].choice == "sales"
    assert len(stub.bodies("/chat/completions")) == 2
    assert len(response.debug["retry_reasons"]) == 1
    assert response.usage.n_calls == 2


def _logprobs_body(tokens: list[dict]) -> dict:
    body = chat_body(content="A")
    body["choices"][0]["logprobs"] = {"content": tokens}
    return body


def test_missing_logprob_on_the_answer_token_is_an_error_not_certainty(stub_server):
    body = _logprobs_body(
        [
            {"token": "A", "logprob": None, "top_logprobs": [{"token": "A", "logprob": None}, {"token": "B", "logprob": -2.0}]},
            {"token": "B", "logprob": -2.0, "top_logprobs": []},
        ]
    )
    stub = stub_server(chat=lambda _: (200, body))
    client = SystemOneClient(openai_client(stub), model="stub", api="chat_completions")

    with pytest.raises(LabelReadoutError) as error:
        client.system_one(state="s", questions={"q": Choice(criteria={"billing": None, "sales": None})})

    assert "no logprob for the answer token 'A'" in str(error.value)
    assert len(stub.bodies("/chat/completions")) == 2  # one corrective retry, not a silent 1.0


def test_alternative_without_a_logprob_is_reported_missing(stub_server):
    body = _logprobs_body(
        [
            {
                "token": "A",
                "logprob": -0.12,
                "top_logprobs": [{"token": "A", "logprob": -0.12}, {"token": "B", "logprob": None}],
            }
        ]
    )
    stub = stub_server(chat=lambda _: (200, body))
    client = SystemOneClient(openai_client(stub), model="stub", api="chat_completions")

    response = client.system_one(
        state="s", questions={"q": Choice(criteria={"billing": None, "sales": None})}
    )

    assert response.answers["q"].probabilities == {"billing": 1.0, "sales": 0.0}
    assert response.debug["labels_missing"] == {"q": ["B"]}


def test_non_finite_logprobs_raise_instead_of_poisoning_the_distribution():
    from jevper.methods import softmax_over_labels

    with pytest.raises(LabelReadoutError) as error:
        softmax_over_labels({"A": float("nan"), "B": -1.0})
    assert "'A'" in str(error.value) and "nan" in str(error.value)

    with pytest.raises(LabelReadoutError):
        softmax_over_labels({"A": float("inf"), "B": -1.0})

    # -inf is a legitimate zero, not an error
    assert softmax_over_labels({"A": float("-inf"), "B": -1.0}) == {"A": 0.0, "B": 1.0}
