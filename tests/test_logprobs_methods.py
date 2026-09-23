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
    assert "I was charged twice this month" in sent["messages"][1]["content"]
    assert "A: billing" in sent["messages"][-1]["content"]


def test_noul_logprobs_has_no_confidence(stub_server):
    stub = stub_server(chat=lambda _: (200, chat_body(content="A", logprobs=[("A", -0.05), ("B", -3.0)])))
    client = SystemOneClient(openai_client(stub), model="stub", api="chat_completions")

    response = client.system_one(state="The sky is blue", questions={"yes": Noul()})

    answer = response.answers["yes"]
    assert answer.noul == pytest.approx(0.950263488, abs=1e-9)
    assert "confidence" not in answer.model_dump()
    question = stub.bodies("/chat/completions")[0]["messages"][-1]["content"]
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
