"""Client behaviour: transient retries, validation, concurrency, async parity, state forms."""

from __future__ import annotations

import asyncio
import json
import threading
import time
import warnings
from types import SimpleNamespace
from typing import Any

import pytest
from fakes import (
    RaisingClient,
    StatusError,
    async_openai_client,
    chat_body,
    openai_client,
    responses_body,
)

from jevper import (
    AsyncSystemOneClient,
    Choice,
    ClientCapabilityError,
    InvalidQuestionError,
    JevperError,
    MalformedAnswerError,
    Noul,
    ProviderError,
    ReasoningConfig,
    RetryPolicy,
    Score,
    SystemOneClient,
    reasoning_text,
)

CHOICE_LOGS = [("A", -0.12), ("B", -2.47), ("C", -3.48)]
CRITERIA = {"billing": None, "technical": None, "sales": None}


def test_transient_failure_is_retried(stub_server):
    calls: list[dict] = []

    def script(body):
        calls.append(body)
        if len(calls) == 1:
            return 429, {"error": {"message": "slow down", "type": "rate_limit_error"}}
        return 200, chat_body(content="A", logprobs=CHOICE_LOGS)

    stub = stub_server(chat=script)
    client = SystemOneClient(
        openai_client(stub),
        model="stub",
        api="chat_completions",
        retry=RetryPolicy(n_retries=2, base_delay=0.0),
    )

    response = client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

    assert response.answers["q"].choice == "billing"
    assert len(stub.bodies("/chat/completions")) == 2
    assert response.usage.n_retries == 1
    assert response.usage.n_calls == 1
    attempts = response.debug["llm_attempts"]
    assert attempts[0]["error"].startswith("RateLimitError")
    assert attempts[0]["response"] is None
    assert attempts[1]["readout"]["source"] == "logprobs"


def test_non_transient_failure_is_not_retried(stub_server):
    stub = stub_server(chat=lambda _: (400, {"error": {"message": "bad request", "type": "invalid_request_error"}}))
    client = SystemOneClient(
        openai_client(stub),
        model="stub",
        api="chat_completions",
        retry=RetryPolicy(n_retries=3, base_delay=0.0),
    )

    with pytest.raises(ProviderError) as error:
        client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

    assert "BadRequestError" in str(error.value)
    assert len(stub.requests) == 1
    assert error.value.attempts[0]["request"]["logprobs"] is True


def test_question_limits_are_enforced_at_construction():
    with pytest.raises(InvalidQuestionError) as error:
        Choice(criteria={f"k{index}": None for index in range(256)})
    assert "255 is the Jev API limit" in str(error.value)

    with pytest.raises(InvalidQuestionError) as error:
        Choice(criteria={"only": None})
    assert "2..255" in str(error.value)

    with pytest.raises(InvalidQuestionError) as error:
        Score(criteria=[f"level {index}" for index in range(11)])
    assert "2..10" in str(error.value)

    with pytest.raises(InvalidQuestionError) as error:
        Noul(criteria={"maybe": "unsure"})  # type: ignore[arg-type]
    assert "'true'/'false'" in str(error.value)


def test_label_readout_methods_stop_at_26_options(stub_server):
    stub = stub_server(chat=lambda _: (200, chat_body(content="A", logprobs=CHOICE_LOGS)))
    client = SystemOneClient(
        openai_client(stub), model="stub", method="logprobs", api="chat_completions"
    )
    wide = {f"k{index}": None for index in range(27)}

    with pytest.raises(InvalidQuestionError) as error:
        client.system_one(state="s", questions={"q": Choice(criteria=wide)})
    assert "27 options exceed the 26 labels" in str(error.value)
    assert "method='structured'" in str(error.value) and "method='discrete'" in str(error.value)
    assert stub.requests == []

    # the same question is fine where the answer is a JSON string rather than a label token
    wide_stub = stub_server(chat=lambda _: (200, chat_body(content=json.dumps({"choice": "AA"}))))
    discrete = SystemOneClient(
        openai_client(wide_stub), model="stub", method="discrete", api="chat_completions"
    )
    response = discrete.system_one(state="s", questions={"q": Choice(criteria=wide)})

    assert response.answers["q"].choice == "k0"


def test_raw_question_dicts_are_validated_before_any_request(stub_server):
    stub = stub_server(chat=lambda _: (200, chat_body(content="A")))
    client = SystemOneClient(openai_client(stub), model="stub", api="chat_completions")

    with pytest.raises(InvalidQuestionError):
        client.system_one(state="s", questions={"q": {"type": "choice", "criteria": {"only": None}}})
    with pytest.raises(InvalidQuestionError) as error:
        client.system_one(state="s", questions={"q": {"type": "bogus"}})
    assert "question 'q' is invalid" in str(error.value)
    with pytest.raises(InvalidQuestionError) as error:
        client.system_one(
            state="s", questions={"q": {"type": "choice", "criteria": CRITERIA, "bogus": 1}}
        )
    assert "bogus" in str(error.value)
    assert stub.requests == []


def test_one_call_per_question_and_answers_keyed_in_order(stub_server):
    stub = stub_server(chat=lambda _: (200, chat_body(content="A", logprobs=CHOICE_LOGS)))
    client = SystemOneClient(openai_client(stub), model="stub", api="chat_completions")

    response = client.system_one(
        state="The charge appeared twice",
        questions={
            "intent": Choice(criteria=CRITERIA),
            "duplicate": Noul(),
            "anger": Score(criteria=["Calm", "Frustrated", "Very angry"]),
        },
    )

    requests = stub.bodies("/chat/completions")
    assert len(requests) == 3
    assert list(response.answers) == ["intent", "duplicate", "anger"]
    assert response.answers["intent"].choice == "billing"
    assert response.answers["duplicate"].noul == pytest.approx(0.912934228, abs=1e-9)
    assert response.answers["anger"].probabilities[0] == pytest.approx(0.884873983, abs=1e-9)
    question_blocks = [request["messages"][-2]["content"] for request in requests]
    assert any("A: billing" in block for block in question_blocks)
    assert any("A: Yes" in block for block in question_blocks)
    assert any("A: 0 — Calm" in block for block in question_blocks)
    assert response.usage.n_calls == 3
    assert list(response.choices) == ["intent"]
    assert list(response.nouls) == ["duplicate"]
    assert list(response.scores) == ["anger"]


def test_one_failing_question_raises_in_insertion_order_after_all_settle(stub_server):
    def script(body):
        question = body["messages"][-2]["content"]
        if "A: billing" in question:
            return 400, {"error": {"message": "boom", "type": "invalid_request_error"}}
        return 200, chat_body(content="A", logprobs=CHOICE_LOGS)

    stub = stub_server(chat=script)
    client = SystemOneClient(openai_client(stub), model="stub", api="chat_completions")

    with pytest.raises(ProviderError):
        client.system_one(
            state="s",
            questions={"intent": Choice(criteria=CRITERIA), "verdict": Noul()},
        )

    assert len(stub.bodies("/chat/completions")) == 2


def test_state_chat_messages_are_preserved_after_the_question_block(stub_server):
    stub = stub_server(chat=lambda _: (200, chat_body(content="A", logprobs=CHOICE_LOGS)))
    client = SystemOneClient(openai_client(stub), model="stub", api="chat_completions")

    client.system_one(
        state={
            "messages": [
                {"role": "user", "content": "I was charged twice."},
                {"role": "assistant", "content": "Let me look into that."},
            ]
        },
        questions={"q": Choice(criteria={"billing": None, "sales": None})},
    )

    messages = stub.bodies("/chat/completions")[0]["messages"]
    # The state ends on the assistant's turn, so the question goes last: llama.cpp's engines refuse a
    # conversation that ends there — ollama and LM Studio both answer 400 "Failed to initialize
    # samplers" — and a server that reads it as a prefill would continue it instead of answering.
    assert [message["role"] for message in messages] == ["system", "user", "assistant", "user"]
    assert messages[0]["content"].startswith("You are a precise classification engine")
    assert messages[1]["content"] == "I was charged twice."
    assert messages[2]["content"] == "Let me look into that."
    assert messages[-1]["content"].startswith("Options:")


def test_json_state_is_pretty_printed_into_one_turn(stub_server):
    stub = stub_server(chat=lambda _: (200, chat_body(content="A", logprobs=CHOICE_LOGS)))
    client = SystemOneClient(openai_client(stub), model="stub", api="chat_completions")

    client.system_one(
        state={"order_id": 42, "lines": [{"sku": "X", "qty": 2}]},
        questions={"q": Choice(criteria={"billing": None, "sales": None})},
    )

    messages = stub.bodies("/chat/completions")[0]["messages"]
    assert len(messages) == 3
    assert messages[1]["content"].startswith("Options:")
    assert messages[2]["role"] == "user"
    assert '"order_id": 42' in messages[2]["content"]
    assert '"qty": 2' in messages[2]["content"]


def test_noul_criteria_are_rendered_as_yes_no_means_lines(stub_server):
    stub = stub_server(chat=lambda _: (200, chat_body(content="A", logprobs=[("A", -0.05), ("B", -3.0)])))
    client = SystemOneClient(openai_client(stub), model="stub", api="chat_completions")

    client.system_one(
        state="s",
        questions={
            "verdict": Noul(
                instructions="Is this message a complaint?",
                criteria={"true": "the customer is unhappy", "false": "the customer is satisfied"},
            )
        },
    )

    block = stub.bodies("/chat/completions")[0]["messages"][-2]["content"]
    assert "Question:\nIs this message a complaint?" in block
    assert "A: Yes" in block and "B: No" in block
    assert "Yes means: the customer is unhappy" in block
    assert "No means: the customer is satisfied" in block


def test_usage_counts_are_none_when_the_provider_omits_them(stub_server):
    stub = stub_server(chat=lambda _: (200, chat_body(content="A", logprobs=CHOICE_LOGS, input_tokens=None, output_tokens=None, reasoning_tokens=None)))
    client = SystemOneClient(openai_client(stub), model="stub", api="chat_completions")

    response = client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

    assert response.usage.input_tokens is None
    assert response.usage.output_tokens is None
    assert response.usage.reasoning_tokens is None
    assert response.usage.n_calls == 1
    assert response.usage.latency >= 0.0


def test_async_client_matches_the_sync_client(stub_server):
    body = responses_body(text="A", logprobs=CHOICE_LOGS)
    stub = stub_server(responses=lambda _: (200, body))
    questions = {"intent": Choice(criteria=CRITERIA), "duplicate": Noul()}

    sync_client = SystemOneClient(openai_client(stub), model="stub")
    async_client = AsyncSystemOneClient(async_openai_client(stub), model="stub")

    sync_response = sync_client.system_one(state="s", questions=questions)
    async_response = asyncio.run(async_client.system_one(state="s", questions=questions))

    assert async_response.model_dump(exclude={"usage"}) == sync_response.model_dump(exclude={"usage"})
    assert async_response.usage.model_dump(exclude={"latency"}) == sync_response.usage.model_dump(
        exclude={"latency"}
    )
    assert len(stub.requests) == 4


def test_missing_client_surface_is_reported(stub_server):
    class ChatOnly:
        pass

    with pytest.raises(Exception) as error:
        SystemOneClient(ChatOnly(), model="stub").system_one(
            state="s", questions={"q": Choice(criteria=CRITERIA)}
        )
    assert "responses.create" in str(error.value) and "chat.completions.create" in str(error.value)


class TransportError(Exception):
    """Stand-in for ``httpx.TransportError`` (same class name, same position in the MRO)."""


class ConnectError(TransportError):
    """Stand-in for ``httpx.ConnectError``: the name carries no Connection/Timeout marker."""


class LocalProtocolError(Exception):
    """Stand-in for ``httpx.LocalProtocolError``: a client-side bug, never transient."""


class _FailingChatClient:
    """Duck client whose transport raises once, then answers — the shape of a dropped connection."""

    def __init__(self, failure: Exception) -> None:
        self.failure = failure
        self.calls = 0
        outer = self

        class Completions:
            def create(self, **kwargs):
                outer.calls += 1
                if outer.calls == 1:
                    raise outer.failure
                return chat_body(content="A", logprobs=CHOICE_LOGS)

        class Chat:
            completions = Completions()

        self.chat = Chat()


def test_transport_level_failures_are_retried():
    duck = _FailingChatClient(ConnectError("connection dropped"))
    client = SystemOneClient(
        duck,
        model="stub",
        api="chat_completions",
        retry=RetryPolicy(n_retries=2, base_delay=0.0),
    )

    response = client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

    assert response.answers["q"].choice == "billing"
    assert duck.calls == 2
    assert response.usage.n_retries == 1
    assert response.debug["llm_attempts"][0]["error"].startswith("ConnectError")


def test_client_side_protocol_errors_are_not_retried():
    duck = _FailingChatClient(LocalProtocolError("bad request construction"))
    client = SystemOneClient(
        duck, model="stub", api="chat_completions", retry=RetryPolicy(n_retries=2, base_delay=0.0)
    )

    with pytest.raises(ProviderError):
        client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

    assert duck.calls == 1


def test_string_status_codes_are_still_transient():
    class SlowDown(Exception):
        status_code = "429"

    duck = _FailingChatClient(SlowDown("slow down"))
    client = SystemOneClient(
        duck, model="stub", api="chat_completions", retry=RetryPolicy(n_retries=1, base_delay=0.0)
    )

    response = client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

    assert response.usage.n_retries == 1
    assert duck.calls == 2


def test_unreadable_status_codes_do_not_mask_the_provider_error():
    class Weird(Exception):
        status_code = [429]  # noqa: RUF012 - an unhashable status is the point

    duck = _FailingChatClient(Weird("boom"))
    client = SystemOneClient(
        duck, model="stub", api="chat_completions", retry=RetryPolicy(n_retries=1, base_delay=0.0)
    )

    with pytest.raises(ProviderError) as error:
        client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

    assert "Weird: boom" in str(error.value)
    assert duck.calls == 1


def test_retry_policy_is_validated(stub_server):
    stub = stub_server(chat=lambda _: (200, chat_body(content="A", logprobs=CHOICE_LOGS)))

    for policy in (RetryPolicy(n_retries=-1), RetryPolicy(base_delay=-1), RetryPolicy(max_delay=-1)):
        with pytest.raises(JevperError) as error:
            SystemOneClient(openai_client(stub), model="stub", retry=policy)
        assert ">= 0" in str(error.value)

    with pytest.raises(JevperError) as error:
        SystemOneClient(openai_client(stub), model="stub", retry={"n_retries": 1})  # type: ignore[arg-type]
    assert "RetryPolicy" in str(error.value)


def test_failed_attempt_records_the_provider_request(stub_server):
    stub = stub_server(chat=lambda _: (400, {"error": {"message": "bad", "type": "invalid_request_error"}}))
    client = SystemOneClient(
        openai_client(stub), model="stub", api="chat_completions", method="structured"
    )

    with pytest.raises(ProviderError) as error:
        client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

    request = error.value.attempts[0]["request"]
    assert request["model"] == "stub"
    assert request["response_format"]["json_schema"]["name"] == "jevper_choice"
    assert request["messages"][-2]["content"].startswith("Options:")


def test_non_string_instructions_and_criteria_render_as_json(stub_server):
    stub = stub_server(chat=lambda _: (200, chat_body(content="A", logprobs=CHOICE_LOGS)))
    client = SystemOneClient(openai_client(stub), model="stub", api="chat_completions")

    response = client.system_one(
        state="s",
        questions={
            "intent": Choice(instructions=42, criteria={"low": 1, "high": 5}),
            "anger": Score(criteria=[1, 2, 3]),
        },
    )

    bodies = stub.bodies("/chat/completions")
    choice_block = next(
        body["messages"][-2]["content"] for body in bodies if "A: low" in body["messages"][-2]["content"]
    )
    assert "Question:\n42" in choice_block
    assert "A: low — 1" in choice_block and "B: high — 5" in choice_block
    score_block = next(
        body["messages"][-2]["content"] for body in bodies if "A: 0" in body["messages"][-2]["content"]
    )
    assert "A: 0 — 1" in score_block and "C: 2 — 3" in score_block
    assert response.scores["anger"].legend == {0: 1, 1: 2, 2: 3}


def test_state_must_be_json_serializable_with_finite_numbers(stub_server):
    stub = stub_server(chat=lambda _: (200, chat_body(content="A", logprobs=CHOICE_LOGS)))
    client = SystemOneClient(openai_client(stub), model="stub", api="chat_completions")

    with pytest.raises(JevperError) as error:
        client.system_one(state={1, 2}, questions={"q": Choice(criteria=CRITERIA)})
    assert "JSON-serializable" in str(error.value)

    with pytest.raises(JevperError) as error:
        client.system_one(state={"latency": float("nan")}, questions={"q": Choice(criteria=CRITERIA)})
    assert "finite" in str(error.value)

    assert stub.requests == []


def test_labels_cover_the_two_letter_range_and_stop_there():
    from jevper.labels import labels_for

    assert labels_for(676)[-1] == "ZZ"
    assert labels_for(27)[26] == "BA"

    with pytest.raises(InvalidQuestionError) as error:
        labels_for(677)
    assert "0..676" in str(error.value)


# -- concurrency -----------------------------------------------------------------------


class _ConcurrencyChatClient:
    """Duck client recording the peak number of in-flight requests; optionally blocks on a barrier."""

    def __init__(self, *, barrier: threading.Barrier | None = None, delay: float = 0.0) -> None:
        self.barrier = barrier
        self.delay = delay
        self.in_flight = 0
        self.peak = 0
        self._lock = threading.Lock()
        outer = self

        class Completions:
            def create(self, **kwargs):
                with outer._lock:
                    outer.in_flight += 1
                    outer.peak = max(outer.peak, outer.in_flight)
                try:
                    if outer.barrier is not None:
                        outer.barrier.wait(timeout=5)
                    if outer.delay:
                        time.sleep(outer.delay)
                    return chat_body(content="A", logprobs=CHOICE_LOGS)
                finally:
                    with outer._lock:
                        outer.in_flight -= 1

        class Chat:
            completions = Completions()

        self.chat = Chat()


def test_questions_run_concurrently_up_to_max_concurrency():
    duck = _ConcurrencyChatClient(barrier=threading.Barrier(4, timeout=5))
    client = SystemOneClient(duck, model="stub", api="chat_completions", max_concurrency=4)

    response = client.system_one(
        state="s", questions={f"q{index}": Choice(criteria=CRITERIA) for index in range(4)}
    )

    # The barrier only releases once all four requests are in flight at the same time.
    assert duck.peak == 4
    assert len(response.answers) == 4


def test_max_concurrency_one_serializes_requests():
    duck = _ConcurrencyChatClient(delay=0.02)
    client = SystemOneClient(duck, model="stub", api="chat_completions", max_concurrency=1)

    client.system_one(
        state="s", questions={f"q{index}": Choice(criteria=CRITERIA) for index in range(4)}
    )

    assert duck.peak == 1


class _AsyncConcurrencyChatClient:
    """Duck async client recording peak in-flight requests; the sleep yields so tasks interleave."""

    def __init__(self, *, delay: float = 0.02) -> None:
        self.delay = delay
        self.in_flight = 0
        self.peak = 0
        outer = self

        class Completions:
            async def create(self, **kwargs):
                outer.in_flight += 1
                outer.peak = max(outer.peak, outer.in_flight)
                try:
                    await asyncio.sleep(outer.delay)
                    return chat_body(content="A", logprobs=CHOICE_LOGS)
                finally:
                    outer.in_flight -= 1

        class Chat:
            completions = Completions()

        self.chat = Chat()


def test_async_client_runs_questions_concurrently():
    duck = _AsyncConcurrencyChatClient()
    client = AsyncSystemOneClient(duck, model="stub", api="chat_completions", max_concurrency=4)

    asyncio.run(
        client.system_one(
            state="s", questions={f"q{index}": Choice(criteria=CRITERIA) for index in range(4)}
        )
    )

    assert duck.peak == 4


def test_async_max_concurrency_one_serializes_requests():
    duck = _AsyncConcurrencyChatClient()
    client = AsyncSystemOneClient(duck, model="stub", api="chat_completions", max_concurrency=1)

    asyncio.run(
        client.system_one(
            state="s", questions={f"q{index}": Choice(criteria=CRITERIA) for index in range(4)}
        )
    )

    assert duck.peak == 1


def test_empty_choices_is_a_client_capability_error(stub_server):
    stub = stub_server(chat=lambda _: (200, {"choices": []}))
    client = SystemOneClient(openai_client(stub), model="stub", api="chat_completions")

    with pytest.raises(ClientCapabilityError) as error:
        client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

    assert "provider returned no choices" in str(error.value)


UPSTREAM_OVERLOAD = {
    "id": "gen-stub",
    "error": {
        "message": "Upstream error from Nvidia: Service temporarily overloaded",
        "code": 503,
        "metadata": {"error_type": "provider_overloaded"},
    },
}


def test_a_provider_error_in_a_200_body_is_a_provider_error(stub_server):
    """OpenRouter reports an overloaded upstream as a 200 whose body carries the error, not a 503."""
    stub = stub_server(chat=lambda _: (200, UPSTREAM_OVERLOAD))
    client = SystemOneClient(
        openai_client(stub),
        model="stub",
        api="chat_completions",
        method="structured",
        retry=RetryPolicy(n_retries=1, base_delay=0.0),
    )

    with pytest.raises(ProviderError) as error:
        client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

    assert "Service temporarily overloaded" in str(error.value)
    assert error.value.status_code == 503
    # The status travelled with the error, so the retry policy saw a transient failure.
    assert len(stub.bodies("/chat/completions")) == 2


def test_a_permanent_provider_error_in_a_200_body_is_not_retried(stub_server):
    body = {"id": "gen-stub", "error": {"message": "upstream rejected the request", "code": 400}}
    stub = stub_server(chat=lambda _: (200, body))
    client = SystemOneClient(
        openai_client(stub),
        model="stub",
        api="chat_completions",
        method="structured",
        retry=RetryPolicy(n_retries=2, base_delay=0.0),
    )

    with pytest.raises(ProviderError) as error:
        client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

    assert error.value.status_code == 400
    assert len(stub.bodies("/chat/completions")) == 1


def test_a_provider_error_in_a_200_response_body_is_a_provider_error(stub_server):
    """The Responses surface carries it the same way: no output, an error, HTTP 200."""
    stub = stub_server(responses=lambda _: (200, UPSTREAM_OVERLOAD))
    client = SystemOneClient(
        openai_client(stub),
        model="stub",
        api="responses",
        method="structured",
        retry=RetryPolicy(n_retries=1, base_delay=0.0),
    )

    with pytest.raises(ProviderError) as error:
        client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

    assert error.value.status_code == 503
    assert len(stub.bodies("/responses")) == 2


def test_non_string_question_id_fails_before_any_request(stub_server):
    stub = stub_server(chat=lambda _: (200, chat_body(content="A", logprobs=CHOICE_LOGS)))
    client = SystemOneClient(openai_client(stub), model="stub", api="chat_completions")

    with pytest.raises(InvalidQuestionError) as error:
        client.system_one(state="s", questions={1: Choice(criteria=CRITERIA)})

    assert "question id must be a string" in str(error.value)
    assert stub.requests == []


def test_408_is_retried(stub_server):
    stub = stub_server(chat=lambda _: (408, {"error": {"message": "Request Timeout"}}))
    client = SystemOneClient(
        openai_client(stub),
        model="stub",
        api="chat_completions",
        retry=RetryPolicy(n_retries=1, base_delay=0.0),
    )

    with pytest.raises(ProviderError) as error:
        client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

    assert len(error.value.attempts) == 2
    assert len(stub.bodies("/chat/completions")) == 2


def test_408_is_not_capability_evidence(stub_server):
    """A timed-out request is not a verdict on the provider, so auto keeps asking for logprobs."""
    stub = stub_server(chat=lambda _: (408, {"error": {"message": "Request Timeout"}}))
    client = SystemOneClient(
        openai_client(stub),
        model="stub",
        api="chat_completions",
        retry=RetryPolicy(n_retries=1, base_delay=0.0),
    )

    with pytest.raises(ProviderError):
        client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

    bodies = stub.bodies("/chat/completions")
    assert len(bodies) == 2
    assert all(body["logprobs"] is True for body in bodies)


def test_status_carried_on_the_response_is_read():
    retry = RetryPolicy(n_retries=1, base_delay=0.0)
    question = {"q": Choice(criteria=CRITERIA)}

    transient = SystemOneClient(
        RaisingClient(StatusError(503)),
        model="m",
        method="discrete",
        api="chat_completions",
        retry=retry,
    )
    with pytest.raises(ProviderError) as error:
        transient.system_one(state="s", questions=question)
    assert len(error.value.attempts) == 2

    permanent = SystemOneClient(
        RaisingClient(StatusError(400)),
        model="m",
        method="discrete",
        api="chat_completions",
        retry=retry,
    )
    with pytest.raises(ProviderError) as error:
        permanent.system_one(state="s", questions=question)
    assert len(error.value.attempts) == 1


def test_discrete_readout_prefers_an_exact_key_over_a_label(stub_server):
    """A model that echoes the option key must not have it read as the first label."""
    stub = stub_server(chat=lambda _: (200, chat_body(content=json.dumps({"choice": "a"}))))
    client = SystemOneClient(
        openai_client(stub), model="stub", method="discrete", api="chat_completions"
    )

    response = client.system_one(state="s", questions={"q": Choice(criteria={"b": None, "a": None})})

    assert response.answers["q"].choice == "a"


def test_per_call_reasoning_must_be_a_config(stub_server):
    stub = stub_server(chat=lambda _: (200, chat_body(content="A", logprobs=CHOICE_LOGS)))
    client = SystemOneClient(openai_client(stub), model="stub", api="chat_completions")

    with pytest.raises(JevperError) as error:
        client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)}, reasoning="native")

    assert "reasoning must be a ReasoningConfig" in str(error.value)
    assert stub.requests == []


def test_extra_body_and_headers_must_be_mappings():
    with pytest.raises(JevperError) as error:
        SystemOneClient(object(), model="m", extra_body=["provider", "flags"])
    assert "extra_body must be a mapping" in str(error.value)

    with pytest.raises(JevperError) as error:
        SystemOneClient(object(), model="m", extra_headers=["Authorization: nope"])
    assert "extra_headers must be a mapping" in str(error.value)


def test_retry_policy_rejects_non_finite_delays():
    with pytest.raises(JevperError) as error:
        SystemOneClient(object(), model="m", retry=RetryPolicy(base_delay=float("nan")))
    assert "finite" in str(error.value)

    with pytest.raises(JevperError):
        SystemOneClient(object(), model="m", retry=RetryPolicy(max_delay=float("inf")))


def test_a_huge_retry_count_does_not_overflow_the_backoff():
    """``3**attempt`` overflows a float past attempt 646: the backoff has to cap, not raise."""
    client = SystemOneClient(
        RaisingClient(StatusError(503)),
        model="m",
        method="discrete",
        api="chat_completions",
        retry=RetryPolicy(n_retries=648, base_delay=0.0, max_delay=0.0),
    )

    with pytest.raises(ProviderError) as error:
        client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

    assert len(error.value.attempts) == 649  # 648 retries past the point the old arithmetic raised


def test_label_methods_reject_a_single_alternative():
    with pytest.raises(JevperError) as error:
        SystemOneClient(object(), model="m", method="logprobs", top_logprobs=1)
    assert "at least 2" in str(error.value)

    # auto answers in JSON when logprobs are unusable, so one alternative is only fatal when pinned.
    SystemOneClient(object(), model="m", method="auto", top_logprobs=1)


def test_a_per_call_method_cannot_ask_for_one_alternative(stub_server):
    stub = stub_server(chat=lambda _: (200, chat_body(content="A", logprobs=CHOICE_LOGS)))
    client = SystemOneClient(
        openai_client(stub), model="stub", api="chat_completions", top_logprobs=1
    )

    with pytest.raises(JevperError) as error:
        client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)}, method="logprobs")

    assert "at least 2" in str(error.value)
    assert stub.requests == []


def test_a_provider_field_the_sdk_types_differently_is_dumped_without_noise(stub_server):
    """SGLang returns a list for ``metadata`` where the SDK declares a string.

    jevper dumps the provider object for ``debug``, so without asking pydantic to keep quiet every
    call against such a server prints a ``PydanticSerializationUnexpectedValue`` warning the caller
    never asked for. The value survives the dump either way: only the noise is removed.
    """
    body = chat_body(content="A", logprobs=CHOICE_LOGS)
    body["metadata"] = [{"version": "default", "start": 0, "end": 4}]
    stub = stub_server(chat=lambda _: (200, body))
    client = SystemOneClient(openai_client(stub), model="stub", api="chat_completions")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        response = client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

    assert [item for item in caught if "PydanticSerializationUnexpectedValue" in str(item.message)] == []
    assert response.debug["llm_attempts"][-1]["response"]["metadata"] == [
        {"version": "default", "start": 0, "end": 4}
    ]
    assert response.answers["q"].choice == "billing"


# --- a server that refuses a field jevper added for capability ---------------------------------


STRUCTURED = '{"probabilities": {"billing": 0.8, "technical": 0.1, "sales": 0.1}}'


def unsupported(field: str) -> tuple[int, dict]:
    """The shape a server uses to refuse a request field it does not implement."""
    return 400, {
        "error": {
            "message": f"{field} is not supported",
            "param": field,
            "code": "unsupported_parameter",
        }
    }


def test_a_refused_json_schema_is_answered_with_json_object(stub_server):
    def script(body):
        if (body.get("response_format") or {}).get("type") == "json_schema":
            return unsupported("response_format")
        return 200, chat_body(content=STRUCTURED)

    stub = stub_server(chat=script)
    client = SystemOneClient(
        openai_client(stub), model="stub", api="chat_completions", method="structured"
    )

    first = client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})
    second = client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

    assert first.answers["q"].choice == "billing"
    assert second.answers["q"].choice == "billing"
    assert [
        body["response_format"]["type"] for body in stub.bodies("/chat/completions")
    ] == ["json_schema", "json_object", "json_object"]
    assert first.debug["server_limits"]["structured"] == "object"
    # The strict request carried the schema in the request itself; the two that fell back carry it in
    # the prompt, because a plain JSON object is not this JSON object.
    assert [
        '"probabilities"' in body["messages"][0]["content"]
        for body in stub.bodies("/chat/completions")
    ] == [False, True, True]


def test_a_refused_json_object_is_answered_without_a_response_format(stub_server):
    def script(body):
        if body.get("response_format") is not None:
            return unsupported("response_format")
        return 200, chat_body(content=STRUCTURED)

    stub = stub_server(chat=script)
    client = SystemOneClient(
        openai_client(stub), model="stub", api="chat_completions", method="structured"
    )

    response = client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

    assert response.answers["q"].choice == "billing"
    bodies = stub.bodies("/chat/completions")
    assert len(bodies) == 3
    assert "response_format" not in bodies[2]  # the prompt still asks for one JSON object
    assert response.debug["server_limits"]["structured"] == "none"


def test_a_refused_reasoning_effort_is_answered_without_it(stub_server):
    def script(body):
        if "reasoning_effort" in body:
            return unsupported("reasoning_effort")
        return 200, chat_body(content="A", logprobs=CHOICE_LOGS)

    stub = stub_server(chat=script)
    client = SystemOneClient(
        openai_client(stub), model="stub", api="chat_completions", method="logprobs"
    )

    response = client.system_one(
        state="s",
        questions={"q": Choice(criteria=CRITERIA)},
        reasoning=ReasoningConfig(effort="low", mode="native"),
    )

    assert response.answers["q"].choice == "billing"
    assert "reasoning_effort" not in stub.bodies("/chat/completions")[1]
    assert response.debug["server_limits"]["reasoning"] is False


def test_a_refused_include_is_answered_without_it(stub_server):
    def script(body):
        if "reasoning.encrypted_content" in (body.get("include") or []):
            return unsupported("include")
        return 200, responses_body(text="A", logprobs=CHOICE_LOGS)

    stub = stub_server(responses=script)
    client = SystemOneClient(
        openai_client(stub), model="stub", api="responses", method="logprobs"
    )

    response = client.system_one(
        state="s",
        questions={"q": Choice(criteria=CRITERIA)},
        reasoning=ReasoningConfig(mode="native"),
    )

    assert response.answers["q"].choice == "billing"
    include = stub.bodies("/responses")[1].get("include") or []
    assert "message.output_text.logprobs" in include  # the field the readout needs stays
    assert "reasoning.encrypted_content" not in include
    assert response.debug["server_limits"]["include"] is False


def test_a_server_that_refuses_everything_still_reports_the_error(stub_server):
    stub = stub_server(chat=lambda _: unsupported("response_format"))
    client = SystemOneClient(
        openai_client(stub),
        model="stub",
        api="chat_completions",
        method="structured",
        retry=RetryPolicy(n_retries=0),
    )

    with pytest.raises(ProviderError) as error:
        client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

    assert "not supported" in str(error.value)
    assert len(stub.bodies("/chat/completions")) == 3  # schema, object, none — then it gives up


def test_the_async_client_drops_a_refused_field_too(stub_server):
    """The async driver has its own copy of the retry, so it needs its own proof."""

    def script(body):
        if (body.get("response_format") or {}).get("type") == "json_schema":
            return unsupported("response_format")
        return 200, chat_body(content=STRUCTURED)

    stub = stub_server(chat=script)
    client = AsyncSystemOneClient(
        async_openai_client(stub), model="stub", api="chat_completions", method="structured"
    )

    response = asyncio.run(client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)}))

    assert response.answers["q"].choice == "billing"
    assert [
        body["response_format"]["type"] for body in stub.bodies("/chat/completions")
    ] == ["json_schema", "json_object"]
    assert response.debug["server_limits"]["structured"] == "object"


@pytest.mark.parametrize("role", ["system", "developer"])
def test_a_state_instruction_turn_is_hoisted_into_the_system_prompt(stub_server, role):
    """No server here accepts a system turn that is not first, and jevper's own prompt is first.

    llama.cpp's template raises "System message must be at the beginning." for one that is not, and vLLM
    and SGLang answer 400 with the same words. The caller's instruction joins the system prompt — its
    content is an instruction, and dropping it would change the question — and the rest of the state
    still goes last, where it can be cached.
    """
    stub = stub_server(chat=lambda _: (200, chat_body(content="A", logprobs=CHOICE_LOGS)))
    client = SystemOneClient(openai_client(stub), model="stub", api="chat_completions")

    client.system_one(
        state=[
            {"role": role, "content": "You are a support triage assistant."},
            {"role": "user", "content": "I was charged twice."},
        ],
        questions={"q": Choice(criteria={"billing": None, "sales": None})},
    )

    messages = stub.bodies("/chat/completions")[0]["messages"]
    assert [message["role"] for message in messages] == ["system", "user", "user"]
    assert messages[0]["content"].startswith("You are a precise classification engine")
    assert messages[0]["content"].endswith("You are a support triage assistant.")
    assert messages[1]["content"].startswith("Options:")
    assert messages[2]["content"] == "I was charged twice."


def test_a_state_without_an_instruction_turn_goes_last(stub_server):
    stub = stub_server(chat=lambda _: (200, chat_body(content="A", logprobs=CHOICE_LOGS)))
    client = SystemOneClient(openai_client(stub), model="stub", api="chat_completions")

    client.system_one(
        state=[{"role": "user", "content": "I was charged twice."}, {"role": "assistant", "content": "Looking."}],
        questions={"q": Choice(criteria={"billing": None, "sales": None})},
    )

    messages = stub.bodies("/chat/completions")[0]["messages"]
    # The last state turn is the assistant's, so the question turn follows it — see
    # ``test_a_state_that_ends_with_the_assistant_keeps_the_question_last``.
    assert [message["role"] for message in messages] == ["system", "user", "assistant", "user"]
    assert messages[-1]["content"].startswith("Options:")


def test_a_state_that_ends_with_the_assistant_keeps_the_question_last(stub_server):
    """The one state shape where the question turn does not come before the state."""
    stub = stub_server(chat=lambda _: (200, chat_body(content="A", logprobs=CHOICE_LOGS)))
    client = SystemOneClient(openai_client(stub), model="stub", api="chat_completions")

    client.system_one(
        state=[
            {"role": "user", "content": "I was charged twice."},
            {"role": "assistant", "content": "Looking."},
            {"role": "user", "content": "It happened again."},
        ],
        questions={"q": Choice(criteria={"billing": None, "sales": None})},
    )

    # Ending on a *user* turn is the common shape and keeps the documented order: the question first, so
    # the state stays the tail that changes and the prefix stays reusable.
    messages = stub.bodies("/chat/completions")[0]["messages"]
    assert [message["role"] for message in messages] == ["system", "user", "user", "assistant", "user"]
    assert messages[1]["content"].startswith("Options:")


def test_a_state_of_only_instructions_leaves_no_state_turn(stub_server):
    """The whole state was instructions: they join the system prompt and nothing is left to place."""
    stub = stub_server(chat=lambda _: (200, chat_body(content="A", logprobs=CHOICE_LOGS)))
    client = SystemOneClient(openai_client(stub), model="stub", api="chat_completions")

    client.system_one(
        state=[{"role": "system", "content": "Answer as a support triage assistant."}],
        questions={"q": Choice(criteria={"billing": None, "sales": None})},
    )

    messages = stub.bodies("/chat/completions")[0]["messages"]
    assert [message["role"] for message in messages] == ["system", "user"]
    assert messages[0]["content"].endswith("Answer as a support triage assistant.")
    assert messages[1]["content"].startswith("Options:")


# -- verdicts belong to the surface and the request that produced them ---------------------------


def test_a_delayed_surface_failure_does_not_write_off_the_working_surface(stub_server):
    """Workers share one call context, so a failure must be judged against its own transport.

    Question 2's Responses call fails only *after* question 1 has failed over to Chat and been
    answered. Read against the shared context — which by then says Chat — the 404 looks like a Chat
    route that does not exist, and the surface that just answered is written off for the rest of the
    client's life.
    """
    answered = threading.Event()
    seen = {"responses": 0}

    def responses(_body):
        seen["responses"] += 1
        if seen["responses"] > 1:
            answered.wait(timeout=10)
        return 404, {"error": {"message": "no such route"}}

    def chat(_body):
        answered.set()
        return 200, chat_body(content="A", logprobs=CHOICE_LOGS)

    stub = stub_server(chat=chat, responses=responses)
    client = SystemOneClient(openai_client(stub), model="stub", max_concurrency=2)

    response = client.system_one(
        state="s",
        questions={"first": Choice(criteria=CRITERIA), "second": Choice(criteria=CRITERIA)},
    )

    assert [answer.choice for answer in response.answers.values()] == ["billing", "billing"]
    assert client._missing_surfaces == {"responses"}


def test_a_downgrade_gives_the_new_request_shape_its_own_retry_budget(stub_server):
    """The attempts one request shape spent say nothing about the shape it is replaced with."""
    seen: list[str] = []

    def script(body):
        kind = (body.get("response_format") or {}).get("type")
        seen.append(str(kind))
        if len(seen) == 1:
            return 500, {"error": {"message": "bad minute"}}
        if kind == "json_schema":
            return unsupported("response_format")
        if len(seen) == 3:
            return 500, {"error": {"message": "another bad minute"}}
        return 200, chat_body(content=STRUCTURED)

    stub = stub_server(chat=script)
    client = SystemOneClient(
        openai_client(stub),
        model="stub",
        api="chat_completions",
        method="structured",
        retry=RetryPolicy(n_retries=1, base_delay=0.0),
    )

    response = client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

    assert response.answers["q"].choice == "billing"
    assert seen == ["json_schema", "json_schema", "json_object", "json_object"]


# -- a caller's own request fields -------------------------------------------------------------


def test_a_caller_cache_key_is_dropped_with_the_refused_field(stub_server):
    """The key the caller put in ``extra_body`` reaches the wire last, so the ladder has to drop it too.

    Otherwise the re-ask is byte-for-byte the same request and the call fails on a field jevper was
    supposed to have removed.
    """
    def script(body):
        if "prompt_cache_key" in body:
            return unsupported("prompt_cache_key")
        return 200, chat_body(content=STRUCTURED)

    stub = stub_server(chat=script)
    client = SystemOneClient(
        openai_client(stub),
        model="stub",
        api="chat_completions",
        method="structured",
        extra_body={"prompt_cache_key": "hand-rolled"},
    )

    response = client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

    assert response.answers["q"].choice == "billing"
    assert [("prompt_cache_key" in body) for body in stub.bodies("/chat/completions")] == [True, False]
    assert response.debug["server_limits"]["cache_key"] is False


def test_a_caller_format_field_puts_the_schema_in_the_prompt(stub_server):
    """The caller's ``response_format`` wins on the wire, so the request carries no schema of ours."""
    stub = stub_server(chat=lambda _: (200, chat_body(content=STRUCTURED)))
    client = SystemOneClient(
        openai_client(stub),
        model="stub",
        api="chat_completions",
        method="structured",
        extra_body={"response_format": {"type": "json_object"}},
    )

    response = client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

    sent = stub.bodies("/chat/completions")[0]
    assert sent["response_format"] == {"type": "json_object"}
    assert "JSON Schema" in sent["messages"][0]["content"]
    assert response.answers["q"].choice == "billing"


# -- refusals that are not about logprobs ------------------------------------------------------


def test_an_include_refusal_on_a_reasoning_request_drops_the_include(stub_server):
    """The Responses ``include`` list carries reasoning as well as logprobs.

    A request that asked for no logprobs can still be refused for that path, and reading it as a logprob
    rejection would skip the include rung of the ladder and fail a call the server would have answered.
    """
    def script(body):
        if "reasoning.encrypted_content" in (body.get("include") or []):
            return 400, {
                "error": {
                    "message": 'Invalid option: expected one of "file_search_call.results"',
                    "param": "include",
                }
            }
        return 200, responses_body(text=STRUCTURED)

    stub = stub_server(responses=script)
    client = SystemOneClient(
        openai_client(stub), model="stub", api="responses", method="structured"
    )

    response = client.system_one(
        state="s",
        questions={"q": Choice(criteria=CRITERIA)},
        reasoning=ReasoningConfig(mode="native"),
    )

    assert response.answers["q"].choice == "billing"
    assert "include" not in stub.bodies("/responses")[1]
    assert response.debug["server_limits"]["include"] is False


def test_a_value_rejected_logprob_request_does_not_move_the_surface(stub_server):
    """A cap on ``top_logprobs`` is not a missing capability, so the surface keeps its turn."""
    def responses(body):
        if body.get("top_logprobs") is not None:
            return 400, {
                "error": {"message": "Invalid 'top_logprobs': integer must be between 0 and 5"}
            }
        return 200, responses_body(text=STRUCTURED)

    stub = stub_server(
        responses=responses, chat=lambda _: (200, chat_body(content="A", logprobs=CHOICE_LOGS))
    )
    client = SystemOneClient(openai_client(stub), model="stub")

    first = client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})
    second = client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

    assert first.answers["q"].choice == "billing"
    assert second.answers["q"].choice == "billing"
    # Each call asks Responses first: one bad value is not a verdict about the surface, so the
    # label readout keeps trying the surface that can carry a distribution.
    assert [path for path in stub.paths if path.endswith("/chat/completions")] == []
    assert len(stub.bodies("/responses")) == 4


# -- answer shapes a provider should not be able to break --------------------------------------


def test_a_sglang_top_level_reasoning_count_is_read(stub_server):
    """SGLang reports ``usage.reasoning_tokens`` beside the OpenAI nested shape, not inside it."""
    body = chat_body(content="A", logprobs=CHOICE_LOGS, reasoning_tokens=None)
    body["usage"]["reasoning_tokens"] = 160
    stub = stub_server(chat=lambda _: (200, body))
    client = SystemOneClient(openai_client(stub), model="stub", api="chat_completions")

    response = client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

    assert response.usage.reasoning_tokens == 160


def test_a_chat_refusal_is_reported_as_a_refusal(stub_server):
    """``content`` is null and ``refusal`` holds the model's own words; both are the API's shape."""
    stub = stub_server(
        chat=lambda _: (200, chat_body(content=None, refusal="I cannot help with that."))
    )
    client = SystemOneClient(
        openai_client(stub),
        model="stub",
        api="chat_completions",
        method="structured",
        n_retry_malformed=0,
    )

    with pytest.raises(MalformedAnswerError) as error:
        client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

    assert "refused to answer" in str(error.value)
    assert "I cannot help with that." in str(error.value)


def test_a_truncated_answer_names_the_budget_even_when_it_parsed_partially(stub_server):
    """The ``{`` is there, the rest is not: the parse error is real and the budget is the reason."""
    stub = stub_server(
        chat=lambda _: (
            200,
            chat_body(content='{"probabilities": {"billing": 0.5', finish_reason="length"),
        )
    )
    client = SystemOneClient(
        openai_client(stub),
        model="stub",
        api="chat_completions",
        method="structured",
        n_retry_malformed=0,
    )

    with pytest.raises(MalformedAnswerError) as error:
        client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

    assert "ran out of output tokens" in str(error.value)
    assert "length" in str(error.value)


def test_an_unusable_first_choice_is_read_as_no_choice_at_all(stub_server):
    """A 200 whose only choice is null, with the provider's failure beside it, is that failure."""
    stub = stub_server(
        chat=lambda _: (200, {"choices": [None], "error": {"message": "overloaded", "code": 503}})
    )
    client = SystemOneClient(
        openai_client(stub),
        model="stub",
        api="chat_completions",
        retry=RetryPolicy(n_retries=0),
    )

    with pytest.raises(ProviderError) as error:
        client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

    assert "overloaded" in str(error.value)
    assert error.value.status_code == 503


def test_a_reasoning_item_whose_dump_is_not_a_mapping_is_skipped():
    """Reasoning is decoration: a client object that cannot be read must not cost the answer.

    ``_as_mapping`` promises that an unreadable object maps to nothing, and a duck-typed client is
    free to be one — the reasoning item here dumps to ``None``, which the old code handed straight to
    a ``.get`` and took the whole answer down with it.
    """
    from jevper.transport import _chat_result

    class Broken:
        type = "reasoning"

        def model_dump(self):
            return None

    body = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="A", reasoning=[Broken()]))],
        usage=None,
    )

    result = _chat_result(body, {})

    assert result.text == "A"
    assert reasoning_text(result.reasoning) == ""


def test_an_unreadable_token_count_is_reported_as_unreported(stub_server):
    """``int(float("inf"))`` raises OverflowError; a provider bug must not escape the call."""
    body = chat_body(content="A", logprobs=CHOICE_LOGS)
    body["usage"]["prompt_tokens"] = float("inf")
    stub = stub_server(chat=lambda _: (200, body))
    client = SystemOneClient(openai_client(stub), model="stub", api="chat_completions")

    response = client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

    assert response.answers["q"].choice == "billing"
    assert response.usage.input_tokens is None
    assert response.usage.output_tokens == 3


# -- what the caller pinned, and what they sent ------------------------------------------------


def test_an_exhausted_surface_fallback_is_a_provider_error(stub_server):
    """Both routes 404: the caller gets the public error, with the status and the attempt history."""
    stub = stub_server(
        chat=lambda _: (404, {"error": {"message": "no such route"}}),
        responses=lambda _: (404, {"error": {"message": "no such route"}}),
    )
    client = SystemOneClient(openai_client(stub), model="stub")

    with pytest.raises(ProviderError) as error:
        client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

    assert error.value.status_code == 404
    assert [attempt["surface"] for attempt in error.value.attempts] == ["responses", "chat_completions"]


def test_no_schema_rung_is_spent_when_structured_outputs_is_off(stub_server):
    """The request already carried ``json_object``, so the next rung is no format field at all."""
    seen: list[Any] = []

    def script(body):
        seen.append((body.get("response_format") or {}).get("type"))
        if "response_format" in body:
            return unsupported("response_format")
        return 200, chat_body(content=STRUCTURED)

    stub = stub_server(chat=script)
    client = SystemOneClient(
        openai_client(stub),
        model="stub",
        api="chat_completions",
        method="structured",
        structured_outputs=False,
    )

    response = client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

    assert response.answers["q"].choice == "billing"
    assert seen == ["json_object", None]
    assert response.debug["server_limits"]["structured"] == "none"


def test_a_same_surface_downgrade_keeps_the_auto_fallback(stub_server):
    """Reloading the method on a same-surface downgrade would undo the fallback this call chose."""
    seen: list[tuple[Any, Any]] = []

    def script(body):
        seen.append((body.get("top_logprobs"), (body.get("response_format") or {}).get("type")))
        if body.get("top_logprobs") is not None:
            return 200, chat_body(content="A")  # answered, but with no logprobs at all
        if (body.get("response_format") or {}).get("type") == "json_schema":
            return unsupported("response_format")
        return 200, chat_body(content=STRUCTURED)

    stub = stub_server(chat=script)
    client = SystemOneClient(
        openai_client(stub), model="stub", api="chat_completions", max_concurrency=1
    )

    response = client.system_one(
        state="s",
        questions={"first": Choice(criteria=CRITERIA), "second": Choice(criteria=CRITERIA)},
    )

    assert [answer.choice for answer in response.answers.values()] == ["billing", "billing"]
    assert response.debug["methods"] == {"first": "structured", "second": "structured"}
    # The first question pays for the probe, the schema refusal and the object rung; the second one
    # takes the fallback and the remembered limit without paying for either again.
    assert seen == [
        (20, None),
        (None, "json_schema"),
        (None, "json_object"),
        (None, "json_object"),
    ]


@pytest.mark.parametrize("override", ["", "bogus"])
def test_an_explicit_api_override_is_validated_before_any_request(stub_server, override):
    stub = stub_server(chat=lambda _: (200, chat_body(content="A", logprobs=CHOICE_LOGS)))
    client = SystemOneClient(openai_client(stub), model="stub")

    with pytest.raises(JevperError):
        client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)}, api=override)

    assert stub.requests == []


def test_an_explicit_empty_method_is_refused_rather_than_ignored(stub_server):
    """An empty string is a caller mistake, not a request for the constructor's default."""
    stub = stub_server(chat=lambda _: (200, chat_body(content="A", logprobs=CHOICE_LOGS)))
    client = SystemOneClient(
        openai_client(stub), model="stub", api="chat_completions", method="logprobs"
    )

    with pytest.raises(JevperError):
        client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)}, method="")

    assert stub.requests == []


def test_extra_headers_reach_the_wire(stub_server):
    stub = stub_server(chat=lambda _: (200, chat_body(content="A", logprobs=CHOICE_LOGS)))
    client = SystemOneClient(
        openai_client(stub),
        model="stub",
        api="chat_completions",
        extra_headers={"X-Jevper-Probe": "1"},
    )

    client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

    sent = {name.lower(): value for name, value in stub.headers[0].items()}
    assert sent.get("x-jevper-probe") == "1"


def test_temperature_reaches_the_wire_on_both_openai_surfaces(stub_server):
    stub = stub_server(
        chat=lambda _: (200, chat_body(content="A", logprobs=CHOICE_LOGS)),
        responses=lambda _: (200, responses_body(text="A", logprobs=CHOICE_LOGS)),
    )
    chat = SystemOneClient(openai_client(stub), model="stub", api="chat_completions", temperature=0.0)
    chat.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})
    assert stub.bodies("/chat/completions")[0]["temperature"] == 0.0

    responses = SystemOneClient(openai_client(stub), model="stub", api="responses", temperature=0.5)
    responses.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})
    assert stub.bodies("/responses")[0]["temperature"] == 0.5


def test_the_backoff_follows_the_policy(stub_server, monkeypatch):
    """``base_delay`` grows by threes and stops at ``max_delay``; the sleeps are the observable."""
    sleeps: list[float] = []
    monkeypatch.setattr(time, "sleep", sleeps.append)

    def script(_body):
        return 503, {"error": {"message": "later"}}

    stub = stub_server(chat=script)
    client = SystemOneClient(
        openai_client(stub),
        model="stub",
        api="chat_completions",
        method="logprobs",
        retry=RetryPolicy(n_retries=3, base_delay=0.5, max_delay=8.0),
    )

    with pytest.raises(ProviderError):
        client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

    assert sleeps == [0.5, 1.5, 4.5]


@pytest.mark.parametrize("status", [502, 504, 529])
def test_every_documented_transient_status_is_retried(stub_server, status):
    seen: list[int] = []

    def script(_body):
        seen.append(status)
        if len(seen) == 1:
            return status, {"error": {"message": "later"}}
        return 200, chat_body(content="A", logprobs=CHOICE_LOGS)

    stub = stub_server(chat=script)
    client = SystemOneClient(
        openai_client(stub),
        model="stub",
        api="chat_completions",
        retry=RetryPolicy(n_retries=1, base_delay=0.0),
    )

    response = client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

    assert response.answers["q"].choice == "billing"
    assert response.usage.n_retries == 1


def test_the_debug_payload_carries_its_documented_keys(stub_server):
    stub = stub_server(chat=lambda _: (200, chat_body(content="A", logprobs=CHOICE_LOGS)))
    client = SystemOneClient(
        openai_client(stub), model="stub", api="chat_completions", method="logprobs"
    )

    response = client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

    assert {
        "method",
        "api",
        "reasoning_mode",
        "llm_attempts",
        "retry_reasons",
        "probability_errors",
        "original_probabilities",
        "labels_missing",
    } <= set(response.debug)
    # The two optional keys appear only when they have something to say.
    assert "methods" not in response.debug and "server_limits" not in response.debug


def test_the_first_failing_question_in_insertion_order_is_the_one_raised(stub_server):
    """Two questions fail; the caller is told about the one they asked about first."""
    other = {"billing": None, "refunds": None}

    def script(body):
        if "refunds" in json.dumps(body):
            return 400, {"error": {"message": "second question is broken"}}
        return 400, {"error": {"message": "first question is broken"}}

    stub = stub_server(chat=script)
    client = SystemOneClient(
        openai_client(stub),
        model="stub",
        api="chat_completions",
        retry=RetryPolicy(n_retries=0),
        max_concurrency=2,
    )

    with pytest.raises(ProviderError) as error:
        client.system_one(
            state="s",
            questions={"first": Choice(criteria=CRITERIA), "second": Choice(criteria=other)},
        )

    assert "first question is broken" in str(error.value)
    assert len(stub.requests) == 2  # both were attempted; the answer is about the first


def test_an_async_transient_failure_is_retried_and_counted(stub_server):
    """The async driver has its own retry loop, so it needs its own proof of it."""
    seen: list[int] = []

    def script(_body):
        seen.append(1)
        if len(seen) == 1:
            return 503, {"error": {"message": "later"}}
        return 200, chat_body(content="A", logprobs=CHOICE_LOGS)

    stub = stub_server(chat=script)
    client = AsyncSystemOneClient(
        async_openai_client(stub),
        model="stub",
        api="chat_completions",
        retry=RetryPolicy(n_retries=1, base_delay=0.0),
    )

    response = asyncio.run(client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)}))

    assert response.answers["q"].choice == "billing"
    assert response.usage.n_retries == 1
    assert [attempt["error"] for attempt in response.debug["llm_attempts"]] == [
        "InternalServerError: Error code: 503 - {'error': {'message': 'later'}}",
        None,
    ]


def test_an_async_malformed_answer_is_retried_with_a_correction(stub_server):
    seen: list[dict] = []

    def script(body):
        seen.append(body)
        if len(seen) == 1:
            return 200, chat_body(content="not json at all")
        return 200, chat_body(content=STRUCTURED)

    stub = stub_server(chat=script)
    client = AsyncSystemOneClient(
        async_openai_client(stub), model="stub", api="chat_completions", method="structured"
    )

    response = asyncio.run(client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)}))

    assert response.answers["q"].choice == "billing"
    assert len(seen) == 2
    assert "invalid" in seen[1]["messages"][-1]["content"]
    (reason,) = response.debug["retry_reasons"]
    assert reason.startswith("no JSON object in the answer (Expecting value: line 1 column 1 (char 0))")


def test_examples_are_validated_before_the_first_provider_call(stub_server):
    """A locally invalid call must cost nothing: the second question's example is checked up front."""
    stub = stub_server(chat=lambda _: (200, chat_body(content=STRUCTURED)))
    client = SystemOneClient(
        openai_client(stub), model="stub", api="chat_completions", method="structured", max_concurrency=1
    )

    with pytest.raises(InvalidQuestionError):
        client.system_one(
            state="s",
            questions={
                "first": Choice(criteria=CRITERIA),
                "second": Choice(
                    criteria=CRITERIA,
                    examples=[{"state": "charged twice", "answer": "not-an-option"}],
                ),
            },
        )

    assert stub.requests == []
