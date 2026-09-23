"""Client behaviour: transient retries, validation, concurrency, async parity, state forms."""

from __future__ import annotations

import asyncio
import json
import threading
import time

import pytest
from fakes import async_openai_client, chat_body, openai_client, responses_body

from jevper import (
    AsyncSystemOneClient,
    Choice,
    InvalidQuestionError,
    JevperError,
    Noul,
    ProviderError,
    RetryPolicy,
    Score,
    SystemOneClient,
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
    client = SystemOneClient(openai_client(stub), model="stub", api="chat_completions")
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
    question_blocks = [request["messages"][-1]["content"] for request in requests]
    assert any("A: billing" in block for block in question_blocks)
    assert any("A: Yes" in block for block in question_blocks)
    assert any("A: 0 — Calm" in block for block in question_blocks)
    assert response.usage.n_calls == 3
    assert list(response.choices) == ["intent"]
    assert list(response.nouls) == ["duplicate"]
    assert list(response.scores) == ["anger"]


def test_one_failing_question_raises_in_insertion_order_after_all_settle(stub_server):
    def script(body):
        question = body["messages"][-1]["content"]
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


def test_state_chat_messages_are_preserved_and_question_block_is_last(stub_server):
    stub = stub_server(chat=lambda _: (200, chat_body(content="A", logprobs=CHOICE_LOGS)))
    client = SystemOneClient(openai_client(stub), model="stub", api="chat_completions")

    client.system_one(
        state={
            "messages": [
                {"role": "system", "content": "You are a support triage assistant."},
                {"role": "user", "content": "I was charged twice."},
                {"role": "assistant", "content": "Let me look into that."},
            ]
        },
        questions={"q": Choice(criteria={"billing": None, "sales": None})},
    )

    messages = stub.bodies("/chat/completions")[0]["messages"]
    assert [message["role"] for message in messages] == ["system", "system", "user", "assistant", "user"]
    assert messages[0]["content"].startswith("You are a precise classification engine")
    assert messages[1]["content"] == "You are a support triage assistant."
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
    assert messages[1]["role"] == "user"
    assert '"order_id": 42' in messages[1]["content"]
    assert '"qty": 2' in messages[1]["content"]


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

    block = stub.bodies("/chat/completions")[0]["messages"][-1]["content"]
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
    assert request["messages"][-1]["content"].startswith("Options:")


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
        body["messages"][-1]["content"] for body in bodies if "A: low" in body["messages"][-1]["content"]
    )
    assert "Question:\n42" in choice_block
    assert "A: low — 1" in choice_block and "B: high — 5" in choice_block
    score_block = next(
        body["messages"][-1]["content"] for body in bodies if "A: 0" in body["messages"][-1]["content"]
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
