"""Integration with MLflow 3.16.1: tracing, hosting, the AI Gateway, and genai evaluation.

These tests need MLflow, which is not a runtime or test dependency of jevper — the whole module skips
without it. Install it with ``uv pip install -e '.[test,mlflow]'``.

What is being verified, beyond "it imports":

* MLflow's ``mlflow.openai.autolog()`` and ``mlflow.anthropic.autolog()`` patch the SDK resource
  classes, so every call jevper makes through a caller-supplied client is traced — including the
  attempts it makes *before* settling on one it can use. A refused field is an error span naming that
  field, so the trace shows exactly which rungs of the fallback ladder the provider turned down.
* MLflow's tracing context lives in a ``ContextVar``, and jevper answers questions on worker threads,
  so spans created inside those threads are roots rather than children. The tests pin that behaviour
  and the documented workaround.
* MLflow's standalone AI Gateway (``mlflow gateway start``, deprecated in 3.16.1 in favour of the
  server-hosted gateway) is a faithful pass-through in both directions for the chat route, but it has
  no ``/v1/responses`` route at all — so ``api="auto"`` against it must fall back, which is a
  behaviour of jevper's, not of MLflow's.
* Hosting jevper as an MLflow model (``pyfunc.ChatModel``, deprecated, and ``pyfunc.ResponsesAgent``,
  the current recommendation) and evaluating it with ``mlflow.genai.evaluate``.
"""

from __future__ import annotations

import asyncio
import json
import socket
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any

import pytest

mlflow = pytest.importorskip("mlflow", reason="MLflow is not installed (uv pip install -e '.[test,mlflow]')")
from mlflow.types.responses import ResponsesAgentRequest

sys.path.insert(0, str(Path(__file__).parent))

from fakes import (
    StubServer,
    anthropic_client,
    async_openai_client,
    chat_body,
    messages_body,
    openai_client,
    responses_body,
)

from jevper import (
    AsyncSystemOneClient,
    Choice,
    LabelReadoutError,
    MalformedAnswerError,
    ProviderError,
    RetryPolicy,
    SystemOneClient,
    reasoning_text,
)

CHOICE_LOGS = [("A", -0.12), ("B", -2.47), ("C", -3.48)]
CRITERIA = {"billing": None, "technical": None, "sales": None}
STRUCTURED_ANSWER = json.dumps(
    {"choice": "billing", "probabilities": {"billing": 0.9, "technical": 0.05, "sales": 0.05}}
)


# -- helpers -------------------------------------------------------------------------------


def status_of(span: Any) -> str:
    code = getattr(getattr(span, "status", None), "status_code", None)
    return str(getattr(code, "value", code))


def traces(experiment_id: str) -> list[Any]:
    """Every trace in the experiment, oldest first, with pending writes flushed."""
    return mlflow.search_traces(locations=[experiment_id], flush=True, return_type="list")


def spans_of(experiment_id: str) -> list[Any]:
    return [span for trace in traces(experiment_id) for span in trace.data.spans]


def span_named(experiment_id: str, name: str) -> Any:
    found = [span for span in spans_of(experiment_id) if span.name == name]
    assert found, f"no {name!r} span; got {[s.name for s in spans_of(experiment_id)]}"
    return found[-1]


def answer_script(body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    """A chat answer good enough for every method jevper tries, in whatever shape it asks for."""
    if body.get("response_format", {}).get("type") == "json_schema":
        return 200, chat_body(content=STRUCTURED_ANSWER, logprobs=CHOICE_LOGS)
    return 200, chat_body(content="A", logprobs=CHOICE_LOGS)


@pytest.fixture
def tracking(tmp_path: Path) -> Any:
    """A private tracking store and experiment, with autolog off on the way out.

    MLflow's tracking URI, active experiment and autolog patches are process-global, so this fixture
    owns them for the duration of one test and hands them back clean.
    """
    previous_uri = mlflow.get_tracking_uri()
    try:
        previous_experiment = mlflow.tracking.fluent._get_experiment_id()
    except Exception:  # noqa: BLE001 - private helper; a missing value only costs a restore
        previous_experiment = None
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    experiment_id = mlflow.set_experiment(f"jevper-{uuid.uuid4().hex[:8]}").experiment_id
    try:
        yield experiment_id
    finally:
        mlflow.openai.autolog(disable=True)
        mlflow.anthropic.autolog(disable=True)
        mlflow.tracing.enable()
        mlflow.set_tracking_uri(previous_uri)
        if previous_experiment is not None:
            mlflow.set_experiment(experiment_id=previous_experiment)


@pytest.fixture
def autolog(tracking: str) -> str:
    mlflow.openai.autolog()
    mlflow.anthropic.autolog()
    return tracking


# -- tracing -------------------------------------------------------------------------------


def test_chat_call_is_traced_with_the_answer_and_its_usage(stub_server, autolog):
    stub = stub_server(
        chat=lambda _: (200, chat_body(content="A", logprobs=CHOICE_LOGS, input_tokens=42, output_tokens=7))
    )
    client = SystemOneClient(openai_client(stub), model="stub", api="chat_completions", method="logprobs")

    response = client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

    span = span_named(autolog, "Completions")
    assert span.span_type == "CHAT_MODEL"
    assert status_of(span) == "OK"
    assert span.attributes["mlflow.llm.model"] == "stub"
    assert span.attributes["mlflow.chat.tokenUsage"] == {
        "input_tokens": 42,
        "output_tokens": 7,
        "total_tokens": 49,
    }
    # The same numbers jevper reports, so the two accounts of one call agree.
    assert response.usage.input_tokens == 42
    assert response.usage.output_tokens == 7


def test_the_trace_records_the_request_fields_jevper_sent(stub_server, autolog):
    stub = stub_server(chat=lambda _: (200, chat_body(content="A", logprobs=CHOICE_LOGS)))
    client = SystemOneClient(openai_client(stub), model="stub", api="chat_completions", method="logprobs")

    client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

    span = span_named(autolog, "Completions")
    assert span.attributes["logprobs"] is True
    assert span.attributes["top_logprobs"] == 20
    # The derived cache key is visible in the trace, which is what makes cache reuse auditable.
    assert span.attributes["prompt_cache_key"].startswith("jevper-")
    assert span.inputs["model"] == "stub"
    assert span.inputs["messages"][0]["role"] == "system"


def test_responses_call_is_traced(stub_server, autolog):
    stub = stub_server(responses=lambda _: (200, responses_body(text="A", logprobs=CHOICE_LOGS)))
    client = SystemOneClient(openai_client(stub), model="stub", api="responses", method="logprobs")

    client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

    span = span_named(autolog, "Responses")
    assert status_of(span) == "OK"
    assert span.attributes["mlflow.chat.tokenUsage"]["input_tokens"] == 10
    assert span.attributes["include"] == ["message.output_text.logprobs"]


def test_messages_call_is_traced(stub_server, autolog):
    stub = stub_server(messages=lambda _: (200, messages_body(text=STRUCTURED_ANSWER)))
    client = SystemOneClient(anthropic_client(stub), model="stub", api="messages", method="structured")

    client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

    span = span_named(autolog, "Messages.create")
    assert span.attributes["mlflow.llm.provider"] == "anthropic"
    assert span.attributes["mlflow.message.format"] == "anthropic"
    assert span.attributes["mlflow.chat.tokenUsage"]["total_tokens"] == 13
    # The Messages route always sends max_tokens, and it lands in span inputs rather than attributes:
    # MLflow promotes request fields to attributes on the OpenAI routes only.
    assert span.inputs["max_tokens"] == 1024
    assert "max_tokens" not in span.attributes


def test_async_call_is_traced(stub_server, autolog):
    stub = stub_server(chat=lambda _: (200, chat_body(content="A", logprobs=CHOICE_LOGS)))
    client = AsyncSystemOneClient(
        async_openai_client(stub), model="stub", api="chat_completions", method="logprobs"
    )

    asyncio.run(client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)}))

    span = span_named(autolog, "AsyncCompletions")
    assert status_of(span) == "OK"


def test_tracing_disabled_still_answers(stub_server, autolog):
    stub = stub_server(chat=lambda _: (200, chat_body(content="A", logprobs=CHOICE_LOGS)))
    client = SystemOneClient(openai_client(stub), model="stub", api="chat_completions", method="logprobs")
    mlflow.tracing.disable()

    response = client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

    assert response.answers["q"].choice == "billing"
    assert traces(autolog) == []


def test_an_unwritable_tracking_store_does_not_break_the_call(stub_server, autolog):
    """MLflow cannot write the span, and the answer still comes back."""
    stub = stub_server(chat=lambda _: (200, chat_body(content="A", logprobs=CHOICE_LOGS)))
    client = SystemOneClient(openai_client(stub), model="stub", api="chat_completions", method="logprobs")
    good_uri = mlflow.get_tracking_uri()
    # A path under a file: no store can be created there, and the span cannot be exported.
    mlflow.set_tracking_uri("sqlite:////dev/null/nope/mlflow.db")
    try:
        response = client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})
    finally:
        mlflow.set_tracking_uri(good_uri)

    assert response.answers["q"].choice == "billing"
    assert response.usage.n_calls == 1
    assert len(traces(autolog)) == 0


def test_switching_the_tracking_store_mid_run(tmp_path, stub_server):
    """Traces follow the tracking URI that is active when the call is made."""
    first = tmp_path / "first.db"
    second = tmp_path / "second.db"
    stub = stub_server(chat=answer_script)

    mlflow.set_tracking_uri(f"sqlite:///{first}")
    mlflow.openai.autolog()
    first_experiment = mlflow.set_experiment("store-one").experiment_id
    SystemOneClient(openai_client(stub), model="stub", api="chat_completions", method="logprobs").system_one(
        state="s", questions={"q": Choice(criteria=CRITERIA)}
    )

    mlflow.set_tracking_uri(f"sqlite:///{second}")
    second_experiment = mlflow.set_experiment("store-two").experiment_id
    SystemOneClient(openai_client(stub), model="stub", api="chat_completions", method="logprobs").system_one(
        state="s", questions={"q": Choice(criteria=CRITERIA)}
    )

    assert len(traces(first_experiment)) == 1
    assert len(traces(second_experiment)) == 1
    mlflow.openai.autolog(disable=True)


def test_a_retried_call_records_both_attempts(stub_server, autolog):
    calls: list[int] = []

    def scripted(body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        calls.append(1)
        if len(calls) == 1:
            return 429, {"error": {"message": "slow down", "type": "rate_limit_error"}}
        return 200, chat_body(content="A", logprobs=CHOICE_LOGS)

    stub = stub_server(chat=scripted)
    client = SystemOneClient(
        openai_client(stub),
        model="stub",
        api="chat_completions",
        method="logprobs",
        retry=RetryPolicy(n_retries=2, base_delay=0.0),
    )

    response = client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

    assert response.usage.n_retries == 1
    recorded = sorted(
        (span for trace in traces(autolog) for span in trace.data.spans), key=lambda s: s.start_time_ns
    )
    assert [status_of(span) for span in recorded] == ["ERROR", "OK"]


def test_two_transient_failures_and_the_retry_are_all_traced(stub_server, autolog):
    calls: list[int] = []

    def scripted(body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        calls.append(1)
        if len(calls) <= 2:
            return 500, {"error": {"message": "upstream exploded", "type": "server_error"}}
        return 200, chat_body(content="A", logprobs=CHOICE_LOGS)

    stub = stub_server(chat=scripted)
    client = SystemOneClient(
        openai_client(stub),
        model="stub",
        api="chat_completions",
        method="logprobs",
        retry=RetryPolicy(n_retries=2, base_delay=0.0),
    )

    response = client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

    assert response.usage.n_retries == 2
    recorded = sorted(
        (span for trace in traces(autolog) for span in trace.data.spans), key=lambda s: s.start_time_ns
    )
    assert [status_of(span) for span in recorded] == ["ERROR", "ERROR", "OK"]


def test_every_refused_rung_of_the_ladder_is_a_traced_error(stub_server, autolog):
    """The structured fallback ladder is visible in the trace, rung by rung."""

    def scripted(body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        if "response_format" in body:
            return 400, {
                "error": {"message": "Unknown field response_format", "type": "invalid_request_error"}
            }
        return 200, chat_body(content=STRUCTURED_ANSWER)

    stub = stub_server(chat=scripted)
    client = SystemOneClient(openai_client(stub), model="stub", api="chat_completions", method="structured")

    response = client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

    assert response.answers["q"].choice == "billing"
    assert response.debug["server_limits"]["structured"] == "none"
    recorded = sorted(
        (span for trace in traces(autolog) for span in trace.data.spans), key=lambda s: s.start_time_ns
    )
    assert [status_of(span) for span in recorded] == ["ERROR", "ERROR", "OK"]
    refused = [span.attributes["response_format"]["type"] for span in recorded[:-1]]
    assert refused == ["json_schema", "json_object"]


def test_the_corrective_retry_is_traced(stub_server, autolog):
    calls: list[int] = []

    def scripted(body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        calls.append(1)
        if len(calls) == 1:
            return 200, chat_body(content="I think the answer is billing.")
        return 200, chat_body(content=STRUCTURED_ANSWER)

    stub = stub_server(chat=scripted)
    client = SystemOneClient(openai_client(stub), model="stub", api="chat_completions", method="structured")

    client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

    recorded = sorted(
        (span for trace in traces(autolog) for span in trace.data.spans), key=lambda s: s.start_time_ns
    )
    assert len(recorded) == 2
    assert [status_of(span) for span in recorded] == ["OK", "OK"]
    assert recorded[0].attributes["mlflow.chat.tokenUsage"] is not None


def test_a_failed_question_records_an_error_span(stub_server, autolog):
    stub = stub_server(
        chat=lambda _: (400, {"error": {"message": "bad request", "type": "invalid_request_error"}})
    )
    client = SystemOneClient(openai_client(stub), model="stub", api="chat_completions", method="logprobs")

    with pytest.raises(ProviderError):
        client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

    recorded = [span for trace in traces(autolog) for span in trace.data.spans]
    assert recorded and all(status_of(span) == "ERROR" for span in recorded)


def test_each_sdk_call_is_its_own_trace(stub_server, autolog):
    stub = stub_server(chat=answer_script)
    client = SystemOneClient(openai_client(stub), model="stub", api="chat_completions", method="logprobs")

    client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

    recorded = traces(autolog)
    assert len(recorded) == 1
    assert [span.name for span in recorded[0].data.spans] == ["Completions"]


def test_a_single_question_stays_inside_the_wrapper_span(stub_server, autolog):
    """One question is answered inline, so its SDK span is a child of the wrapper's."""
    stub = stub_server(chat=answer_script)
    client = SystemOneClient(openai_client(stub), model="stub", api="chat_completions", method="logprobs")

    @mlflow.trace(name="system_one")
    def traced_call() -> Any:
        return client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

    traced_call()

    grouped = next(t for t in traces(autolog) if any(s.name == "system_one" for s in t.data.spans))
    names = {span.name for span in grouped.data.spans}
    assert names == {"system_one", "Completions"}
    completions = next(span for span in grouped.data.spans if span.name == "Completions")
    wrapper = next(span for span in grouped.data.spans if span.name == "system_one")
    assert completions.parent_id == wrapper.span_id


def test_two_questions_become_their_own_traces_inside_the_wrapper(stub_server, autolog):
    """Two questions are answered on worker threads, and MLflow's context does not cross threads.

    So the SDK spans are roots of their own rather than children of the wrapper span — the wrapper
    trace keeps the answer, and the per-call traces keep the provider's view of it.
    """
    stub = stub_server(chat=answer_script)
    client = SystemOneClient(openai_client(stub), model="stub", api="chat_completions", method="logprobs")

    @mlflow.trace(name="system_one")
    def traced_call() -> Any:
        return client.system_one(
            state="s",
            questions={"a": Choice(criteria=CRITERIA), "b": Choice(criteria=CRITERIA)},
        )

    traced_call()

    recorded = traces(autolog)
    grouped = next(t for t in recorded if any(s.name == "system_one" for s in t.data.spans))
    assert [span.name for span in grouped.data.spans] == ["system_one"]
    assert len(recorded) == 3


def test_a_chat_only_client_is_still_traced(stub_server, autolog):
    """A caller-supplied object with only ``chat.completions``: jevper detects the surface, and
    autolog still sees the SDK call underneath, because the object delegates to the real client."""
    stub = stub_server(chat=answer_script)

    class ChatOnly:
        def __init__(self, client: Any) -> None:
            self.chat = client.chat

    client = SystemOneClient(ChatOnly(openai_client(stub)), model="stub", method="auto")

    response = client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

    assert response.answers["q"].choice == "billing"
    assert response.debug["api"] == "chat_completions"
    assert span_named(autolog, "Completions") is not None


def test_concurrent_calls_record_separate_traces(stub_server, autolog):
    stub = stub_server(chat=answer_script)
    barrier = threading.Barrier(3)
    errors: list[BaseException] = []

    def call() -> None:
        client = SystemOneClient(openai_client(stub), model="stub", api="chat_completions", method="logprobs")
        barrier.wait(timeout=10)
        try:
            client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})
        except BaseException as exc:  # noqa: BLE001 - reported through `errors`
            errors.append(exc)

    threads = [threading.Thread(target=call) for _ in range(2)]
    for thread in threads:
        thread.start()
    barrier.wait(timeout=10)
    for thread in threads:
        thread.join(timeout=30)

    assert not errors
    assert len(traces(autolog)) == 2


def test_an_answer_without_usage_still_traces(stub_server, autolog):
    stub = stub_server(chat=lambda _: (200, chat_body(content="A", logprobs=CHOICE_LOGS, input_tokens=None, output_tokens=None)))
    client = SystemOneClient(openai_client(stub), model="stub", api="chat_completions", method="logprobs")

    response = client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

    assert response.answers["q"].choice == "billing"
    span = span_named(autolog, "Completions")
    assert status_of(span) == "OK"
    # MLflow records the key with null counts rather than omitting it.
    assert span.attributes["mlflow.chat.tokenUsage"] == {
        "input_tokens": None,
        "output_tokens": None,
        "total_tokens": None,
    }


def test_two_experiments_keep_their_own_traces(stub_server, autolog):
    """Traces land in whichever experiment is active, and the two do not mix."""
    stub = stub_server(chat=answer_script)
    first = autolog
    second = mlflow.set_experiment(f"jevper-second-{uuid.uuid4().hex[:8]}").experiment_id

    mlflow.set_experiment(experiment_id=first)
    SystemOneClient(openai_client(stub), model="stub", api="chat_completions", method="logprobs").system_one(
        state="one", questions={"q": Choice(criteria=CRITERIA)}
    )
    mlflow.set_experiment(experiment_id=second)
    SystemOneClient(openai_client(stub), model="stub", api="chat_completions", method="logprobs").system_one(
        state="two", questions={"q": Choice(criteria=CRITERIA)}
    )

    assert len(traces(first)) == 1
    assert len(traces(second)) == 1
    assert "one" in json.dumps(traces(first)[0].data.spans[0].inputs)
    assert "two" in json.dumps(traces(second)[0].data.spans[0].inputs)


def test_trace_metadata_can_be_attached_to_a_jevper_call(stub_server, autolog):
    stub = stub_server(chat=answer_script)
    client = SystemOneClient(openai_client(stub), model="stub", api="chat_completions", method="logprobs")

    @mlflow.trace(name="triage")
    def traced_call(text: str) -> Any:
        mlflow.update_current_trace(user="tester", session_id="session-1")
        mlflow.update_current_trace(tags={"question": "intent"})
        return client.system_one(state=text, questions={"q": Choice(criteria=CRITERIA)})

    traced_call("I was charged twice")

    trace = next(t for t in traces(autolog) if any(s.name == "triage" for s in t.data.spans))
    assert trace.info.tags["question"] == "intent"
    assert trace.info.trace_metadata["mlflow.trace.user"] == "tester"
    assert trace.info.trace_metadata["mlflow.trace.session"] == "session-1"


def test_reasoning_is_readable_from_a_traced_call(stub_server, autolog):
    """The reasoning text jevper returns is also in the span, so both agree."""
    stub = stub_server(
        chat=lambda _: (
            200,
            chat_body(content="A", logprobs=CHOICE_LOGS, reasoning="The label is billing."),
        )
    )
    client = SystemOneClient(openai_client(stub), model="stub", api="chat_completions", method="logprobs")

    response = client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

    assert response.reasoning is not None
    assert "billing" in reasoning_text(response.reasoning)
    assert "billing" in json.dumps(span_named(autolog, "Completions").outputs)


# -- hosting jevper as an MLflow model -----------------------------------------------------


FIXTURES = Path(__file__).parent / "fixtures"


def _log_model(stub: Any, source: str, name: str = "jevper", input_example: Any = None) -> str:
    """Log one of the fixture model files, the models-from-code way."""
    if input_example is None:
        input_example = {"messages": [{"role": "user", "content": "I was charged twice"}]}
    with mlflow.start_run():
        info = mlflow.pyfunc.log_model(
            name=name,
            python_model=str(FIXTURES / source),
            model_config={"base_url": stub.base_url, "model": "stub"},
            input_example=input_example,
        )
    return info.model_uri


def test_chat_model_predicts_through_the_pyfunc_interface(stub_server, tracking):
    stub = stub_server(chat=answer_script)
    model_uri = _log_model(stub, "jevper_chat_model.py")

    loaded = mlflow.pyfunc.load_model(model_uri)
    result = loaded.predict({"messages": [{"role": "user", "content": "I was charged twice"}]})

    assert result["choices"][0]["message"]["content"] == "billing"


def test_chat_model_survives_the_predict_helper(stub_server, tracking, tmp_path):
    """``mlflow.models.predict`` is the MLflow-standard way to invoke a logged model in place."""
    stub = stub_server(chat=answer_script)
    model_uri = _log_model(stub, "jevper_chat_model.py")
    output_path = tmp_path / "prediction.json"

    mlflow.models.predict(
        model_uri=model_uri,
        input_data={"messages": [{"role": "user", "content": "I was charged twice"}]},
        output_path=str(output_path),
        env_manager="local",
    )

    assert json.loads(output_path.read_text())["choices"][0]["message"]["content"] == "billing"


def test_chat_model_reports_a_readout_failure(stub_server, tracking):
    """A model that answers prose where a label is required fails loudly, not silently."""
    stub = stub_server(chat=answer_script)
    model_uri = _log_model(stub, "jevper_chat_model.py")
    stub.chat = lambda _: (200, chat_body(content="Probably billing."))

    loaded = mlflow.pyfunc.load_model(model_uri)
    with pytest.raises(MalformedAnswerError) as caught:
        loaded.predict({"messages": [{"role": "user", "content": "I was charged twice"}]})

    assert "no JSON object in the answer" in str(caught.value)


def test_responses_agent_predicts_the_state_it_was_given(stub_server, tracking):
    """The state MLflow sends must reach the provider — the agent answers from the request, not a default."""
    stub = stub_server(chat=label_script)
    model_uri = _log_model(
        stub,
        "jevper_responses_agent.py",
        name="jevper-agent",
        input_example={"input": [{"role": "user", "content": "the app crashes on login"}]},
    )

    loaded = mlflow.pyfunc.load_model(model_uri)
    result = loaded.predict({"input": [{"role": "user", "content": "the app crashes on login"}]})

    assert result["output"][0]["content"][0]["text"] == "technical"
    assert result["output"][0]["role"] == "assistant"
    # And the state reached the backend, so the label was not a coincidence.
    assert "crashes" in json.dumps(stub.bodies("/chat/completions")[-1])


def test_responses_agent_streams_one_completed_item(stub_server, tracking):
    """``predict_stream`` is what MLflow's serving path asks for; the base class raises without it.

    The pyfunc ``predict`` interface answers in one piece, so this drives the agent the way the serving
    layer does — through the model's own streaming method.
    """
    from types import SimpleNamespace

    stub = stub_server(chat=label_script)
    sys.path.insert(0, str(FIXTURES))
    from jevper_responses_agent import JevperResponsesAgent

    agent = JevperResponsesAgent()
    agent.load_context(SimpleNamespace(model_config={"base_url": stub.base_url, "model": "stub"}))
    request = ResponsesAgentRequest(
        input=[{"role": "user", "content": "the app crashes on login"}]
    )

    events = list(agent.predict_stream(request))

    assert len(events) == 1
    item = events[0].item
    content = item["content"] if isinstance(item, dict) else item.content
    first = content[0]
    text = first["text"] if isinstance(first, dict) else first.text
    assert text == "technical"
    assert "crashes" in json.dumps(stub.bodies("/chat/completions")[-1])


# -- the standalone AI Gateway --------------------------------------------------------------


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class Gateway:
    """The standalone gateway: `/health` at the root, the OpenAI surface under `/v1`."""

    def __init__(self, root: str, backend: StubServer) -> None:
        self.root = root
        self.base_url = f"{root}/v1"
        self.backend = backend


def _wait_for(url: str, process: subprocess.Popen, seconds: float = 60.0) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if process.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(url, timeout=2):
                return True
        except Exception:  # noqa: BLE001 - any failure just means "not yet"
            time.sleep(0.25)
    return False


@pytest.fixture(scope="module")
def gateway() -> Any:
    pytest.importorskip("fastapi", reason="the gateway extra is not installed")
    pytest.importorskip("uvicorn", reason="the gateway extra is not installed")

    stub = StubServer(chat=answer_script, responses=lambda _: (200, responses_body(text="A")))
    port = free_port()
    # A directory of our own: the gateway watches the config's *directory*, and on a systemd host that
    # walk trips over /tmp/systemd-private-* with a permission error and the server never starts.
    config_dir = Path(tempfile.mkdtemp(prefix="jevper-gateway-"))
    config = config_dir / "gateway.yaml"
    config.write_text(
        textwrap.dedent(
            f"""
            endpoints:
              - name: stub-route
                endpoint_type: llm/v1/chat
                model:
                  provider: openai
                  name: stub-model
                  config:
                    openai_api_key: test-key
                    openai_api_base: {stub.base_url}/v1
            """
        )
    )
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "mlflow",
            "gateway",
            "start",
            "--config-path",
            str(config),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    base_url = f"http://127.0.0.1:{port}"
    if not _wait_for(f"{base_url}/health", process):
        process.kill()
        pytest.skip(f"the MLflow gateway did not start: {(process.stdout.read() or '')[-400:]}")
    try:
        yield Gateway(base_url, stub)
    finally:
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
        stub.close()
        config.unlink(missing_ok=True)
        config_dir.rmdir()


def test_gateway_is_healthy(gateway):
    with urllib.request.urlopen(f"{gateway.root}/health", timeout=10) as response:
        assert json.loads(response.read()) == {"status": "OK"}


def test_jevper_answers_through_the_gateway(gateway):
    client = SystemOneClient(
        openai_client(gateway), model="stub-route", api="chat_completions", method="structured"
    )

    response = client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

    assert response.answers["q"].choice == "billing"
    assert response.debug["api"] == "chat_completions"


def test_the_gateway_forwards_the_fields_jevper_sends(gateway):
    """The request is a faithful pass-through, even though the answer is not (see below)."""
    client = SystemOneClient(
        openai_client(gateway), model="stub-route", api="chat_completions", method="logprobs"
    )

    with pytest.raises(LabelReadoutError):
        client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

    received = gateway.backend.bodies("/chat/completions")[-1]
    assert received["logprobs"] is True
    assert received["top_logprobs"] == 20
    assert received["prompt_cache_key"].startswith("jevper-")
    # The gateway replaces the client's model with the configured backend model.
    assert received["model"] == "stub-model"


def test_auto_falls_back_when_the_gateway_has_no_responses_route(gateway):
    """The gateway answers 404 for /v1/responses; jevper notices and answers on chat instead."""
    client = SystemOneClient(openai_client(gateway), model="stub-route", method="auto")

    response = client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

    assert response.answers["q"].choice == "billing"
    assert response.debug["api"] == "chat_completions"
    assert response.debug["method"] == "structured"
    assert response.usage.n_calls == 2


def test_structured_answers_through_the_gateway(gateway):
    client = SystemOneClient(
        openai_client(gateway), model="stub-route", api="chat_completions", method="structured"
    )

    response = client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

    assert response.answers["q"].choice == "billing"
    assert gateway.backend.bodies("/chat/completions")[-1]["response_format"]["type"] == "json_schema"


def test_extra_body_reaches_the_backend_through_the_gateway(gateway):
    """The gateway's request model allows extra fields, so vendor knobs survive it."""
    client = SystemOneClient(
        openai_client(gateway),
        model="stub-route",
        api="chat_completions",
        method="structured",
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )

    client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

    received = gateway.backend.bodies("/chat/completions")[-1]
    assert received["chat_template_kwargs"] == {"enable_thinking": False}


def test_a_backend_error_is_relayed_with_its_status(gateway):
    """A backend 400 arrives as a 400 whose message the caller can read."""
    calls: list[int] = []

    def scripted(body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        calls.append(1)
        return 400, {"error": {"message": "Unknown field top_logprobs", "type": "invalid_request_error"}}

    gateway.backend.chat = scripted
    try:
        client = SystemOneClient(
            openai_client(gateway),
            model="stub-route",
            api="chat_completions",
            method="logprobs",
            retry=RetryPolicy(n_retries=0),
        )
        with pytest.raises(ProviderError) as caught:
            client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})
        assert "top_logprobs" in str(caught.value)
    finally:
        gateway.backend.chat = answer_script


def test_the_gateway_keeps_finish_reason_and_usage(gateway):
    """What survives the gateway's response model: the choice shape and the token counts."""
    client = SystemOneClient(
        openai_client(gateway), model="stub-route", api="chat_completions", method="structured"
    )

    response = client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

    assert response.usage.input_tokens == 10
    assert response.usage.output_tokens == 3


def test_the_gateway_response_contract_is_what_the_docs_say(gateway):
    """Read the gateway's own JSON: what it keeps, and what its response model throws away."""
    gateway.backend.chat = lambda _: (
        200,
        chat_body(
            content="A",
            logprobs=CHOICE_LOGS,
            reasoning="thinking out loud",
            finish_reason="length",
            cached_tokens=61,
            reasoning_tokens=5,
        ),
    )
    try:
        request = urllib.request.Request(
            f"{gateway.base_url}/chat/completions",
            data=json.dumps(
                {"model": "stub-route", "messages": [{"role": "user", "content": "hi"}]}
            ).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            body = json.loads(response.read())
    finally:
        gateway.backend.chat = answer_script

    assert response.status == 200
    assert body["choices"][0]["finish_reason"] == "length"
    assert body["choices"][0]["message"]["refusal"] is None
    assert body["usage"]["prompt_tokens_details"]["cached_tokens"] == 61
    assert body["usage"]["completion_tokens_details"]["reasoning_tokens"] == 5
    assert "logprobs" not in body["choices"][0]
    assert "reasoning_content" not in body["choices"][0]["message"]


def test_the_gateway_drops_logprobs_so_auto_settles_on_structured(gateway):
    """The gateway re-shapes the response, and logprobs is not one of its fields.

    The request reaches the backend with logprobs set, but the answer comes back through MLflow's
    ``ChatCompletionResponse``, which has no place for them — so jevper reads no distribution, says so,
    and falls back to a JSON schema.
    """
    client = SystemOneClient(openai_client(gateway), model="stub-route", method="auto")

    response = client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

    assert response.answers["q"].choice == "billing"
    assert response.debug["method"] == "structured"
    sent = gateway.backend.bodies("/chat/completions")
    assert sent[-2]["logprobs"] is True
    assert "logprobs" not in sent[-1]
    assert sent[-1]["response_format"]["type"] == "json_schema"


def test_the_gateway_relays_a_field_refusal_so_auto_can_fall_back(gateway):
    """A backend 400 naming a field arrives as a 400 naming that field, so the ladder still runs."""
    refusals: list[int] = []

    def scripted(body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        if "logprobs" in body:
            refusals.append(1)
            return 400, {"error": {"message": "Unknown field logprobs", "type": "invalid_request_error"}}
        return 200, chat_body(content=STRUCTURED_ANSWER)

    gateway.backend.chat = scripted
    try:
        client = SystemOneClient(openai_client(gateway), model="stub-route", method="auto")

        response = client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

        assert response.answers["q"].choice == "billing"
        assert response.debug["method"] == "structured"
        assert refusals, "the backend never saw the logprob request"
        assert "logprobs" not in gateway.backend.bodies("/chat/completions")[-1]
    finally:
        gateway.backend.chat = answer_script


def test_the_gateway_drops_reasoning_content(gateway):
    """A reasoning model's thinking is not part of the gateway's response model either."""
    gateway.backend.chat = lambda _: (
        200,
        chat_body(content=STRUCTURED_ANSWER, reasoning="The label is billing."),
    )
    try:
        client = SystemOneClient(
            openai_client(gateway), model="stub-route", api="chat_completions", method="structured"
        )
        response = client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})
        assert response.answers["q"].choice == "billing"
        assert not response.reasoning
    finally:
        gateway.backend.chat = answer_script


def test_non_ascii_labels_survive_the_gateway(gateway):
    """UTF-8 criteria, state and answer, through the gateway and back."""
    labels = {"請求書": None, "技術的な問題": None, "営業": None}
    answer = json.dumps(
        {"choice": "技術的な問題", "probabilities": {"請求書": 0.05, "技術的な問題": 0.9, "営業": 0.05}}
    )
    gateway.backend.chat = lambda _: (200, chat_body(content=answer))
    try:
        client = SystemOneClient(
            openai_client(gateway), model="stub-route", api="chat_completions", method="structured"
        )

        response = client.system_one(state="アプリがクラッシュします", questions={"q": Choice(criteria=labels)})

        assert response.answers["q"].choice == "技術的な問題"
        sent = json.dumps(gateway.backend.bodies("/chat/completions")[-1], ensure_ascii=False)
        assert "アプリがクラッシュします" in sent
    finally:
        gateway.backend.chat = answer_script


def test_the_deployments_client_does_not_match_the_standalone_gateway(gateway):
    """``MlflowDeploymentClient`` posts to ``/v1/endpoints/...``; this gateway serves ``/endpoints/...``.

    Pinned because the mismatch is silent until the call is made: the client and the standalone
    gateway in 3.16.1 do not agree on the path.
    """
    import mlflow.deployments

    client = mlflow.deployments.get_deploy_client(gateway.base_url)
    with pytest.raises(Exception) as caught:
        client.predict(endpoint="stub-route", inputs={"messages": [{"role": "user", "content": "hi"}]})
    assert "404" in str(caught.value)


def test_an_unknown_route_name_is_refused_without_reaching_the_backend(gateway):
    """The endpoint is chosen by the request's `model`, so a wrong name is a gateway error, not a
    silent answer from whichever backend happens to be configured."""
    before = len(gateway.backend.bodies("/chat/completions"))
    client = SystemOneClient(
        openai_client(gateway),
        model="not-a-route",
        api="chat_completions",
        method="structured",
        retry=RetryPolicy(n_retries=0),
    )

    with pytest.raises(ProviderError):
        client.system_one(state="s", questions={"q": Choice(criteria=CRITERIA)})

    assert len(gateway.backend.bodies("/chat/completions")) == before


def test_the_legacy_endpoints_route_answers_chat(gateway):
    """The route the standalone gateway does serve, used directly."""
    request = urllib.request.Request(
        f"{gateway.root}/endpoints/stub-route/invocations",
        data=json.dumps({"messages": [{"role": "user", "content": "I was charged twice"}]}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        body = json.loads(response.read())
    assert response.status == 200
    assert body["choices"][0]["message"]["content"] in {"A", STRUCTURED_ANSWER}


# -- genai evaluation -----------------------------------------------------------------------


def label_script(body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    """Answer with the label the state calls for, in whatever shape jevper asks for."""
    text = json.dumps(body.get("messages") or body.get("input") or "")
    label, letter = ("technical", "B") if "crashes" in text else ("billing", "A")
    if body.get("response_format", {}).get("type") == "json_schema":
        probabilities = {name: (0.9 if name == label else 0.05) for name in CRITERIA}
        return 200, chat_body(content=json.dumps({"choice": label, "probabilities": probabilities}))
    # The readout reads the distribution, so the sampled token has to be the most likely one.
    others = [(name, -2.47) for name, _ in CHOICE_LOGS if name != letter]
    return 200, chat_body(
        content=letter, logprobs=[(letter, -0.12)], alternatives=[(letter, -0.12), *others]
    )


def judge_or_jevper(jevper_answer: Any, judge_body: dict[str, Any]) -> Any:
    """One backend, two callers: jevper's prompt and a judge's prompt are told apart by their text."""

    def scripted(body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        text = json.dumps(body.get("messages") or body.get("input") or "")
        if "classification engine" in text:
            return label_script(body)
        return 200, chat_body(content=json.dumps(judge_body))

    return scripted


def file_store(monkeypatch: Any, tmp_path: Path, name: str) -> str:
    """A file tracking store, opted in explicitly.

    MLflow 3.16.1 refuses the filesystem backend unless ``MLFLOW_ALLOW_FILE_STORE=true``, and on a SQLite
    store an evaluation that logs an LLM-judge metric fails inside MLflow with
    ``logged_model_metrics`` foreign-key error. This is MLflow's, not jevper's.
    """
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    mlflow.set_tracking_uri(f"file://{tmp_path / 'file-store'}")
    return mlflow.set_experiment(name).experiment_id


def _evaluate_dataset() -> list[dict[str, Any]]:
    return [
        {"inputs": {"text": "I was charged twice"}, "expectations": {"label": "billing"}},
        {"inputs": {"text": "the app crashes on login"}, "expectations": {"label": "technical"}},
    ]


def _jevper_predict_fn(stub: Any) -> Any:
    client = SystemOneClient(openai_client(stub), model="stub", api="chat_completions", method="logprobs")

    @mlflow.trace(name="jevper.system_one")
    def predict(text: str) -> str:
        response = client.system_one(state=text, questions={"intent": Choice(criteria=CRITERIA)})
        return response.answers["intent"].choice or ""

    return predict


def test_evaluate_scores_a_jevper_predict_fn(stub_server, tracking):
    from mlflow.genai.scorers import scorer

    stub = stub_server(chat=label_script)

    @scorer
    def label_matches(outputs: str, expectations: dict[str, Any]) -> bool:
        return outputs == expectations["label"]

    result = mlflow.genai.evaluate(
        data=_evaluate_dataset(), scorers=[label_matches], predict_fn=_jevper_predict_fn(stub)
    )

    assert result.metrics["label_matches/mean"] == 1.0
    assert len(result.result_df) == 2
    # Every row was answered by jevper, and its call is the row's trace.
    assert set(result.result_df["response"]) == {"billing", "technical"}


def test_evaluate_aborts_when_the_predict_fn_raises(stub_server, tracking):
    """A prose answer where a label is required fails the row, and MLflow fails the run."""
    from mlflow.exceptions import MlflowException
    from mlflow.genai.scorers import scorer

    stub = stub_server(chat=lambda _: (200, chat_body(content="Probably billing.")))

    @scorer
    def always_true(outputs: str) -> bool:
        return True

    with pytest.raises(MlflowException, match="Failed to run the prediction function"):
        mlflow.genai.evaluate(
            data=_evaluate_dataset()[:1], scorers=[always_true], predict_fn=_jevper_predict_fn(stub)
        )


def test_a_total_predict_fn_keeps_the_evaluation_running(stub_server, tracking):
    """The pattern that survives a model which will not answer in the required shape."""
    from mlflow.genai.scorers import scorer

    stub = stub_server(chat=lambda _: (200, chat_body(content="Probably billing.")))
    client = SystemOneClient(
        openai_client(stub), model="stub", api="chat_completions", method="structured"
    )

    @mlflow.trace(name="jevper.system_one")
    def predict(text: str) -> str:
        try:
            response = client.system_one(state=text, questions={"intent": Choice(criteria=CRITERIA)})
        except MalformedAnswerError:
            return ""
        return response.answers["intent"].choice or ""

    @scorer
    def answered(outputs: str) -> bool:
        return outputs != ""

    result = mlflow.genai.evaluate(data=_evaluate_dataset()[:1], scorers=[answered], predict_fn=predict)

    assert result.metrics["answered/mean"] == 0.0
    assert len(result.result_df) == 1


def test_evaluate_with_a_failing_scorer(stub_server, tracking):
    from mlflow.genai.scorers import scorer

    stub = stub_server(chat=answer_script)

    @scorer
    def explodes(outputs: str) -> bool:
        raise RuntimeError("scorer boom")

    result = mlflow.genai.evaluate(
        data=_evaluate_dataset()[:1], scorers=[explodes], predict_fn=_jevper_predict_fn(stub)
    )

    assert len(result.result_df) == 1


def test_a_gateway_judge_needs_the_server_hosted_gateway(gateway, tracking, monkeypatch):
    """``gateway:/<route>`` resolves to the *server-hosted* gateway's passthrough path.

    ``MLFLOW_GATEWAY_URI`` (or an HTTP tracking URI) plus ``gateway/mlflow/v1/`` is what the judge
    adapter posts to; the standalone gateway serves ``/v1/chat/completions`` instead, so the judge's
    call misses and the run records an assessment error rather than a score. ``Correctness`` also
    needs ``expectations.expected_response`` (or ``expected_facts``) before it will run at all.
    """
    from mlflow.genai.scorers import Correctness

    monkeypatch.setenv("MLFLOW_GATEWAY_URI", gateway.root)

    gateway.backend.chat = judge_or_jevper(None, {"result": "yes", "rationale": "matches"})
    try:
        result = mlflow.genai.evaluate(
            data=[{"inputs": {"text": "I was charged twice"}, "expectations": {"expected_response": "billing"}}],
            scorers=[Correctness(model="gateway:/stub-route")],
            predict_fn=_jevper_predict_fn(gateway.backend),
        )
        assessments = result.result_df["assessments"].iloc[0]
        assert assessments, "the judge recorded no assessment at all"
        detail = [{"name": a.get("assessment_name"), "feedback": a.get("feedback")} for a in assessments]
        assert not [key for key in result.metrics if key.startswith("correctness")], detail
        assert assessments, "the judge recorded no assessment at all"
        # A failed judge leaves an assessment with no successful feedback behind it.
        assert [
            a for a in assessments if a.get("feedback") is None or (a["feedback"] or {}).get("error")
        ], detail
    finally:
        gateway.backend.chat = answer_script


def test_the_deprecated_openai_flavour_points_at_a_local_server(stub_server, tracking, tmp_path, monkeypatch):
    """``mlflow.openai`` is deprecated since 3.8.0 but still works, and reads the env at call time."""
    pytest.importorskip("openai")
    import mlflow.openai

    stub = stub_server(chat=lambda _: (200, chat_body(content="billing")))
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setenv("OPENAI_API_BASE", f"{stub.base_url}/v1")
    path = tmp_path / "openai-flavour"

    mlflow.openai.save_model(
        model="stub",
        task="chat.completions",
        path=str(path),
        messages=[{"role": "user", "content": "{text}"}],
    )

    loaded = mlflow.pyfunc.load_model(str(path))
    assert loaded.predict({"text": "I was charged twice"}) == ["billing"]


def test_a_jevper_backed_langchain_model_logs_and_predicts(stub_server, tracking):
    """The LangChain flavour hosts any ``BaseChatModel`` — including one that answers with jevper."""
    pytest.importorskip("langchain_core")
    from mlflow.langchain import log_model

    stub = stub_server(chat=answer_script)
    with mlflow.start_run():
        info = log_model(
            lc_model=str(FIXTURES / "jevper_langchain_model.py"),
            name="jevper-langchain",
            model_config={"base_url": stub.base_url, "model": "stub"},
            input_example=["I was charged twice"],
        )

    loaded = mlflow.pyfunc.load_model(info.model_uri)
    result = loaded.predict(["I was charged twice"])
    assert "billing" in json.dumps(result)


def test_evaluate_with_expected_facts(stub_server, tracking, tmp_path, monkeypatch):
    """The other expectation shape the docs mention: facts rather than a reference response."""
    from mlflow.genai.scorers import Correctness

    stub = stub_server(chat=label_script)
    judge_stub = stub_server(
        chat=lambda _: (200, chat_body(content=json.dumps({"result": "yes", "rationale": "supported"})))
    )
    file_store(monkeypatch, tmp_path, "expected-facts")
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setenv("OPENAI_API_BASE", f"{judge_stub.base_url}/v1")

    result = mlflow.genai.evaluate(
        data=[
            {
                "inputs": {"text": "I was charged twice"},
                "expectations": {"expected_facts": ["the customer was charged twice"]},
            }
        ],
        scorers=[Correctness(model="openai:/stub")],
        predict_fn=_jevper_predict_fn(stub),
    )

    correctness = [key for key in result.metrics if key.startswith("correctness")]
    assert correctness, f"no correctness metric in {sorted(result.metrics)}"
    assert judge_stub.bodies("/chat/completions"), "the judge was never called"


def test_evaluate_a_dataset_that_already_has_outputs(stub_server, tracking, tmp_path, monkeypatch):
    """No predict_fn at all: the answers are already in the dataset.

    On a SQLite store this form fails inside MLflow (``logged_model_metrics`` references a run that was
    never created), so it runs against a file store — the same shape, without the constraint.
    """
    from mlflow.genai.scorers import scorer

    file_store(monkeypatch, tmp_path, "static-outputs")

    @scorer
    def label_matches(outputs: str, expectations: dict[str, Any]) -> bool:
        return outputs == expectations["label"]

    result = mlflow.genai.evaluate(
        data=[
            {"inputs": {"text": "I was charged twice"}, "outputs": "billing", "expectations": {"label": "billing"}},
            {"inputs": {"text": "crashes"}, "outputs": "billing", "expectations": {"label": "technical"}},
        ],
        scorers=[label_matches],
    )

    assert result.metrics["label_matches/mean"] == 0.5


def test_a_judge_can_run_against_a_local_openai_compatible_server(stub_server, tracking, monkeypatch):
    """`openai:/<model>` plus OPENAI_API_BASE is how a judge reaches a local server."""
    from mlflow.genai.judges import is_correct

    stub = stub_server(
        chat=lambda _: (200, chat_body(content=json.dumps({"result": "yes", "rationale": "the same"})))
    )
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setenv("OPENAI_API_BASE", stub.base_url + "/v1")

    feedback = is_correct(request="What is 2+2?", response="4", expected_response="4", model="openai:/stub")

    assert feedback is not None
    assert str(getattr(feedback, "value", feedback)).lower() in {"yes", "true", "1"}
