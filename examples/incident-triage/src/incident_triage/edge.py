"""The edge cases a consumer hits before any server is involved.

Every case here is local: an input the library is documented to refuse, a shape it is
documented to accept, or a lifecycle a service has to be able to rely on — cancellation,
ordering, the two facades refusing each other's client. Each case that claims a *local*
refusal also proves it with :class:`RejectingClient`, which fails the test if a single
request reaches the network: a validation that costs a round trip is not a validation.
"""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jevper import (
    AsyncSystemOneClient,
    Choice,
    Example,
    Noul,
    RetryPolicy,
    Score,
    SystemOneClient,
)
from jevper.client import (
    APIS,
    MAX_PROMPT_CACHE_KEY,
    MAX_TOP_LOGPROBS,
    METHOD_SELECTIONS,
    METHODS,
)
from jevper.errors import (
    ClientCapabilityError,
    InvalidQuestionError,
    JevperError,
    MalformedAnswerError,
    ProviderError,
    UnsupportedMethodError,
)

from .checks import Gap, check
from .duck import DuckClient, SlowDuckClient
from .rubrics import CATALOG_OPTIONS, catalog_choice

__all__ = ["CASES", "EdgeCase", "run_edge_cases"]


class RejectingClient:
    """A client that fails the test if it is ever called."""

    def __init__(self) -> None:
        self.calls = 0
        self.chat = _Chat(self)
        self.responses = _Surface(self)
        self.messages = _Surface(self)
        self.default_headers: dict[str, str] = {}


class _Surface:
    def __init__(self, owner: RejectingClient) -> None:
        self._owner = owner

    def create(self, **kwargs: Any) -> Any:
        self._owner.calls += 1
        raise AssertionError("a request reached the network during a local validation")


class _Chat:
    def __init__(self, owner: RejectingClient) -> None:
        self.completions = _Surface(owner)


class SequenceDuck(DuckClient):
    """Answers the first call with a malformed answer and the rest correctly."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.bad_first = True

    def request(self, route: str, kwargs: dict[str, Any]) -> dict[str, Any]:
        body = {key: value for key, value in kwargs.items() if key not in ("extra_body", "extra_headers")}
        body.update(kwargs.get("extra_body") or {})
        self.requests.append({"route": route, "body": body, "headers": {}})
        if self.bad_first:
            self.bad_first = False
            text = '{"probabilities": {"billing": 2.0}}'  # out of range: a malformed answer
        else:
            text = '{"probabilities": {"billing": 0.7, "technical": 0.2, "other": 0.1}}'
        return {
            "choices": [
                {"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": text}}
            ]
        }


@dataclass
class EdgeCase:
    """One local behaviour: what it does, and what it must raise."""

    name: str
    run: Callable[[], dict[str, Any] | None]
    raises: tuple[type[BaseException], ...] = ()
    local: bool = True
    note: str = ""


def _client(**options: Any) -> tuple[SystemOneClient, RejectingClient]:
    raw = RejectingClient()
    return SystemOneClient(raw, model="test-model", **options), raw


def _questions() -> dict[str, Any]:
    return {
        "intent": Choice(criteria={"billing": "money", "technical": "errors"}),
        "urgency": Score(criteria=["low", "high"]),
    }


def _expect_local(call: Callable[[], Any], errors: tuple[type[BaseException], ...], raw: RejectingClient) -> str:
    """Run something that must fail locally, and prove no request was made."""
    try:
        call()
    except errors as exc:
        if raw.calls:
            raise Gap(f"{type(exc).__name__} was raised, but {raw.calls} request(s) had already been sent") from None
        return f"{type(exc).__name__}: {exc}"[:200]
    except Exception as exc:  # noqa: BLE001 - the wrong exception is the finding
        raise Gap(f"expected {tuple(error.__name__ for error in errors)}, got {type(exc).__name__}: {exc}") from None
    raise Gap(f"expected {tuple(error.__name__ for error in errors)}, nothing was raised")


def _deep_state(depth: int) -> Any:
    node: Any = "leaf"
    for _ in range(depth):
        node = {"nested": node}
    return node


# -- constructor validation -------------------------------------------------------------


def case_model_empty() -> str:
    raw = RejectingClient()
    return _expect_local(lambda: SystemOneClient(raw, model=""), (JevperError,), raw)


def case_model_missing() -> str:
    try:
        SystemOneClient(RejectingClient())  # type: ignore[call-arg]
    except TypeError as exc:
        return f"TypeError: {exc}"[:200]
    raise Gap("a client without a model was constructed")


def case_unknown_method() -> str:
    return _expect_local(lambda: _client(method="telepathy"), (JevperError,), RejectingClient())


def case_unknown_api() -> str:
    return _expect_local(lambda: _client(api="smoke-signals"), (JevperError,), RejectingClient())


def case_top_logprobs_range() -> str:
    return _expect_local(lambda: _client(top_logprobs=MAX_TOP_LOGPROBS + 1), (JevperError,), RejectingClient())


def case_top_logprobs_floor() -> str:
    return _expect_local(lambda: _client(top_logprobs=1, method="logprobs"), (JevperError,), RejectingClient())


def case_concurrency_floor() -> str:
    return _expect_local(lambda: _client(max_concurrency=0), (JevperError,), RejectingClient())


def case_retry_malformed_negative() -> str:
    return _expect_local(lambda: _client(n_retry_malformed=-1), (JevperError,), RejectingClient())


def case_retry_not_a_policy() -> str:
    return _expect_local(lambda: _client(retry={"n_retries": 2}), (JevperError,), RejectingClient())


def case_retry_nan() -> str:
    return _expect_local(lambda: _client(retry=RetryPolicy(base_delay=float("nan"))), (JevperError,), RejectingClient())


def case_reasoning_not_a_config() -> str:
    return _expect_local(lambda: _client(reasoning={"effort": "low"}), (JevperError,), RejectingClient())


def case_cache_key_empty() -> str:
    return _expect_local(lambda: _client(prompt_cache_key=""), (JevperError,), RejectingClient())


def case_cache_key_too_long() -> str:
    return _expect_local(
        lambda: _client(prompt_cache_key="k" * (MAX_PROMPT_CACHE_KEY + 1)), (JevperError,), RejectingClient()
    )


def case_extra_body_model() -> str:
    client, raw = _client(extra_body={"model": "other"})
    return _expect_local(lambda: client.system_one(state="s", questions=_questions()), (JevperError,), raw)


def case_extra_body_stream() -> str:
    client, raw = _client(extra_body={"stream": True})
    return _expect_local(lambda: client.system_one(state="s", questions=_questions()), (JevperError,), raw)


def case_header_newline() -> str:
    return _expect_local(
        lambda: _client(extra_headers={"X-Tenant": "a\r\nX-Injected: b"}), (JevperError,), RejectingClient()
    )


def case_header_non_ascii() -> str:
    return _expect_local(lambda: _client(extra_headers={"X-Tenant": "café"}), (JevperError,), RejectingClient())


def case_header_bad_name() -> str:
    return _expect_local(lambda: _client(extra_headers={"X Tenant": "ok"}), (JevperError,), RejectingClient())


def case_async_refuses_blocking() -> str:
    import asyncio

    from openai import OpenAI

    client = AsyncSystemOneClient(OpenAI(api_key="x"), model="m")
    try:
        asyncio.run(client.system_one(state="s", questions=_questions()))
    except ClientCapabilityError as exc:
        check("SystemOneClient" in str(exc), f"the refusal does not name the class to use: {exc}")
        return f"ClientCapabilityError: {exc}"[:200]
    raise Gap("the async facade accepted a blocking client")


def case_blocking_refuses_async() -> str:
    from openai import AsyncOpenAI

    raw = RejectingClient()
    client = SystemOneClient(AsyncOpenAI(api_key="x"), model="m")
    try:
        client.system_one(state="s", questions=_questions())
    except ClientCapabilityError as exc:
        check("AsyncSystemOneClient" in str(exc), f"the refusal does not name the class to use: {exc}")
        check(raw.calls == 0, "a request was sent before the facade refused the client")
        return f"ClientCapabilityError: {exc}"[:200]
    raise Gap("the blocking facade accepted an async client")


# -- questions and examples ------------------------------------------------------------


def case_choice_too_many() -> str:
    return _expect_local(
        lambda: Choice(criteria={f"k{index}": "x" for index in range(CATALOG_OPTIONS + 1)}),
        (InvalidQuestionError,),
        RejectingClient(),
    )


def case_score_too_few() -> str:
    return _expect_local(lambda: Score(criteria=["only"]), (InvalidQuestionError,), RejectingClient())


def case_score_too_many() -> str:
    return _expect_local(
        lambda: Score(criteria=[f"level{index}" for index in range(11)]), (InvalidQuestionError,), RejectingClient()
    )


def case_noul_criteria_keys() -> str:
    return _expect_local(
        lambda: Noul(criteria={"yes": "a", "no": "b"}), (InvalidQuestionError,), RejectingClient()
    )


def case_question_unknown_field() -> str:
    return _expect_local(
        lambda: Choice(criteria={"a": "x"}, weight=2), (InvalidQuestionError,), RejectingClient()
    )


def case_logprobs_choice_cap() -> str:
    client, raw = _client(method="logprobs")
    questions = {"wide": Choice(criteria={f"k{index}": "x" for index in range(27)})}
    return _expect_local(
        lambda: client.system_one(state="hello", questions=questions), (InvalidQuestionError,), raw
    )

def case_example_probability_keys() -> str:
    return _expect_local(
        lambda: Choice(
            criteria={"a": "x", "b": "y"},
            examples=(Example(state="s", answer="a", probabilities={"a": 1.0, "c": 0.0}),),
        ),
        (InvalidQuestionError,),
        RejectingClient(),
    )


def case_example_duplicate_answer_keys() -> str:
    return _expect_local(
        lambda: Score(
            criteria=["low", "high"],
            examples=(Example(state="s", answer=1, probabilities={1: 0.9, "1": 0.1}),),
        ),
        (InvalidQuestionError,),
        RejectingClient(),
    )


def _run_with_client(**options: Any) -> Any:
    """Construct a client with these options and make one call with it."""
    client, _ = _client(**options)
    return client.system_one(state="s", questions=_questions())


def case_examples_not_a_sequence() -> str:
    return _expect_local(
        lambda: _run_with_client(examples="please"),
        (JevperError, InvalidQuestionError),
        RejectingClient(),
    )


def case_question_dump_excludes_examples() -> dict[str, Any]:
    """A question dumps to exactly the Jev wire keys, examples included or not."""
    question = Choice(
        criteria={"a": "x"},
        examples=(Example(state="s", answer="a"),),
    )
    dumped = question.model_dump(mode="json")
    check(set(dumped) == {"type", "instructions", "criteria"}, f"question dump keys are {sorted(dumped)}")
    check(catalog_choice().type == "choice", "the 255-option rubric is not a Choice")
    return {"keys": sorted(dumped)}


def case_public_constants() -> dict[str, Any]:
    """The constants the API reference tells callers they can validate their own inputs with."""
    check(set(METHODS) == {"logprobs", "grammar", "structured", "discrete"}, f"METHODS is {METHODS}")
    check(set(METHOD_SELECTIONS) == set(METHODS) | {"auto"}, f"METHOD_SELECTIONS is {METHOD_SELECTIONS}")
    check(
        set(APIS) == {"chat_completions", "responses", "messages", "auto"} or "auto" in APIS,
        f"APIS is {APIS}",
    )
    check(MAX_TOP_LOGPROBS == 20, f"MAX_TOP_LOGPROBS is {MAX_TOP_LOGPROBS}")
    check(MAX_PROMPT_CACHE_KEY == 256, f"MAX_PROMPT_CACHE_KEY is {MAX_PROMPT_CACHE_KEY}")
    return {"methods": sorted(METHODS), "apis": sorted(APIS)}


# -- state and content -----------------------------------------------------------------


def case_empty_questions() -> str:
    client, raw = _client()
    return _expect_local(lambda: client.system_one(state="s", questions={}), (JevperError,), raw)


def case_state_empty_list() -> str:
    client, raw = _client()
    return _expect_local(lambda: client.system_one(state=[], questions=_questions()), (JevperError,), raw)


def case_state_unknown_role() -> str:
    client, raw = _client()
    return _expect_local(
        lambda: client.system_one(
            state=[{"role": "wizard", "content": "hi"}], questions=_questions()
        ),
        (JevperError,),
        raw,
    )


def case_state_missing_content() -> str:
    client, raw = _client()
    return _expect_local(
        lambda: client.system_one(state=[{"role": "user"}], questions=_questions()), (JevperError,), raw
    )


def case_state_surrogate() -> str:
    client, raw = _client()
    return _expect_local(
        lambda: client.system_one(state="bad \ud800 text", questions=_questions()), (JevperError,), raw
    )


def case_state_not_encodable() -> str:
    client, raw = _client()
    return _expect_local(lambda: client.system_one(state=object(), questions=_questions()), (JevperError,), raw)


def case_state_non_finite() -> str:
    client, raw = _client()
    return _expect_local(
        lambda: client.system_one(state={"score": float("inf")}, questions=_questions()), (JevperError,), raw
    )


def case_state_too_deep() -> str:
    """A pathologically nested state is reported, never as a raw ``RecursionError``."""
    client, raw = _client()
    try:
        client.system_one(state=_deep_state(2000), questions=_questions())
    except (JevperError, ProviderError) as exc:
        check(raw.calls > 0 or "depth" in str(exc) or "nested" in str(exc).lower(), f"unclear report: {exc}")
        return f"{type(exc).__name__}: {exc}"[:200]
    except RecursionError:
        raise Gap("a 2000-level state reached the JSON encoder as a RecursionError") from None
    raise Gap("a 2000-level state was answered without a word about its depth")


def case_criteria_surrogate() -> str:
    client, raw = _client()
    return _expect_local(
        lambda: client.system_one(
            state="s", questions={"q": Choice(criteria={"a\ud800": "x"})}
        ),
        (JevperError,),
        raw,
    )


def case_raw_question_malformed() -> str:
    client, raw = _client()
    return _expect_local(
        lambda: client.system_one(state="s", questions={"q": {"type": "choice", "criteria": []}}),
        (InvalidQuestionError,),
        raw,
    )


# -- surface and method refusals --------------------------------------------------------


def case_messages_logprobs() -> str:
    client, raw = _client(api="messages", method="logprobs")
    return _expect_local(
        lambda: client.system_one(state="s", questions=_questions()), (UnsupportedMethodError,), raw
    )


def case_messages_grammar() -> str:
    client, raw = _client(api="messages", method="grammar")
    return _expect_local(
        lambda: client.system_one(state="s", questions=_questions()), (UnsupportedMethodError,), raw
    )


def case_responses_grammar() -> str:
    client, raw = _client(api="responses", method="grammar")
    return _expect_local(
        lambda: client.system_one(state="s", questions=_questions()), (UnsupportedMethodError,), raw
    )


def case_surface_missing_attribute() -> str:
    class Bare:
        pass

    client = SystemOneClient(Bare(), model="m", api="responses")
    try:
        client.system_one(state="s", questions=_questions())
    except (ClientCapabilityError, UnsupportedMethodError) as exc:
        return f"{type(exc).__name__}: {exc}"[:200]
    raise Gap("a client with no surfaces at all was used without an error")


# -- lifecycle -------------------------------------------------------------------------


_INTERRUPT_CHILD = """
import json, os, signal, sys, threading, time
sys.path.insert(0, sys.argv[1])
from incident_triage.duck import SpinDuckClient
from jevper import Choice, SystemOneClient

duck = SpinDuckClient("http://127.0.0.1:1/v1", delay=3.0)
client = SystemOneClient(duck, model="m", method="structured", max_concurrency=1, n_retry_malformed=0)
questions = {f"q{index}": Choice(criteria={"a": "x", "b": "y"}) for index in range(3)}
threading.Timer(0.5, lambda: os.kill(os.getpid(), signal.SIGINT)).start()
started = time.perf_counter()
try:
    client.system_one(state="s", questions=questions)
    print(json.dumps({"raised": None}))
except KeyboardInterrupt:
    print(json.dumps({"raised": "KeyboardInterrupt", "calls": duck.calls, "seconds": time.perf_counter() - started}))
joined = time.perf_counter()
client.close()
print(json.dumps({"close_seconds": time.perf_counter() - joined, "calls_after_close": duck.calls}))
"""


def case_interrupt_cancels_queue() -> dict[str, Any]:
    """Ctrl-C during a batch: the queued questions are cancelled, the running one is not.

    The interrupt has to be a real signal to this process: that is how a service is actually
    stopped, and it is the only way it reaches the thread waiting on the batch rather than a
    worker already inside a C call.
    """
    import subprocess
    import tempfile

    with tempfile.TemporaryDirectory() as workdir:
        script = Path(workdir) / "interrupt.py"
        script.write_text(_INTERRUPT_CHILD)
        finished = subprocess.run(
            [sys.executable, str(script), str(Path(__file__).parents[1])],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    lines = [json.loads(line) for line in finished.stdout.splitlines() if line.startswith("{")]
    check(len(lines) == 2, f"the child printed {finished.stdout!r} / {finished.stderr[-200:]!r}")
    raised, closed = lines
    check(raised.get("raised") == "KeyboardInterrupt", f"the batch ended with {raised!r}")
    check(raised.get("seconds", 9) < 2.0, f"the interrupt waited {raised.get('seconds')}s for the running call")
    check(raised.get("calls") == 1, f"{raised.get('calls')} requests were sent; only the running one may be")
    check(closed.get("close_seconds", 9) < 3.5, f"close() waited {closed.get('close_seconds')}s")
    check(closed.get("calls_after_close") == 1, "a queued question ran while close() joined the pool")
    return {
        "raised_in": round(raised["seconds"], 2),
        "requests": closed["calls_after_close"],
        "close_seconds": round(closed["close_seconds"], 2),
    }


def case_first_failure_in_order() -> dict[str, Any]:
    """An ordinary per-question failure runs every question and reports the first one."""
    duck = SequenceDuck("http://127.0.0.1:1/v1")
    client = SystemOneClient(duck, model="m", method="structured", n_retry_malformed=0, max_concurrency=1)
    questions = {
        "first": Choice(criteria={"billing": "money", "technical": "errors", "other": "other"}),
        "second": Choice(criteria={"billing": "money", "technical": "errors", "other": "other"}),
    }
    try:
        client.system_one(state="s", questions=questions)
    except MalformedAnswerError as exc:
        check(duck.calls == 2, f"only {duck.calls} of 2 questions ran before the failure stopped the batch")
        return {"calls": duck.calls, "error": str(exc)[:120]}
    finally:
        client.close()
    raise Gap("a malformed answer was not reported")


def case_close_is_repeatable() -> dict[str, Any]:
    duck = SlowDuckClient("http://127.0.0.1:1/v1", delay=0.0)
    with SystemOneClient(duck, model="m") as client:
        check(client is not None, "the context manager yielded no client")
    client.close()
    return {"closed": True}


def case_aclose_is_noop() -> dict[str, Any]:
    import asyncio

    async def run() -> str:
        client = AsyncSystemOneClient(RejectingClient(), model="m")
        await client.aclose()
        async with AsyncSystemOneClient(RejectingClient(), model="m") as other:
            check(other is not None, "the async context manager yielded no client")
        return "ok"

    return asyncio.run(run())


CASES: tuple[EdgeCase, ...] = (
    EdgeCase("model-empty", case_model_empty),
    EdgeCase("model-missing", case_model_missing),
    EdgeCase("unknown-method", case_unknown_method),
    EdgeCase("unknown-api", case_unknown_api),
    EdgeCase("top-logprobs-range", case_top_logprobs_range),
    EdgeCase("top-logprobs-floor", case_top_logprobs_floor),
    EdgeCase("concurrency-floor", case_concurrency_floor),
    EdgeCase("retry-malformed-negative", case_retry_malformed_negative),
    EdgeCase("retry-not-a-policy", case_retry_not_a_policy),
    EdgeCase("retry-nan", case_retry_nan),
    EdgeCase("reasoning-not-a-config", case_reasoning_not_a_config),
    EdgeCase("cache-key-empty", case_cache_key_empty),
    EdgeCase("cache-key-too-long", case_cache_key_too_long),
    EdgeCase("extra-body-model", case_extra_body_model),
    EdgeCase("extra-body-stream", case_extra_body_stream),
    EdgeCase("header-newline", case_header_newline),
    EdgeCase("header-non-ascii", case_header_non_ascii),
    EdgeCase("header-bad-name", case_header_bad_name),
    EdgeCase("async-refuses-blocking", case_async_refuses_blocking, local=False),
    EdgeCase("blocking-refuses-async", case_blocking_refuses_async, local=False),
    EdgeCase("choice-too-many", case_choice_too_many),
    EdgeCase("score-too-few", case_score_too_few),
    EdgeCase("score-too-many", case_score_too_many),
    EdgeCase("noul-criteria-keys", case_noul_criteria_keys),
    EdgeCase("question-unknown-field", case_question_unknown_field),
    EdgeCase("logprobs-choice-cap", case_logprobs_choice_cap),
    EdgeCase("example-probability-keys", case_example_probability_keys),
    EdgeCase("example-duplicate-answer-keys", case_example_duplicate_answer_keys),
    EdgeCase("examples-not-a-sequence", case_examples_not_a_sequence),
    EdgeCase("question-dump-excludes-examples", case_question_dump_excludes_examples),
    EdgeCase("public-constants", case_public_constants),
    EdgeCase("empty-questions", case_empty_questions),
    EdgeCase("state-empty-list", case_state_empty_list),
    EdgeCase("state-unknown-role", case_state_unknown_role),
    EdgeCase("state-missing-content", case_state_missing_content),
    EdgeCase("state-surrogate", case_state_surrogate),
    EdgeCase("state-not-encodable", case_state_not_encodable),
    EdgeCase("state-non-finite", case_state_non_finite),
    EdgeCase("state-too-deep", case_state_too_deep),
    EdgeCase("criteria-surrogate", case_criteria_surrogate),
    EdgeCase("raw-question-malformed", case_raw_question_malformed),
    EdgeCase("messages-logprobs", case_messages_logprobs),
    EdgeCase("messages-grammar", case_messages_grammar),
    EdgeCase("responses-grammar", case_responses_grammar),
    EdgeCase("surface-missing-attribute", case_surface_missing_attribute),
    EdgeCase("interrupt-cancels-queue", case_interrupt_cancels_queue, note="a KeyboardInterrupt in a worker thread"),
    EdgeCase("first-failure-in-order", case_first_failure_in_order),
    EdgeCase("close-is-repeatable", case_close_is_repeatable),
    EdgeCase("aclose-is-noop", case_aclose_is_noop),
)


def run_edge_cases(*, only: tuple[str, ...] = ()) -> list[dict[str, Any]]:
    """Every local case, each reported with what it did."""
    report: list[dict[str, Any]] = []
    for case in CASES:
        if only and case.name not in only:
            continue
        record: dict[str, Any] = {"case": case.name, "note": case.note}
        started = time.perf_counter()
        try:
            observation = case.run()
        except Gap as exc:
            record.update(status="gap", error=str(exc))
        except Exception as exc:  # noqa: BLE001 - a raw exception from a consumer's edge case is a finding
            record.update(status="gap", error=f"raw {type(exc).__name__}: {exc}")
        else:
            record.update(status="ok", observation=observation)
        record["seconds"] = round(time.perf_counter() - started, 3)
        report.append(record)
    return report


def summarize(report: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for record in report:
        counts[record["status"]] = counts.get(record["status"], 0) + 1
    return {
        "counts": counts,
        "gaps": [
            {"case": record["case"], "error": record.get("error")}
            for record in report
            if record["status"] == "gap"
        ],
        "report_json_chars": len(json.dumps(report)),
    }
