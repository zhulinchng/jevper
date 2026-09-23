"""Sync and async client facades.

The per-question call sequence (analysis pass, answer pass, corrective retries) is expressed once as a
generator of ``CallSpec``; two small drivers execute it against the sync or async transport.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import math
import threading
import time
from collections.abc import Generator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

from . import methods
from .errors import (
    InvalidQuestionError,
    JevperError,
    LabelReadoutError,
    MalformedAnswerError,
    ProviderError,
)
from .labels import labels_for
from .normalize import (
    PROBABILITY_TOLERANCE,
    choice_confidence,
    probability_error,
    rescale,
    score_confidence,
)
from .prompts import (
    ANSWER_CUE,
    build_analysis_messages,
    build_messages,
    correction_message,
    example_answer_label,
    render_state_messages,
    structured_correction_message,
)
from .reasoning import (
    ReasoningConfig,
    ReasoningContentPart,
    ReasoningSummaryPart,
    reasoning_text,
    resolve_reasoning_mode,
)
from .transport import CallResult, CallSpec, Transport, make_transport, select_surface
from .types import (
    Answer,
    Api,
    ChoiceAnswer,
    Example,
    Method,
    NoulAnswer,
    Question,
    ScoreAnswer,
    SystemOneResponse,
    Usage,
    parse_question,
)

TRANSIENT_STATUS_CODES = frozenset({429, 500, 502, 503, 504, 529})
_TRANSIENT_NAME_MARKERS = ("Connection", "Timeout")
_TRANSIENT_TRANSPORT_CLASSES = frozenset({"TransportError", "TimeoutException"})
METHODS: tuple[Method, ...] = ("logprobs", "grammar", "structured", "discrete")
APIS: tuple[Api, ...] = ("auto", "chat_completions", "responses")
MAX_TOP_LOGPROBS = 20
TOKEN_FIELDS = ("input_tokens", "output_tokens", "reasoning_tokens")

Examples = Sequence[Example] | Mapping[str, Sequence[Example]]


class RetryPolicy(BaseModel):
    """Transient-failure retries: status codes 429/5xx and connection/timeout errors."""

    n_retries: int = 2
    base_delay: float = 0.5
    max_delay: float = 8.0


def _status_code(exc: BaseException) -> int | None:
    """The provider's HTTP status, when the exception carries one in a readable form."""
    status = getattr(exc, "status_code", None)
    if status is None or isinstance(status, bool):
        return None
    try:
        return int(status)
    except (TypeError, ValueError):
        return None


def _is_transient(exc: BaseException) -> bool:
    if _status_code(exc) in TRANSIENT_STATUS_CODES:
        return True
    names = {cls.__name__ for cls in type(exc).__mro__}
    # Transport failures of the httpx family (ConnectError, ReadError, RemoteProtocolError, ...) are
    # named after neither "Connection" nor "Timeout"; their base classes are the reliable marker.
    if names & _TRANSIENT_TRANSPORT_CLASSES:
        return True
    return any(marker in name for name in names for marker in _TRANSIENT_NAME_MARKERS)


def _dump_model(obj: Any) -> Any:
    dump = getattr(obj, "model_dump", None)
    if callable(dump):
        try:
            return dump(mode="json")
        except TypeError:  # pragma: no cover - non-pydantic objects with a model_dump()
            return dump()
    return obj


def _pick_examples(examples: Examples, question_id: str) -> Sequence[Example]:
    if isinstance(examples, Mapping):
        return examples.get(question_id, ())
    return examples


@dataclass
class _CallContext:
    """Everything one ``system_one`` call needs, resolved once."""

    transport: Transport
    model: str
    method: Method
    mode: str
    temperature: float | None
    examples: Examples
    answer_reasoning: ReasoningConfig | None
    analysis_reasoning: ReasoningConfig | None


@dataclass
class _CallLog:
    """Per-question provider-call accounting; owned by one worker, so no locking is needed."""

    attempts: list[dict[str, Any]] = field(default_factory=list)
    n_calls: int = 0
    n_retries: int = 0
    tokens: dict[str, int | None] = field(
        default_factory=lambda: {name: 0 for name in TOKEN_FIELDS}
    )

    def add_attempt(
        self,
        question_id: str,
        *,
        surface: str,
        request: Any,
        response: Any = None,
        error: str | None = None,
    ) -> dict[str, Any]:
        record = {
            "question_id": question_id,
            "surface": surface,
            "request": request,
            "response": response,
            "error": error,
            "readout": None,
        }
        self.attempts.append(record)
        return record

    def add_result(self, result: CallResult) -> None:
        self.n_calls += 1
        for name in TOKEN_FIELDS:
            value = getattr(result, name)
            current = self.tokens[name]
            self.tokens[name] = None if value is None or current is None else current + int(value)


@dataclass
class _QuestionOutcome:
    answer: Answer
    reasoning: tuple[ReasoningContentPart, ...]
    tokens: dict[str, int | None]
    n_calls: int
    n_retries: int
    attempts: list[dict[str, Any]]
    retry_reasons: list[str]
    readout_debug: dict[str, Any]
    probability_error: float | None = None
    original_probabilities: dict[str, float] | None = None
    missing_labels: tuple[str, ...] = ()


class _BaseClient:
    def __init__(
        self,
        client: Any,
        *,
        model: str,
        method: Method = "logprobs",
        api: Api = "auto",
        reasoning: ReasoningConfig | None = None,
        examples: Examples = (),
        structured_outputs: bool = True,
        normalize_probabilities: bool = True,
        top_logprobs: int = MAX_TOP_LOGPROBS,
        max_concurrency: int = 8,
        n_retry_malformed: int = 1,
        retry: RetryPolicy | None = None,
        temperature: float | None = None,
        extra_body: Mapping[str, Any] | None = None,
        extra_headers: Mapping[str, str] | None = None,
    ) -> None:
        if method not in METHODS:
            raise JevperError(f"method must be one of {METHODS!r}, got {method!r}")
        if api not in APIS:
            raise JevperError(f"api must be one of {APIS!r}, got {api!r}")
        if not 0 <= top_logprobs <= MAX_TOP_LOGPROBS:
            raise JevperError(f"top_logprobs must be in [0, {MAX_TOP_LOGPROBS}], got {top_logprobs!r}")
        if max_concurrency < 1:
            raise JevperError(f"max_concurrency must be >= 1, got {max_concurrency!r}")
        if n_retry_malformed < 0:
            raise JevperError(f"n_retry_malformed must be >= 0, got {n_retry_malformed!r}")
        if reasoning is not None and not isinstance(reasoning, ReasoningConfig):
            raise JevperError(f"reasoning must be a ReasoningConfig, got {type(reasoning).__name__}")
        if retry is not None and not isinstance(retry, RetryPolicy):
            raise JevperError(f"retry must be a RetryPolicy, got {type(retry).__name__}")
        policy = retry or RetryPolicy()
        if policy.n_retries < 0 or policy.base_delay < 0 or policy.max_delay < 0:
            raise JevperError(
                f"retry needs n_retries, base_delay and max_delay >= 0, got {policy!r}"
            )
        self.client = client
        self.model = model
        self.method = method
        self.api = api
        self.reasoning = reasoning
        self.examples = examples
        self.structured_outputs = structured_outputs
        self.normalize_probabilities = normalize_probabilities
        self.top_logprobs = top_logprobs
        self.max_concurrency = max_concurrency
        self.n_retry_malformed = n_retry_malformed
        self.retry = policy
        self.temperature = temperature
        self.extra_body = extra_body
        self.extra_headers = extra_headers
        self._executor: concurrent.futures.ThreadPoolExecutor | None = None
        self._executor_lock = threading.Lock()

    # -- shared helpers ----------------------------------------------------------------

    def _prepare(
        self,
        state: Any,
        questions: Mapping[str, Question | Mapping[str, Any]],
        examples: Examples,
        model: str | None,
        method: Method | None,
        api: Api | None,
        reasoning: ReasoningConfig | None,
        temperature: float | None,
    ) -> tuple[_CallContext, dict[str, Question]]:
        if not questions:
            raise InvalidQuestionError("at least one question is required")
        render_state_messages(state)  # fail fast, before any provider call
        parsed = {question_id: parse_question(question_id, raw) for question_id, raw in questions.items()}
        effective_method = method or self.method
        if effective_method not in METHODS:
            raise JevperError(f"method must be one of {METHODS!r}, got {effective_method!r}")
        surface = select_surface(self.client, api or self.api, effective_method)
        if effective_method == "grammar":
            methods.require_grammar_surface(surface)
        for question_id, question in parsed.items():
            methods.require_label_readout(effective_method, question, question_id)
        effective_reasoning = reasoning if reasoning is not None else self.reasoning
        mode = resolve_reasoning_mode(effective_reasoning, surface)
        context = _CallContext(
            transport=make_transport(
                self.client,
                surface,
                structured_outputs=self.structured_outputs,
                extra_body=self.extra_body,
                extra_headers=self.extra_headers,
            ),
            model=model or self.model,
            method=effective_method,
            mode=mode,
            temperature=temperature if temperature is not None else self.temperature,
            examples=examples,
            answer_reasoning=effective_reasoning if mode == "native" else None,
            analysis_reasoning=(
                effective_reasoning if mode == "two_step" and surface == "responses" else None
            ),
        )
        return context, parsed

    def _resolve_examples(
        self, question: Question, labels: Sequence[str], question_id: str, examples: Examples
    ) -> tuple[Example, ...]:
        """Question-level examples win, then per-call, then constructor defaults."""
        resolved: tuple[Example, ...] = ()
        for candidate in (
            question.examples,
            _pick_examples(examples, question_id),
            _pick_examples(self.examples, question_id),
        ):
            if candidate:
                resolved = tuple(candidate)
                break
        for index, example in enumerate(resolved):
            try:
                example_answer_label(question, labels, example.answer, index)
            except InvalidQuestionError as exc:
                raise InvalidQuestionError(f"question {question_id!r}: {exc}") from exc
        return resolved

    def _question_steps(
        self,
        question_id: str,
        question: Question,
        state: Any,
        context: _CallContext,
        log: _CallLog,
    ) -> Generator[CallSpec, CallResult, _QuestionOutcome]:
        labels = labels_for(2 if question.type == "noul" else len(question.criteria))
        examples = self._resolve_examples(question, labels, question_id, context.examples)
        messages = build_messages(state, question, labels, examples=examples, method=context.method)
        native_reasoning: tuple[ReasoningContentPart, ...] = ()
        trace: str | None = None
        retry_reasons: list[str] = []
        if context.mode == "two_step":
            analysis = build_analysis_messages(state, question, labels, examples, method=context.method)
            result = yield CallSpec(messages=analysis, reasoning=context.analysis_reasoning)
            native_reasoning = result.reasoning
            # A model that reasons without writing output leaves `text` empty; its reasoning items are
            # then the analysis. An empty assistant turn is never sent: several OpenAI-compatible
            # servers reject empty content, and it would teach the answer pass nothing.
            trace = result.text.strip() or None
            trace_text = trace or reasoning_text(native_reasoning).strip() or None
            messages = messages + (
                [
                    {"role": "assistant", "content": trace_text},
                    {"role": "user", "content": ANSWER_CUE},
                ]
                if trace_text is not None
                else [{"role": "user", "content": ANSWER_CUE}]
            )

        correction: str | None = None
        for attempt in range(self.n_retry_malformed + 1):
            spec = methods.build_spec(
                context.method,
                messages + ([{"role": "user", "content": correction}] if correction else []),
                question,
                labels,
                top_logprobs=self.top_logprobs,
                temperature=context.temperature,
                reasoning=context.answer_reasoning,
            )
            result = yield spec
            try:
                readout = methods.readout(context.method, result, question, labels)
            except (LabelReadoutError, MalformedAnswerError) as exc:
                if attempt >= self.n_retry_malformed:
                    raise
                retry_reasons.append(str(exc))
                correction = (
                    structured_correction_message(str(exc))
                    if context.method in ("structured", "discrete")
                    else correction_message(str(exc), labels)
                )
                continue
            break
        native_reasoning = native_reasoning + result.reasoning
        return self._finalize(
            question, labels, readout, context, trace, native_reasoning, retry_reasons, log
        )

    def _finalize(
        self,
        question: Question,
        labels: Sequence[str],
        readout: methods.Readout,
        context: _CallContext,
        trace: str | None,
        native_reasoning: tuple[ReasoningContentPart, ...],
        retry_reasons: list[str],
        log: _CallLog,
    ) -> _QuestionOutcome:
        probabilities = readout.probabilities
        error: float | None = None
        original: dict[str, float] | None = None
        if context.method == "structured":
            error = probability_error(probabilities)
            if error > PROBABILITY_TOLERANCE:
                if self.normalize_probabilities:
                    original = {str(key): float(value) for key, value in probabilities.items()}
                    probabilities = rescale(probabilities)
            else:
                error = None
        if question.type == "choice":
            order = list(question.criteria)
            distribution = {key: probabilities[key] for key in order}
            answer: Answer = ChoiceAnswer(
                choice=max(order, key=distribution.__getitem__),
                probabilities=distribution,
                confidence=choice_confidence(list(distribution.values())),
            )
        elif question.type == "noul":
            answer = NoulAnswer(noul=probabilities[True])
        else:
            levels = list(range(len(question.criteria)))
            distribution = {level: probabilities[level] for level in levels}
            answer = ScoreAnswer(
                score=math.fsum(level * probability for level, probability in distribution.items()),
                legend={level: description for level, description in enumerate(question.criteria)},
                probabilities=distribution,
                confidence=score_confidence(list(distribution.values())),
            )
        reasoning: tuple[ReasoningContentPart, ...] = native_reasoning
        if trace is not None:
            reasoning = (
                ReasoningContentPart(summary=[ReasoningSummaryPart(text=trace)]),
            ) + native_reasoning
        readout_debug = {
            "source": readout.source,
            "probabilities": {str(key): float(value) for key, value in readout.probabilities.items()},
            "missing_labels": list(readout.missing_labels),
            "observed_text": readout.observed_text,
        }
        if log.attempts:
            log.attempts[-1]["readout"] = readout_debug
        return _QuestionOutcome(
            answer=answer,
            reasoning=reasoning,
            tokens=log.tokens,
            n_calls=log.n_calls,
            n_retries=log.n_retries,
            attempts=log.attempts,
            retry_reasons=retry_reasons,
            readout_debug=readout_debug,
            probability_error=error,
            original_probabilities=original,
            missing_labels=readout.missing_labels,
        )

    def _assemble(
        self,
        context: _CallContext,
        parsed: Mapping[str, Question],
        outcomes: Mapping[str, _QuestionOutcome],
        latency: float,
    ) -> SystemOneResponse:
        answers: dict[str, Answer] = {}
        reasoning: list[ReasoningContentPart] = []
        attempts: list[dict[str, Any]] = []
        retry_reasons: list[str] = []
        probability_errors: dict[str, float] = {}
        original_probabilities: dict[str, dict[str, float]] = {}
        labels_missing: dict[str, list[str]] = {}
        tokens: dict[str, int | None] = {name: 0 for name in TOKEN_FIELDS}
        n_calls = 0
        n_retries = 0
        for question_id in parsed:
            outcome = outcomes[question_id]
            answers[question_id] = outcome.answer
            reasoning.extend(outcome.reasoning)
            attempts.extend(outcome.attempts)
            retry_reasons.extend(outcome.retry_reasons)
            n_calls += outcome.n_calls
            n_retries += outcome.n_retries
            if outcome.probability_error is not None:
                probability_errors[question_id] = outcome.probability_error
            if outcome.original_probabilities is not None:
                original_probabilities[question_id] = outcome.original_probabilities
            if outcome.missing_labels:
                labels_missing[question_id] = list(outcome.missing_labels)
            for name in TOKEN_FIELDS:
                value = outcome.tokens[name]
                current = tokens[name]
                tokens[name] = None if value is None or current is None else current + value
        debug: dict[str, Any] = {
            "method": context.method,
            "api": context.transport.surface,
            "reasoning_mode": context.mode,
            "llm_attempts": attempts,
            "retry_reasons": retry_reasons,
            "probability_errors": probability_errors,
            "original_probabilities": original_probabilities,
            "labels_missing": labels_missing,
        }
        return SystemOneResponse(
            model=context.model,
            answers=answers,
            usage=Usage(
                **tokens, n_calls=n_calls, n_retries=n_retries, latency=latency
            ),
            reasoning=tuple(reasoning),
            debug=debug,
        )


class SystemOneClient(_BaseClient):
    """Blocking Jev-shaped client over any OpenAI-compatible client object.

    ``method`` selects how the decision is elicited; ``api`` selects the surface (auto-detected by
    default). ``examples`` provides few-shot demonstrations for every question in the call, and each
    question may carry its own ``examples``, which take precedence.

    ``temperature`` is not sent unless you pass it. ``temperature=0.0`` is recommended for
    ``structured`` and ``discrete``, where the answer is a single sampled JSON object; ``logprobs``
    reads the model's own distribution and needs no setting.
    """

    def _run(
        self, question_id: str, question: Question, state: Any, context: _CallContext
    ) -> _QuestionOutcome:
        log = _CallLog()
        steps = self._question_steps(question_id, question, state, context, log)
        try:
            spec = next(steps)
            while True:
                spec = steps.send(self._call(log, question_id, spec, context))
        except StopIteration as stop:
            return stop.value

    def _call(
        self, log: _CallLog, question_id: str, spec: CallSpec, context: _CallContext
    ) -> CallResult:
        attempt = 0
        while True:
            try:
                result = context.transport.call(spec, context.model)
            except Exception as exc:
                log.add_attempt(
                    question_id,
                    surface=context.transport.surface,
                    request=context.transport.kwargs(spec, context.model),
                    error=f"{type(exc).__name__}: {exc}",
                )
                if not _is_transient(exc) or attempt >= self.retry.n_retries:
                    raise ProviderError(f"{type(exc).__name__}: {exc}", attempts=log.attempts) from exc
                log.n_retries += 1
                time.sleep(min(self.retry.base_delay * 3**attempt, self.retry.max_delay))
                attempt += 1
                continue
            log.add_result(result)
            log.add_attempt(
                question_id,
                surface=result.surface,
                request=result.request,
                response=_dump_model(result.response),
            )
            return result

    def system_one(
        self,
        *,
        state: Any,
        questions: Mapping[str, Question | Mapping[str, Any]],
        examples: Examples = (),
        model: str | None = None,
        method: Method | None = None,
        api: Api | None = None,
        reasoning: ReasoningConfig | None = None,
        temperature: float | None = None,
    ) -> SystemOneResponse:
        start = time.perf_counter()
        context, parsed = self._prepare(state, questions, examples, model, method, api, reasoning, temperature)
        outcomes: dict[str, _QuestionOutcome] = {}
        failure: BaseException | None = None
        if len(parsed) == 1:
            question_id, question = next(iter(parsed.items()))
            try:
                outcomes[question_id] = self._run(question_id, question, state, context)
            except BaseException as exc:  # noqa: BLE001 - re-raised below, in question order
                failure = exc
        else:
            with self._executor_lock:
                if self._executor is None:
                    self._executor = concurrent.futures.ThreadPoolExecutor(
                        max_workers=self.max_concurrency
                    )
                executor = self._executor
            futures = {
                question_id: executor.submit(self._run, question_id, question, state, context)
                for question_id, question in parsed.items()
            }
            for question_id, future in futures.items():
                try:
                    outcomes[question_id] = future.result()
                except BaseException as exc:  # noqa: BLE001 - collected, then re-raised in order
                    if failure is None:
                        failure = exc
        if failure is not None:
            raise failure
        return self._assemble(context, parsed, outcomes, time.perf_counter() - start)

    def close(self) -> None:
        """Shut down the internal thread pool. The caller owns ``client``."""
        with self._executor_lock:
            executor, self._executor = self._executor, None
        if executor is not None:
            executor.shutdown(wait=True)

    def __enter__(self) -> SystemOneClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


class AsyncSystemOneClient(_BaseClient):
    """Asynchronous twin of ``SystemOneClient``; identical constructor and semantics."""

    async def _run(
        self, question_id: str, question: Question, state: Any, context: _CallContext
    ) -> _QuestionOutcome:
        log = _CallLog()
        steps = self._question_steps(question_id, question, state, context, log)
        try:
            spec = next(steps)
            while True:
                spec = steps.send(await self._call(log, question_id, spec, context))
        except StopIteration as stop:
            return stop.value

    async def _call(
        self, log: _CallLog, question_id: str, spec: CallSpec, context: _CallContext
    ) -> CallResult:
        attempt = 0
        while True:
            try:
                result = await context.transport.acall(spec, context.model)
            except Exception as exc:
                log.add_attempt(
                    question_id,
                    surface=context.transport.surface,
                    request=context.transport.kwargs(spec, context.model),
                    error=f"{type(exc).__name__}: {exc}",
                )
                if not _is_transient(exc) or attempt >= self.retry.n_retries:
                    raise ProviderError(f"{type(exc).__name__}: {exc}", attempts=log.attempts) from exc
                log.n_retries += 1
                await asyncio.sleep(min(self.retry.base_delay * 3**attempt, self.retry.max_delay))
                attempt += 1
                continue
            log.add_result(result)
            log.add_attempt(
                question_id,
                surface=result.surface,
                request=result.request,
                response=_dump_model(result.response),
            )
            return result

    async def system_one(
        self,
        *,
        state: Any,
        questions: Mapping[str, Question | Mapping[str, Any]],
        examples: Examples = (),
        model: str | None = None,
        method: Method | None = None,
        api: Api | None = None,
        reasoning: ReasoningConfig | None = None,
        temperature: float | None = None,
    ) -> SystemOneResponse:
        start = time.perf_counter()
        context, parsed = self._prepare(state, questions, examples, model, method, api, reasoning, temperature)
        semaphore = asyncio.Semaphore(self.max_concurrency)

        async def run(question_id: str, question: Question) -> _QuestionOutcome:
            async with semaphore:
                return await self._run(question_id, question, state, context)

        results = await asyncio.gather(
            *(run(question_id, question) for question_id, question in parsed.items()),
            return_exceptions=True,
        )
        outcomes: dict[str, _QuestionOutcome] = {}
        failure: BaseException | None = None
        for (question_id, _), result in zip(parsed.items(), results):
            if isinstance(result, BaseException):
                if failure is None:
                    failure = result
            else:
                outcomes[question_id] = result
        if failure is not None:
            raise failure
        return self._assemble(context, parsed, outcomes, time.perf_counter() - start)

    async def aclose(self) -> None:
        """The async client holds no resources of its own; the caller owns ``client``."""

    async def __aenter__(self) -> AsyncSystemOneClient:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()
