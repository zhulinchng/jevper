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
from dataclasses import dataclass, field, replace
from typing import Any

from pydantic import BaseModel

from . import methods
from .errors import (
    ClientCapabilityError,
    InvalidQuestionError,
    JevperError,
    LabelReadoutError,
    MalformedAnswerError,
    ProviderError,
    UnsupportedMethodError,
    _LogprobsUnavailable,
    _SurfaceUnavailable,
)
from .labels import MAX_LABEL_OPTIONS, labels_for
from .normalize import (
    PROBABILITY_TOLERANCE,
    choice_confidence,
    probability_error,
    rescale,
    score_confidence,
)
from .prompts import (
    ANALYSIS_SYSTEM_PROMPT,
    answer_cue,
    assemble,
    build_parts,
    correction_message,
    derived_cache_key,
    example_answer_label,
    render_state_messages,
    structured_correction_message,
    system_prompt,
    validate_example_probabilities,
)
from .reasoning import (
    ReasoningConfig,
    ReasoningContentPart,
    ReasoningSummaryPart,
    reasoning_text,
    resolve_reasoning_mode,
)
from .transport import (
    SURFACES,
    CallResult,
    CallSpec,
    Limits,
    Surface,
    Transport,
    _has_attribute,
    make_transport,
    select_surface,
)
from .types import (
    Answer,
    Api,
    ChoiceAnswer,
    Example,
    Method,
    MethodSelection,
    NoulAnswer,
    Question,
    ScoreAnswer,
    SystemOneResponse,
    Usage,
    parse_question,
)

TRANSIENT_STATUS_CODES = frozenset({408, 429, 500, 502, 503, 504, 529})
# 429 means "slow down" and 408 means the request timed out; only the server's own failures can be the
# logprob request's fault, so neither counts as capability evidence.
_SERVER_ERROR_STATUS_CODES = TRANSIENT_STATUS_CODES - {408, 429}
_LOGPROB_REJECTION_STATUS_CODES = frozenset({400, 403, 422})
# Text that says the provider refuses the *field itself*: the capability evidence ``auto`` remembers.
# "requires" and "must be set" are how a server states a structural condition rather than a bad value:
# llama.cpp's Responses shim answers ``top_logprobs requires logprobs to be set to true`` and vLLM
# ``when using `top_logprobs`, `logprobs` must be set to true`` — on a surface where there is no field
# to set, that is a statement about what the route can do, not about the number that was sent.
_UNSUPPORTED_MARKERS = (
    "not supported",
    "unsupported",
    "unknown name",
    "cannot find field",
    "does not support",
    "unrecognized",
    "not recognized",
    "not allowed",
    "must be set",
    "requires",
)
# Text that says only the *value* was wrong: the field exists, so this is not a capability verdict.
# Any "between" phrasing bounds a value — ollama answers ``top_logprobs must be between 0 and 20``.
# The "must be" phrasings are how a server that knows the field rejects the number in it: SGLang
# answers ``budget_tokens: must be at least 1024`` and a gateway with a narrower effort enum answers
# ``reasoning_effort must be one of low, medium, high``.
_VALUE_MARKERS = (
    "between",
    "out of range",
    "maximum",
    "max_logprobs",
    "exceeds",
    "must be at least",
    "must be less than",
    "must be greater than",
    "must be one of",
    "invalid value",
    "not a valid",
)
# Text that names a *field jevper added for capability* rather than a bad value: the server does not
# implement structured outputs, the reasoning parameters, or the Responses ``include`` list. None of the
# three is needed to answer a question — the prompt already asks for one JSON object — so the field is
# dropped and the call is re-asked, and the server's limit is remembered.
_SCHEMA_MARKERS = (
    "response_format",
    "json_schema",
    "text.format",
    "structured output",
    "structured_output",
)
_REASONING_MARKERS = ("reasoning_effort", "reasoning")
# The Responses surface carries logprobs in ``include``, so a provider that does not offer that
# includable refuses the *field* without ever naming logprobs: OpenRouter answers ``Invalid option:
# expected one of "file_search_call.results"|...`` for ``path: ["include", 0]``. This vocabulary is
# only read next to ``include``, where it is the refusal of the field rather than of a value.
_INCLUDE_REJECTION_MARKERS = ("invalid option", "invalid_value", "expected one of")
# Text that names the cache key rather than a bad value. A server that does not know the field may say
# so either way — vLLM's request models reject unknown fields outright — and the key is optional, so it
# is dropped and the call re-asked like the other capability fields.
_CACHE_KEY_MARKERS = ("prompt_cache_key", "prompt cache key", "cache key")
# Text that names the Messages API's thinking budget rather than a bad value. vLLM's protocol has no
# ``thinking`` field at all, so a request carrying one is refused there and re-asked without it.
_THINKING_MARKERS = ("thinking", "budget_tokens")
# A 404 that names the model *and* talks about a model is about the model: the other surface would
# answer the same way, so switching would only hide the real problem. Both halves are required —
# an unrelated message can easily contain a short model name, and a server that does not implement
# the Responses route answers 404 there with a message about the route.
_MODEL_404_MARKERS = ("model", "no such", "not exist")
_TRANSIENT_NAME_MARKERS = ("Connection", "Timeout")
_TRANSIENT_TRANSPORT_CLASSES = frozenset({"TransportError", "TimeoutException"})
METHODS: tuple[Method, ...] = ("logprobs", "grammar", "structured", "discrete")
METHOD_SELECTIONS: tuple[MethodSelection, ...] = ("auto", *METHODS)
APIS: tuple[Api, ...] = ("auto", "chat_completions", "responses", "messages")
AUTO_METHOD: Method = "logprobs"  # what method="auto" tries first
FALLBACK_METHOD: Method = "structured"  # what it answers with when logprobs are unavailable
# Readout-level absences auto needs before it treats a provider as unable to return logprobs. One
# response with unreadable logprobs is a bad minute; two are a pattern.
AUTO_ABSENCES_BEFORE_REMEMBERING = 2
MAX_TOP_LOGPROBS = 20
MAX_PROMPT_CACHE_KEY = 256
"""The provider cap jevper enforces on a cache key. OpenRouter documents 256 characters for the
``session_id`` that shares this routing role, and a key that is too long is a caller mistake worth
catching before a request is spent on it."""
TOKEN_FIELDS = ("input_tokens", "output_tokens", "reasoning_tokens", "cached_tokens")

Examples = Sequence[Example] | Mapping[str, Sequence[Example]]


class RetryPolicy(BaseModel):
    """Transient-failure retries: status codes 429/5xx and connection/timeout errors."""

    n_retries: int = 2
    base_delay: float = 0.5
    max_delay: float = 8.0


def _status_code(exc: BaseException) -> int | None:
    """The provider's HTTP status, when the exception carries one in a readable form."""
    status = getattr(exc, "status_code", None)
    if status is None:
        # httpx.HTTPStatusError keeps it on the response instead, and its MRO carries no transport
        # marker, so without this it would be neither retried nor classified.
        status = getattr(getattr(exc, "response", None), "status_code", None)
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


def _retry_delay(policy: RetryPolicy, attempt: int) -> float:
    """Exponential backoff for transient retries, capped by the policy."""
    if policy.base_delay >= policy.max_delay:
        return policy.max_delay
    # The cap is reached by multiplying, not by computing ``3**attempt``: that overflows a float past
    # attempt 646 (and ``base_delay * inf`` cannot be recovered from), while this stops the moment the
    # policy's own cap is reached — a handful of steps for any usable policy, and attempt steps at
    # worst when the delay never grows.
    delay = policy.base_delay
    for _ in range(attempt):
        delay *= 3.0
        if delay >= policy.max_delay:
            return policy.max_delay
    return delay


def _require_top_logprobs(method: Method, top_logprobs: int) -> None:
    """A label readout needs the sampled token and at least one alternative to compare it with."""
    if method in ("logprobs", "grammar") and top_logprobs < 2:
        raise JevperError(
            f"top_logprobs must be at least 2 for method={method!r}: a distribution needs the sampled "
            f"token and at least one alternative, got {top_logprobs!r}"
        )


def _require_prompt_cache_key(key: Any) -> None:
    """A cache key is an opaque routing label: any non-blank string within the provider's cap."""
    if not isinstance(key, str) or not key.strip():
        raise JevperError(f"prompt_cache_key must be a non-empty string, got {key!r}")
    if len(key) > MAX_PROMPT_CACHE_KEY:
        raise JevperError(
            f"prompt_cache_key must be at most {MAX_PROMPT_CACHE_KEY} characters, got {len(key)}"
        )


def _add_count(current: int | None, value: Any) -> int | None:
    """Add one provider's count to a running total; a count nobody can read counts as unreported.

    A token count arrives as whatever the provider put in its JSON. ``int`` already accepts a numeric
    string and truncates a float, which is as much tolerance as the shape deserves; anything else — a
    nested object, a word, a boolean — is a provider bug, and reporting the total as ``None`` says "not
    reported" rather than inventing a number. The raw body is still in ``debug["llm_attempts"]``.
    """
    if value is None or isinstance(value, bool) or current is None:
        return None
    try:
        return current + int(value)
    except (TypeError, ValueError, OverflowError):
        # ``int(float("inf"))`` raises OverflowError, which is the same kind of provider bug as a word
        # in a token count: "not reported" is the honest answer, and a raw OverflowError escaping the
        # call would not be.
        return None


def _error_evidence(exc: BaseException) -> str:
    """Everything the provider said about the failure: its message, and the param/code it named."""
    parts = [str(exc)]
    for attr in ("param", "code"):
        value = getattr(exc, attr, None)
        if isinstance(value, str) and value:
            parts.append(value)
    return " ".join(parts).lower()


def _route_missing(exc: BaseException, *, surface: Surface, model: str) -> bool:
    """The server has no route for this surface, rather than a problem with what was sent.

    An OpenAI-compatible server that does not implement a surface's route answers 404 for it, and the
    ``openai`` client object exposes ``responses.create`` either way. A 404 that names the model is
    about the model — the same error would come back from the other surface — so it is left alone.
    """
    if _status_code(exc) != 404:
        return False
    evidence = _error_evidence(exc)
    names_the_model = model.lower() in evidence
    return not (names_the_model and any(marker in evidence for marker in _MODEL_404_MARKERS))


def _include_refused(evidence: str) -> bool:
    """The provider refused the Responses surface's logprob carrier, the ``include`` entry.

    ``include`` is the only place a Responses request can ask for logprobs, and a provider that does
    not offer that includable rejects the path rather than the word: OpenRouter answers ``Invalid
    option: expected one of "file_search_call.results"|...`` for ``path: ["include", 0]``.
    """
    return "include" in evidence and any(
        marker in evidence for marker in _INCLUDE_REJECTION_MARKERS
    )


def _logprobs_rejected(exc: BaseException, spec: CallSpec | None = None) -> bool:
    """The provider refused the request because of the logprob fields it carried.

    Both shapes seen in the wild name the field: Gemini's OpenAI-compatibility layer answers
    ``Unknown name "logprobs": Cannot find field.`` and a reasoning model behind an OpenAI-shaped
    gateway answers ``logprobs are not supported with reasoning models.`` A Responses request asks for
    logprobs through ``include`` instead, so the field it can be refused for is that one — but only
    when ``spec`` says this request asked for logprobs at all: the same ``include`` list also carries
    the reasoning include, and a refusal of that path is about reasoning, not about logprobs.
    """
    if _status_code(exc) not in _LOGPROB_REJECTION_STATUS_CODES:
        return False
    evidence = _error_evidence(exc)
    if "logprob" in evidence:
        return True
    return bool(spec is not None and spec.logprobs and _include_refused(evidence))


def _logprobs_unsupported(exc: BaseException, spec: CallSpec | None = None) -> bool:
    """The rejection reads as a missing capability rather than a bad value, so it is worth remembering.

    ``logprobs are not supported with reasoning models.`` and ``Unknown name "logprobs": Cannot find
    field.`` both refuse the field. ``Invalid 'top_logprobs': integer must be between 0 and 5, but got
    20.`` — a server whose cap is lower than the default — refuses the value, so the question is
    answered with a logprob-free method without writing off logprobs for the rest of the client's life.
    """
    if not _logprobs_rejected(exc, spec):
        return False
    evidence = _error_evidence(exc)
    if any(marker in evidence for marker in _VALUE_MARKERS):
        return False
    if _include_refused(evidence):
        return True
    return any(marker in evidence for marker in _UNSUPPORTED_MARKERS)


def _value_refused(evidence: str) -> bool:
    """The provider complained about the value it was sent, not about the field's existence.

    The distinction decides whether a capability field may be dropped. ``budget_tokens: must be at
    least 1024`` names a field the server knows and a number it will not take, so re-asking without the
    field would answer the question with the caller's reasoning quietly switched off — and remember that
    as this server's limit for the rest of the client's life. ``reasoning_effort: Extra inputs are not
    permitted`` is the other case: the field itself is what is missing.

    A refusal of the schema is deliberately not covered: dropping to ``json_object`` does not lose the
    schema, which travels in the prompt from then on, so the ladder is worth a rung even when the
    complaint is about the schema's contents.
    """
    return any(marker in evidence for marker in _VALUE_MARKERS)


def _dump_model(obj: Any) -> Any:
    """The provider object as plain data for ``debug``, without the provider's typing noise.

    A server can put a value the SDK's model does not expect in a field it still has to serialize —
    SGLang returns a list for ``metadata``, which the SDK types as a string — and pydantic answers
    every ``model_dump`` with a ``PydanticSerializationUnexpectedValue`` warning. The value survives
    the dump either way, and the dump exists for ``debug``, so the warning is noise the caller never
    asked for; ``warnings=False`` keeps it out of their logs without touching global warning state.
    """
    dump = getattr(obj, "model_dump", None)
    if callable(dump):
        try:
            return dump(mode="json", warnings=False)
        except TypeError:  # pragma: no cover - a model_dump() without those keywords
            try:
                return dump(mode="json")
            except TypeError:
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
    reasoning: ReasoningConfig | None
    """The effective config, kept so a surface switch can re-derive ``mode`` for the new surface."""
    temperature: float | None
    examples: Examples
    state_messages: tuple[dict[str, str], ...]
    answer_reasoning: ReasoningConfig | None
    analysis_reasoning: ReasoningConfig | None
    auto: bool = False
    """``method="auto"``: choose per question, and fall back when logprobs are unavailable. ``method``
    then holds the resolved default rather than what the caller asked for."""
    api_auto: bool = False
    """``api="auto"``: the surface was chosen here, so this call may re-choose it. An explicit
    ``api="responses"`` is a decision: its 404 belongs to the caller, not to a fallback."""
    prompt_cache_key: str | None = None
    """The caller's cache key, or ``None`` to derive one per question from the prefix it can reuse."""


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
            self.tokens[name] = _add_count(self.tokens[name], getattr(result, name))


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
    method: Method
    probability_error: float | None = None
    original_probabilities: dict[str, float] | None = None
    missing_labels: tuple[str, ...] = ()


class _BaseClient:
    def __init__(
        self,
        client: Any,
        *,
        model: str,
        method: MethodSelection = "auto",
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
        prompt_cache_key: str | None = None,
    ) -> None:
        if method not in METHOD_SELECTIONS:
            raise JevperError(f"method must be one of {METHOD_SELECTIONS!r}, got {method!r}")
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
        if extra_body is not None and not isinstance(extra_body, Mapping):
            raise JevperError(f"extra_body must be a mapping, got {type(extra_body).__name__}")
        if extra_headers is not None and not isinstance(extra_headers, Mapping):
            raise JevperError(f"extra_headers must be a mapping, got {type(extra_headers).__name__}")
        if prompt_cache_key is not None:
            _require_prompt_cache_key(prompt_cache_key)
        if method in ("logprobs", "grammar"):
            _require_top_logprobs(method, top_logprobs)
        policy = retry or RetryPolicy()
        if (
            policy.n_retries < 0
            or policy.base_delay < 0
            or policy.max_delay < 0
            or not (math.isfinite(policy.base_delay) and math.isfinite(policy.max_delay))
        ):
            raise JevperError(
                f"retry needs n_retries >= 0 and finite base_delay/max_delay >= 0, got {policy!r}"
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
        self.prompt_cache_key = prompt_cache_key
        self._executor: concurrent.futures.ThreadPoolExecutor | None = None
        self._executor_lock = threading.Lock()
        # What method="auto" has learned about this provider, per (model, surface).
        self._auto_methods: dict[tuple[str, Surface], Method] = {}
        self._auto_misses: dict[tuple[str, Surface], int] = {}
        # Surfaces this server has answered 404 for, learned by trying them once each.
        self._missing_surfaces: set[Surface] = set()
        # Request fields this server has refused, per surface: structured output, reasoning, include.
        self._limits: dict[Surface, Limits] = {}
        self._auto_lock = threading.Lock()

    # -- shared helpers ----------------------------------------------------------------

    def _record_failure(
        self,
        log: _CallLog,
        question_id: str,
        spec: CallSpec,
        context: _CallContext,
        transport: Transport,
        exc: Exception,
    ) -> bool:
        """Record a failed attempt and report whether it may be retried.

        The record names the transport that made the call — not whatever the shared context holds by the
        time the failure is handled — so a delayed failure still shows the surface and the request that
        produced it.
        """
        log.add_attempt(
            question_id,
            surface=transport.surface,
            request=transport.kwargs(spec, context.model),
            error=f"{type(exc).__name__}: {exc}",
        )
        return _is_transient(exc)

    # -- method="auto" -----------------------------------------------------------------

    def _auto_method(self, model: str, surface: Surface) -> Method:
        """The method ``auto`` uses here: what this client learned, or the preferred method.

        The Messages API carries no logprobs at all, so a label readout has no first attempt to make
        there: ``auto`` starts at the JSON answer instead of spending a request to discover this.
        """
        if surface == "messages":
            return FALLBACK_METHOD
        with self._auto_lock:
            return self._auto_methods.get((model, surface), AUTO_METHOD)

    def _remember_logprobs_unavailable(self, model: str, surface: Surface) -> None:
        """Remember that this provider cannot return logprobs, for the rest of this client's life."""
        with self._auto_lock:
            self._auto_methods[(model, surface)] = FALLBACK_METHOD

    def _note_absent_logprobs(self, model: str, surface: Surface) -> int:
        """Count one response whose logprobs could not be read, for this (model, surface)."""
        with self._auto_lock:
            misses = self._auto_misses.get((model, surface), 0) + 1
            self._auto_misses[(model, surface)] = misses
            return misses

    def _note_logprobs_present(self, model: str, surface: Surface) -> None:
        """A readable distribution retires the absences counted so far, and any verdict against it."""
        with self._auto_lock:
            self._auto_misses.pop((model, surface), None)
            # A distribution that arrived here proves this surface can carry one, so a verdict that
            # moved the readout away from it is stale.
            if self._auto_methods.get((model, surface)) == FALLBACK_METHOD:
                self._auto_methods.pop((model, surface), None)

    def _surface_missing(self, surface: Surface) -> bool:
        """Whether this server has already answered 404 for a surface's route."""
        with self._auto_lock:
            return surface in self._missing_surfaces

    def _remember_surface_missing(self, surface: Surface) -> None:
        """Remember that this server has no route for a surface, for the rest of this client's life."""
        with self._auto_lock:
            self._missing_surfaces.add(surface)

    def _logprobs_absent_here(self, model: str, surface: Surface) -> bool:
        """Whether the label readout is already known to have no future on this surface."""
        with self._auto_lock:
            return self._auto_methods.get((model, surface), AUTO_METHOD) == FALLBACK_METHOD

    def _limits_for(self, surface: Surface) -> Limits:
        """What this server has been observed to accept on this surface."""
        with self._auto_lock:
            return self._limits.get(surface, Limits())

    def _remember_limits(self, surface: Surface, limits: Limits) -> None:
        """Remember a server's limit for the rest of this client's life, like any other verdict."""
        with self._auto_lock:
            self._limits[surface] = limits

    def _downgrade(self, exc: Exception, spec: CallSpec, transport: Transport) -> Limits | None:
        """The next request shape to try when a server refuses a field jevper added for capability.

        A server that does not implement structured outputs, the reasoning parameters or the Responses
        ``include`` list answers 400 naming the field it refuses. None of the three is required to
        answer the question, so the field is dropped and the same call is re-asked — one step down the
        ladder at a time, remembered afterwards, so the discovery is paid once. The ladder is bounded,
        so a server that refuses everything still ends in a ``ProviderError``. A rejection that names
        the logprob fields belongs to the label-readout fallback, not here.

        The verdict is read against the transport that made the call, so a failure that arrives after
        another question moved the shared context still descends *its own* ladder rather than the one
        belonging to a surface it never touched.
        """
        if _status_code(exc) not in _LOGPROB_REJECTION_STATUS_CODES or _logprobs_rejected(exc, spec):
            return None
        evidence = _error_evidence(exc)
        # A complaint about the number is not a complaint about the field: dropping the field there
        # would answer the question with the caller's reasoning off, and remember that as the server's
        # limit. The provider's own error says which number it wanted, so it travels back instead.
        value_complaint = _value_refused(evidence)
        limits = transport.limits
        if spec.json_schema is not None and any(marker in evidence for marker in _SCHEMA_MARKERS):
            if limits.structured == "schema":
                # With ``structured_outputs=False`` the request already carried a plain
                # ``json_object``: the next rung is no format field at all, and re-sending the same
                # bytes would only spend a second call on the same refusal.
                if not transport.structured_outputs:
                    return replace(limits, structured="none")
                return replace(limits, structured="object")
            if limits.structured == "object":
                return replace(limits, structured="none")
        if (
            not value_complaint
            and spec.reasoning is not None
            and limits.reasoning
            and any(marker in evidence for marker in _REASONING_MARKERS)
        ):
            return replace(limits, reasoning=False)
        if spec.reasoning is not None and limits.include and "include" in evidence:
            return replace(limits, include=False)
        if (
            not value_complaint
            and spec.prompt_cache_key is not None
            and limits.cache_key
            and any(marker in evidence for marker in _CACHE_KEY_MARKERS)
        ):
            return replace(limits, cache_key=False)
        if (
            not value_complaint
            and spec.reasoning is not None
            and limits.thinking
            and any(marker in evidence for marker in _THINKING_MARKERS)
        ):
            return replace(limits, thinking=False)
        return None

    def _logprob_surface_alternative(
        self, surface: Surface, reasoning: ReasoningConfig | None
    ) -> Surface | None:
        """The other surface to try for a label readout, when the move is free of side effects.

        ``native`` reasoning exists only on the Responses surface, so a caller who asked for it — or
        whose ``mode="auto"`` resolved to it — keeps the surface that choice implies. Otherwise the
        move costs nothing but a request, and it is worth one: a server can implement a surface
        without carrying logprobs through it — ollama answers the Responses route with an empty
        logprob list, llama.cpp rejects the logprob fields there outright, and OpenAI's Responses
        logprobs hold the sampled token and no alternatives — while Chat Completions on the same
        server carries a full distribution.
        """
        other: Surface = "chat_completions" if surface == "responses" else "responses"
        if self._surface_missing(other) or resolve_reasoning_mode(reasoning, surface) == "native":
            return None
        return other if _has_attribute(self.client, f"{SURFACES[other][2]}.create") else None

    def _note_surface_absence(
        self, model: str, surface: Surface, exc: _LogprobsUnavailable
    ) -> None:
        """Count one absence for this surface, remembering the verdict once it is a pattern."""
        if exc.capability and (
            exc.evidence == "provider"
            or self._note_absent_logprobs(model, surface) >= AUTO_ABSENCES_BEFORE_REMEMBERING
        ):
            self._remember_logprobs_unavailable(model, surface)

    def _switch_surface(self, context: _CallContext, surface: Surface) -> None:
        """Rebuild the transport and the reasoning plan for another surface, in place.

        ``_assemble`` reports the surface from the context it was handed, and the driver reads the
        transport per call, so the switch has to be visible on the object they already hold.
        """
        moved = surface != context.transport.surface
        context.transport = make_transport(
            self.client,
            surface,
            structured_outputs=self.structured_outputs,
            extra_body=self.extra_body,
            extra_headers=self.extra_headers,
            limits=self._limits_for(surface),
        )
        if context.auto and moved:
            # The method verdict is keyed by surface, so a surface that just changed has its own:
            # coming back to one that is known to withhold logprobs must not ask for them again. Only a
            # real move reloads it: on a same-surface downgrade the reload would undo the fallback this
            # call just chose, because a first weak absence is deliberately not cached and the verdict
            # still says "logprobs" — the next question would pay for the same probe again.
            context.method = self._auto_method(context.model, surface)
        context.mode = resolve_reasoning_mode(context.reasoning, surface)
        context.answer_reasoning = context.reasoning if context.mode == "native" else None
        context.analysis_reasoning = (
            context.reasoning if context.mode == "two_step" and surface == "responses" else None
        )

    def _call_failure(
        self,
        exc: Exception,
        spec: CallSpec,
        context: _CallContext,
        transport: Transport,
        log: _CallLog,
    ) -> JevperError:
        """The error to give up on a call with: a logprob verdict under ``auto``, else the provider's.

        A rejection that reads as a missing capability is a fact about the provider, so it is
        remembered; one that only complains about the value it was sent is answered in JSON without
        being remembered. A server error that survived every retry is only a bad minute, so it is not.
        """
        if context.auto and spec.logprobs:
            failure = f"{type(exc).__name__}: {exc}"
            if _logprobs_rejected(exc, spec):
                capability = _logprobs_unsupported(exc, spec)
                note = (
                    ""
                    if capability
                    else " (not remembered: the rejection names a bad value, not a missing capability)"
                )
                return _LogprobsUnavailable(
                    f"the provider rejected the logprob request ({failure}){note}",
                    capability=capability,
                    surface=transport.surface,
                )
            if _status_code(exc) in _SERVER_ERROR_STATUS_CODES:
                return _LogprobsUnavailable(
                    f"the provider failed every attempt at the logprob request ({failure})",
                    capability=False,
                    # Not evidence about the surface: another surface would have failed too, so this
                    # must not move the label readout anywhere.
                    evidence="transient",
                    surface=transport.surface,
                )
        if isinstance(exc, ProviderError):
            # The transport already built the right error — an embedded provider failure — so keep
            # its message and status and hand it the attempt history instead of wrapping it again.
            exc.attempts = log.attempts
            return exc
        return ProviderError(
            f"{type(exc).__name__}: {exc}",
            attempts=log.attempts,
            status_code=_status_code(exc),
        )

    def _question_method(self, question: Question, context: _CallContext) -> Method:
        """The method one question is answered with."""
        if not context.auto:
            return context.method
        if question.type == "choice" and len(question.criteria) > MAX_LABEL_OPTIONS:
            # A label token cannot tell "AA" from "A", so a wide choice never uses a label readout.
            return FALLBACK_METHOD
        return context.method

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
        prompt_cache_key: str | None = None,
    ) -> tuple[_CallContext, dict[str, Question]]:
        if not questions:
            raise InvalidQuestionError("at least one question is required")
        if reasoning is not None and not isinstance(reasoning, ReasoningConfig):
            # Without this the mode lookup dereferences a string and the caller sees an AttributeError.
            raise JevperError(f"reasoning must be a ReasoningConfig, got {type(reasoning).__name__}")
        if prompt_cache_key is not None:
            _require_prompt_cache_key(prompt_cache_key)
        # Fail fast, before any provider call, and reuse the rendered turns for every question.
        state_messages = tuple(render_state_messages(state))
        parsed = {question_id: parse_question(question_id, raw) for question_id, raw in questions.items()}
        # Every few-shot example is checked here too. A question whose example is invalid would
        # otherwise only fail inside its own worker — after the questions ahead of it had already spent
        # provider calls on a call that was locally invalid from the start.
        for question_id, question in parsed.items():
            labels = labels_for(2 if question.type == "noul" else len(question.criteria))
            self._resolve_examples(question, labels, question_id, examples)
        requested = self.method if method is None else method
        if requested not in METHOD_SELECTIONS:
            raise JevperError(f"method must be one of {METHOD_SELECTIONS!r}, got {requested!r}")
        auto = requested == "auto"
        # ``is None`` rather than truthiness: an explicit empty string is a caller mistake the eager
        # validation below has to catch, not a request to fall back to the constructor's value.
        effective_api = self.api if api is None else api
        if effective_api not in APIS:
            raise JevperError(f"api must be one of {APIS!r}, got {effective_api!r}")
        api_auto = effective_api == "auto"
        effective_model = self.model if model is None else model
        effective_reasoning = reasoning if reasoning is not None else self.reasoning
        # The surface is picked for the method auto tries first; the fallback runs on either surface.
        surface = select_surface(self.client, effective_api, AUTO_METHOD if auto else requested)
        if api_auto and self._surface_missing(surface):
            # This server answered 404 for the route before: do not pay for the discovery again. The
            # flip is only worth making if the client can speak the other surface — a messages-only
            # client has no ``responses`` to move to, and moving anyway turned the provider's 404 into
            # an AttributeError for an attribute the caller never had. Staying put re-asks the surface
            # that is known to 404, and the in-call handler reports that 404, exactly as it did on the
            # call that learned it.
            other: Surface = "chat_completions" if surface == "responses" else "responses"
            if _has_attribute(self.client, f"{SURFACES[other][2]}.create"):
                surface = other
        elif api_auto and auto and self._logprobs_absent_here(effective_model, surface):
            alternative = self._logprob_surface_alternative(surface, effective_reasoning)
            if alternative is not None and not self._logprobs_absent_here(
                effective_model, alternative
            ):
                # The label readout has no future on this surface and the other one is unmarked:
                # start there instead of paying for the same discovery on every call.
                surface = alternative
        if requested == "grammar":
            methods.require_grammar_surface(surface)
        effective_method = self._auto_method(effective_model, surface) if auto else requested
        if surface == "messages" and effective_method in ("logprobs", "grammar"):
            # Checked before any request: no server that implements this API returns logprobs through
            # it, so an explicit label readout there could only ever fail — and would fail late.
            raise UnsupportedMethodError(
                "the Messages API returns no logprobs: use method='structured' or 'discrete', or an "
                "OpenAI-compatible client for a label readout"
            )
        if not auto:
            # auto never reaches the label-readout cap: a wide Choice is answered in JSON.
            for question_id, question in parsed.items():
                methods.require_label_readout(effective_method, question, question_id)
            _require_top_logprobs(effective_method, self.top_logprobs)
        mode = resolve_reasoning_mode(effective_reasoning, surface)
        context = _CallContext(
            transport=make_transport(
                self.client,
                surface,
                structured_outputs=self.structured_outputs,
                extra_body=self.extra_body,
                extra_headers=self.extra_headers,
                limits=self._limits_for(surface),
            ),
            model=effective_model,
            method=effective_method,
            mode=mode,
            reasoning=effective_reasoning,
            temperature=temperature if temperature is not None else self.temperature,
            examples=examples,
            state_messages=state_messages,
            prompt_cache_key=prompt_cache_key if prompt_cache_key is not None else self.prompt_cache_key,
            answer_reasoning=effective_reasoning if mode == "native" else None,
            analysis_reasoning=(
                effective_reasoning if mode == "two_step" and surface == "responses" else None
            ),
            auto=auto,
            api_auto=api_auto,
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
                if example.probabilities is not None:
                    # Checked whatever the method renders: a bad distribution must not wait for the
                    # one method that happens to show it.
                    validate_example_probabilities(question, example.probabilities, index)
            except InvalidQuestionError as exc:
                raise InvalidQuestionError(f"question {question_id!r}: {exc}") from exc
        return resolved

    def _question_steps(
        self,
        question_id: str,
        question: Question,
        context: _CallContext,
        log: _CallLog,
    ) -> Generator[CallSpec, CallResult, _QuestionOutcome]:
        """One question's calls, restarted once with a logprob-free method when the provider needs it."""
        retry_reasons: list[str] = []
        fell_back = False
        while True:
            try:
                return (
                    yield from self._answer_steps(question_id, question, context, log, retry_reasons)
                )
            except _LogprobsUnavailable as exc:
                if not context.auto or fell_back:
                    raise
                # The surface that produced the verdict, which is not necessarily the one the shared
                # context holds now: another question's worker may have moved it while this call was
                # in flight.
                surface = exc.surface or context.transport.surface
                if exc.capability and exc.evidence != "transient" and context.api_auto:
                    alternative = self._logprob_surface_alternative(surface, context.reasoning)
                    if alternative is not None and not self._logprobs_absent_here(
                        context.model, alternative
                    ):
                        # The verdict is about this surface — it answered without logprobs, or refused
                        # the fields outright, as llama.cpp's Responses shim does. Mark it, so later
                        # calls start where the distribution is, and answer this question there.
                        self._remember_logprobs_unavailable(context.model, surface)
                        retry_reasons.append(
                            f"{exc} — retrying the label readout on api={alternative!r}"
                        )
                        self._switch_surface(context, alternative)
                        continue
                fell_back = True
                retry_reasons.append(f"{exc} — answering with method={FALLBACK_METHOD!r}")
                self._note_surface_absence(context.model, surface, exc)
                # Questions that have not started yet take the fallback without paying for it.
                context.method = FALLBACK_METHOD
            except _SurfaceUnavailable as exc:
                if not context.api_auto:
                    raise
                self._remember_surface_missing(exc.surface)
                other: Surface = "chat_completions" if exc.surface == "responses" else "responses"
                if self._surface_missing(other) or not _has_attribute(
                    self.client, f"{SURFACES[other][2]}.create"
                ):
                    # Nowhere left to go: this 404 is the answer, and it is a provider failure like any
                    # other — a public error carrying the status and the attempt history, not the
                    # private verdict this fallback runs on.
                    raise ProviderError(str(exc), attempts=log.attempts, status_code=404) from exc
                retry_reasons.append(f"{exc} — answering on api={other!r}")
                self._switch_surface(context, other)

    def _answer_steps(
        self,
        question_id: str,
        question: Question,
        context: _CallContext,
        log: _CallLog,
        retry_reasons: list[str],
    ) -> Generator[CallSpec, CallResult, _QuestionOutcome]:
        method = self._question_method(question, context)
        labels = labels_for(2 if question.type == "noul" else len(question.criteria))
        examples = self._resolve_examples(question, labels, question_id, context.examples)
        parts = build_parts(context.state_messages, question, labels, examples, method=method)
        messages = assemble(parts, system=system_prompt(method))
        # One key per question, stable across the states it is asked about: a provider routes requests
        # that share a prefix by this key, and jevper's varying part is the state, never the rubric.
        cache_key = context.prompt_cache_key or derived_cache_key(context.model, parts)
        native_reasoning: tuple[ReasoningContentPart, ...] = ()
        trace: str | None = None
        if context.mode == "two_step":
            analysis = assemble(parts, system=ANALYSIS_SYSTEM_PROMPT)
            result = yield CallSpec(
                messages=analysis, reasoning=context.analysis_reasoning, prompt_cache_key=cache_key
            )
            native_reasoning = result.reasoning
            # A model that reasons without writing output leaves `text` empty; its reasoning items are
            # then the analysis. An empty assistant turn is never sent: several OpenAI-compatible
            # servers reject empty content, and it would teach the answer pass nothing.
            trace = result.text.strip() or None
            trace_text = trace or reasoning_text(native_reasoning).strip() or None
            cue = answer_cue(method)
            messages = messages + (
                [
                    {"role": "assistant", "content": trace_text},
                    {"role": "user", "content": cue},
                ]
                if trace_text is not None
                else [{"role": "user", "content": cue}]
            )

        correction: str | None = None
        for attempt in range(self.n_retry_malformed + 1):
            spec = methods.build_spec(
                method,
                messages + ([{"role": "user", "content": correction}] if correction else []),
                question,
                labels,
                top_logprobs=self.top_logprobs,
                temperature=context.temperature,
                reasoning=context.answer_reasoning,
                prompt_cache_key=cache_key,
            )
            result = yield spec
            try:
                readout = methods.readout(method, result, question, labels)
            except _LogprobsUnavailable:
                raise  # the provider cannot supply logprobs; correcting the model cannot help
            except (LabelReadoutError, MalformedAnswerError) as exc:
                if attempt >= self.n_retry_malformed:
                    raise
                retry_reasons.append(str(exc))
                correction = (
                    structured_correction_message(str(exc))
                    if method in ("structured", "discrete")
                    else correction_message(str(exc), labels)
                )
                continue
            break
        if method in ("logprobs", "grammar"):
            # A readable distribution is proof the provider can supply one: retire earlier absences.
            self._note_logprobs_present(context.model, context.transport.surface)
        native_reasoning = native_reasoning + result.reasoning
        return self._finalize(
            question, labels, readout, method, context, trace, native_reasoning, retry_reasons, log
        )

    def _finalize(
        self,
        question: Question,
        labels: Sequence[str],
        readout: methods.Readout,
        method: Method,
        context: _CallContext,
        trace: str | None,
        native_reasoning: tuple[ReasoningContentPart, ...],
        retry_reasons: list[str],
        log: _CallLog,
    ) -> _QuestionOutcome:
        probabilities = readout.probabilities
        error: float | None = None
        original: dict[str, float] | None = None
        if method == "structured":
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
            method=method,
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
        if context.auto:
            # The method was chosen rather than pinned, so report what each question actually used.
            debug["methods"] = {
                question_id: outcome.method for question_id, outcome in outcomes.items()
            }
        limits = self._limits_for(context.transport.surface)
        if limits != Limits():
            # The server refused a request field jevper added, and the answer came without it.
            debug["server_limits"] = {
                "structured": limits.structured,
                "reasoning": limits.reasoning,
                "include": limits.include,
                "cache_key": limits.cache_key,
                "thinking": limits.thinking,
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

    ``method`` selects how the decision is elicited — ``"auto"`` (the default) uses logprobs where the
    provider returns them and ``structured`` where it does not; ``api`` selects the surface (auto-detected
    by default). ``examples`` provides few-shot demonstrations for every question in the call, and each
    question may carry its own ``examples``, which take precedence.

    ``temperature`` is not sent unless you pass it. ``temperature=0.0`` is recommended for
    ``structured`` and ``discrete``, where the answer is a single sampled JSON object; ``logprobs``
    reads the model's own distribution and needs no setting.
    """

    def _run(
        self, question_id: str, question: Question, context: _CallContext
    ) -> _QuestionOutcome:
        log = _CallLog()
        steps = self._question_steps(question_id, question, context, log)
        try:
            spec = next(steps)
            while True:
                try:
                    result = self._call(log, question_id, spec, context)
                except (_LogprobsUnavailable, _SurfaceUnavailable) as exc:
                    # Hand the verdict back to the generator: it owns the fallback decision.
                    spec = steps.throw(exc)
                else:
                    spec = steps.send(result)
        except StopIteration as stop:
            return stop.value

    def _call(
        self, log: _CallLog, question_id: str, spec: CallSpec, context: _CallContext
    ) -> CallResult:
        attempt = 0
        while True:
            # The transport that issues *this* attempt. Another question's worker can switch the shared
            # context to another surface while this call is in flight, so a failure has to be judged
            # against the surface that produced it: judging it against whatever the context holds now
            # records a Responses 404 as a Chat one and writes off the surface that was working.
            transport = context.transport
            try:
                result = transport.call(spec, context.model)
            except Exception as exc:
                if not self._record_failure(log, question_id, spec, context, transport, exc) or (
                    attempt >= self.retry.n_retries
                ):
                    if isinstance(exc, ClientCapabilityError):
                        raise  # the client cannot read this surface; another attempt cannot help
                    if context.api_auto and _route_missing(
                        exc, surface=transport.surface, model=context.model
                    ):
                        # A server with no such route: hand the verdict to the generator, which owns
                        # the fallback decision, exactly as with an unavailable logprob readout.
                        raise _SurfaceUnavailable(
                            f"the server has no {transport.surface!r} route: {exc}",
                            surface=transport.surface,
                        ) from exc
                    downgraded = self._downgrade(exc, spec, transport)
                    if downgraded is not None:
                        # The field is optional; the question is not. Remember the limit and re-ask —
                        # with a fresh retry budget, because the attempts the old request shape spent
                        # say nothing about this one.
                        attempt = 0
                        self._remember_limits(transport.surface, downgraded)
                        self._switch_surface(context, transport.surface)
                        continue
                    failure = self._call_failure(exc, spec, context, transport, log)
                    # An embedded provider error is the caught exception itself; raising it `from`
                    # itself would print as its own cause.
                    raise failure from (None if failure is exc else exc)
                log.n_retries += 1
                time.sleep(_retry_delay(self.retry, attempt))
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
        method: MethodSelection | None = None,
        api: Api | None = None,
        reasoning: ReasoningConfig | None = None,
        temperature: float | None = None,
        prompt_cache_key: str | None = None,
    ) -> SystemOneResponse:
        start = time.perf_counter()
        context, parsed = self._prepare(
            state, questions, examples, model, method, api, reasoning, temperature, prompt_cache_key
        )
        outcomes: dict[str, _QuestionOutcome] = {}
        failure: BaseException | None = None
        if len(parsed) == 1:
            question_id, question = next(iter(parsed.items()))
            try:
                outcomes[question_id] = self._run(question_id, question, context)
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
                question_id: executor.submit(self._run, question_id, question, context)
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
        self, question_id: str, question: Question, context: _CallContext
    ) -> _QuestionOutcome:
        log = _CallLog()
        steps = self._question_steps(question_id, question, context, log)
        try:
            spec = next(steps)
            while True:
                try:
                    result = await self._call(log, question_id, spec, context)
                except (_LogprobsUnavailable, _SurfaceUnavailable) as exc:
                    # Hand the verdict back to the generator: it owns the fallback decision.
                    spec = steps.throw(exc)
                else:
                    spec = steps.send(result)
        except StopIteration as stop:
            return stop.value

    async def _call(
        self, log: _CallLog, question_id: str, spec: CallSpec, context: _CallContext
    ) -> CallResult:
        attempt = 0
        while True:
            # Same as the sync driver: the verdict belongs to the surface that produced the failure.
            transport = context.transport
            try:
                result = await transport.acall(spec, context.model)
            except Exception as exc:
                if not self._record_failure(log, question_id, spec, context, transport, exc) or (
                    attempt >= self.retry.n_retries
                ):
                    if isinstance(exc, ClientCapabilityError):
                        raise  # the client cannot read this surface; another attempt cannot help
                    if context.api_auto and _route_missing(
                        exc, surface=transport.surface, model=context.model
                    ):
                        # A server with no such route: hand the verdict to the generator, which owns
                        # the fallback decision, exactly as with an unavailable logprob readout.
                        raise _SurfaceUnavailable(
                            f"the server has no {transport.surface!r} route: {exc}",
                            surface=transport.surface,
                        ) from exc
                    downgraded = self._downgrade(exc, spec, transport)
                    if downgraded is not None:
                        # The field is optional; the question is not. Remember the limit and re-ask —
                        # with a fresh retry budget for the new request shape.
                        attempt = 0
                        self._remember_limits(transport.surface, downgraded)
                        self._switch_surface(context, transport.surface)
                        continue
                    failure = self._call_failure(exc, spec, context, transport, log)
                    # An embedded provider error is the caught exception itself; raising it `from`
                    # itself would print as its own cause.
                    raise failure from (None if failure is exc else exc)
                log.n_retries += 1
                await asyncio.sleep(_retry_delay(self.retry, attempt))
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
        method: MethodSelection | None = None,
        api: Api | None = None,
        reasoning: ReasoningConfig | None = None,
        temperature: float | None = None,
        prompt_cache_key: str | None = None,
    ) -> SystemOneResponse:
        start = time.perf_counter()
        context, parsed = self._prepare(
            state, questions, examples, model, method, api, reasoning, temperature, prompt_cache_key
        )
        semaphore = asyncio.Semaphore(self.max_concurrency)

        async def run(question_id: str, question: Question) -> _QuestionOutcome:
            async with semaphore:
                return await self._run(question_id, question, context)

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
