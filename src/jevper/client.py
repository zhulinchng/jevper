"""Sync and async client facades.

The per-question call sequence (analysis pass, answer pass, corrective retries) is expressed once as a
generator of ``CallSpec``; two small drivers execute it against the sync or async transport.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextvars
import math
import threading
import time
from collections.abc import Callable, Generator, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from dataclasses import fields as dataclass_fields
from datetime import timezone
from email.utils import parsedate_to_datetime
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
    _provider_error_from,
    afetch_models,
    fetch_models,
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
    ModelMetadata,
    NoulAnswer,
    NoulCriteria,
    Question,
    ScoreAnswer,
    SystemOnePayload,
    SystemOneResponse,
    Usage,
    bounded_text,
    ensure_encodable,
    noul_carries_a_question,
    parse_answer,
    parse_models,
    parse_question,
)

TRANSIENT_STATUS_CODES = frozenset({408, 409, 429})
"""The statuses below 500 that both official SDKs retry: 408 (the request timed out), 409 (OpenAI's
lock timeout, which Anthropic also retries) and 429 (slow down). Every 5xx is retried too, which is
why this is a set and not the whole rule — see :func:`_is_transient_status`. 429 and 408 can be the
logprob request's own fault, so neither counts as capability evidence; a 5xx can be nothing but the
server's, and is."""


def _is_transient_status(status: int | None) -> bool:
    """Whether this HTTP status is the provider's own transient failure, by the SDKs' rule."""
    return status is not None and (status in TRANSIENT_STATUS_CODES or status >= 500)


def _is_server_error(status: int | None) -> bool:
    """Whether the provider answered with a 5xx of its own: never the request's fault."""
    return status is not None and status >= 500
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
# implement structured outputs — ``response_format``/``text.format`` on the OpenAI surfaces,
# ``output_config`` on the Messages API — the reasoning parameters, or the Responses ``include`` list.
# None of them is needed to answer a question — the prompt already asks for one JSON object — so the
# field is dropped and the call is re-asked, and the server's limit is remembered.
_SCHEMA_MARKERS = (
    "response_format",
    "json_schema",
    "text.format",
    "structured output",
    "structured_output",
    "output_config",
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
# A provider that says which field was wrong is believed: these codes name the model itself, and a
# route that is missing says nothing of the kind. Read from the whole evidence, since the code can
# arrive as the exception's ``code`` attribute or inside the body it carries.
_MODEL_ERROR_CODES = (
    "model_not_found",
    "model_not_exist",
    "model_does_not_exist",
    "unknown_model",
    "invalid_model",
    "unsupported_model",
)
# A refusal that says the *protocol* is the problem, rather than the request or the model. Measured on
# opencode Zen 2026-09-26: `muse-spark-1.3-contributor` answers every Chat Completions request with
# `400 {"type": "ModelProtocolUnsupported", "message": "Model does not support this protocol."}` and
# answers the same question on Responses without complaint, so a client that reads this as a dead end
# never reaches the surface that works. Both a route and a protocol are "this server will not speak
# this surface here", and the marker is deliberately about the protocol rather than the model: a
# 400 naming a model is about the model, and the next surface would refuse it the same way.
_PROTOCOL_REFUSAL_MARKERS = (
    "does not support this protocol",
    "modelprotocolunsupported",
    "unsupported protocol",
    "protocol not supported",
    "does not support this endpoint",
    "does not support this route",
    "unsupported endpoint",
    "unsupported route",
    "not implemented for this model",
)
# A 400 that says only "unsupported" is about a field, not a surface, and is left alone: the field
# downgrade path reads those. A protocol refusal has to say so.
_PROTOCOL_REFUSAL_STATUSES = frozenset({400, 404, 405, 415, 422})
_SHOULD_RETRY = "x-should-retry"
_TRANSIENT_EXCEPTION_CLASSES = frozenset(
    {
        # httpx: every transport failure derives from TransportError, named individually anyway so a
        # client that raises its own look-alike is still recognised.
        "ConnectError",
        "ConnectTimeout",
        "ReadError",
        "ReadTimeout",
        "WriteError",
        "WriteTimeout",
        "PoolTimeout",
        "RemoteProtocolError",
        "TransportError",
        "TimeoutException",
        # openai / anthropic
        "APIConnectionError",
        "APITimeoutError",
        # urllib / http.client / ssl, and the socket errors they raise
        "URLError",
        "HTTPException",
        "IncompleteRead",
        "ChunkedEncodingError",
        "SSLError",
        "SSLEOFError",
        "ConnectionResetError",
        "ConnectionAbortedError",
        "ConnectionRefusedError",
        "BrokenPipeError",
        "NewConnectionError",
    }
)
"""Exception *class names* that mean the request never reached an answer: the httpx transport family,
the two SDKs' connection errors, and the standard library's. Matched whole — a name that merely
contains one of these words is somebody else's error, and retrying a programming error wastes the
caller's money to tell them nothing new."""
METHODS: tuple[Method, ...] = ("logprobs", "grammar", "structured", "discrete")
METHOD_SELECTIONS: tuple[MethodSelection, ...] = ("auto", *METHODS)
APIS: tuple[Api, ...] = ("auto", "chat_completions", "responses", "messages", "systemone")
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
    """Transient-failure retries: status codes 429/5xx and connection/timeout errors.

    A rate-limited provider says when to come back in ``Retry-After`` — or the millisecond
    ``retry-after-ms`` some send instead — and the TypeSafe clients honor it by default: the header
    replaces the computed backoff, so jevper waits as long as the server asked rather than as long as
    its own curve says. Set ``respect_retry_after=False`` for the curve alone, or ``n_retries=0`` to
    fail on the first rate-limited response. The honored delay is the server's to set: a header asking
    for minutes is waited out in minutes, and ``max_delay`` caps the computed curve rather than it.
    """

    n_retries: int = 2
    base_delay: float = 0.5
    max_delay: float = 8.0
    respect_retry_after: bool = True


def _response_of(exc: BaseException) -> Any:
    """The response an exception carries, by attribute or by mapping key — both are shapes in the wild."""
    response = getattr(exc, "response", None)
    if response is None:
        response = getattr(exc, "_response", None)
    if response is None and isinstance(getattr(exc, "__dict__", None), Mapping):
        response = exc.__dict__.get("response")
    return response


def _field(source: Any, name: str) -> Any:
    """One field of a duck-typed object, read as an attribute or as a mapping key."""
    value = getattr(source, name, None)
    if value is None and isinstance(source, Mapping):
        value = source.get(name)
    return value


def _read_status(value: Any) -> int | None:
    """One status value as an int, or ``None`` when it is not a status at all."""
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        # ``int(float("inf"))`` raises OverflowError, and a duck client is free to put anything in the
        # attribute: an unreadable status is an absent one, never a reason to fail the call here.
        return None


def _status_code(exc: BaseException) -> int | None:
    """The provider's HTTP status, when the exception carries one in a readable form."""
    status = _read_status(_field(exc, "status_code"))
    if status is None:
        # httpx.HTTPStatusError keeps it on the response instead, and its MRO carries no transport
        # marker, so without this it would be neither retried nor classified. A response is consulted
        # whenever the direct attribute is missing *or* unreadable: the response is the more reliable
        # of the two, and skipping it on a malformed direct value loses a status that was right there.
        status = _read_status(_field(_response_of(exc), "status_code"))
    return status


def _should_retry(exc: BaseException) -> bool | None:
    """What the provider asked for through ``x-should-retry``, when it said anything at all.

    Both official SDKs give this header precedence over their status defaults: OpenAI returns it when
    a request is worth repeating despite the status, and Anthropic does the same, so a gateway in
    front of either can decide for the client. Reading it as a plain status rule makes the opposite
    decision on both counts.
    """
    headers = _field(exc, "headers")
    if _header(headers, _SHOULD_RETRY) is None:
        headers = _field(_response_of(exc), "headers")
    asked = _header(headers, _SHOULD_RETRY)
    if asked is None:
        return None
    return asked.strip().casefold() == "true"


def _is_transient(exc: BaseException) -> bool:
    decided = _should_retry(exc)
    if decided is not None:
        return decided
    if _is_transient_status(_status_code(exc)):
        return True
    # Matched whole, not by substring: a caller's own ``ConnectionProgrammingError`` is a programming
    # error, and repeating the request cannot make it go away. The set is the httpx transport family
    # (whose failures are named after neither "Connection" nor "Timeout", which is why their base
    # classes are listed too), the two SDKs' connection errors, and the standard library's.
    names = {cls.__name__ for cls in type(exc).__mro__}
    return bool(names & _TRANSIENT_EXCEPTION_CLASSES) or isinstance(exc, (TimeoutError, ConnectionError))


def _retry_delay(policy: RetryPolicy, attempt: int, exc: BaseException | None = None) -> float:
    """How long to wait before retrying: what the provider asked for, else exponential backoff."""
    if policy.respect_retry_after and exc is not None:
        asked = _retry_after_seconds(exc)
        if asked is not None:
            # A provider that says when to come back is taken at its word, uncapped: the point of the
            # header is that coming back sooner is another request it will refuse. `max_delay` is the
            # caller's ceiling on the *computed* curve, not on the server's own instruction.
            return asked
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


def _sleep_before_retry(delay: float, policy: RetryPolicy) -> None:
    """Wait out the delay, degrading to the policy's own ceiling if this runtime cannot sleep it.

    ``MAX_RETRY_AFTER`` is the ceiling measured on the platforms this library runs on, but a runtime
    whose ``time_t`` is narrower still exists (32-bit musl), and there ``time.sleep`` raises
    ``OverflowError`` instead of sleeping. The caller's own ``max_delay`` is then the next best
    answer: a wait nobody can take is not a wait worth dying over.
    """
    try:
        time.sleep(delay)
    except OverflowError:
        time.sleep(min(policy.max_delay, delay))


_RETRY_AFTER_MS = "retry-after-ms"
_RETRY_AFTER = "retry-after"


def _header(headers: Any, name: str) -> str | None:
    """One response header as text, from an ``httpx.Headers``, a plain mapping, or a pair sequence.

    HTTP field names are case-insensitive, and a plain mapping is not: a client that hands its
    exceptions a ``{"RETRY-AFTER": "120"}`` dict is saying the same thing as one that spells it
    ``Retry-After``, so the lookup compares names case-folded rather than probing two spellings. A
    list of ``(name, value)`` pairs is the third shape a hand-rolled client ends up with — it has no
    ``.get`` at all, and a header jevper cannot see is one it cannot obey.
    """
    if headers is None:
        return None
    wanted = name.casefold()
    items = getattr(headers, "items", None)
    try:
        if callable(items):
            for key, value in items():
                if str(key).casefold() == wanted and isinstance(value, str) and value.strip():
                    return value.strip()
            return None
        if isinstance(headers, Mapping):
            for key in (name, name.title()):
                value = headers.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
            return None
        for entry in headers:  # a sequence of pairs, or anything else iterable
            if not isinstance(entry, (tuple, list)) or len(entry) < 2:
                continue
            if str(entry[0]).casefold() == wanted and isinstance(entry[1], str) and entry[1].strip():
                return entry[1].strip()
    except (AttributeError, IndexError, KeyError, TypeError, ValueError):  # not headers at all
        return None
    return None


_RETRY_AFTER_MS = "retry-after-ms"
_RETRY_AFTER = "retry-after"

# The longest wait a retry will take from a header: an operational ceiling, not a policy cap, so
# ``max_delay`` still does not apply to a header jevper honors. A day is longer than any provider asks
# for — OpenAI and Anthropic rate-limit windows are seconds to a minute, and a gateway that says
# "come back when the quota resets" means hours — and short enough that a header nobody sane sends
# cannot wedge a call for a century: a provider (or a proxy with a bug) answering ``Retry-After:
# 4294967295`` would otherwise park a thread for 136 years, which is representable as a float and as
# a C ``time_t`` and is therefore not caught by any representability check. Past the ceiling the
# header is not an instruction any client should carry out, and the backoff curve answers.
MAX_RETRY_AFTER = 24 * 60 * 60.0


def _retry_after_seconds(exc: BaseException) -> float | None:
    """How long the failed response asked jevper to wait, when it said.

    ``retry-after-ms`` is the millisecond form OpenRouter and Anthropic send; ``Retry-After`` is
    either delta-seconds or an HTTP date, which is how a proxy states the same wait. Delta-seconds are
    read as the integer the HTTP grammar defines — ``1*DIGIT`` — so a header spelled ``1e3``, ``+2`` or
    ``1.5`` is not a wait of a thousand, two or one and a half seconds, it is a header that is not
    one of the two documented forms. Anything unreadable falls back to the backoff rather than being
    guessed at.

    Only an exception the SDK raised for a failed status carries a response to read the headers from.
    A provider failure carried in the body of a ``200`` — OpenRouter's way of reporting an overloaded
    upstream — is a plain model object with no headers on it, so that case uses the backoff too.
    """
    headers = getattr(exc, "headers", None)
    if _header(headers, _RETRY_AFTER_MS) is None and _header(headers, _RETRY_AFTER) is None:
        headers = _field(_response_of(exc), "headers")
    for name, scale in ((_RETRY_AFTER_MS, 1e-3), (_RETRY_AFTER, 1.0)):
        raw = _header(headers, name)
        if raw is None:
            continue
        if raw.isdigit():
            seconds = int(raw) * scale
        elif name == _RETRY_AFTER:
            try:
                when = parsedate_to_datetime(raw)
            except (TypeError, ValueError, OverflowError):
                continue
            if when.tzinfo is None:
                when = when.replace(tzinfo=timezone.utc)
            # A date already past means come back now, which is what the header asked for.
            seconds = max(0.0, when.timestamp() - time.time())
        else:
            continue
        if 0 <= seconds <= MAX_RETRY_AFTER:
            return seconds
    return None


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
    ensure_encodable(key, where="prompt_cache_key")


def _require_count(name: str, value: Any, *, minimum: int | None = None, maximum: int | None = None) -> None:
    """A count option is an integer, in range, before anything else reads it as one.

    A float here is not a rounding opportunity: it reaches ``range()`` as a loop bound, a thread pool
    size, or a provider field, and a ``TypeError`` from any of those places is a worse report of a
    caller's slip than the one line this raises. ``bool`` is an ``int`` in Python and means nothing as
    a count here, so it is refused with the rest.
    """
    if not isinstance(value, int) or isinstance(value, bool):
        raise JevperError(f"{name} must be an integer, got {value!r}")
    if minimum is not None and value < minimum:
        raise JevperError(f"{name} must be >= {minimum}, got {value!r}")
    if maximum is not None and value > maximum:
        raise JevperError(f"{name} must be <= {maximum}, got {value!r}")


def _require_model(model: Any) -> None:
    """A model id is provider-ready text: something to send, and something to key state on.

    The key is the reason this is checked at all. The absence memory, the surface verdicts and the
    derived prompt-cache key are all indexed by the model string, and a number or an empty string
    reaches that indexing as an ``AttributeError`` on ``.encode()`` — long after construction, from a
    frame the caller did not write.
    """
    if not isinstance(model, str) or not model.strip():
        raise JevperError(f"model must be a non-empty string, got {model!r}")
    ensure_encodable(model, where="the model id")


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
        count = int(value)
    except (TypeError, ValueError, OverflowError):
        # ``int(float("inf"))`` raises OverflowError, which is the same kind of provider bug as a word
        # in a token count: "not reported" is the honest answer, and a raw OverflowError escaping the
        # call would not be.
        return None
    if count < 0:
        # No provider counts the tokens it did not use. A negative total is a provider bug, and
        # reporting one says less than "not reported" — and two of them can cancel into a plausible
        # positive number, which is worse than either.
        return None

    return current + count


_HEADER_NAME_CHARS = frozenset(
    "!#$%&'*+-.^_`|~0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
)


def _validate_headers(headers: Mapping[Any, Any], *, where: str) -> None:
    """Refuse a header jevper cannot put on the wire, before any provider sees the client.

    The official SDKs hand the mapping to httpx, which encodes a header as ASCII and raises from
    inside the request when it cannot — so a value with a newline, a NUL, a DEL, a non-ASCII
    character or an unpaired surrogate arrives as a ``ProviderError`` about a provider that never
    answered. A header that can inject a second one is worse than a crash, on a duck client that
    forwards it. All of these are the caller's own mistake and are reported as one, with the header
    named. RFC 9110's own rule is the one applied: a field name is a token, and a field value is
    printable ASCII with horizontal tabs.
    """
    for name, value in headers.items():
        if not isinstance(name, str) or not isinstance(value, str):
            raise JevperError(
                f"{where} names and values must be strings, got {name!r}: {value!r}"
            )
        if not name or not set(name) <= _HEADER_NAME_CHARS:
            raise JevperError(
                f"{where} name {name!r} is not a valid HTTP header name (letters, digits and "
                "!#$%&'*+-.^_`|~ only)"
            )
        for position, char in enumerate(value):
            code = ord(char)
            if code == 0x09 or 0x20 <= code <= 0x7E:
                continue
            raise JevperError(
                f"{where}[{name!r}] contains {char!r} at position {position}, which a header value "
                "cannot carry: the SDKs encode header values as ASCII, and a value may hold only "
                "printable ASCII and horizontal tabs"
            )


def _exception_text(exc: BaseException, limit: int = 500) -> str:
    """The exception's own text, bounded, and without letting its ``__str__`` fail the call.

    Two things go wrong when a provider's message is interpolated as-is. A duck-typed client can
    raise an exception whose ``__str__`` raises, which turns a provider failure into a programming
    error somewhere else entirely; and a gateway that answers with a megabyte of HTML makes every
    error message, every attempt record and every log line that quotes it a megabyte too. The first
    few hundred characters carry the status, the field and the reason — which is all a reader uses.
    """
    try:
        text = str(exc)
    except Exception:  # noqa: BLE001 - an exception that cannot describe itself is still an exception
        return f"<{type(exc).__name__} raised while formatting its own message>"
    return bounded_text(text, limit)


def _describe(exc: BaseException) -> str:
    """The failure as one line: its class, and its own bounded text."""
    return f"{type(exc).__name__}: {_exception_text(exc)}"


def _error_evidence(exc: BaseException) -> str:
    """Everything the provider said about the failure: its message, and the param/code it named."""
    parts = [_exception_text(exc, limit=2000)]
    for attr in ("param", "code"):
        value = getattr(exc, attr, None)
        if isinstance(value, str) and value:
            parts.append(value)
    return " ".join(parts).lower()


def _route_missing(exc: BaseException, *, surface: Surface, model: str) -> bool:
    """This surface is not spoken here, rather than a problem with what was sent.

    An OpenAI-compatible server that does not implement a surface's route answers 404 for it, and
    the ``openai`` client object exposes ``responses.create`` either way. A gateway whose *model*
    speaks only one protocol says so instead: opencode Zen answers every Chat Completions request for
    ``muse-spark-1.3-contributor`` with ``400 ModelProtocolUnsupported — Model does not support this
    protocol.`` and answers the same question on Responses, so a client that reads that as a dead end
    never reaches the surface that works. Both are the same fact — this server, this model, this
    surface — and both mean the next surface is worth trying.

    What is *not* that fact is a refusal about the model rather than the protocol, or about a field.
    A 404 that names the model is about the model, and so is ``{"error": {"message": "Unknown
    model", "code": "model_not_found"}}``, which names no id at all: the other surface would answer
    the same way, so switching would hide the real problem, and a protocol refusal has to say
    ``protocol`` or ``route`` in as many words before this reads one.
    """
    status = _status_code(exc)
    if status not in _PROTOCOL_REFUSAL_STATUSES:
        return False
    evidence = _error_evidence(exc)
    if getattr(exc, "embedded", False):
        # The status came from the body of a ``200``, not from the status line: a body that says
        # ``404`` is the provider reporting a failure inside a successful response, and treating it
        # as a missing route would answer the question on another surface and hide the error.
        return False
    if any(marker in evidence for marker in _PROTOCOL_REFUSAL_MARKERS):
        # Said in as many words, whatever the status. The evidence is read before the model markers
        # below because a protocol refusal usually names the model too — opencode's own message
        # starts "Model does not support this protocol." — and that is exactly the case where the
        # next surface is the answer rather than a repeat of the same error.
        return True
    if status != 404:
        return False
    if any(marker in evidence for marker in _MODEL_ERROR_CODES):
        return False
    names_the_model = model.lower() in evidence
    return not (names_the_model and any(marker in evidence for marker in _MODEL_404_MARKERS))


def _include_refused(evidence: str) -> bool:
    """The provider refused the Responses surface's logprob carrier, the ``include`` entry.

    ``include`` is the only place a Responses request can ask for logprobs, and a provider that does
    not offer that includable refuses the path rather than the word. OpenRouter answers ``Invalid
    option: expected one of "file_search_call.results"|...`` for ``path: ["include", 0]``; OpenAI's
    own wording for a model that offers no includable is ``Unsupported parameter: 'include' is not
    supported with this model.`` Both name the field and both say the same thing about it, so both
    read as the refusal they are — the vocabulary a server chooses to say so is not the fact.
    """
    return "include" in evidence and (
        any(marker in evidence for marker in _INCLUDE_REJECTION_MARKERS)
        or any(marker in evidence for marker in _UNSUPPORTED_MARKERS)
    )


def _include_value_refused(evidence: str) -> bool:
    """The refusal is about the *value* of an include entry: this provider offers other includables.

    OpenRouter answers ``Invalid option: expected one of "file_search_call.results"|"reasoning.
    encrypted_content"`` for ``path: ["include", 0]`` — it has listed what it accepts, and the logprob
    entry is not on the list, so no rearrangement of the other entries will satisfy it.
    """
    if "message.output_text.logprobs" in evidence:
        # A server that lists the logprob entry among the values it accepts is refusing the *other*
        # entry — OpenRouter and vLLM answer ``include[1]: expected one of
        # "message.output_text.logprobs"`` for ``reasoning.encrypted_content`` — and the logprob
        # word in that sentence is the server naming what it will carry, not what it refuses.
        return False
    return any(marker in evidence for marker in _INCLUDE_REJECTION_MARKERS)


def _logprobs_rejected(
    exc: BaseException, spec: CallSpec | None = None, surface: Surface | None = None
) -> bool:
    """The provider refused the request because of the logprob fields it carried.

    Both shapes seen in the wild name the field: Gemini's OpenAI-compatibility layer answers
    ``Unknown name "logprobs": Cannot find field.`` and a reasoning model behind an OpenAI-shaped
    gateway answers ``logprobs are not supported with reasoning models.`` A Responses request asks for
    logprobs through ``include`` instead, so the field it can be refused for is that one — and only
    there: on Chat Completions the carrier is the ``logprobs`` field itself, so a message that merely
    mentions ``include`` is about something else, however unsupported it says that something is.
    The ``spec`` has to say the request asked for logprobs either way: the same ``include`` list also
    carries the reasoning entry, and a refusal of that one is about reasoning, not about logprobs.
    """
    if _status_code(exc) not in _LOGPROB_REJECTION_STATUS_CODES:
        return False
    evidence = _error_evidence(exc)
    if "logprob" in evidence:
        return True
    return bool(
        spec is not None
        and spec.logprobs
        and surface == "responses"
        and _include_refused(evidence)
    )


def _logprobs_unsupported(
    exc: BaseException, spec: CallSpec | None = None, surface: Surface | None = None
) -> bool:
    """The rejection reads as a missing capability rather than a bad value, so it is worth remembering.

    ``logprobs are not supported with reasoning models.`` and ``Unknown name "logprobs": Cannot find
    field.`` both refuse the field. ``Invalid 'top_logprobs': integer must be between 0 and 5, but got
    20.`` — a server whose cap is lower than the default — refuses the value, so the question is
    answered with a logprob-free method without writing off logprobs for the rest of the client's life.
    """
    if not _logprobs_rejected(exc, spec, surface):
        return False
    evidence = _error_evidence(exc)
    if any(marker in evidence for marker in _VALUE_MARKERS):
        return False
    if surface == "responses" and _include_refused(evidence):
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

# A recorded request or response is debugging evidence, not a data channel: the answers carry what
# the caller asked for. These bounds keep a hostile or careless body from making one response
# unserializable or unprintable — a provider that pads a field with megabytes, or nests a value
# deeper than the interpreter's own recursion limit, cannot decide whether
# ``SystemOneResponse.model_dump_json()`` works.
MAX_DEBUG_STRING = 1 << 16
MAX_DEBUG_DEPTH = 24


def _dump_model(obj: Any) -> Any:
    """The provider object as plain data for ``debug``, without the provider's typing noise.

    A server can put a value the SDK's model does not expect in a field it still has to serialize —
    SGLang returns a list for ``metadata``, which the SDK types as a string — and pydantic answers
    every ``model_dump`` with a ``PydanticSerializationUnexpectedValue`` warning. The value survives
    the dump either way, and the dump exists for ``debug``, so the warning is noise the caller never
    asked for; ``warnings=False`` keeps it out of their logs without touching global warning state.

    A duck client that answers with a plain mapping is sanitized like a typed model. An object that
    cannot be dumped, or that is neither plain data nor a model at all, leaves a name behind rather
    than itself: the provider's own object in ``debug`` is exactly the thing ``model_dump_json`` then
    has to serialize, and a duck client is free to return anything.
    """
    dump = getattr(obj, "model_dump", None)
    if callable(dump):
        for keywords in ({"mode": "json", "warnings": False}, {"mode": "json"}, {}):
            try:
                return _sanitize_debug(dump(**keywords))
            except TypeError:
                continue  # this dump does not take those keywords; try the next spelling
            except Exception:  # noqa: BLE001 - debug data never decides whether a call succeeded
                return {"undumpable": f"<{type(obj).__name__} could not be dumped for debug>"}
    if isinstance(obj, (Mapping, list, tuple)):
        return _sanitize_debug(obj)
    if _is_plain_data(obj):
        return obj
    return {"undumpable": f"<{type(obj).__name__} is not data jevper can keep for debug>"}


def _is_plain_data(value: Any) -> bool:
    """Whether a value is something ``model_dump_json`` can already write."""
    return value is None or isinstance(value, (str, int, float, bool))


def _debug_entries(items: Any) -> dict[Any, Any]:
    """A mapping's entries with printable, non-colliding keys.

    Escaping and truncation can map two distinct provider keys onto the same text — ``"x\\ud800"``
    and the literal ``"x\\\\ud800"`` both become ``x\\ud800``, and a key cut at the bound can land on
    another key's marker. Silently keeping one of them would lose a field the provider sent, so a
    repeat is numbered: the record stays printable and every field the provider sent is still there.
    """
    entries: dict[Any, Any] = {}
    for key, item in items:
        name = _debug_key(key)
        if name in entries:
            suffix = 2
            while f"{name}#{suffix}" in entries:
                suffix += 1
            name = f"{name}#{suffix}"
        entries[name] = item
    return entries


def _debug_key(key: Any) -> Any:
    """A mapping key as JSON can write it: text bounded and escaped, anything else as its name."""
    if isinstance(key, str):
        return bounded_text(key, MAX_DEBUG_STRING)
    if isinstance(key, (int, float, bool)) or key is None:
        return key
    return bounded_text(str(key), MAX_DEBUG_STRING)


def _sanitize_debug(value: Any, depth: int = 0) -> Any:
    """The dumped provider object as printable, bounded, JSON-writable data.

    A body can carry an escaped lone surrogate, which the SDK decodes into a Python string no UTF-8
    encoder accepts; left in ``debug`` it would make ``SystemOneResponse.model_dump_json()`` raise
    long after the call succeeded. The escapes the wire carried are shown instead, so the debug
    record still says what the provider sent. Keys are text too, a duck client's raw mapping goes
    through here as well, and a value that is neither data nor a model — an ``object()`` in a
    metadata field, a set, a client object — becomes a marker, so no path into ``debug`` can leave
    the response unserializable.
    """
    if isinstance(value, str):
        return bounded_text(value, MAX_DEBUG_STRING)
    if _is_plain_data(value):
        return value
    if depth >= MAX_DEBUG_DEPTH:
        return f"<{type(value).__name__} nested deeper than {MAX_DEBUG_DEPTH} levels>"
    if isinstance(value, Mapping):
        return _debug_entries((key, _sanitize_debug(item, depth + 1)) for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return [_sanitize_debug(item, depth + 1) for item in value]
    # A model or an object of the caller's own, nested inside a duck client's mapping: ask it for
    # its data, and keep a name if it cannot answer.
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        try:
            return _sanitize_debug(dump(), depth + 1)
        except Exception:  # noqa: BLE001 - the record is evidence, never the call's outcome
            return f"<{type(value).__name__} could not be dumped for debug>"
    return f"<{type(value).__name__} is not data jevper can keep for debug>"


# A header whose name says it carries a credential is not recorded with its value: the attempt
# history reaches ``debug`` on every response and onto every ``ProviderError``, and a bearer token
# or tenant key does not belong in a structure callers log. The name is kept, because which header
# was sent is the part a reader needs.
_CREDENTIAL_HEADER_PARTS = (
    "authorization",
    "api-key",
    "apikey",
    "token",
    "secret",
    "cookie",
    "credential",
    "password",
    "signature",
)


def _redact_headers(headers: Mapping[Any, Any]) -> dict[Any, Any]:
    return {
        name: (
            "<redacted>"
            if isinstance(name, str)
            and any(part in name.casefold() for part in _CREDENTIAL_HEADER_PARTS)
            else value
        )
        for name, value in headers.items()
    }


def _recorded_request(request: Any) -> Any:
    """The request kwargs as they are kept: credential headers redacted, everything else bounded."""
    if not isinstance(request, Mapping):
        return _sanitize_debug(request)
    recorded = {
        key: _redact_headers(value) if key in ("extra_headers", "default_headers") else value
        for key, value in request.items()
    }
    return _sanitize_debug(recorded)


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
            "request": _recorded_request(request),
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


def _labels_for(question: Question) -> Sequence[str]:
    """A question's labels: two for a noul, one per option or level otherwise."""
    return labels_for(2 if question.type == "noul" else len(question.criteria))


def _attempts_for(log: _CallLog, question_id: str) -> list[dict[str, Any]]:
    """The one request's attempt records, attributed to the question being reported.

    A batch answers N questions with a single request, and jevper records one attempt per question it
    is answering: the request was the same, the question it was read for was not, and a trail naming
    only the first would make the other answers look unattributed.
    """
    return [{**record, "question_id": question_id} for record in log.attempts]


def _systemone_spec(
    state: Any,
    questions: Mapping[str, Question],
    question_ids: Sequence[str],
) -> CallSpec:
    """The request for ``question_ids``, and only those.

    The service takes a map, and which entries it carries is the batching decision rather than the
    wire format's.
    """
    selected = {question_id: questions[question_id] for question_id in question_ids}
    return CallSpec(messages=[], systemone=SystemOnePayload(state=state, questions=selected))


def _systemone_outcome(
    log: _CallLog,
    result: CallResult,
    question_id: str,
    question: Question,
    totals: tuple[dict[str, int | None], int, int] | None = None,
    attempts: list[dict[str, Any]] | None = None,
) -> _QuestionOutcome:
    """One answer as the service sent it, as the outcome the rest of the library expects.

    The service's own ``score``, ``confidence`` and ``choice`` are kept: it computed them from a
    model trained for the decision, and jevper's formulas — which agree with it to within 0.015,
    see ``docs/jev-comparison.md`` — are not a reason to replace them. ``totals`` carries the
    request's usage when several questions shared one request, so that summing the outcomes
    counts the call once rather than once per question it answered.
    """
    answers = result.response.get("answers") if isinstance(result.response, Mapping) else None
    if not isinstance(answers, Mapping) or question_id not in answers:
        raise MalformedAnswerError(
            f"question {question_id!r}: the service's response carries no answer for it"
            + (f"; it answered {sorted(answers)}" if isinstance(answers, Mapping) else "")
        )
    answer = parse_answer(question_id, question, answers[question_id])
    tokens, n_calls, n_retries = (
        totals if totals is not None else (log.tokens, log.n_calls, log.n_retries)
    )
    return _QuestionOutcome(
        answer=answer,
        reasoning=(),
        tokens=tokens,
        n_calls=n_calls,
        n_retries=n_retries,
        attempts=log.attempts if attempts is None else attempts,
        retry_reasons=[],
        readout_debug={
            "source": "systemone",
            "probabilities": {
                str(key): float(value) for key, value in getattr(answer, "probabilities", {}).items()
            },
            "missing_labels": [],
            "observed_text": None,
        },
        method="systemone",
    )


def _batch_failure(exc: BaseException, log: _CallLog, question_ids: Sequence[str]) -> None:
    """Attribute a failed batch request's trail to every question it was asked for.

    One request was made for the whole batch, so a trail naming only the first question would read as
    though the rest were never asked. A failure carrying no trail — a capability refusal, a
    connection error before the first try — is left alone, because there is nothing to reattribute.
    """
    if getattr(exc, "attempts", None):
        exc.attempts = [
            record for question_id in question_ids for record in _attempts_for(log, question_id)
        ]


def _batch_outcomes(
    log: _CallLog,
    result: CallResult,
    parsed: Mapping[str, Question],
) -> tuple[dict[str, _QuestionOutcome], tuple[dict[str, int | None], int, int]]:
    """Every question answered by one request, and that request's totals, which they all share.

    The point of the System One surface is that the questions travel together: measured against the
    live endpoint on 2026-09-26, eleven questions came back from one request in 1.08 s, where the
    per-question path spends a round trip each. The price is granularity — a transient failure re-asks
    the whole batch, and a missing or mistyped answer fails the call rather than one question of it —
    and the totals come back with the outcomes because counted per question they would report N
    times what the service charged.
    """
    totals = (log.tokens, log.n_calls, log.n_retries)
    outcomes = {
        question_id: _systemone_outcome(
            log,
            result,
            question_id,
            parsed[question_id],
            totals=totals,
            attempts=_attempts_for(log, question_id),
        )
        for question_id in parsed
    }
    return outcomes, totals


def _wire_offenders(state: Any, parsed: Mapping[str, Question]) -> list[str]:
    """What the System One wire format refuses, found before a request rather than after a 422.

    One rule, measured field by field against the live service on 2026-09-26 and matching the
    published API reference: every structured field — ``state``, a question's ``instructions``, a
    noul's two criteria sides, a choice option's description, a score's level — takes a string, an
    object or an array. A bare number or boolean is 422 "Input should be a valid string" naming the
    exact path, in all five places. An array answers 200 where a string would, as does an object.

    Null is the one value the rule treats differently per field, and each of these is measured too: a
    null ``state`` is refused as missing, a null score level is 422, and a null on a noul side or a
    choice option is 200 — the Jev API documents the last as "use null when an option needs no extra
    detail". An empty question id answers 400 "Question key cannot be empty.".

    jevper's own question types are deliberately wider than this, because a prompt surface can render
    a number as text and no prompt surface has such a limit, so the check belongs here rather than in
    the models: nothing about ``Noul(instructions=0)`` is wrong until it goes on this wire.
    """
    offenders: list[str] = []
    if state is None or _is_bare_scalar(state):
        offenders.append(f"state={state!r}")
    for question_id, question in parsed.items():
        if not question_id:
            offenders.append("an empty question id")
        if _is_bare_scalar(question.instructions):
            offenders.append(f"question {question_id!r}: instructions={question.instructions!r}")
        if question.type == "noul":
            # A noul may carry no criteria at all, leaning on its instructions, which the noul rule
            # has already settled; there is nothing on a wire to check in that case.
            criteria = question.criteria
            if isinstance(criteria, NoulCriteria):
                for side in ("true", "false"):
                    value = getattr(criteria, side)
                    if _is_bare_scalar(value):
                        offenders.append(f"question {question_id!r}: criteria {side}={value!r}")
        elif question.type == "choice":
            for option, description in question.criteria.items():
                if _is_bare_scalar(description):
                    offenders.append(
                        f"question {question_id!r}: the description of option {option!r} "
                        f"is {description!r}"
                    )
        else:
            for index, level in enumerate(question.criteria):
                if level is None or _is_bare_scalar(level):
                    offenders.append(f"question {question_id!r}: score level {index} = {level!r}")
    return offenders


def _is_bare_scalar(value: Any) -> bool:
    """Whether a value is a number or a boolean, which this wire format has nowhere to put.

    ``bool`` before ``int`` for the reader rather than for the behaviour: ``isinstance`` already
    covers it, and naming both says what is being excluded.
    """
    return isinstance(value, (bool, int, float))


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
        # The count is a provider field, so its own range is the floor: OpenAI and every server here
        # answer [0, 20]. Only the label readouts need the lower bound of 2 (see _require_top_logprobs);
        # a negative count is not a small count, and letting it reach the provider spends a request to
        # be told what a client can say for free.
        _require_count("top_logprobs", top_logprobs, minimum=0, maximum=MAX_TOP_LOGPROBS)
        _require_count("max_concurrency", max_concurrency, minimum=1)
        _require_count("n_retry_malformed", n_retry_malformed, minimum=0)
        _require_model(model)
        if reasoning is not None and not isinstance(reasoning, ReasoningConfig):
            raise JevperError(f"reasoning must be a ReasoningConfig, got {type(reasoning).__name__}")
        if retry is not None and not isinstance(retry, RetryPolicy):
            raise JevperError(f"retry must be a RetryPolicy, got {type(retry).__name__}")
        if extra_body is not None and not isinstance(extra_body, Mapping):
            raise JevperError(f"extra_body must be a mapping, got {type(extra_body).__name__}")
        if extra_headers is not None:
            if not isinstance(extra_headers, Mapping):
                raise JevperError(
                    f"extra_headers must be a mapping, got {type(extra_headers).__name__}"
                )
            _validate_headers(extra_headers, where="extra_headers")
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
        # A copy, so a caller that keeps and later edits the mapping cannot slip a header past the
        # validation above: what was checked is exactly what is sent, and later edits are the
        # caller's own new client. Same for the body, whose text is checked on every call anyway.
        self.extra_body = dict(extra_body) if extra_body is not None else None
        self.extra_headers = dict(extra_headers) if extra_headers is not None else None
        self.prompt_cache_key = prompt_cache_key
        self._executor: concurrent.futures.ThreadPoolExecutor | None = None
        self._executor_lock = threading.Lock()
        # What method="auto" has learned about this provider, per (model, surface).
        self._auto_methods: dict[tuple[str, Surface], Method] = {}
        self._auto_misses: dict[tuple[str, Surface], int] = {}
        # Surfaces this server has answered 404 for, learned by trying them once each.
        self._missing_surfaces: set[Surface] = set()
        # Request fields this server has refused, per (model, surface): structured output, reasoning,
        # include. A refusal is usually about the model that earned it, so it is remembered there.
        self._limits: dict[tuple[str, Surface], Limits] = {}
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
        produced it. The provider's own words go through the same scrub as the header values: a gateway
        that quotes the key it rejected puts a credential in every error message and every log line
        that quotes one.
        """
        log.add_attempt(
            question_id,
            surface=transport.surface,
            request=transport.kwargs(spec, context.model),
            error=self._scrub(_describe(exc)),
        )
        return _is_transient(exc)

    def _secret_values(self) -> tuple[str, ...]:
        """The caller's own credential values, from the headers whose names carry one.

        Only the caller's: the SDK client's own key is not reachable from here, and a value too short
        to be a credential is not scrubbed — replacing ``"1"`` or ``"test"`` everywhere would corrupt
        every message that happens to contain those letters.
        """
        if not self.extra_headers:
            return ()
        return tuple(
            value
            for name, value in self.extra_headers.items()
            if isinstance(name, str)
            and isinstance(value, str)
            and len(value) >= 8
            and any(part in name.casefold() for part in _CREDENTIAL_HEADER_PARTS)
        )

    def _scrub(self, text: str) -> str:
        """The caller's own credential values removed from text the provider wrote."""
        for secret in self._secret_values():
            text = text.replace(secret, "<redacted>")
        return text

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

    def _next_surface(self, surface: Surface, method: Method) -> Surface | None:
        """The next surface ``api="auto"`` may answer on after this one stopped answering.

        The order is the documented one: Responses, then Chat Completions, then Messages, and the
        search wraps — a surface the client can still speak and this server has not already given
        a 404 for is worth a request wherever it sits in the order, because a label readout moving
        off a surface without logprobs lands on the other one and needs to come back when that one
        turns out to have no route. The 404 memory is what stops the walk from circling: a surface
        that has answered 404 is never tried again. Messages carries no logprobs, so a pinned
        ``logprobs`` or ``grammar`` method skips it and keeps looking.
        """
        order: tuple[Surface, ...] = ("responses", "chat_completions", "messages")
        start = order.index(surface)
        for step in range(1, len(order)):
            candidate = order[(start + step) % len(order)]
            if self._surface_missing(candidate):
                continue
            if not _has_attribute(self.client, SURFACES[candidate][2]):
                continue
            if candidate == "messages" and method in ("logprobs", "grammar"):
                continue
            return candidate
        return None

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

    def _limits_for(self, model: str, surface: Surface) -> Limits:
        """What this server has been observed to accept for this model on this surface.

        Keyed by the model as well as the surface, because a refusal of a request field is a fact
        about a model far more often than about a server: ``reasoning_effort is not supported for
        this model`` says so in as many words, and a strict-schema refusal is usually a particular
        model's serving stack. Remembering it provider-wide would quietly switch off a field the
        caller asked for on every other model this client ever names — and ``debug`` would still say
        the field was requested. The cost of the narrower key is one refusal per (model, surface),
        which is a request the caller would have spent anyway on a server that refuses it everywhere.
        """
        with self._auto_lock:
            return self._limits.get((model, surface), Limits())

    def _remember_limits(self, model: str, surface: Surface, base: Limits, limits: Limits) -> None:
        """Remember a server's limit for this model on this surface, like any other verdict.

        A downgrade is computed from the snapshot its transport was built with, and the questions run
        concurrently, so writing that result wholesale would let a later write put back a field
        another question has meanwhile learned to leave out — and the next call would pay the same
        refusal again. Only the fields this downgrade actually changed are applied, onto whatever is
        remembered now.
        """
        key = (model, surface)
        with self._auto_lock:
            current = self._limits.get(key, base)
            changed = {
                entry.name: getattr(limits, entry.name)
                for entry in dataclass_fields(Limits)
                if getattr(limits, entry.name) != getattr(base, entry.name)
            }
            if changed:
                self._limits[key] = replace(current, **changed)

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
        if _status_code(exc) not in _LOGPROB_REJECTION_STATUS_CODES:
            return None
        evidence = _error_evidence(exc)
        limits = transport.limits
        if (
            spec.reasoning is not None
            and limits.include
            and "include" in evidence
            and not (spec.logprobs and _include_value_refused(evidence))
        ):
            # The list carried the reasoning entry too, and a server that refuses the *field* may
            # well accept the one entry a label readout needs — dropping the reasoning entry is one
            # request to find out, and the only thing there is to drop. The exception is a request
            # that wanted logprobs from a server which has already listed the includables it offers
            # and left ours out: rearranging the other entries earns the identical refusal, so the
            # verdict below takes it at once.
            return replace(limits, include=False)
        if _logprobs_rejected(exc, spec, transport.surface):
            return None
        # A complaint about the number is not a complaint about the field: dropping the field there
        # would answer the question with the caller's reasoning off, and remember that as the server's
        # limit. The provider's own error says which number it wanted, so it travels back instead.
        value_complaint = _value_refused(evidence)
        if spec.json_schema is not None and any(marker in evidence for marker in _SCHEMA_MARKERS):
            if transport.surface == "messages":
                # Anthropic's own schema field, and the only one this surface has. There is no
                # ``json_object`` rung to step down to: dropping the field leaves the prompt carrying
                # the schema, which is where it lived before the field existed. A complaint about the
                # schema's contents is worth the rung too, for the reason the comment above gives.
                if limits.output_config:
                    return replace(limits, output_config=False)
            elif limits.structured == "schema":
                # With ``structured_outputs=False`` the request already carried a plain
                # ``json_object``: the next rung is no format field at all, and re-sending the same
                # bytes would only spend a second call on the same refusal.
                if not transport.structured_outputs:
                    return replace(limits, structured="none")
                return replace(limits, structured="object")
            elif limits.structured == "object":
                return replace(limits, structured="none")
        if (
            not value_complaint
            and spec.reasoning is not None
            and limits.reasoning
            and any(marker in evidence for marker in _REASONING_MARKERS)
        ):
            return replace(limits, reasoning=False)
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
        self, surface: Surface, reasoning: ReasoningConfig | None, method: Method | None = None
    ) -> Surface | None:
        """The other surface to try for a label readout, when the move is free of side effects.

        ``native`` reasoning exists only on the Responses surface, so a caller who asked for it — or
        whose ``mode="auto"`` resolved to it — keeps the surface that choice implies, and ``grammar``
        is a Chat Completions convention with no counterpart anywhere, so a request that carries one
        has nowhere to go. Otherwise the move costs nothing but a request, and it is worth one: a
        server can implement a surface without carrying logprobs through it — ollama answers the
        Responses route with an empty logprob list, llama.cpp rejects the logprob fields there
        outright, and OpenAI's Responses logprobs hold the sampled token and no alternatives — while
        Chat Completions on the same server carries a full distribution.
        """
        other: Surface = "chat_completions" if surface == "responses" else "responses"
        if (
            method == "grammar"
            or self._surface_missing(other)
            or resolve_reasoning_mode(reasoning, surface) == "native"
        ):
            return None
        return other if _has_attribute(self.client, SURFACES[other][2]) else None

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
            limits=self._limits_for(context.model, surface),
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
        if (context.auto or context.api_auto) and spec.logprobs:
            failure = self._scrub(_describe(exc))
            if _logprobs_rejected(exc, spec, transport.surface):
                capability = _logprobs_unsupported(exc, spec, transport.surface)
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
            if _is_server_error(_status_code(exc)) and context.auto:
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
            scrubbed = self._scrub(str(exc))
            if scrubbed != str(exc) and type(exc) is ProviderError:
                # The body the provider sent can quote the credential it rejected, and this is the
                # message every caller and every log line will carry. Only the plain class is rebuilt,
                # so a subclass keeps its identity rather than being flattened by the scrub.
                return ProviderError(
                    scrubbed,
                    attempts=log.attempts,
                    status_code=exc.status_code,
                    embedded=exc.embedded,
                )
            return exc
        return ProviderError(
            self._scrub(_describe(exc)),
            attempts=log.attempts,
            status_code=_status_code(exc),
        )

    def _models_failure(self, exc: BaseException, attempt: int) -> BaseException | None:
        """The failure to report for a model-list request, or ``None`` to try it again.

        A jevper error is already the verdict — a provider failure with the provider's own status,
        or a capability refusal naming the caller's client, neither of which is the other and
        neither of which repeating the request would change. Anything else is the SDK's exception
        type, which is re-raised the way every other request jevper makes reports one: a
        ``ProviderError`` carrying the status the provider gave.

        A ``ProviderError`` is returned scrubbed rather than re-raised, because a provider can quote
        the credential it rejected and this message is what every log line downstream carries.
        """
        failure = _provider_error_from(exc)
        if isinstance(failure, ProviderError):
            # No attempt history to clear: a ProviderError built here has none, and a duck client
            # that raised one of its own brought its own trail, which is the caller's to keep. A
            # subclass is returned as it is, carrying the subclass a caller may be catching.
            scrubbed = self._scrub(str(failure))
            if scrubbed != str(failure) and type(failure) is ProviderError:
                return ProviderError(
                    scrubbed,
                    status_code=failure.status_code,
                    embedded=failure.embedded,
                )
            return failure
        if isinstance(failure, JevperError):
            # A capability verdict is about the caller's client, not the provider's answer.
            return failure
        if _is_transient(failure) and attempt < self.retry.n_retries:
            return None
        return ProviderError(self._scrub(_describe(failure)), status_code=_status_code(failure))

    def _fetch_models_retrying(self, fetch: Callable[[], Any]) -> Any:
        """One model-list request under the caller's retry policy, or the provider's own failure.

        This is jevper's request, so the policy that applies to it is the caller's and the SDK's own
        loop is off (see ``transport.fetch_models``). It is not an evaluation, so there is no attempt
        history to report — but a provider failure is still reported the way one is everywhere else,
        as a ``ProviderError`` carrying the status, rather than as the SDK's own exception type
        escaping a method documented to return a list.
        """
        attempt = 0
        while True:
            try:
                return fetch()
            except Exception as exc:  # noqa: BLE001 - re-raised below, carrying the provider's status
                failure = self._models_failure(exc, attempt)
                if failure is not None:
                    raise failure from None
                _sleep_before_retry(_retry_delay(self.retry, attempt, exc), self.retry)
                attempt += 1

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
        if self.extra_body is not None:
            ensure_encodable(self.extra_body, where="extra_body")
        if reasoning is not None and not isinstance(reasoning, ReasoningConfig):
            # Without this the mode lookup dereferences a string and the caller sees an AttributeError.
            raise JevperError(f"reasoning must be a ReasoningConfig, got {type(reasoning).__name__}")
        if prompt_cache_key is not None:
            _require_prompt_cache_key(prompt_cache_key)
        # Fail fast, before any provider call. The state is validated per surface: rendering it as
        # prompt turns is what a prompt surface needs, and judging it by that renderer would refuse
        # a state the System One wire format takes — an empty list is a wire-format state, and the
        # renderer refuses it as an empty conversation.
        parsed = {question_id: parse_question(question_id, raw) for question_id, raw in questions.items()}
        state_messages: tuple[dict[str, str], ...] = ()
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
        _require_model(effective_model)
        effective_reasoning = reasoning if reasoning is not None else self.reasoning
        # The surface is picked for the method auto tries first; the fallback runs on either surface.
        surface = select_surface(self.client, effective_api, AUTO_METHOD if auto else requested)
        if surface == "systemone":
            # Everything below this line is about choosing how to ask a model a question in a prompt.
            # This surface is the wire format itself: the questions go in the body, the service
            # answers them, and none of the prompt machinery below applies.
            return self._prepare_systemone(
                state,
                parsed,
                examples,
                effective_model,
                requested,
                reasoning,
                temperature,
                prompt_cache_key,
            ), parsed
        # The rendered turns are reused for every question, so the state is rendered once.
        state_messages = tuple(render_state_messages(state))

        if api_auto and self._surface_missing(surface):
            # This server answered 404 for the route before: do not pay for the discovery again, and
            # start on the next surface in the documented order. When there is none — a messages-only
            # client, or a method that only an OpenAI surface can carry — staying put re-asks the
            # surface that is known to 404, and the in-call handler reports that 404, exactly as it
            # did on the call that learned it. Moving anyway used to turn the provider's 404 into an
            # AttributeError for an attribute the caller never had.
            other = self._next_surface(surface, requested)
            if other is not None:
                surface = other
        elif (
            api_auto
            and (auto or requested == "logprobs")
            and self._logprobs_absent_here(effective_model, surface)
        ):
            alternative = self._logprob_surface_alternative(
                surface, effective_reasoning, AUTO_METHOD if auto else requested
            )
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
                limits=self._limits_for(effective_model, surface),
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

    def _prepare_systemone(
        self,
        state: Any,
        parsed: Mapping[str, Question],
        examples: Examples,
        model: str,
        requested: Method,
        reasoning: ReasoningConfig | None,
        temperature: float | None,
        prompt_cache_key: str | None,
    ) -> _CallContext:
        """A call on the System One surface, which is the wire format rather than a prompt.

        Four of jevper's options have no field on this wire, and the service ignores what it does
        not know — which is exactly why they are refused here rather than dropped in silence. One
        error naming all of them, so a caller fixes the call in one go instead of one complaint per
        attempt. ``method="auto"`` and no ``temperature`` are what this branch accepts, because they
        are the values that ask for nothing.
        """
        for question_id, question in parsed.items():
            # The one question rule the service itself enforces, mirrored where it is enforced: a
            # prompt surface can render the question from a few-shot example, this wire format cannot.
            if question.type == "noul" and not noul_carries_a_question(question):
                raise InvalidQuestionError(
                    f"question {question_id!r}: a noul must carry instructions or criteria; the Jev "
                    "API answers 400 for one with neither"
                )
        # What the wire format asks for that a prompt surface has no opinion about, refused here
        # rather than sent: a structured field takes a string, an object or an array, so a bare
        # number, a boolean or null is a 422 there, and a question id has to be a non-empty string.
        # All of them in one error, so a caller fixes the call in one go rather than one complaint
        # per attempt.
        offenders = _wire_offenders(state, parsed)
        if offenders:
            raise InvalidQuestionError(
                "the Jev API refuses "
                + "; ".join(offenders)
                + " — on this wire a structured field takes a string, an object or an array, and a "
                "question id is a non-empty string"
            )
        # The prompt renderer is not consulted on this surface, so the one thing it did check is
        # checked here: a lone surrogate in the state is a value no request could carry.
        ensure_encodable(state, where="state")
        refused: list[str] = []
        if requested != "auto":
            refused.append(f"method={requested!r} (the service picks its own method here)")
        if reasoning is not None or self.reasoning is not None:
            refused.append("reasoning= (this wire format has no thinking field)")
        if any(
            question.examples
            or _pick_examples(examples, question_id)
            or _pick_examples(self.examples, question_id)
            for question_id, question in parsed.items()
        ):
            # Asked per question, the way ``_resolve_examples`` asks: an example set keyed to other
            # questions never reaches this wire either, and refusing the call over one would leave
            # the caller no way to clear it but rebuilding the client.
            refused.append("examples= (the System One request has no examples field)")
        effective_temperature = temperature if temperature is not None else self.temperature
        if effective_temperature is not None:
            refused.append(f"temperature={effective_temperature!r} (this wire format takes none)")
        effective_cache_key = (
            prompt_cache_key if prompt_cache_key is not None else self.prompt_cache_key
        )
        if effective_cache_key is not None:
            refused.append("prompt_cache_key= (there is no prompt here to cache)")
        if refused:
            raise ClientCapabilityError(
                "api='systemone' cannot carry "
                + ", ".join(refused)
                + f"; drop {'it' if len(refused) == 1 else 'them'}, or use api='auto' with an "
                "OpenAI-compatible client"
            )
        return _CallContext(
            transport=make_transport(
                self.client,
                "systemone",
                structured_outputs=self.structured_outputs,
                extra_body=self.extra_body,
                extra_headers=self.extra_headers,
                limits=self._limits_for(model, "systemone"),
            ),
            model=model,
            method="systemone",
            mode="off",
            reasoning=None,
            temperature=None,
            examples=(),
            state_messages=(),
            answer_reasoning=None,
            analysis_reasoning=None,
            prompt_cache_key=None,
        )

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
                if isinstance(candidate, (str, bytes)) or not isinstance(candidate, Iterable):
                    raise InvalidQuestionError(
                        f"question {question_id!r}: examples must be a sequence of Example objects, "
                        f"got {type(candidate).__name__}"
                    )
                resolved = tuple(candidate)
                break
        for index, example in enumerate(resolved):
            if not isinstance(example, Example):
                # A tuple or a dict here is a caller guessing at the shape: the AttributeError that
                # reading `.answer` would raise is the least useful way to hear about it.
                raise InvalidQuestionError(
                    f"question {question_id!r}: example {index} must be an Example, "
                    f"got {type(example).__name__}"
                )
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
                if fell_back or not (context.auto or context.api_auto):
                    raise
                # The surface that produced the verdict, which is not necessarily the one the shared
                # context holds now: another question's worker may have moved it while this call was
                # in flight.
                surface = exc.surface or context.transport.surface
                if exc.capability and exc.evidence != "transient" and context.api_auto:
                    alternative = self._logprob_surface_alternative(surface, context.reasoning, context.method)
                    if alternative is not None and not self._logprobs_absent_here(
                        context.model, alternative
                    ):
                        # The verdict is about this surface — it answered without logprobs, or refused
                        # the fields outright, as llama.cpp's Responses shim does. Mark it, so later
                        # calls start where the distribution is, and answer this question there. The
                        # move keeps the caller's method, so it is made even for an explicit
                        # ``method="logprobs"``: that method asked for a distribution, not for a
                        # particular surface to produce it.
                        self._remember_logprobs_unavailable(context.model, surface)
                        retry_reasons.append(
                            f"{exc} — retrying the label readout on api={alternative!r}"
                        )
                        self._switch_surface(context, alternative)
                        continue
                if not context.auto:
                    # An explicit method keeps its own contract: where no surface is left to carry it,
                    # it reports the provider's refusal rather than being swapped for another readout.
                    # The verdict is still real, so it is remembered like any other: without that, a
                    # later ``method="auto"`` call pays a request to rediscover it on this surface.
                    self._note_surface_absence(context.model, surface, exc)
                    raise
                fell_back = True
                retry_reasons.append(f"{exc} — answering with method={FALLBACK_METHOD!r}")
                self._note_surface_absence(context.model, surface, exc)
                # Questions that have not started yet take the fallback without paying for it.
                context.method = FALLBACK_METHOD
            except _SurfaceUnavailable as exc:
                if not context.api_auto:
                    raise
                self._remember_surface_missing(exc.surface)
                other = self._next_surface(exc.surface, context.method)
                if other is None:
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
            unavailable = methods.answer_failure(result)
            if unavailable is not None:
                # The analysis is a request like any other: a trace cut off by the output budget, or
                # filtered, is not a trace. Quoting a partial one into the answer prompt teaches the
                # model to answer from a half-thought, and the second call would be spent proving it.
                unavailable.attempts = log.attempts
                raise unavailable
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
            unavailable = methods.answer_failure(result)
            if unavailable is not None:
                # Truncated, filtered and refused responses are not malformed answers: reading one
                # would turn a generation the provider itself cut short into a typed decision, and a
                # corrective retry spends a call to be cut short again. Reported as the provider's
                # failure, with the attempt history the other provider errors carry.
                unavailable.attempts = log.attempts
                raise unavailable
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
            self._note_logprobs_present(context.model, result.surface)
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
            # The score is an expected value, so it is only meaningful over a distribution that sums
            # to 1: with `normalize_probabilities=False` the reported numbers are the provider's own,
            # and a distribution off 1 would otherwise put `score` off the 0..N-1 line the Jev answer
            # schema documents. The TypeSafe reference adapter rescales for exactly this and leaves
            # the reported probabilities untouched, so the answer stays verbatim and the score does not.
            score_distribution = rescale(distribution)
            answer = ScoreAnswer(
                score=math.fsum(level * probability for level, probability in score_distribution.items()),
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
            # The answer text is the provider's, and it is as unbounded as anything else it sent.
            log.attempts[-1]["readout"] = _sanitize_debug(readout_debug)
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
        totals: tuple[dict[str, int | None], int, int] | None = None,
    ) -> SystemOneResponse:
        """The call's response, from its per-question outcomes.

        ``totals`` replaces the summed usage when every outcome came from one shared request — a
        System One batch answers N questions with one request, and counting its tokens once per
        question would report N times what the service charged.
        """
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
        if totals is not None:
            shared_tokens, shared_calls, shared_retries = totals
            tokens = dict(shared_tokens)
            n_calls = shared_calls
            n_retries = shared_retries
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
        surfaces = {
            attempt["surface"] for attempt in attempts if isinstance(attempt.get("surface"), str)
        }
        if len(surfaces) > 1:
            # Questions run concurrently against one shared context, so a 404 on one surface can
            # move the client while another question is still answering on the old one. The scalar
            # ``api`` is the surface the call ended on; this says where each question was answered.
            debug["apis"] = {
                question_id: outcome.attempts[-1]["surface"]
                for question_id, outcome in outcomes.items()
                if outcome.attempts and isinstance(outcome.attempts[-1].get("surface"), str)
            }
            # The limits are per (model, surface), and a mixed call learned some on each; the scalar
            # ``server_limits`` speaks for the surface the call ended on.
            debug["server_limits_by_api"] = {
                name: {
                    entry.name: getattr(self._limits_for(context.model, name), entry.name)
                    for entry in dataclass_fields(Limits)
                }
                for name in sorted(surfaces)
            }
            # The mode follows from the surface a question was answered on — native on Responses,
            # the two-step path elsewhere — so the scalar ``reasoning_mode`` cannot speak for a
            # batch that used both. Derived from each question's own last attempt, the same way
            # ``apis`` is, so it cannot disagree with what was sent.
            debug["reasoning_modes"] = {
                question_id: resolve_reasoning_mode(context.reasoning, surface)
                for question_id, surface in debug["apis"].items()
            }
        if context.auto:
            # The method was chosen rather than pinned, so report what each question actually used.
            debug["methods"] = {
                question_id: outcome.method for question_id, outcome in outcomes.items()
            }
        limits = self._limits_for(context.model, context.transport.surface)
        if limits != Limits():
            # The server refused a request field jevper added, and the answer came without it. Every
            # field of the limits is reported, so a new rung cannot be forgotten here.
            debug["server_limits"] = {
                entry.name: getattr(limits, entry.name) for entry in dataclass_fields(Limits)
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

    def _run_systemone_batch(
        self,
        state: Any,
        parsed: Mapping[str, Question],
        context: _CallContext,
    ) -> tuple[dict[str, _QuestionOutcome], tuple[dict[str, int | None], int, int]]:
        """Every question in one request; see ``_batch_outcomes`` for what that costs and buys."""
        log = _CallLog()
        question_ids = list(parsed)
        spec = _systemone_spec(state, parsed, question_ids)
        try:
            result = self._call(log, question_ids[0], spec, context)
        except Exception as exc:
            _batch_failure(exc, log, question_ids)
            raise
        return _batch_outcomes(log, result, parsed)

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
                        self._remember_limits(
                            context.model, transport.surface, transport.limits, downgraded
                        )
                        self._switch_surface(context, transport.surface)
                        continue
                    failure = self._call_failure(exc, spec, context, transport, log)
                    # An embedded provider error is the caught exception itself; raising it `from`
                    # itself would print as its own cause.
                    raise failure from (None if failure is exc else exc)
                log.n_retries += 1
                _sleep_before_retry(_retry_delay(self.retry, attempt, exc), self.retry)
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
        if context.transport.surface == "systemone":
            # One request for every question, which is the shape the service is built to be asked:
            # eleven questions came back from a single request in 1.08 s when measured on 2026-09-26.
            # There is no per-question option here on purpose — on this surface the questions travel
            # together because the service evaluates them together, and a caller who wants them apart
            # is asking a different service.
            outcomes, totals = self._run_systemone_batch(state, parsed, context)
            return self._assemble(
                context, parsed, outcomes, time.perf_counter() - start, totals=totals
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
            # Each worker runs the caller's context rather than a fresh one, so a trace or span the
            # caller has open — MLflow's, OpenTelemetry's, or their own — still holds on this thread.
            # A fresh context per task is what makes that safe: one Context cannot be entered twice
            # at once, and the questions run concurrently. Without the hand-off, an enclosing span
            # sees one question land inside it and the rest land as roots of their own, which is
            # worse than either being consistent.
            futures = {
                question_id: executor.submit(
                    contextvars.copy_context().run, self._run, question_id, question, context
                )
                for question_id, question in parsed.items()
            }
            for question_id, future in futures.items():
                try:
                    outcomes[question_id] = future.result()
                except Exception as exc:  # noqa: BLE001 - collected, then re-raised in order
                    if failure is None:
                        failure = exc
                except BaseException:
                    # A KeyboardInterrupt or a SystemExit from a worker is not one question failing:
                    # the caller asked for the work to stop, so the questions still queued are
                    # cancelled and this one is raised now, rather than held until a blocked sibling
                    # finishes and then possibly dropped in favour of an earlier provider error.
                    for pending in futures.values():
                        pending.cancel()
                    raise
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

    def list_models(self) -> list[ModelMetadata]:
        """The models this deployment offers, from the System One endpoint's ``GET /v1/models``.

        The one request on this library's account that is not an evaluation, and it needs no
        ``api=``: the path is the service's, and the client object is the caller's own, so the same
        object that answers a question lists the models it would answer with. The schema read here
        is the one the service's OpenAPI declares — ``{"models": [{"name", "description",
        "release_date"}]}`` — so a gateway that answers the path with its own list is reported as the
        wrong shape rather than half-read.
        """
        return parse_models(self._fetch_models_retrying(lambda: fetch_models(self.client)))


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
                        self._remember_limits(
                            context.model, transport.surface, transport.limits, downgraded
                        )
                        self._switch_surface(context, transport.surface)
                        continue
                    failure = self._call_failure(exc, spec, context, transport, log)
                    # An embedded provider error is the caught exception itself; raising it `from`
                    # itself would print as its own cause.
                    raise failure from (None if failure is exc else exc)
                log.n_retries += 1
                await asyncio.sleep(_retry_delay(self.retry, attempt, exc))
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

    async def _run_systemone_batch(
        self,
        state: Any,
        parsed: Mapping[str, Question],
        context: _CallContext,
    ) -> tuple[dict[str, _QuestionOutcome], tuple[dict[str, int | None], int, int]]:
        """The async twin of the batching path; see ``_batch_outcomes`` for what it costs and buys."""
        log = _CallLog()
        question_ids = list(parsed)
        spec = _systemone_spec(state, parsed, question_ids)
        try:
            result = await self._call(log, question_ids[0], spec, context)
        except Exception as exc:
            _batch_failure(exc, log, question_ids)
            raise
        return _batch_outcomes(log, result, parsed)

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
        if context.transport.surface == "systemone":
            outcomes, totals = await self._run_systemone_batch(state, parsed, context)
            return self._assemble(
                context, parsed, outcomes, time.perf_counter() - start, totals=totals
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
            if isinstance(result, BaseException) and not isinstance(result, Exception):
                # A cancellation or an interrupt that reached a question is not that question's
                # provider failing, and `return_exceptions` hands it back as data. It is raised here
                # instead of being counted as a failure another question already reported — and
                # before `_assemble`, which would otherwise be handed a response missing an answer.
                raise result
            if isinstance(result, BaseException):
                if failure is None:
                    failure = result
            else:
                outcomes[question_id] = result
        if failure is not None:
            raise failure
        return self._assemble(context, parsed, outcomes, time.perf_counter() - start)

    async def _afetch_models_retrying(self, fetch: Callable[[], Any]) -> Any:
        """``_fetch_models_retrying`` for a coroutine request.

        The wait between tries is handed to the executor rather than slept on the loop, so a retry
        does not stall the tasks sharing it.
        """
        attempt = 0
        while True:
            try:
                return await fetch()
            except Exception as exc:  # noqa: BLE001 - re-raised below, carrying the provider's status
                failure = self._models_failure(exc, attempt)
                if failure is not None:
                    raise failure from None
                delay = _retry_delay(self.retry, attempt, exc)
                await asyncio.get_running_loop().run_in_executor(
                    None, _sleep_before_retry, delay, self.retry
                )
                attempt += 1

    async def alist_models(self) -> list[ModelMetadata]:
        """The async twin of ``list_models``; the same request, awaited."""
        return parse_models(await self._afetch_models_retrying(lambda: afetch_models(self.client)))

    async def aclose(self) -> None:
        """The async client holds no resources of its own; the caller owns ``client``."""

    async def __aenter__(self) -> AsyncSystemOneClient:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()
