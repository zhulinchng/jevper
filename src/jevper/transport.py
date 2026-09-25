"""Surfaces, request building, provider result normalization.

Everything provider-specific lives here: the two builders decide which request fields each surface
gets, and the normalizers turn either provider response object into one ``CallResult``.
"""

from __future__ import annotations

import inspect
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

from .errors import ClientCapabilityError, JevperError, ProviderError
from .reasoning import ReasoningConfig, ReasoningContentPart, ReasoningTextPart
from .types import Method, bounded_text

Surface = Literal["chat_completions", "responses", "messages"]

JSON_SCHEMA_FORMAT = "json_schema"

# The Messages API has no default for ``max_tokens`` — Anthropic, vLLM and SGLang all refuse a
# request without one — while jevper's other two surfaces leave the budget to the server. A
# classification answer is a label or a small object, so this is generous for the answer and small
# enough that a model which ignores the prompt cannot run away; ``extra_body={"max_tokens": n}``
# overrides it. A caller who also sets ``ReasoningConfig(budget_tokens=n)`` gets this much *plus* the
# budget, because Anthropic requires the budget to be strictly below ``max_tokens`` and answers 400
# when it is not — a fixed 1024 would refuse the 1024 the docs call the floor.
DEFAULT_MAX_TOKENS = 1024


@dataclass(frozen=True)
class CallSpec:
    messages: list[dict[str, str]]
    logprobs: bool = False
    top_logprobs: int | None = None
    json_schema: dict[str, Any] | None = None
    schema_name: str | None = None
    grammar: str | None = None
    reasoning: ReasoningConfig | None = None
    temperature: float | None = None
    prompt_cache_key: str | None = None
    """The provider's cache-routing key for this request, or ``None`` to send no key at all."""


@dataclass(frozen=True)
class TokenLogprob:
    token: str
    logprob: float | None
    top_logprobs: tuple[tuple[str, float], ...] = ()
    reported_alternatives: int = 0
    """Entries the provider put in ``top_logprobs``, whether or not they were usable."""


@dataclass(frozen=True)
class CallResult:
    text: str
    token_logprobs: tuple[TokenLogprob, ...]
    reasoning: tuple[ReasoningContentPart, ...]
    surface: Surface
    request: dict[str, Any]
    response: Any
    input_tokens: int | None = None
    output_tokens: int | None = None
    reasoning_tokens: int | None = None
    cached_tokens: int | None = None
    """Prompt tokens the provider read from its cache, when it says. ``None`` is "not reported":
    a server that has prefix caching off reports a plain zero, and that is a different fact."""
    stop: str | None = None
    """Why generation ended: ``finish_reason`` on Chat Completions, the incompleteness reason on the
    Responses surface. A reasoning model can spend the whole budget thinking — vLLM and SGLang then
    report ``status: "incomplete"`` with an empty answer — and the caller deserves to be told that
    rather than left with "no non-whitespace token in the response"."""
    refusal: str | None = None
    """The model's own refusal, when the provider reports one instead of an answer. OpenAI puts a safety
    refusal in a ``refusal`` sibling of ``content`` and leaves ``content`` null, so without this the
    answer reads as "no JSON object in the answer" — true, and useless."""


@dataclass(frozen=True)
class Limits:
    """What a server has been observed to accept, per surface.

    A server that does not implement structured outputs, the reasoning parameters or the Responses
    ``include`` list answers 400 naming the field it refuses. None of those fields is required to
    answer a question — the prompt already asks for one JSON object — so jevper drops the field and
    re-asks, one step down the ladder at a time, and remembers the server's limit the way it remembers
    a missing surface or a withheld distribution.
    """

    structured: Literal["schema", "object", "none"] = "schema"
    """``schema`` sends ``json_schema``, ``object`` sends ``json_object``, ``none`` sends nothing."""
    reasoning: bool = True
    include: bool = True
    cache_key: bool = True
    """Whether the server accepts ``prompt_cache_key``. Like the others, a field jevper may drop."""
    thinking: bool = True
    """Whether the server accepts the Messages API's ``thinking`` field. vLLM's protocol has no such
    field at all, so a request carrying one is refused there and re-asked without it."""
    output_config: bool = True
    """Whether the server accepts the Messages API's ``output_config`` structured-output field. The
    OpenAI surfaces carry their schema in ``response_format``/``text``; Anthropic's native equivalent
    is ``output_config.format``, and a server implementing Messages without it refuses the field."""


def _schema_instruction(spec: CallSpec) -> str:
    schema = json.dumps(spec.json_schema, sort_keys=True, separators=(",", ":"))
    return f"Reply with a single JSON object that validates against this JSON Schema:\n{schema}"


def _schema_in_prompt(messages: list[dict[str, str]], spec: CallSpec) -> list[dict[str, str]]:
    """The schema in the prompt, for a surface or a server that cannot carry one in the request.

    Both OpenAI surfaces can constrain an answer with a schema field, and jevper sends one wherever the
    server accepts it. Where it cannot — the Messages API has no such field at all, and a server that
    refuses the strict schema is answered with a plain JSON object — the prompt is the only place left.
    Without this the structured system prompt's "matches the provided schema exactly" would refer to
    nothing that was ever provided, which is what a 4B model answers ``{"intent": "A"}`` to.

    The schema joins the leading system message rather than becoming a turn of its own, so the
    instruction stays where the other instructions are and the question block keeps its place.
    """
    if spec.json_schema is None:
        return messages
    instruction = _schema_instruction(spec)
    if not messages or messages[0].get("role") != "system":
        return [{"role": "system", "content": instruction}, *messages]
    first, rest = messages[0], messages[1:]
    return [{**first, "content": f"{first.get('content') or ''}\n\n{instruction}"}, *rest]


# Anthropic's structured outputs implement a subset of JSON Schema: their documentation lists
# numerical constraints among the unsupported features, and an unsupported keyword is a ``400``, not
# a warning. Their own SDK helpers answer it the same way — drop the constraint, keep its meaning in
# the description — so the wire schema is one the API accepts and the model is still told the bound.
_UNSUPPORTED_BOUNDS = {
    "minimum": "at least {value}",
    "maximum": "at most {value}",
    "exclusiveMinimum": "greater than {value}",
    "exclusiveMaximum": "less than {value}",
    "multipleOf": "a multiple of {value}",
    "minItems": "at least {value} items",
    "maxItems": "at most {value} items",
    "minLength": "at least {value} characters",
    "maxLength": "at most {value} characters",
    "minProperties": "at least {value} properties",
    "maxProperties": "at most {value} properties",
}


def _api_schema(schema: Any) -> Any:
    """The schema as this API accepts it: every unsupported bound folded into a description.

    jevper's own schemas bound every probability at ``minimum: 0`` and every Noul answer from 0 to 1,
    so this is not a shape the raw schema can skip: sending it untransformed costs a refused request
    on the first schema-bearing call and turns the field off for the rest of the client's life. The
    prompt keeps the full schema — text can say what a constraint says — so nothing is lost.
    """
    if isinstance(schema, Mapping):
        out: dict[str, Any] = {}
        bounds: list[str] = []
        for key, value in schema.items():
            template = _UNSUPPORTED_BOUNDS.get(key)
            if template is not None and isinstance(value, (int, float)) and not isinstance(value, bool):
                bounds.append(template.format(value=value))
            else:
                out[key] = _api_schema(value)
        if bounds:
            note = ", ".join(bounds)
            said = out.get("description")
            out["description"] = f"{said} (must be {note})" if said else f"Must be {note}."
        return out
    if isinstance(schema, list):
        return [_api_schema(item) for item in schema]
    return schema


def _caller_body(extra_body: Mapping[str, Any] | None, limits: Limits) -> dict[str, Any]:
    """The caller's own request fields, minus the capability fields this server has refused.

    The SDK merges ``extra_body`` into the request *after* the typed parameters, so a key the caller
    names there is the value that reaches the wire — including a key jevper would otherwise set. The
    builders therefore leave those fields alone and read the caller's value as the effective one, which
    is also why a capability field the server has refused is removed here: without that, the "re-ask
    without it" the ladder promises would send the same bytes a second time.
    """
    body = dict(extra_body or {})
    if "model" in body:
        # The wire model would then disagree with everything jevper keys state on: the derived
        # prompt-cache key, the absence memory, the surface verdicts, and the model the public
        # response reports. One model per client, named where jevper validates it.
        raise JevperError(
            "extra_body cannot carry 'model': pass model= to the constructor or to system_one(), "
            "where it is validated and used for the cache key and the public response"
        )
    if body.get("stream"):
        # The SDK would merge a stream request over the typed ``stream`` jevper never sets, and the
        # server's event stream would reach a parser built for one whole response. Refused here,
        # where it costs nothing, rather than as an unreadable answer after the call.
        raise JevperError(
            "extra_body cannot ask for a stream: jevper reads the answer from one non-streaming "
            "response, so remove 'stream' (or set it to false)"
        )
    if limits.structured != "schema":
        # The server refused the schema field once, so the caller's format field goes too — whatever it
        # holds, jevper can no longer promise it reaches a server that just rejected the field. The
        # schema itself is not lost: it travels in the prompt from that point on.
        body.pop("response_format", None)
        body.pop("text", None)
    if not limits.reasoning:
        body.pop("reasoning_effort", None)
        body.pop("reasoning", None)
    if not limits.include:
        body.pop("include", None)
    if not limits.cache_key:
        body.pop("prompt_cache_key", None)
    if not limits.output_config:
        body.pop("output_config", None)
    if not limits.thinking:
        body.pop("thinking", None)
    return body


def build_chat_kwargs(
    spec: CallSpec,
    *,
    model: str,
    structured_outputs: bool = True,
    extra_body: Mapping[str, Any] | None = None,
    extra_headers: Mapping[str, str] | None = None,
    limits: Limits | None = None,
) -> dict[str, Any]:
    limits = limits or Limits()
    body = _caller_body(extra_body, limits)
    kwargs: dict[str, Any] = {"model": model, "messages": spec.messages}
    if spec.logprobs:
        # ``logprobs`` and ``top_logprobs`` are two fields, and a caller who names only the first in
        # ``extra_body`` is still asking for a distribution: dropping the alternatives would answer
        # with the sampled token's own logprob and no rivals, which the label readout cannot read.
        # A caller who turned logprobs *off* keeps them off — ``top_logprobs`` without ``logprobs``
        # is a 400 on OpenAI — and a caller who named both keeps their numbers.
        wanted = body.get("logprobs", True)
        if wanted and "logprobs" not in body:
            kwargs["logprobs"] = True
        if wanted and "top_logprobs" not in body:
            kwargs["top_logprobs"] = spec.top_logprobs
    schema_sent = False
    if spec.json_schema is not None:
        if "response_format" in body:
            # The caller named the format, so theirs is the value on the wire and the schema is not in
            # the request at all — which is exactly when it has to travel in the prompt.
            schema_sent = False
        elif structured_outputs and limits.structured == "schema":
            kwargs["response_format"] = {
                "type": JSON_SCHEMA_FORMAT,
                "json_schema": {"name": spec.schema_name, "schema": spec.json_schema, "strict": True},
            }
            schema_sent = True
        elif limits.structured != "none":
            kwargs["response_format"] = {"type": "json_object"}
    if spec.json_schema is not None and not schema_sent:
        # ``json_object`` constrains the answer to be *an* object, not to be *this* object.
        kwargs["messages"] = _schema_in_prompt(spec.messages, spec)
    if (
        spec.reasoning is not None
        and spec.reasoning.effort is not None
        and limits.reasoning
        and "reasoning_effort" not in body
    ):
        kwargs["reasoning_effort"] = spec.reasoning.effort
    if spec.prompt_cache_key is not None and limits.cache_key and "prompt_cache_key" not in body:
        kwargs["prompt_cache_key"] = spec.prompt_cache_key
    if spec.temperature is not None and "temperature" not in body:
        kwargs["temperature"] = spec.temperature
    if spec.grammar is not None and "grammar" not in body:
        body["grammar"] = spec.grammar
    if body:
        kwargs["extra_body"] = body
    if extra_headers:
        kwargs["extra_headers"] = dict(extra_headers)
    return kwargs


def _responses_input(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """The prompt turns as Responses input items, each carrying the item type its spec requires.

    OpenAI accepts the ``{"role": ..., "content": ...}`` shorthand for an input message, but the
    OpenResponses schema discriminates its input union on ``type`` and lists it as required, so a
    server that validates against the spec refuses every turn without it. The type is what the
    shorthand means; naming it costs nothing on OpenAI and is the portable form.
    """
    return [{"type": "message", **turn} for turn in messages]


def build_responses_kwargs(
    spec: CallSpec,
    *,
    model: str,
    structured_outputs: bool = True,
    extra_body: Mapping[str, Any] | None = None,
    extra_headers: Mapping[str, str] | None = None,
    limits: Limits | None = None,
) -> dict[str, Any]:
    limits = limits or Limits()
    body = _caller_body(extra_body, limits)
    # ``logprobs`` is Chat Completions' on/off switch; this surface carries the same request through
    # ``top_logprobs`` plus the ``include`` entry, so the caller's switch is read here and the
    # switch itself is left where the caller put it.
    wanted_logprobs = spec.logprobs and bool(body.get("logprobs", True))
    # A key the caller names in ``extra_body`` is the value that reaches the wire, so jevper does
    # not also set the typed one: two keys for one field make the request recorded in ``debug``
    # differ from the body the provider received.
    kwargs: dict[str, Any] = {}
    if "model" not in body:
        kwargs["model"] = model
    if "input" not in body:
        kwargs["input"] = _responses_input(spec.messages)
    if "store" not in body:
        kwargs["store"] = False
    if wanted_logprobs and "top_logprobs" not in body:
        kwargs["top_logprobs"] = spec.top_logprobs
    include: list[str] = []
    if wanted_logprobs:
        include.append("message.output_text.logprobs")
    if spec.reasoning is not None and limits.include:
        include.append("reasoning.encrypted_content")
    if include and "include" not in body:
        kwargs["include"] = include
    if spec.reasoning is not None and limits.reasoning and "reasoning" not in body:
        reasoning = {
            name: value
            for name, value in (
                ("effort", spec.reasoning.effort),
                ("summary", spec.reasoning.summary),
                ("context", spec.reasoning.context),
            )
            if value is not None
        }
        if reasoning:
            kwargs["reasoning"] = reasoning
    schema_sent = False
    if spec.json_schema is not None:
        if "text" in body:
            # The caller named the format, so theirs is the value on the wire.
            schema_sent = False
        elif structured_outputs and limits.structured == "schema":
            kwargs["text"] = {
                "format": {
                    "type": JSON_SCHEMA_FORMAT,
                    "name": spec.schema_name,
                    "schema": spec.json_schema,
                    "strict": True,
                }
            }
            schema_sent = True
        elif limits.structured != "none":
            kwargs["text"] = {"format": {"type": "json_object"}}
    if spec.json_schema is not None and not schema_sent and "input" not in body:
        # The caller's own ``input`` is authoritative when they named one: rewriting the typed key
        # would not reach the wire, so the schema travels only where jevper still owns the input.
        kwargs["input"] = _responses_input(_schema_in_prompt(spec.messages, spec))
    if spec.prompt_cache_key is not None and limits.cache_key and "prompt_cache_key" not in body:
        kwargs["prompt_cache_key"] = spec.prompt_cache_key
    if spec.temperature is not None and "temperature" not in body:
        kwargs["temperature"] = spec.temperature
    if body:
        kwargs["extra_body"] = body
    if extra_headers:
        kwargs["extra_headers"] = dict(extra_headers)
    return kwargs


def _get(obj: Any, name: str, default: Any = None) -> Any:
    """Attribute or mapping read, so a non-``openai`` client object works unchanged."""
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _as_mapping(obj: Any) -> dict[str, Any]:
    """Best-effort mapping view of a provider object; an unreadable object maps to nothing.

    A ``model_dump`` that answers with something that is not a mapping is one of the unreadable
    objects this promises to map to nothing: returning it would fail later, in a ``.get`` far from
    here, and take an otherwise readable answer down with it. A dump that is a mapping but leaves
    out the fields the object plainly has is read the other way round — a partial dump is a
    serialization quirk, not a missing answer, and the attributes are right there.
    """
    if isinstance(obj, Mapping):
        return dict(obj)
    dump = getattr(obj, "model_dump", None)
    if callable(dump):
        try:
            dumped = dump()
        except Exception:  # noqa: BLE001 - a dump that raises is an unreadable object, not a failure
            dumped = None
        if isinstance(dumped, Mapping) and dumped:
            return dict(dumped)
        attributes: dict[str, Any] = {}
        try:
            attributes = dict(vars(obj))
        except TypeError:
            attributes = {}
        if isinstance(dumped, Mapping):
            attributes = {**attributes, **dumped}
        return attributes
    try:
        return dict(vars(obj))
    except TypeError:
        # Objects with __slots__ and plain scalars have no __dict__.
        return {}


def _usable_choice(choice: Any) -> bool:
    """Whether anything at all can be read from a choice.

    A body whose first choice is null, or an empty object, carries no answer — the provider answered
    ``200`` with a shape it did not fill in. Reading that as a blank answer would hide an embedded
    provider error, which is how OpenRouter reports an overloaded upstream, and would turn a retryable
    failure into a malformed answer after the corrective retries were spent.

    The carrier has to be there, not merely named: a ``message`` that is null, with no legacy
    ``text`` beside it, is the shape a provider sends when it has nothing to say, and reading it as
    an empty answer is a malformed-answer verdict spent on a body that never carried one. A
    ``delta`` is a streaming chunk's carrier and is not this reader's; ``_chat_result`` names that
    mismatch for what it is.
    """
    payload = _as_mapping(choice)
    if not payload:
        return False
    if _get(choice, "message") is not None:
        return True
    return isinstance(_get(choice, "text"), str)


def _chat_reasoning_tokens(usage: Any, details: Any) -> Any:
    """The reasoning-token count, from the nested detail object or the top level of ``usage``.

    OpenAI nests it under ``completion_tokens_details``; SGLang's Chat surface reports it at the top
    level — the recorded body in this repo's fixtures does exactly that — and the count is real either
    way, so a nested absence falls back to the top level instead of reporting "not reported".
    """
    nested = _get(details, "reasoning_tokens")
    if nested is not None:
        return nested
    return _get(usage, "reasoning_tokens")


_REASONING_TEXT_TYPES = ("summary_text", "reasoning_text", "output_text", "input_text", "text")
"""Every part type the two specs put reasoning text in. OpenAI names them ``summary_text`` and
``reasoning_text``; OpenResponses allows the same text under ``text``, ``output_text`` and
``input_text``, and what the caller reads is the text either way."""


def _reasoning_part(obj: Any) -> ReasoningContentPart | None:
    """One reasoning part, or ``None`` when the provider sent something unreadable.

    Reasoning is decoration: a null ``summary``/``content``, a part that is not a reasoning item at
    all, or text that cannot be encoded as UTF-8 must never cost the caller an answer that is
    otherwise right there — nor leave a string in the public response that cannot be serialized.

    The part types are normalized on the way in because the OpenResponses spec spells the same two
    things several ways: a summary item is ``summary_text`` or plain ``text``, and reasoning content
    is ``reasoning_text``, ``output_text`` or ``text``. What the caller reads — a summary text and a
    reasoning text — is the same either way.
    """
    payload = _as_mapping(obj)
    for name in ("summary", "content"):
        if payload.get(name) is None and name in payload:
            payload[name] = []
    _normalize_reasoning_items(payload, "summary", "summary_text", _REASONING_TEXT_TYPES)
    _normalize_reasoning_items(payload, "content", "reasoning_text", _REASONING_TEXT_TYPES)
    try:
        part = ReasoningContentPart.model_validate(payload)
    except ValueError:
        return None
    for group in (part.summary, part.content):
        for item in group:
            if not _is_encodable(item.text):
                return None
    return part


def _is_encodable(text: str) -> bool:
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def _normalize_reasoning_items(
    payload: dict[str, Any], name: str, canonical: str, accepted: tuple[str, ...]
) -> None:
    """Rewrite readable text parts of a reasoning item to the one type the public model validates."""
    items = payload.get(name)
    if not isinstance(items, (list, tuple)):
        return
    normalized: list[Any] = []
    for item in items:
        if isinstance(item, Mapping) and _get(item, "type") in accepted[1:]:
            item = {**item, "type": canonical}
        normalized.append(item)
    payload[name] = normalized


def _reasoning_parts(items: Any) -> tuple[ReasoningContentPart, ...]:
    """The readable reasoning parts of a provider list, skipping everything else."""
    parts = []
    for item in items or ():
        if _get(item, "type") != "reasoning":
            continue
        part = _reasoning_part(item)
        if part is not None:
            parts.append(part)
    return tuple(parts)


def _logprob_token(entry: Any) -> str | None:
    """The token's own text, read from the bytes the provider sent beside it when it sent any.

    OpenAI's Chat Completions and the OpenResponses spec both carry a ``bytes`` array next to every
    logprob token: the exact UTF-8 bytes of that token, whatever spelling the tokenizer uses. A
    server whose tokens are byte-level (``ĠA`` for " A", ``ĊA`` for a newline before A) would have
    its sampled token refused as unreadable and its alternatives dropped as non-labels, so the bytes
    are the authority where they are there and the string is the fallback where they are not.
    """
    raw = _get(entry, "bytes")
    if isinstance(raw, (list, tuple)) and raw and all(
        isinstance(value, int) and not isinstance(value, bool) and 0 <= value < 256 for value in raw
    ):
        return bytes(raw).decode("utf-8", "replace")
    token = _get(entry, "token")
    return None if token is None else str(token)


def _token_logprobs(logprobs: Any) -> tuple[TokenLogprob, ...]:
    entries = _get(logprobs, "content")
    if not entries:
        return ()
    tokens: list[TokenLogprob] = []
    for entry in entries:
        token = _logprob_token(entry)
        if token is None:
            continue
        logprob = _get(entry, "logprob")
        tops: list[tuple[str, float]] = []
        reported = 0
        for top in _get(entry, "top_logprobs") or ():
            reported += 1
            top_token = _logprob_token(top)
            top_logprob = _get(top, "logprob")
            if top_token is None or top_logprob is None:
                continue
            tops.append((str(top_token), float(top_logprob)))
        tokens.append(
            TokenLogprob(
                token=str(token),
                # A missing logprob stays missing: 0.0 would read as certainty.
                logprob=None if logprob is None else float(logprob),
                top_logprobs=tuple(tops),
                reported_alternatives=reported,
            )
        )
    return tuple(tokens)


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, (list, tuple)):
        return "".join(
            str(_get(part, "text") or "") for part in content if _get(part, "type") == "text"
        )
    return ""


def _chat_refusal(message: Any) -> str | None:
    """The model's own refusal, when the provider reports one instead of an answer.

    ``refusal`` is a sibling of ``content`` on an assistant message, and a refusal leaves ``content``
    null. The SDK types it as ``Optional[str]``; a provider that sends something else is ignored, since
    a refusal is a diagnostic and not worth failing a readable answer over.
    """
    refusal = _get(message, "refusal")
    if isinstance(refusal, str) and refusal.strip():
        return refusal.strip()
    return None


def _chat_reasoning(message: Any) -> tuple[ReasoningContentPart, ...]:
    for name in ("reasoning_content", "thinking", "reasoning"):
        value = _get(message, name)
        if isinstance(value, str) and value.strip() and _is_encodable(value):
            return (ReasoningContentPart(content=[ReasoningTextPart(text=value)]),)
        if isinstance(value, (list, tuple)) and value:
            parts = _reasoning_parts(value)
            if parts:
                return parts
    content = _get(message, "content")
    if isinstance(content, (list, tuple)):
        return _reasoning_parts(content)
    return ()


def _embedded_error(response: Any) -> ProviderError | None:
    """The provider's own failure, carried in the body of a ``200``.

    OpenRouter answers an overloaded upstream that way — ``{"id": ..., "error": {"message":
    "Upstream error from Nvidia: Service temporarily overloaded", "code": 503}}``, with no
    ``choices`` at all — so without this the call reads as a surface the client cannot parse. The
    status travels with the error, which keeps a transient upstream failure retryable, and the code
    arrives as a string about as often as a number (OpenRouter sends ``"503"``), which is read the
    same way. An error beside a readable answer is still the error: a body that says both is a body
    whose answer cannot be trusted, and a 200 is not a provider's way of reporting success with an
    error attached.
    """
    error = _get(response, "error")
    if error is None:
        return None
    message = _get(error, "message")
    code = _get(error, "code")
    detail = bounded_text(message if isinstance(message, str) and message else "no message")
    status: int | None = None
    if isinstance(code, int) and not isinstance(code, bool):
        status = code
    elif isinstance(code, str) and code.strip().isdigit():
        status = int(code.strip())
    return ProviderError(f"provider reported an error: {detail}", status_code=status, embedded=True)


def _event_stream_error(text: str) -> ProviderError | None:
    """The provider's own failure carried in an SSE frame, where a response object belongs.

    A server that answers a non-streaming request with ``text/event-stream`` has gone wrong about the
    request, but not always about the answer, and the two streaming dialects report a failure
    differently. OpenResponses sends ``event: error`` with the status and message in the payload;
    OpenAI sends a typed ``response.failed`` whose ``response`` object carries ``status: "failed"``
    and the same ``error``. Both hold the one thing a caller needs — what failed, and with which
    status, which is what decides whether retrying is worth anything — so both are read, and only a
    stream that carries no failure of its own is reported as the protocol mismatch it is.
    """
    for frame in text.split("\n\n"):
        lines = [line for line in frame.splitlines() if line.strip()]
        if not lines:
            continue
        event = ""
        payloads: list[str] = []
        for line in lines:
            name, separator, value = line.partition(":")
            if not separator:
                continue
            value = value.removeprefix(" ")
            if name.strip() == "event":
                event = value.strip()
            elif name.strip() == "data":
                payloads.append(value)
        for payload in payloads:
            if payload.strip() in ("[DONE]", ""):
                continue
            try:
                body = json.loads(payload)
            except ValueError:
                continue
            if not isinstance(body, Mapping):
                continue
            failure = _frame_failure(body, event=event)
            if failure is not None:
                return failure
    return None


def _frame_failure(body: Mapping[str, Any], *, event: str) -> ProviderError | None:
    """The failure one decoded event frame carries, in either dialect, with its status if it had one."""
    # OpenAI: {"type": "response.failed", "response": {"status": "failed", "error": {...}}}.
    # OpenResponses: {"type": "error", "status": 429, "error": {...}}, and the plain proxy shape
    # {"error": {...}} with no type at all.
    candidates = [body]
    nested = _get(body, "response")
    if isinstance(nested, Mapping):
        candidates.append(nested)
    named_error = event == "error" or _get(body, "type") in ("error", "response.failed")
    for candidate in candidates:
        error = _get(candidate, "error")
        if error is None and named_error and _get(candidate, "type") == "error":
            error = candidate
        if error is None:
            continue
        failure = _embedded_error({"error": error})
        if failure is None:
            continue
        for source in candidates:
            status = _get(source, "status") or _get(source, "code")
            if failure.status_code is None and isinstance(status, int) and not isinstance(status, bool):
                failure.status_code = status
        return failure
    return None


def _event_stream_failure(response: str | bytes, surface: str) -> ProviderError:
    """What to raise for a body that is an event stream, error frame or not."""
    text = response.decode("utf-8", "backslashreplace") if isinstance(response, bytes) else response
    failure = _event_stream_error(text)
    if failure is not None:
        return failure
    return ProviderError(
        f"the provider answered the {surface} request with a streaming event stream instead of a "
        "response object; jevper reads the answer from one non-streaming response, so a streaming "
        "route cannot be read here"
    )


def _stop_text(value: Any) -> str | None:
    """A stop reason as a hashable name, whatever the provider sent in the field.

    The documented values are strings, and a provider that sends a list or a dict there is broken —
    but a broken stop reason must not be a ``TypeError`` from a set lookup halfway through reading
    the answer. Naming it as text keeps the failure where it belongs: an incomplete answer, reported
    as one.
    """
    if value is None or isinstance(value, str):
        return value
    return str(value)


def _chat_result(response: Any, request: dict[str, Any]) -> CallResult:
    if isinstance(response, (str, bytes)):
        # A server that streams where a whole response belongs has not answered; an error frame
        # inside that stream is the provider's own failure, and its status is what makes it
        # retryable, so the frame is read before the mismatch is reported.
        raise _event_stream_failure(response, "Chat Completions")
    failure = _embedded_error(response)
    if failure is not None:
        raise failure
    choices = _get(response, "choices") or []
    choice = choices[0] if choices else None
    if not _usable_choice(choice):
        if _get(choice, "delta") is not None:
            raise ProviderError(
                "the provider answered a non-streaming Chat Completions request with a streaming "
                "chunk (a choice carrying 'delta' and no message); jevper reads the whole answer "
                "from one response, so this route cannot be read here"
            )
        raise ClientCapabilityError("provider returned no choices")
    # ``text`` is the legacy completions carrier, and a proxy that answers a chat request with one
    # puts the answer there: the choice is already accepted as usable for carrying it, so read it.
    message = _get(choice, "message")
    if message is None:
        message = {"content": _get(choice, "text")}
    usage = _get(response, "usage")
    details = _get(usage, "completion_tokens_details")
    prompt_details = _get(usage, "prompt_tokens_details")
    finish = _get(choice, "finish_reason")
    # Chat Completions types the field as a required, non-null literal, so a body without one is a
    # generation whose completion the provider never reported — a snapshot, not an answer. Reading
    # the text anyway would report a decision from a body that says it stopped for an unknown
    # reason; the name carries the field the caller can look for.
    stop = "finish_reason_missing" if finish is None else _stop_text(finish)
    return CallResult(
        text=_content_text(_get(message, "content")),
        token_logprobs=_token_logprobs(_get(choice, "logprobs")),
        reasoning=_chat_reasoning(message),
        surface="chat_completions",
        request=request,
        response=response,
        input_tokens=_get(usage, "prompt_tokens"),
        output_tokens=_get(usage, "completion_tokens"),
        reasoning_tokens=_chat_reasoning_tokens(usage, details),
        cached_tokens=_get(prompt_details, "cached_tokens"),
        stop=stop,
        refusal=_chat_refusal(message),
    )


def _message_text(item: Any) -> str:
    """One message item's text, from every content part type the two specs give text on.

    OpenAI names the answer part ``output_text``; OpenResponses also allows ``text`` and
    ``input_text`` in a message's content union, all three carrying the same ``text`` field. Only a
    part that is one of those and whose text is a string counts, so a refusal part and an image
    stay out of the answer, as they do on the other two surfaces.
    """
    chunks: list[str] = []
    for part in _get(item, "content") or []:
        if _get(part, "type") not in ("output_text", "text", "input_text"):
            continue
        text = _get(part, "text")
        if isinstance(text, str):
            chunks.append(text)
    return "".join(chunks)


def _message_logprobs(item: Any) -> Any:
    """The first logprob list on a message item's content parts, where the carrier puts it."""
    for part in _get(item, "content") or []:
        entries = _get(part, "logprobs")
        if entries:
            return entries
    return None


def _answer_messages(response: Any) -> list[Any]:
    """The message items that carry the answer, with OpenResponses' ``phase`` honoured.

    The 2026-04-24 spec lets a model emit several assistant messages in one response, each labelled
    ``commentary`` (intermediate) or ``final_answer``. Reading them all as one answer concatenates
    "here is my thinkingB" with the answer, so when any item is labelled ``final_answer`` only
    those count; with no labels every message counts, as before. A response whose every message is
    commentary has no answer to read, which the empty-answer path then reports.
    """
    messages = [item for item in _get(response, "output") or [] if _get(item, "type") == "message"]
    final = [item for item in messages if _get(item, "phase") == "final_answer"]
    if final:
        return final
    return [item for item in messages if _get(item, "phase") is None]


def _answer_message(response: Any) -> Any:
    """The output item that carries the answer: the last such message with text or logprobs."""
    answer = None
    for item in _answer_messages(response):
        if _message_text(item) or _message_logprobs(item):
            answer = item
    return answer


def _responses_text(response: Any) -> str:
    messages = _answer_messages(response)
    if messages:
        item = _answer_message(response)
        return _message_text(item) if item is not None else ""
    if any(_get(item, "type") == "message" for item in _get(response, "output") or []):
        # Every message is commentary: the response has items, and none of them is the answer. The
        # SDK's convenience property would still join their text, so it is not consulted here.
        return ""
    try:
        text = _get(response, "output_text")
    except Exception:  # noqa: BLE001 - the SDK's convenience property walks output and joins it
        text = None  # ``output`` null or shaped unlike a list; there is nothing else to read
    return text if isinstance(text, str) else ""


def _responses_refusal(response: Any) -> str | None:
    """The model's own refusal, where this surface puts it: a content part of its own type.

    Chat Completions carries a refusal as a sibling of ``content`` and leaves the content null, and the
    Messages API as a ``refusal`` block. The Responses surface has the third shape, and reading only
    ``output_text`` turns a refusal into an empty answer — true, and useless, the same way the other
    two would be without their own reader.
    """
    for item in _get(response, "output") or []:
        if _get(item, "type") != "message":
            continue
        for part in _get(item, "content") or []:
            if _get(part, "type") == "refusal":
                refusal = str(_get(part, "refusal") or "").strip()
                if refusal:
                    return refusal
    return None


def _responses_token_logprobs(response: Any) -> tuple[TokenLogprob, ...]:
    item = _answer_message(response)
    entries = _message_logprobs(item) if item is not None else None
    return _token_logprobs({"content": entries}) if entries else ()


def _responses_reasoning(response: Any) -> tuple[ReasoningContentPart, ...]:
    return _reasoning_parts(_get(response, "output"))


def _responses_result(response: Any, request: dict[str, Any]) -> CallResult:
    if isinstance(response, (str, bytes)):
        # A server that answers a non-streaming request with an event stream (the OpenResponses
        # streaming dialect: ``event: response.completed`` with a ``data:`` line) is not sending a
        # response object at all, and the SDK hands the raw text through. An ``event: error`` frame
        # in that stream is the provider's own failure, status and all, so it is read first.
        raise _event_stream_failure(response, "Responses")
    # An error carried in a 200 wins over anything the body also carries: a response that says both
    # "here is your answer" and "the upstream failed" is not an answer jevper can vouch for.
    failure = _embedded_error(response)
    if failure is not None:
        raise failure
    usage = _get(response, "usage")
    status = _get(response, "status")
    if status is not None and status not in ("completed", "incomplete"):
        # A generation the provider did not finish is a provider failure, not an answer. Reading
        # ``failed`` or ``cancelled`` as an empty answer spends corrective retries re-asking a request
        # that was never going to arrive, and a failed generation that still carries text would be
        # reported as one. The reference adapter raises for any status but ``completed``.
        failure = _get(response, "error") or _get(_get(response, "incomplete_details"), "reason")
        detail = _get(failure, "message") or failure
        raise ProviderError(
            f"the provider reported status={status!r} before the answer was complete"
            + (f": {bounded_text(detail)}" if isinstance(detail, str) and detail.strip() else "")
        )
    details = _get(usage, "output_tokens_details")
    input_details = _get(usage, "input_tokens_details")
    # A Responses call that hit the output budget says so here rather than in a finish_reason.
    stop = None
    if status == "incomplete":
        reason = _get(_get(response, "incomplete_details"), "reason")
        stop = _stop_text(reason) or "incomplete"
    # OpenResponses gives every output item its own lifecycle, and a message item that says it is
    # ``incomplete`` says the model ran out of room mid-answer even when the response around it
    # claims completion. Its partial text is not an answer — a cut-off JSON object that happens to
    # parse is exactly the shape a model produces when the budget ran out mid-write.
    item = _answer_message(response)
    item_status = _stop_text(_get(item, "status")) if item is not None else None
    if item_status is not None and item_status != "completed" and stop is None:
        if item_status == "incomplete":
            reason = _get(_get(response, "incomplete_details"), "reason")
            stop = _stop_text(reason) or "incomplete"
        else:
            failure = _get(response, "error") or _get(item, "error")
            detail = _get(failure, "message") or failure
            raise ProviderError(
                f"the provider left the answer item in status={item_status!r} before the answer "
                "was complete"
                + (f": {bounded_text(detail)}" if isinstance(detail, str) and detail.strip() else "")
            )
    return CallResult(
        text=_responses_text(response),
        token_logprobs=_responses_token_logprobs(response),
        reasoning=_responses_reasoning(response),
        surface="responses",
        request=request,
        response=response,
        input_tokens=_get(usage, "input_tokens"),
        output_tokens=_get(usage, "output_tokens"),
        reasoning_tokens=_get(details, "reasoning_tokens"),
        cached_tokens=_get(input_details, "cached_tokens"),
        stop=stop,
        refusal=_responses_refusal(response),
    )


def _split_system(messages: list[dict[str, str]]) -> tuple[str, list[dict[str, str]]]:
    """The Messages API's top-level ``system``, and the turns that go in ``messages``.

    Anthropic has no ``system`` role inside ``messages``: a server that accepts one renders it
    positionally into the chat template, which llama.cpp's Qwen template refuses outright and vLLM
    and SGLang answer 400 for. jevper's own prompt is always the first turn, and
    ``prompts.hoist_instructions`` folds a state's instruction turns into it, so in practice there is
    exactly one — anything else that calls itself ``system`` is joined to it rather than dropped.
    """
    system: list[str] = []
    turns: list[dict[str, str]] = []
    for message in messages:
        if message.get("role") == "system":
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                system.append(content)
            continue
        turns.append(message)
    return "\n\n".join(system), turns


def build_messages_kwargs(
    spec: CallSpec,
    *,
    model: str,
    structured_outputs: bool = True,
    extra_body: Mapping[str, Any] | None = None,
    extra_headers: Mapping[str, str] | None = None,
    limits: Limits | None = None,
) -> dict[str, Any]:
    """The Anthropic Messages request.

    There is no logprob carrier here at all, so a label readout never reaches this builder. The answer's
    shape has a field of its own — ``output_config.format``, this API's schema-constrained output and the
    counterpart of ``response_format`` on Chat Completions, which the TypeSafe reference adapter sends
    here — and jevper sends it whenever ``structured_outputs`` is on, the server has not refused it, and
    the caller has not named it in ``extra_body``.

    The system prompt carries the schema either way. A server can refuse the field, which the ladder
    catches, or accept it and drop it without a word — vLLM's Messages request model has no such field
    and pydantic discards extras — and a shape that reached neither place is a shape the model never saw.
    """
    limits = limits or Limits()
    system, turns = _split_system(spec.messages)
    body = _caller_body(extra_body, limits)
    native_schema = (
        spec.json_schema is not None
        and structured_outputs
        and limits.output_config
        and "output_config" not in body
    )
    if spec.json_schema is not None:
        instruction = _schema_instruction(spec)
        system = f"{system}\n\n{instruction}" if system else instruction
    budget = spec.reasoning.budget_tokens if spec.reasoning is not None else None
    caller_thinking = body.get("thinking")
    caller_budget = (
        caller_thinking.get("budget_tokens")
        if isinstance(caller_thinking, Mapping) and caller_thinking.get("type") != "disabled"
        else None
    )
    if not isinstance(caller_budget, int) or isinstance(caller_budget, bool):
        caller_budget = None
    thinking = budget is not None and limits.thinking and "thinking" not in body
    # A thinking budget is a thinking budget whoever set it: the caller's own ``thinking`` object in
    # ``extra_body`` carries the same rule Anthropic enforces (budget strictly below max_tokens), so
    # jevper sizes max_tokens for it exactly as it does for a ReasoningConfig — otherwise the request
    # it builds is one the API refuses before a token is generated. A budget the server has already
    # refused is not one: the request carries no thinking block, so nothing has to be made room for.
    sent_budget = max(budget if thinking else 0, caller_budget or 0) or None
    if sent_budget is not None:
        # Anthropic requires the budget to be strictly below ``max_tokens`` and answers 400 otherwise,
        # and jevper owns this default: a fixed 1024 would refuse the 1024 the docs call the floor. The
        # answer keeps the whole default and the thinking is paid for out of the extra.
        default_max_tokens = DEFAULT_MAX_TOKENS + sent_budget
    else:
        default_max_tokens = DEFAULT_MAX_TOKENS
    max_tokens = body.pop("max_tokens", default_max_tokens)
    if sent_budget is not None and isinstance(max_tokens, int) and max_tokens <= sent_budget:
        # Anthropic requires the budget to be strictly below max_tokens, and jevper owns both numbers
        # unless the caller took one over. Spending a request on a request this code can already prove
        # the API will refuse is the one failure mode a local check is strictly better at.
        raise JevperError(
            f"max_tokens={max_tokens} must be greater than the thinking budget_tokens="
            f"{sent_budget}; raise max_tokens (extra_body={{'max_tokens': n}}) or lower "
            "ReasoningConfig(budget_tokens=n) / extra_body['thinking']['budget_tokens']"
        )
    kwargs: dict[str, Any] = {
        "model": model,
        # Required by this API, with no server-side default anywhere that implements it.
        "max_tokens": max_tokens,
        "messages": turns,
    }
    if system:
        kwargs["system"] = system
    if thinking:
        kwargs["thinking"] = {"type": "enabled", "budget_tokens": budget}
    if native_schema:
        # Through ``extra_body`` rather than as a typed keyword: the field postdates the oldest
        # Anthropic SDK jevper supports (``anthropic>=0.49`` has no ``output_config`` parameter and
        # answers ``TypeError: unexpected keyword argument`` before a request is sent), and the SDK
        # merges the body into the same top-level JSON either way.
        body["output_config"] = {
            "format": {"type": JSON_SCHEMA_FORMAT, "schema": _api_schema(spec.json_schema)}
        }
    if spec.temperature is not None and sent_budget is None:
        # Not a typed parameter of the SDK's ``messages.create`` — the newest Claude models refuse a
        # non-default temperature, so the client stopped naming it — but the API itself still accepts
        # one, and every local server implementing this API reads it. Extended thinking is the
        # exception: the API refuses a temperature that is not its default alongside a thinking
        # budget, so a request that enables thinking leaves it out entirely. The caller's own body
        # wins either way.
        body.setdefault("temperature", spec.temperature)
    if body:
        # Whatever is left of the caller's body reaches the request on every path, not only when a
        # temperature happened to be sent: ``extra_body`` is the only way to name a field this
        # builder does not type, and dropping it would silently ignore the caller's request.
        kwargs["extra_body"] = body
    if extra_headers:
        kwargs["extra_headers"] = dict(extra_headers)
    return kwargs


def _messages_text(response: Any) -> str:
    return "".join(
        str(_get(block, "text") or "")
        for block in _get(response, "content") or ()
        if _get(block, "type") == "text"
    )


def _messages_reasoning(response: Any) -> tuple[ReasoningContentPart, ...]:
    """The response's thinking blocks, in the shape the other surfaces' reasoning arrives in.

    ``signature`` and anything else the provider sent is kept in the part's extra fields, so a caller
    replaying a turn into a later request does not lose what the provider wants back. An empty
    ``thinking`` string is not a trace: Anthropic answers that way when the block is deliberately
    omitted, and a part with no text would make ``reasoning_text`` return an empty line.
    """
    parts: list[ReasoningContentPart] = []
    for block in _get(response, "content") or ():
        if _get(block, "type") != "thinking":
            continue
        text = _get(block, "thinking")
        if not isinstance(text, str) or not text.strip():
            continue
        payload = {
            name: value
            for name, value in _as_mapping(block).items()
            if name not in ("type", "thinking")
        }
        payload["type"] = "reasoning"
        payload["content"] = [{"type": "reasoning_text", "text": text}]
        try:
            parts.append(ReasoningContentPart.model_validate(payload))
        except ValueError:
            continue
    return tuple(parts)


def _messages_result(response: Any, request: dict[str, Any]) -> CallResult:
    if isinstance(response, (str, bytes)):
        # The same mismatch the other two surfaces name: an event stream where a whole Message
        # belongs. Reading it as an empty message would report a malformed answer and spend the
        # corrective retries on a protocol problem, and an ``event: error`` frame inside it is the
        # provider's own failure, which is worth reading before either verdict.
        raise _event_stream_failure(response, "Messages")
    failure = _embedded_error(response)
    if failure is not None or _get(response, "type") == "error":
        raise failure or ClientCapabilityError("provider reported an error")
    usage = _get(response, "usage")
    details = _get(usage, "output_tokens_details")
    return CallResult(
        text=_messages_text(response),
        # No server that implements this API returns logprobs through it.
        token_logprobs=(),
        reasoning=_messages_reasoning(response),
        surface="messages",
        request=request,
        response=response,
        input_tokens=_get(usage, "input_tokens"),
        output_tokens=_get(usage, "output_tokens"),
        reasoning_tokens=_get(details, "thinking_tokens"),
        # Anthropic reports what it read from its cache under this name; a server without prompt
        # caching reports nothing at all, which stays ``None`` rather than becoming a zero.
        cached_tokens=_get(usage, "cache_read_input_tokens"),
        stop=_stop_text(_get(response, "stop_reason")),
    )


SurfaceBuilder = Callable[..., dict[str, Any]]
SurfaceNormalizer = Callable[[Any, dict[str, Any]], CallResult]

# surface -> (request builder, response normalizer, dotted path to the create method on the client)
SURFACES: dict[Surface, tuple[SurfaceBuilder, SurfaceNormalizer, str]] = {
    "chat_completions": (build_chat_kwargs, _chat_result, "chat.completions"),
    "responses": (build_responses_kwargs, _responses_result, "responses"),
    "messages": (build_messages_kwargs, _messages_result, "messages"),
}


def _without_sdk_retries(client: Any) -> Any:
    """The client with its own retry loop switched off, where the client can be asked for that.

    The official SDKs retry twice by default and understand nothing of jevper's policy, so a client
    passed unchanged multiplies every attempt: three SDK tries inside each of jevper's three, nine
    requests for one call, with the SDK's retries invisible in ``usage.n_retries``. jevper's loop
    is the one that knows ``x-should-retry``, the attempt history and the backoff the caller chose,
    so the SDK's is turned off on a copy — the caller's own client keeps the setting it was built
    with. The check is on the type, so a mock that fabricates any attribute is left alone.
    """
    with_options = getattr(type(client), "with_options", None)
    if not inspect.isfunction(with_options):
        return client
    try:
        return client.with_options(max_retries=0)
    except Exception:  # noqa: BLE001 - a client that cannot be copied is used as it is
        return client


_SDK_AUTH_HEADERS = {"openai": ("Authorization",), "anthropic": ("x-api-key",)}
"""The credential header each official SDK adds per request, under its own spelling. They are not in
``default_headers`` — the SDKs attach them when the request is built — so a caller's differently
spelled override would otherwise be *added* to the credential rather than replacing it."""


def _merge_headers(extra: Mapping[str, str] | None, client: Any) -> dict[str, str]:
    """The caller's headers under the spelling the client already uses, one value per name.

    The SDKs merge their default headers with the request's case-sensitively and only then hand
    the result to a case-insensitive header mapping, so a caller who spells ``authorization`` where
    the client sends ``Authorization`` puts two credentials on the wire. Renaming the caller's
    header to the spelling the client uses makes it replace the default, which is what overriding a
    header means.
    """
    if not extra:
        return {}
    merged = dict(extra)
    defaults = getattr(client, "default_headers", None)
    canonical = (
        {str(name).lower(): name for name in defaults} if isinstance(defaults, Mapping) else {}
    )
    for cls in type(client).__mro__:  # a subclass of OpenAI is still an OpenAI client
        names = _SDK_AUTH_HEADERS.get(cls.__module__.split(".")[0])
        if names:
            for name in names:
                canonical.setdefault(name.lower(), name)
            break
    for name in list(merged):
        spelling = canonical.get(name.lower())
        if spelling is not None and spelling != name:
            merged[spelling] = merged.pop(name)
    return merged


def _provider_error_from(exc: BaseException) -> BaseException:
    """A strict-validation failure whose body carries the provider's error, as that error.

    The SDKs parse the body before jevper sees it, and a client built with strict response
    validation refuses a body that does not fit the typed model — so a ``200`` whose body is only
    an error arrives as ``APIResponseValidationError`` and the embedded-error reader never runs.
    The body is on the exception; reading the same error from it keeps the provider's message and
    its status, which is what makes an overloaded upstream retryable rather than terminal.
    """
    status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(getattr(exc, "response", None), "status_code", None)
    if status not in (200, None):
        # A real status error already says everything: the SDK has the status, the headers and the
        # body, and jevper's readers work from the exception. Only a successful response whose body
        # failed to parse is a failure the embedded-error reader has not seen yet.
        return exc
    body = getattr(exc, "body", None)
    if isinstance(body, Mapping):
        failure = _embedded_error(body)
        if failure is not None:
            return failure
    return exc


class Transport:
    """One surface: the request builder, the response normalizer and where the call goes.

    The client is validated by ``select_surface`` before a transport is built, so the attribute walk
    in ``_endpoint`` cannot fail for a client that passed selection.
    """

    def __init__(
        self,
        client: Any,
        surface: Surface,
        *,
        structured_outputs: bool = True,
        extra_body: Mapping[str, Any] | None = None,
        extra_headers: Mapping[str, str] | None = None,
        limits: Limits | None = None,
    ) -> None:
        self.client = _without_sdk_retries(client)
        self.surface = surface
        self.build_kwargs, self.normalize, self.endpoint_path = SURFACES[surface]
        self.structured_outputs = structured_outputs
        self.extra_body = extra_body
        self.extra_headers = _merge_headers(extra_headers, client)
        self.limits = limits or Limits()

    def kwargs(self, spec: CallSpec, model: str) -> dict[str, Any]:
        return self.build_kwargs(
            spec,
            model=model,
            structured_outputs=self.structured_outputs,
            extra_body=self.extra_body,
            extra_headers=self.extra_headers,
            limits=self.limits,
        )

    def _endpoint(self) -> Any:
        endpoint = self.client
        for name in self.endpoint_path.split("."):
            endpoint = getattr(endpoint, name)
        return endpoint

    def call(self, spec: CallSpec, model: str) -> CallResult:
        kwargs = self.kwargs(spec, model)
        try:
            response = self._endpoint().create(**kwargs)
        except Exception as exc:  # noqa: BLE001 - re-raised below, carrying the provider's own error
            raise _provider_error_from(exc) from None
        if inspect.isawaitable(response):
            # An async client handed to the blocking facade: no request was made, and the coroutine
            # would otherwise be left un-awaited with a warning nobody asked for.
            close = getattr(response, "close", None)
            if callable(close):
                close()
            raise ClientCapabilityError(
                "this client answers asynchronously; use AsyncSystemOneClient for it"
            )
        return self.normalize(response, kwargs)

    async def acall(self, spec: CallSpec, model: str) -> CallResult:
        kwargs = self.kwargs(spec, model)
        create = self._endpoint().create
        if (
            not inspect.iscoroutinefunction(create)
            and type(self.client).__module__.split(".")[0] in ("openai", "anthropic")
            and not inspect.iscoroutinefunction(getattr(type(self.client), "__aenter__", None))
        ):
            # An official blocking client in the async facade: asking anyway would make a blocking
            # request on the event loop before the mismatch is noticed. Duck clients are left to the
            # check on the result below, because a duck ``create`` may be an ordinary function that
            # returns an awaitable, and a mock of one is neither.
            raise ClientCapabilityError(
                "this client answers synchronously; use SystemOneClient for it"
            )
        try:
            response = create(**kwargs)
        except Exception as exc:  # noqa: BLE001 - re-raised below, carrying the provider's own error
            raise _provider_error_from(exc) from None
        if not inspect.isawaitable(response):
            raise ClientCapabilityError(
                "this client answers synchronously; use SystemOneClient for it"
            )
        try:
            response = await response
        except Exception as exc:  # noqa: BLE001 - re-raised below, carrying the provider's own error
            raise _provider_error_from(exc) from None
        return self.normalize(response, kwargs)


def _has_attribute(client: Any, path: str) -> bool:
    """Whether the client exposes a callable at this dotted path, without letting it raise.

    A property that raises is a client that cannot be used — a closed SDK client, a lazy loader that
    failed to load — and the answer to "can this client do this?" is no. Letting its exception
    escape would hand the caller the SDK's error from a place that promised a capability verdict.
    """
    current = client
    for name in path.split("."):
        try:
            current = getattr(current, name, None)
        except Exception:  # noqa: BLE001 - any failure to reach the attribute means it is not there
            return False
        if current is None:
            return False
    return callable(current)


def select_surface(client: Any, api: str, method: Method) -> Surface:
    has_chat = _has_attribute(client, "chat.completions.create")
    has_responses = _has_attribute(client, "responses.create")
    has_messages = _has_attribute(client, "messages.create")
    if api == "responses":
        if not has_responses:
            raise ClientCapabilityError("client has no responses.create; pass api='chat_completions'")
        return "responses"
    if api == "chat_completions":
        if not has_chat:
            raise ClientCapabilityError("client has no chat.completions.create; pass api='responses'")
        return "chat_completions"
    if api == "messages":
        if not has_messages:
            raise ClientCapabilityError(
                "client has no messages.create; pass an Anthropic-compatible client, or "
                "api='chat_completions'/'responses' for an OpenAI-compatible one"
            )
        return "messages"
    if method == "grammar":
        if not has_chat:
            raise ClientCapabilityError(
                "grammar requires a client with chat.completions.create (llama-cpp-python and similar "
                "OpenAI-compatible servers)"
            )
        return "chat_completions"
    if has_responses:
        return "responses"
    if has_chat:
        return "chat_completions"
    if has_messages:
        # The only surface this client has. No label readout exists on it, which is the caller's
        # method resolution to handle: ``auto`` answers in JSON there, an explicit ``logprobs`` is
        # refused before any request is sent.
        return "messages"
    raise ClientCapabilityError(
        "client exposes none of responses.create, chat.completions.create or messages.create"
    )


def make_transport(
    client: Any,
    surface: Surface,
    *,
    structured_outputs: bool = True,
    extra_body: Mapping[str, Any] | None = None,
    extra_headers: Mapping[str, str] | None = None,
    limits: Limits | None = None,
) -> Transport:
    return Transport(
        client,
        surface,
        structured_outputs=structured_outputs,
        extra_body=extra_body,
        extra_headers=extra_headers,
        limits=limits,
    )
