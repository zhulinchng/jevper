"""Surfaces, request building, provider result normalization.

Everything provider-specific lives here: the two builders decide which request fields each surface
gets, and the normalizers turn either provider response object into one ``CallResult``.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

from .errors import ClientCapabilityError, ProviderError
from .reasoning import ReasoningConfig, ReasoningContentPart, ReasoningTextPart
from .types import Method

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


def _caller_body(extra_body: Mapping[str, Any] | None, limits: Limits) -> dict[str, Any]:
    """The caller's own request fields, minus the capability fields this server has refused.

    The SDK merges ``extra_body`` into the request *after* the typed parameters, so a key the caller
    names there is the value that reaches the wire — including a key jevper would otherwise set. The
    builders therefore leave those fields alone and read the caller's value as the effective one, which
    is also why a capability field the server has refused is removed here: without that, the "re-ask
    without it" the ladder promises would send the same bytes a second time.
    """
    body = dict(extra_body or {})
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
    if spec.logprobs and "logprobs" not in body:
        kwargs["logprobs"] = True
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
    kwargs: dict[str, Any] = {"model": model, "input": spec.messages, "store": False}
    if spec.logprobs:
        kwargs["top_logprobs"] = spec.top_logprobs
    include: list[str] = []
    if spec.logprobs:
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
    if spec.json_schema is not None and not schema_sent:
        kwargs["input"] = _schema_in_prompt(spec.messages, spec)
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
    """Best-effort mapping view of a provider object; an unreadable object maps to nothing."""
    if isinstance(obj, Mapping):
        return dict(obj)
    dump = getattr(obj, "model_dump", None)
    if callable(dump):
        # A ``model_dump`` that answers with something that is not a mapping is one of the unreadable
        # objects this promises to map to nothing: returning it would fail later, in a ``.get`` far from
        # here, and take an otherwise readable answer down with it.
        dumped = dump()
        return dict(dumped) if isinstance(dumped, Mapping) else {}
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
    """
    payload = _as_mapping(choice)
    return bool(payload) and any(name in payload for name in ("message", "text", "delta"))


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


def _reasoning_part(obj: Any) -> ReasoningContentPart | None:
    """One reasoning part, or ``None`` when the provider sent something unreadable.

    Reasoning is decoration: a null ``summary``/``content``, or a part that is not a reasoning item at
    all, must never cost the caller an answer that is otherwise right there.
    """
    payload = _as_mapping(obj)
    for name in ("summary", "content"):
        if payload.get(name) is None and name in payload:
            payload[name] = []
    try:
        return ReasoningContentPart.model_validate(payload)
    except ValueError:
        return None


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


def _token_logprobs(logprobs: Any) -> tuple[TokenLogprob, ...]:
    entries = _get(logprobs, "content")
    if not entries:
        return ()
    tokens: list[TokenLogprob] = []
    for entry in entries:
        token = _get(entry, "token")
        if token is None:
            continue
        logprob = _get(entry, "logprob")
        tops: list[tuple[str, float]] = []
        reported = 0
        for top in _get(entry, "top_logprobs") or ():
            reported += 1
            top_token = _get(top, "token")
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
        if isinstance(value, str) and value.strip():
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
    status travels with the error, which keeps a transient upstream failure retryable.
    """
    error = _get(response, "error")
    if error is None:
        return None
    message = _get(error, "message")
    code = _get(error, "code")
    detail = message if isinstance(message, str) and message else "no message"
    status = code if isinstance(code, int) and not isinstance(code, bool) else None
    return ProviderError(f"provider reported an error: {detail}", status_code=status)


def _chat_result(response: Any, request: dict[str, Any]) -> CallResult:
    choices = _get(response, "choices") or []
    choice = choices[0] if choices else None
    if not _usable_choice(choice):
        raise _embedded_error(response) or ClientCapabilityError("provider returned no choices")
    message = _get(choice, "message")
    usage = _get(response, "usage")
    details = _get(usage, "completion_tokens_details")
    prompt_details = _get(usage, "prompt_tokens_details")
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
        stop=_get(choice, "finish_reason"),
        refusal=_chat_refusal(message),
    )


def _responses_text(response: Any) -> str:
    text = _get(response, "output_text")
    if isinstance(text, str) and text:
        return text
    chunks: list[str] = []
    for item in _get(response, "output") or []:
        if _get(item, "type") != "message":
            continue
        for part in _get(item, "content") or []:
            if _get(part, "type") == "output_text":
                chunks.append(str(_get(part, "text") or ""))
    return "".join(chunks)


def _responses_token_logprobs(response: Any) -> tuple[TokenLogprob, ...]:
    for item in _get(response, "output") or []:
        if _get(item, "type") != "message":
            continue
        for part in _get(item, "content") or []:
            if _get(part, "type") != "output_text":
                continue
            entries = _get(part, "logprobs")
            if entries:
                return _token_logprobs({"content": entries})
    return ()


def _responses_reasoning(response: Any) -> tuple[ReasoningContentPart, ...]:
    return _reasoning_parts(_get(response, "output"))


def _responses_result(response: Any, request: dict[str, Any]) -> CallResult:
    if not (_get(response, "output") or []):
        failure = _embedded_error(response)
        if failure is not None:
            raise failure
    usage = _get(response, "usage")
    details = _get(usage, "output_tokens_details")
    input_details = _get(usage, "input_tokens_details")
    # A Responses call that hit the output budget says so here rather than in a finish_reason.
    stop = None
    if _get(response, "status") == "incomplete":
        stop = _get(_get(response, "incomplete_details"), "reason") or "incomplete"
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

    Two fields of the other surfaces have no equivalent here, so neither is ever sent: there is no
    logprob carrier at all, and no schema field — so the JSON Schema travels in the system prompt
    instead, which is the only place this request can state the answer's shape. ``structured_outputs``
    is therefore accepted and unused, so that every surface builder keeps one signature.
    """
    limits = limits or Limits()
    system, turns = _split_system(spec.messages)
    if spec.json_schema is not None:
        # This API has no schema field at all, so the shape has to travel in the top-level system
        # prompt — the only place a request on this surface can state the answer's format.
        instruction = _schema_instruction(spec)
        system = f"{system}\n\n{instruction}" if system else instruction
    body = _caller_body(extra_body, limits)
    budget = spec.reasoning.budget_tokens if spec.reasoning is not None else None
    thinking = budget is not None and limits.thinking and "thinking" not in body
    if thinking:
        # Anthropic requires the budget to be strictly below ``max_tokens`` and answers 400 otherwise,
        # and jevper owns this default: a fixed 1024 would refuse the 1024 the docs call the floor. The
        # answer keeps the whole default and the thinking is paid for out of the extra.
        default_max_tokens = DEFAULT_MAX_TOKENS + (budget or 0)
    else:
        default_max_tokens = DEFAULT_MAX_TOKENS
    kwargs: dict[str, Any] = {
        "model": model,
        # Required by this API, with no server-side default anywhere that implements it.
        "max_tokens": body.pop("max_tokens", default_max_tokens),
        "messages": turns,
    }
    if system:
        kwargs["system"] = system
    if thinking:
        kwargs["thinking"] = {"type": "enabled", "budget_tokens": budget}
    if spec.temperature is not None:
        # Not a typed parameter of the SDK's ``messages.create`` — the newest Claude models refuse a
        # non-default temperature, so the client stopped naming it — but the API itself still accepts
        # one, and every local server implementing this API reads it. The caller's own body wins.
        body.setdefault("temperature", spec.temperature)
    if body:
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
    if _get(response, "type") == "error":
        raise _embedded_error(response) or ClientCapabilityError("provider reported an error")
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
        stop=_get(response, "stop_reason"),
    )


SurfaceBuilder = Callable[..., dict[str, Any]]
SurfaceNormalizer = Callable[[Any, dict[str, Any]], CallResult]

# surface -> (request builder, response normalizer, dotted path to the create method on the client)
SURFACES: dict[Surface, tuple[SurfaceBuilder, SurfaceNormalizer, str]] = {
    "chat_completions": (build_chat_kwargs, _chat_result, "chat.completions"),
    "responses": (build_responses_kwargs, _responses_result, "responses"),
    "messages": (build_messages_kwargs, _messages_result, "messages"),
}


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
        self.client = client
        self.surface = surface
        self.build_kwargs, self.normalize, self.endpoint_path = SURFACES[surface]
        self.structured_outputs = structured_outputs
        self.extra_body = extra_body
        self.extra_headers = extra_headers
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
        return self.normalize(self._endpoint().create(**kwargs), kwargs)

    async def acall(self, spec: CallSpec, model: str) -> CallResult:
        kwargs = self.kwargs(spec, model)
        return self.normalize(await self._endpoint().create(**kwargs), kwargs)


def _has_attribute(client: Any, path: str) -> bool:
    current = client
    for name in path.split("."):
        current = getattr(current, name, None)
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
