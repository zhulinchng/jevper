"""Surfaces, request building, provider result normalization.

Everything provider-specific lives here: the two builders decide which request fields each surface
gets, and the normalizers turn either provider response object into one ``CallResult``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

from .errors import ClientCapabilityError, ProviderError
from .reasoning import ReasoningConfig, ReasoningContentPart, ReasoningTextPart
from .types import Method

Surface = Literal["chat_completions", "responses"]

JSON_SCHEMA_FORMAT = "json_schema"


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
    kwargs: dict[str, Any] = {"model": model, "messages": spec.messages}
    if spec.logprobs:
        kwargs["logprobs"] = True
        kwargs["top_logprobs"] = spec.top_logprobs
    if spec.json_schema is not None:
        if structured_outputs and limits.structured == "schema":
            kwargs["response_format"] = {
                "type": JSON_SCHEMA_FORMAT,
                "json_schema": {"name": spec.schema_name, "schema": spec.json_schema, "strict": True},
            }
        elif limits.structured != "none":
            kwargs["response_format"] = {"type": "json_object"}
    if spec.reasoning is not None and spec.reasoning.effort is not None and limits.reasoning:
        kwargs["reasoning_effort"] = spec.reasoning.effort
    if spec.prompt_cache_key is not None and limits.cache_key:
        kwargs["prompt_cache_key"] = spec.prompt_cache_key
    if spec.temperature is not None:
        kwargs["temperature"] = spec.temperature
    body = dict(extra_body or {})
    if spec.grammar is not None:
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
    kwargs: dict[str, Any] = {"model": model, "input": spec.messages, "store": False}
    if spec.logprobs:
        kwargs["top_logprobs"] = spec.top_logprobs
    include: list[str] = []
    if spec.logprobs:
        include.append("message.output_text.logprobs")
    if spec.reasoning is not None and limits.include:
        include.append("reasoning.encrypted_content")
    if include:
        kwargs["include"] = include
    if spec.reasoning is not None and limits.reasoning:
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
    if spec.json_schema is not None:
        if structured_outputs and limits.structured == "schema":
            kwargs["text"] = {
                "format": {
                    "type": JSON_SCHEMA_FORMAT,
                    "name": spec.schema_name,
                    "schema": spec.json_schema,
                    "strict": True,
                }
            }
        elif limits.structured != "none":
            kwargs["text"] = {"format": {"type": "json_object"}}
    if spec.prompt_cache_key is not None and limits.cache_key:
        kwargs["prompt_cache_key"] = spec.prompt_cache_key
    if spec.temperature is not None:
        kwargs["temperature"] = spec.temperature
    if extra_body:
        kwargs["extra_body"] = dict(extra_body)
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
        return dump()
    try:
        return dict(vars(obj))
    except TypeError:
        # Objects with __slots__ and plain scalars have no __dict__.
        return {}


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
    if not choices:
        raise _embedded_error(response) or ClientCapabilityError("provider returned no choices")
    choice = choices[0]
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
        reasoning_tokens=_get(details, "reasoning_tokens"),
        cached_tokens=_get(prompt_details, "cached_tokens"),
        stop=_get(choice, "finish_reason"),
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


SurfaceBuilder = Callable[..., dict[str, Any]]
SurfaceNormalizer = Callable[[Any, dict[str, Any]], CallResult]

# surface -> (request builder, response normalizer, dotted path to the create method on the client)
SURFACES: dict[Surface, tuple[SurfaceBuilder, SurfaceNormalizer, str]] = {
    "chat_completions": (build_chat_kwargs, _chat_result, "chat.completions"),
    "responses": (build_responses_kwargs, _responses_result, "responses"),
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
    if api == "responses":
        if not has_responses:
            raise ClientCapabilityError("client has no responses.create; pass api='chat_completions'")
        return "responses"
    if api == "chat_completions":
        if not has_chat:
            raise ClientCapabilityError("client has no chat.completions.create; pass api='responses'")
        return "chat_completions"
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
    raise ClientCapabilityError("client exposes neither responses.create nor chat.completions.create")


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
