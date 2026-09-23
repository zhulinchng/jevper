"""Surfaces, request building, provider result normalization.

Everything provider-specific lives here: the two builders decide which request fields each surface
gets, and the normalizers turn either provider response object into one ``CallResult``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

from .errors import ClientCapabilityError
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


@dataclass(frozen=True)
class TokenLogprob:
    token: str
    logprob: float | None
    top_logprobs: tuple[tuple[str, float], ...] = ()


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


def build_chat_kwargs(
    spec: CallSpec,
    *,
    model: str,
    structured_outputs: bool = True,
    extra_body: Mapping[str, Any] | None = None,
    extra_headers: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"model": model, "messages": spec.messages}
    if spec.logprobs:
        kwargs["logprobs"] = True
        kwargs["top_logprobs"] = spec.top_logprobs
    if spec.json_schema is not None:
        if structured_outputs:
            kwargs["response_format"] = {
                "type": JSON_SCHEMA_FORMAT,
                "json_schema": {"name": spec.schema_name, "schema": spec.json_schema, "strict": True},
            }
        else:
            kwargs["response_format"] = {"type": "json_object"}
    if spec.reasoning is not None and spec.reasoning.effort is not None:
        kwargs["reasoning_effort"] = spec.reasoning.effort
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
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"model": model, "input": spec.messages, "store": False}
    if spec.logprobs:
        kwargs["top_logprobs"] = spec.top_logprobs
    include: list[str] = []
    if spec.logprobs:
        include.append("message.output_text.logprobs")
    if spec.reasoning is not None:
        include.append("reasoning.encrypted_content")
    if include:
        kwargs["include"] = include
    if spec.reasoning is not None:
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
        if structured_outputs:
            kwargs["text"] = {
                "format": {
                    "type": JSON_SCHEMA_FORMAT,
                    "name": spec.schema_name,
                    "schema": spec.json_schema,
                    "strict": True,
                }
            }
        else:
            kwargs["text"] = {"format": {"type": "json_object"}}
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
    if isinstance(obj, Mapping):
        return dict(obj)
    dump = getattr(obj, "model_dump", None)
    if callable(dump):
        return dump()
    return dict(vars(obj))


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
        for top in _get(entry, "top_logprobs") or ():
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
            parts = [
                ReasoningContentPart.model_validate(_as_mapping(part))
                for part in value
                if _get(part, "type") == "reasoning"
            ]
            if parts:
                return tuple(parts)
    content = _get(message, "content")
    if isinstance(content, (list, tuple)):
        parts = [
            ReasoningContentPart.model_validate(_as_mapping(part))
            for part in content
            if _get(part, "type") == "reasoning"
        ]
        if parts:
            return tuple(parts)
    return ()


def _chat_result(response: Any, request: dict[str, Any]) -> CallResult:
    choices = _get(response, "choices") or []
    if not choices:
        raise ClientCapabilityError("provider returned no choices")
    choice = choices[0]
    message = _get(choice, "message")
    usage = _get(response, "usage")
    details = _get(usage, "completion_tokens_details")
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
    parts = []
    for item in _get(response, "output") or []:
        if _get(item, "type") == "reasoning":
            parts.append(ReasoningContentPart.model_validate(_as_mapping(item)))
    return tuple(parts)


def _responses_result(response: Any, request: dict[str, Any]) -> CallResult:
    usage = _get(response, "usage")
    details = _get(usage, "output_tokens_details")
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
    ) -> None:
        self.client = client
        self.surface = surface
        self.build_kwargs, self.normalize, self.endpoint_path = SURFACES[surface]
        self.structured_outputs = structured_outputs
        self.extra_body = extra_body
        self.extra_headers = extra_headers

    def kwargs(self, spec: CallSpec, model: str) -> dict[str, Any]:
        return self.build_kwargs(
            spec,
            model=model,
            structured_outputs=self.structured_outputs,
            extra_body=self.extra_body,
            extra_headers=self.extra_headers,
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
) -> Transport:
    return Transport(
        client,
        surface,
        structured_outputs=structured_outputs,
        extra_body=extra_body,
        extra_headers=extra_headers,
    )
