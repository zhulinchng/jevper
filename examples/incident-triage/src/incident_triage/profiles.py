"""Server profiles and the clients the service talks to them with.

A profile is what a deployment would configure: the base URL, the model id and the request
fields that make the served model behave like a classifier. The defaults are the recipes
measured for the five local servers; every one of them can be overridden from the command
line, because a profile is configuration, not code.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

__all__ = ["PROFILES", "ServerProfile", "build_anthropic_client", "build_openai_client", "profile_for"]

# The field each server reads to turn thinking off, and what the Responses route reads instead
# where the two disagree. Empty means the served model needs nothing (LM Studio: a
# non-thinking model is the switch).
_CHAT_TEMPLATE_OFF: dict[str, Any] = {"chat_template_kwargs": {"enable_thinking": False}}


@dataclass(frozen=True)
class ServerProfile:
    """Everything a deployment needs to point the service at one server."""

    name: str
    base_url: str
    model: str
    extra_body: dict[str, Any] = field(default_factory=dict)
    responses_extra_body: dict[str, Any] = field(default_factory=dict)
    api_key: str = "local"
    surfaces: tuple[str, ...] = ("responses", "chat_completions", "messages")
    messages_max_tokens: int = 1024
    """The output budget for the Messages route, where jevper's 1024 default is the server's to
    spend. A thinking model spends it on the trace before the answer, so a server that serves
    one on this route needs more; the default matches what the API asks for."""
    responses_json_schema: bool = True
    """Whether the server enforces ``text.format`` on ``/v1/responses``. llama.cpp's converter never
    reads the field and LM Studio accepts and ignores it, so a JSON-schema answer there comes back
    in whatever shape the prompt implies — a server limit the service routes around with
    ``api="chat_completions"``, not a library failure."""
    messages_thinks: bool = False
    """Whether the server's Messages route runs a thinking model that ignores the OpenAI-style
    thinking-off field. Where it does, the trace is paid for out of the output budget and a long
    prompt can spend all of it before the answer starts — reported as a spent budget, which is
    what happened, with the knob that opens it named."""
    notes: str = ""

    def body_for(self, api: str) -> dict[str, Any]:
        """The caller's own request fields for one surface."""
        if api == "responses" and self.responses_extra_body:
            return dict(self.responses_extra_body)
        body = dict(self.extra_body)
        if api == "messages":
            body.setdefault("max_tokens", self.messages_max_tokens)
        return body


PROFILES: dict[str, ServerProfile] = {
    "ollama": ServerProfile(
        name="ollama",
        base_url="http://127.0.0.1:11434/v1",
        model="qwen3.5:9b",
        extra_body={"reasoning_effort": "none"},
        responses_extra_body={"reasoning": {"effort": "none"}},
        messages_max_tokens=2048,
        messages_thinks=True,
        notes="Chat Completions carries logprobs; the Responses route returns an empty logprob list. "
        "The Messages route runs a thinking model that ignores reasoning_effort, so its trace is paid "
        "for out of the output budget.",
    ),
    "llamacpp": ServerProfile(
        name="llamacpp",
        base_url="http://127.0.0.1:8080/v1",
        model="qwen3.5-9b",
        extra_body=dict(_CHAT_TEMPLATE_OFF),
        responses_json_schema=False,
        notes="The one server that takes jevper's GBNF grammar field, and the one whose Responses "
        "route ignores text.format.",
    ),
    "vllm": ServerProfile(
        name="vllm",
        base_url="http://127.0.0.1:8000/v1",
        model="qwen3-4b-instruct",
        extra_body=dict(_CHAT_TEMPLATE_OFF),
        notes="top_logprobs is capped by --max-logprobs; served with --reasoning-parser qwen3.",
    ),
    "sglang": ServerProfile(
        name="sglang",
        base_url="http://127.0.0.1:30000/v1",
        model="qwen3.5-9b",
        extra_body=dict(_CHAT_TEMPLATE_OFF),
        messages_thinks=True,
        notes="Its Responses route needs top_logprobs sent explicitly; with a reasoning parser its "
        "Messages route runs the trace into the output budget, and its context is 4096.",
    ),
    "lmstudio": ServerProfile(
        name="lmstudio",
        base_url="http://127.0.0.1:1234/v1",
        model="qwen3-4b-instruct-2507",
        extra_body=dict(_CHAT_TEMPLATE_OFF),
        responses_json_schema=False,
        notes="Non-thinking model; text.format is ignored on the Responses route.",
    ),
}


def profile_for(name: str, **overrides: Any) -> ServerProfile:
    """The named profile, with any field overridden from configuration."""
    base = PROFILES.get(name)
    if base is None:
        raise SystemExit(f"unknown server {name!r}; known: {', '.join(sorted(PROFILES))}")
    if not overrides:
        return base
    fields = {
        "base_url": base.base_url,
        "model": base.model,
        "extra_body": dict(base.extra_body),
        "responses_extra_body": dict(base.responses_extra_body),
        "api_key": base.api_key,
        "surfaces": base.surfaces,
        "messages_max_tokens": base.messages_max_tokens,
        "notes": base.notes,
    }
    unknown = set(overrides) - set(fields)
    if unknown:
        raise SystemExit(f"unknown profile field(s): {', '.join(sorted(unknown))}")
    fields.update(overrides)
    if isinstance(fields["extra_body"], str):
        fields["extra_body"] = json.loads(fields["extra_body"])
    if isinstance(fields["responses_extra_body"], str):
        fields["responses_extra_body"] = json.loads(fields["responses_extra_body"])
    return ServerProfile(name=base.name, **fields)


def _http_module():
    """The HTTP library the installed SDKs speak.

    ``openai`` 3.19 and ``anthropic`` 1.8 are built on ``httpx2`` and refuse a client object from
    the ``httpx`` package by name, so the counter has to come from whichever one is installed —
    the same reason a deployment cannot hard-code its own transport.
    """
    try:
        import httpx2

        return httpx2
    except ImportError:
        import httpx

        return httpx


def _counting_http_client(counter: RequestCounter, timeout: float):
    """An HTTP client that counts the requests that actually reach the network.

    Counting at the HTTP layer is the only honest place: a consumer's bill is the wire, not
    the calls its client object was handed.
    """
    return _http_module().Client(timeout=timeout, event_hooks={"request": [counter.record]})


def build_openai_client(profile: ServerProfile, *, counter: RequestCounter | None = None, timeout: float = 300.0):
    """The OpenAI SDK client, pointed at the profile's server.

    The SDK's own retry loop is left at its default on purpose: the library turns it off on a
    copy, and the request counter is how this project checks that it did.
    """
    from openai import OpenAI

    kwargs: dict[str, Any] = {"base_url": profile.base_url, "api_key": profile.api_key, "timeout": timeout}
    if counter is not None:
        kwargs["http_client"] = _counting_http_client(counter, timeout)
    return OpenAI(**kwargs)


def build_anthropic_client(profile: ServerProfile, *, counter: RequestCounter | None = None, timeout: float = 300.0):
    """The Anthropic SDK client for the Messages surface, pointed at the same server.

    The SDK appends its own ``/v1/messages``, so an OpenAI-style base URL loses its ``/v1`` here:
    the same server, at the path the Messages API actually lives on.
    """
    from anthropic import Anthropic

    kwargs: dict[str, Any] = {
        "base_url": profile.base_url.removesuffix("/v1"),
        "api_key": profile.api_key,
        "timeout": timeout,
    }
    if counter is not None:
        kwargs["http_client"] = _counting_http_client(counter, timeout)
    return Anthropic(**kwargs)


class RequestCounter:
    """Counts requests on the wire and keeps the last few for the record."""

    def __init__(self) -> None:
        self.total = 0
        self.paths: list[str] = []

    def record(self, request: Any) -> None:
        self.total += 1
        self.paths.append(str(getattr(request, "url", request)))

    def reset(self) -> None:
        self.total = 0
        self.paths.clear()
