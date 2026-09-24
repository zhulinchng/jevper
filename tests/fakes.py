"""Stub HTTP server for the OpenAI chat-completions and responses endpoints.

Tests point a real ``openai.OpenAI`` / ``openai.AsyncOpenAI`` client at this server, so the SDK's own
serialization and parsing path is exercised instead of a hand-rolled client stub. The wrapper is
duck-typed, so a stdlib ``urllib`` client object works too if the openai package is unavailable.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable, Sequence
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from typing import Any

Script = Callable[[dict[str, Any]], "tuple[int, dict[str, Any]]"]


def chat_body(
    *,
    content: str | None,
    logprobs: Sequence[tuple[str, float]] | None = None,
    alternatives: Sequence[tuple[str, float]] | None = None,
    reasoning: str | None = None,
    refusal: str | None = None,
    finish_reason: str = "stop",
    input_tokens: int | None = 10,
    output_tokens: int | None = 3,
    reasoning_tokens: int | None = 0,
    cached_tokens: int | None = None,
) -> dict[str, Any]:
    """A ``chat.completion`` body. ``logprobs`` is the generated token stream for the answer, first
    entry first; the answer token's ``top_logprobs`` is the whole sequence unless ``alternatives``
    says otherwise — a stream that carries reasoning tokens needs the distribution spelled out.

    ``content`` may be ``None``: that is what the API returns when a model refuses (with ``refusal``
    set beside it) or answers with tool calls only.
    """
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if reasoning is not None:
        message["reasoning_content"] = reasoning
    if refusal is not None:
        message["refusal"] = refusal
    choice: dict[str, Any] = {"index": 0, "finish_reason": finish_reason, "message": message}
    if logprobs is not None:
        distribution = logprobs if alternatives is None else alternatives
        choice["logprobs"] = {
            "content": [
                {
                    "token": token,
                    "logprob": logprob,
                    "bytes": list(token.encode()),
                    "top_logprobs": [
                        {"token": name, "logprob": value, "bytes": list(name.encode())}
                        for name, value in distribution
                    ],
                }
                for token, logprob in logprobs
            ]
        }
    usage: dict[str, Any] = {}
    if input_tokens is not None:
        usage["prompt_tokens"] = input_tokens
    if output_tokens is not None:
        usage["completion_tokens"] = output_tokens
    if input_tokens is not None and output_tokens is not None:
        usage["total_tokens"] = input_tokens + output_tokens
    if reasoning_tokens is not None:
        usage["completion_tokens_details"] = {"reasoning_tokens": reasoning_tokens}
    if cached_tokens is not None:
        usage["prompt_tokens_details"] = {"cached_tokens": cached_tokens}
    return {
        "id": "chatcmpl-stub",
        "object": "chat.completion",
        "created": 0,
        "model": "stub",
        "choices": [choice],
        "usage": usage,
    }


def reasoning_item(text: str, *, id: str = "rs_stub", encrypted_content: str = "encrypted") -> dict[str, Any]:
    return {
        "type": "reasoning",
        "id": id,
        "summary": [{"type": "summary_text", "text": text}],
        "content": [],
        "encrypted_content": encrypted_content,
        "status": "completed",
    }


def responses_body(
    *,
    text: str,
    logprobs: Sequence[tuple[str, float]] | None = None,
    reasoning: Sequence[dict[str, Any]] = (),
    input_tokens: int | None = 10,
    output_tokens: int | None = 3,
    reasoning_tokens: int | None = 0,
    cached_tokens: int | None = 0,
    refusal: str | None = None,
) -> dict[str, Any]:
    """A ``response`` body with one assistant message and any reasoning items."""
    output_text: dict[str, Any] = {"type": "output_text", "text": text, "annotations": []}
    content: list[dict[str, Any]] = []
    if refusal is not None:
        # This surface's refusal shape: a content part of its own type, and no output_text at all.
        content.append({"type": "refusal", "refusal": refusal})
    else:
        content.append(output_text)
    if logprobs is not None:
        output_text["logprobs"] = [
            {
                "token": token,
                "logprob": logprob,
                "bytes": list(token.encode()),
                "top_logprobs": [
                    {"token": name, "logprob": value, "bytes": list(name.encode())}
                    for name, value in logprobs
                ],
            }
            for token, logprob in logprobs
        ]
    output = list(reasoning) + [
        {
            "type": "message",
            "id": "msg_stub",
            "role": "assistant",
            "content": content,
        }
    ]
    usage: dict[str, Any] = {}
    if cached_tokens is not None:
        usage["input_tokens_details"] = {"cached_tokens": cached_tokens, "cache_write_tokens": 0}
    if input_tokens is not None:
        usage["input_tokens"] = input_tokens
    if output_tokens is not None:
        usage["output_tokens"] = output_tokens
    if input_tokens is not None and output_tokens is not None:
        usage["total_tokens"] = input_tokens + output_tokens
    if reasoning_tokens is not None:
        usage["output_tokens_details"] = {"reasoning_tokens": reasoning_tokens}
    return {
        "id": "resp_stub",
        "object": "response",
        "created_at": 0,
        "model": "stub",
        "status": "completed",
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "output": output,
        "usage": usage,
    }


def messages_body(
    *,
    text: str,
    thinking: str | None = None,
    signature: str | None = None,
    stop_reason: str | None = "end_turn",
    input_tokens: int | None = 10,
    output_tokens: int | None = 3,
    thinking_tokens: int | None = None,
    cached_tokens: int | None = None,
) -> dict[str, Any]:
    """An Anthropic ``message`` body: a content-block list, thinking before the answer text."""
    content: list[dict[str, Any]] = []
    if thinking is not None:
        block: dict[str, Any] = {"type": "thinking", "thinking": thinking}
        if signature is not None:
            block["signature"] = signature
        content.append(block)
    content.append({"type": "text", "text": text})
    usage: dict[str, Any] = {}
    if input_tokens is not None:
        usage["input_tokens"] = input_tokens
    if output_tokens is not None:
        usage["output_tokens"] = output_tokens
    if thinking_tokens is not None:
        usage["output_tokens_details"] = {"thinking_tokens": thinking_tokens}
    if cached_tokens is not None:
        usage["cache_read_input_tokens"] = cached_tokens
    return {
        "id": "msg_stub",
        "type": "message",
        "role": "assistant",
        "model": "stub",
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": usage,
    }


class StubServer:
    """Threaded HTTP server recording every request body it receives."""

    def __init__(
        self,
        *,
        chat: Script | None = None,
        responses: Script | None = None,
        messages: Script | None = None,
    ) -> None:
        self.chat = chat
        self.responses = responses
        self.messages = messages
        self.requests: list[dict[str, Any]] = []
        self.paths: list[str] = []
        self.headers: list[dict[str, Any]] = []
        """Request headers per call, so a test can prove an ``extra_headers`` value reached the wire."""
        stub = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length) if length else b"{}"
                body = json.loads(raw or b"{}")
                stub.requests.append(body)
                stub.paths.append(self.path)
                stub.headers.append(dict(self.headers))
                if self.path.endswith("/responses"):
                    script = stub.responses
                elif self.path.endswith("/messages"):
                    script = stub.messages
                else:
                    script = stub.chat
                if script is None:
                    status, payload = 404, {"error": {"message": f"no stub script for {self.path}"}}
                else:
                    status, payload = script(body)
                if isinstance(payload, str):
                    # A string payload is sent verbatim: real servers answer with plain text
                    # (ollama's ``404 page not found``) or with a bare JSON string (SGLang's
                    # validation errors), and both must reach the client byte for byte.
                    data = payload.encode()
                    try:
                        json.loads(payload)
                        content_type = "application/json"
                    except ValueError:
                        content_type = "text/plain"
                else:
                    data = json.dumps(payload).encode()
                    content_type = "application/json"
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args: Any) -> None:
                """Silence the default stderr access log."""

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._server.stub = self  # type: ignore[attr-defined]
        self._thread = threading.Thread(
            target=self._server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
        )
        self._thread.start()

    @property
    def base_url(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}/v1"

    def bodies(self, path_suffix: str) -> list[dict[str, Any]]:
        return [body for path, body in zip(self.paths, self.requests) if path.endswith(path_suffix)]

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)

    def __enter__(self) -> StubServer:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


class StatusError(Exception):
    """An ``httpx.HTTPStatusError``-shaped failure: the status lives on the response, not the exception."""

    def __init__(self, status_code: int, message: str | None = None) -> None:
        super().__init__(message or f"server error '{status_code}'")
        self.response = SimpleNamespace(status_code=status_code)


class RaisingClient:
    """A duck-typed client whose chat call always raises the given exception."""

    def __init__(self, exc: Exception) -> None:
        class Completions:
            def create(self, **kwargs: Any) -> Any:
                raise exc

        class Chat:
            completions = Completions()

        self.chat = Chat()


def openai_client(stub: StubServer) -> Any:
    from openai import OpenAI

    return OpenAI(base_url=stub.base_url, api_key="test", max_retries=0, timeout=10)


def async_openai_client(stub: StubServer) -> Any:
    from openai import AsyncOpenAI

    return AsyncOpenAI(base_url=stub.base_url, api_key="test", max_retries=0, timeout=10)


def anthropic_client(stub: StubServer) -> Any:
    """The real SDK, pointed at the stub: it appends ``/v1/messages`` to the base URL itself."""
    from anthropic import Anthropic

    return Anthropic(base_url=stub.base_url, api_key="test", max_retries=0, timeout=10)


def async_anthropic_client(stub: StubServer) -> Any:
    from anthropic import AsyncAnthropic

    return AsyncAnthropic(base_url=stub.base_url, api_key="test", max_retries=0, timeout=10)
