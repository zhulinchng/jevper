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
    content: str,
    logprobs: Sequence[tuple[str, float]] | None = None,
    alternatives: Sequence[tuple[str, float]] | None = None,
    reasoning: str | None = None,
    input_tokens: int | None = 10,
    output_tokens: int | None = 3,
    reasoning_tokens: int | None = 0,
) -> dict[str, Any]:
    """A ``chat.completion`` body. ``logprobs`` is the generated token stream for the answer, first
    entry first; the answer token's ``top_logprobs`` is the whole sequence unless ``alternatives``
    says otherwise — a stream that carries reasoning tokens needs the distribution spelled out."""
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if reasoning is not None:
        message["reasoning_content"] = reasoning
    choice: dict[str, Any] = {"index": 0, "finish_reason": "stop", "message": message}
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
) -> dict[str, Any]:
    """A ``response`` body with one assistant message and any reasoning items."""
    output_text: dict[str, Any] = {"type": "output_text", "text": text, "annotations": []}
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
            "status": "completed",
            "content": [output_text],
        }
    ]
    usage: dict[str, Any] = {
        "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
    }
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


class StubServer:
    """Threaded HTTP server recording every request body it receives."""

    def __init__(self, *, chat: Script | None = None, responses: Script | None = None) -> None:
        self.chat = chat
        self.responses = responses
        self.requests: list[dict[str, Any]] = []
        self.paths: list[str] = []
        stub = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length) if length else b"{}"
                body = json.loads(raw or b"{}")
                stub.requests.append(body)
                stub.paths.append(self.path)
                script = stub.responses if self.path.endswith("/responses") else stub.chat
                if script is None:
                    status, payload = 404, {"error": {"message": f"no stub script for {self.path}"}}
                else:
                    status, payload = script(body)
                data = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
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

    def __init__(self, status_code: int) -> None:
        super().__init__(f"server error '{status_code}'")
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
