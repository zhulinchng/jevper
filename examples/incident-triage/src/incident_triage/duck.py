"""A hand-rolled client: jevper's headline claim, used for real.

Nothing here imports the OpenAI or Anthropic SDK. The object exposes the attributes the
library asks for — ``chat.completions.create``, ``responses.create``,
``messages.create`` — and answers with the provider's own JSON, which the library reads
because it reads attributes *or* mapping keys. The counter is the point of the exercise:
a consumer can see exactly what went on the wire, which the SDKs will not tell them.
"""

from __future__ import annotations

import json
import time
from typing import Any

__all__ = ["DuckClient", "SlowDuckClient"]


class _Namespace:
    """Attribute access over parsed JSON, for the few places a dotted path reads better."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def __getattr__(self, name: str) -> Any:
        try:
            return self._payload[name]
        except KeyError:
            raise AttributeError(name) from None

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<body {sorted(self._payload)}>"


class _Completions:
    def __init__(self, client: DuckClient) -> None:
        self._client = client

    def create(self, **kwargs: Any) -> dict[str, Any]:
        return self._client.request("chat/completions", kwargs)


class _Chat:
    def __init__(self, client: DuckClient) -> None:
        self.completions = _Completions(client)


class _Responses:
    def __init__(self, client: DuckClient) -> None:
        self._client = client

    def create(self, **kwargs: Any) -> dict[str, Any]:
        return self._client.request("responses", kwargs)


class _Messages:
    def __init__(self, client: DuckClient) -> None:
        self._client = client

    def create(self, **kwargs: Any) -> dict[str, Any]:
        return self._client.request("messages", kwargs)


class DuckClient:
    """A minimal OpenAI-compatible client over ``httpx``, with no SDK underneath."""

    def __init__(self, base_url: str, *, api_key: str = "local", timeout: float = 300.0, default_headers: dict[str, str] | None = None) -> None:
        import httpx

        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.default_headers = dict(default_headers or {})
        self._http = httpx.Client(timeout=timeout)
        self.chat = _Chat(self)
        self.responses = _Responses(self)
        self.messages = _Messages(self)
        self.requests: list[dict[str, Any]] = []
        self.extra_body: dict[str, Any] = {}

    @property
    def calls(self) -> int:
        return len(self.requests)

    def request(self, route: str, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Send one request, with the SDKs' own ``extra_body`` merge rule applied."""
        body = {key: value for key, value in kwargs.items() if key not in ("extra_body", "extra_headers")}
        body.update(kwargs.get("extra_body") or {})
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            **self.default_headers,
            **(kwargs.get("extra_headers") or {}),
        }
        self.requests.append({"route": route, "body": body, "headers": headers})
        if self.extra_body.get("__delay__"):
            time.sleep(float(self.extra_body["__delay__"]))
        response = self._http.post(f"{self.base_url}/{route}", json=body, headers=headers)
        if response.status_code >= 400:
            # The shape the official SDKs raise, so jevper's error handling is exercised the
            # same way it is with them.
            try:
                parsed = response.json()
            except ValueError:
                parsed = {"message": response.text}
            raise RuntimeError(f"HTTP {response.status_code}: {json.dumps(parsed)[:400]}")
        return response.json()

    def close(self) -> None:
        self._http.close()


class SlowDuckClient(DuckClient):
    """A client that answers after a delay, for the cancellation and concurrency paths.

    A real service cancels work when a request is withdrawn; the only honest way to test that
    is against a call that is still in flight, and the wait has to be somewhere the test
    controls.
    """

    def __init__(self, *args: Any, delay: float = 5.0, answer: dict[str, Any] | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.delay = delay
        self.answer = answer or {
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": '{"probabilities": {"billing": 1.0}}'},
                    "logprobs": None,
                }
            ]
        }

    def request(self, route: str, kwargs: dict[str, Any]) -> dict[str, Any]:
        self.requests.append({"route": route, "body": kwargs.get("extra_body", kwargs), "headers": {}})
        time.sleep(self.delay)
        return self.answer


class SpinDuckClient(SlowDuckClient):
    """A slow client whose wait is interruptible.

    ``time.sleep`` is a C call, so an exception delivered to a thread blocked in one waits for
    it to return — which would make a cancellation test measure the sleep instead of the
    cancellation. This one spends its delay in Python, so the signal lands within milliseconds
    of the moment it is sent, which is what a real in-flight request does.
    """

    def request(self, route: str, kwargs: dict[str, Any]) -> dict[str, Any]:
        self.requests.append({"route": route, "body": kwargs.get("extra_body", kwargs), "headers": {}})
        deadline = time.monotonic() + self.delay
        while time.monotonic() < deadline:
            time.sleep(0.005)
        return self.answer
