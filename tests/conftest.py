"""Shared fixtures: every stub server started by a test is closed at teardown."""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from fakes import StubServer


@pytest.fixture
def stub_server() -> Any:
    servers: list[StubServer] = []

    def make(
        *,
        chat: Callable[..., Any] | None = None,
        responses: Callable[..., Any] | None = None,
        messages: Callable[..., Any] | None = None,
        systemone: Callable[..., Any] | None = None,
        models: Any = None,
        decide: Callable[..., Any] | None = None,
    ) -> StubServer:
        server = StubServer(
            chat=chat,
            responses=responses,
            messages=messages,
            systemone=systemone,
            models=models,
            decide=decide,
        )
        servers.append(server)
        return server

    yield make
    for server in servers:
        server.close()
