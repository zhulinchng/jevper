"""The service itself: one call per ticket, reports, explanations and a traced async path.

This is the layer a deployment owns. It knows the rubric, the profile and the shape of the
report; it does not know how an answer is elicited, and it does not need to — which is the
claim the rest of this project exists to test.
"""

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from jevper import (
    AsyncSystemOneClient,
    ChoiceAnswer,
    NoulAnswer,
    ScoreAnswer,
    SystemOneClient,
    SystemOneResponse,
    reasoning_text,
)

from . import tickets
from .profiles import ServerProfile, build_anthropic_client, build_openai_client
from .rubrics import build_rubric

__all__ = ["AsyncTriage", "Triage", "TriageReport", "build_client"]


def build_client(
    profile: ServerProfile,
    *,
    messages: bool = False,
    counter: Any | None = None,
    timeout: float = 300.0,
) -> Any:
    """The SDK client for one profile: OpenAI for both OpenAI surfaces, Anthropic for Messages."""
    if messages:
        return build_anthropic_client(profile, counter=counter, timeout=timeout)
    return build_openai_client(profile, counter=counter, timeout=timeout)


def build_async_client(profile: ServerProfile, *, messages: bool = False, timeout: float = 300.0) -> Any:
    """The async SDK client for one profile — the other half of the same two objects.

    The two facades refuse each other's client, so the async service cannot borrow the blocking
    one; it needs its own, built the same way.
    """
    if messages:
        from anthropic import AsyncAnthropic

        return AsyncAnthropic(
            base_url=profile.base_url.removesuffix("/v1"),
            api_key=profile.api_key,
            timeout=timeout,
        )
    from openai import AsyncOpenAI

    return AsyncOpenAI(base_url=profile.base_url, api_key=profile.api_key, timeout=timeout)


@dataclass
class TriageReport:
    """What the service hands its callers: a decision, not a provider object."""

    ticket_id: str
    intent: str | None
    intent_confidence: float | None
    urgency: float | None
    urgency_level: str | None
    needs_human: float | None
    method: str | None
    surface: str | None
    calls: int
    retries: int
    cached_tokens: int | None
    latency: float
    reasoning_chars: int

    @classmethod
    def from_response(cls, response: SystemOneResponse, ticket_id: str = "-") -> "TriageReport":
        intent = response.answers.get("intent")
        urgency = response.answers.get("urgency")
        noul = response.answers.get("needs_human")
        level = None
        if isinstance(urgency, ScoreAnswer) and urgency.legend:
            level = urgency.legend.get(round(urgency.score))
        debug = response.debug
        trace = reasoning_text(response.reasoning)
        return cls(
            ticket_id=ticket_id,
            intent=intent.choice if isinstance(intent, ChoiceAnswer) else None,
            intent_confidence=intent.confidence if isinstance(intent, ChoiceAnswer) else None,
            urgency=urgency.score if isinstance(urgency, ScoreAnswer) else None,
            urgency_level=level,
            needs_human=noul.noul if isinstance(noul, NoulAnswer) else None,
            method=debug.get("method"),
            surface=debug.get("api"),
            calls=response.usage.n_calls,
            retries=response.usage.n_retries,
            cached_tokens=response.usage.cached_tokens,
            latency=round(response.usage.latency, 3),
            reasoning_chars=len(trace),
        )

    def to_json(self) -> str:
        return json.dumps(self.__dict__, sort_keys=True)


@dataclass
class Triage:
    """The blocking service."""

    client: Any
    model: str
    options: dict[str, Any] = field(default_factory=dict)
    rubric: dict[str, Any] = field(default_factory=build_rubric)

    def classify(
        self,
        state: Any,
        *,
        questions: Mapping[str, Any] | None = None,
        **overrides: Any,
    ) -> SystemOneResponse:
        """One ``system_one`` call, with the rubric as the default question set."""
        return SystemOneClient(self.client, model=self.model, **self.options).system_one(
            state=state, questions=dict(questions or self.rubric), **overrides
        )

    def session(self) -> SystemOneClient:
        """A client the caller drives directly — the same object, kept for several calls."""
        return SystemOneClient(self.client, model=self.model, **self.options)


    def triage_all(self, work: Iterable[Mapping[str, Any]], **overrides: Any) -> list[TriageReport]:
        with self.session() as client:
            reports = []
            for item in work:
                response = client.system_one(state=dict(item), questions=self.rubric, **overrides)
                reports.append(TriageReport.from_response(response, str(item.get("id", "-"))))
            return reports

    def explain(self, response: SystemOneResponse) -> dict[str, Any]:
        """The debug record, reduced to what an on-call engineer asks for first."""
        debug = response.debug
        return {
            "method": debug.get("method"),
            "methods": debug.get("methods"),
            "api": debug.get("api"),
            "apis": debug.get("apis"),
            "reasoning_mode": debug.get("reasoning_mode"),
            "server_limits": debug.get("server_limits"),
            "retry_reasons": debug.get("retry_reasons"),
            "labels_missing": debug.get("labels_missing"),
            "attempts": [
                {
                    "question_id": attempt.get("question_id"),
                    "surface": attempt.get("surface"),
                    "error": attempt.get("error"),
                    "readout": (attempt.get("readout") or {}).get("source"),
                }
                for attempt in debug.get("llm_attempts", [])
            ],
        }


@dataclass
class AsyncTriage:
    """The async service: one semaphore, many tickets in flight."""

    client: Any
    model: str
    options: dict[str, Any] = field(default_factory=dict)
    rubric: dict[str, Any] = field(default_factory=build_rubric)

    def session(self) -> AsyncSystemOneClient:
        return AsyncSystemOneClient(self.client, model=self.model, **self.options)

    async def triage_all(
        self,
        work: Iterable[Mapping[str, Any]],
        *,
        questions: Mapping[str, Any] | None = None,
        **overrides: Any,
    ) -> list[TriageReport]:
        import asyncio

        client = self.session()
        async def one(item: Mapping[str, Any]) -> TriageReport:
            response = await client.system_one(
                state=dict(item), questions=dict(questions or self.rubric), **overrides
            )
            return TriageReport.from_response(response, str(item.get("id", "-")))

        return list(await asyncio.gather(*(one(item) for item in work)))


def sample_work() -> tuple[dict, ...]:
    """The tickets the CLI runs by default: a JSON ticket, a long one and a conversation."""
    return (
        dict(tickets.SHORT_TICKET),
        dict(tickets.LONG_TICKET),
        {"id": "INC-1043", "conversation": tickets.CONVERSATION},
    )
