"""Reasoning configuration and provider-agnostic reasoning content types."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

ReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh", "max"]
ReasoningSummary = Literal["auto", "concise", "detailed"]
ReasoningContext = Literal["auto", "current_turn", "all_turns"]
ReasoningMode = Literal["off", "native", "two_step"]


class ReasoningSummaryPart(BaseModel):
    """One ``summary_text`` item, as returned by the Responses API."""

    model_config = ConfigDict(extra="allow")

    type: Literal["summary_text"] = "summary_text"
    text: str


class ReasoningTextPart(BaseModel):
    """One ``reasoning_text`` item (llama.cpp and other self-hosted servers)."""

    type: Literal["reasoning_text"] = "reasoning_text"
    text: str


class ReasoningContentPart(BaseModel):
    """MLflow-shaped reasoning part.

    ``extra="allow"`` is load-bearing: Responses API reasoning items carry ``id``, ``status`` and
    ``encrypted_content``, which callers replay when resuming a stored=False conversation.
    """

    model_config = ConfigDict(extra="allow")

    type: Literal["reasoning"] = "reasoning"
    summary: list[ReasoningSummaryPart] = Field(default_factory=list)
    content: list[ReasoningTextPart] = Field(default_factory=list)


class ReasoningConfig(BaseModel):
    """Reasoning request. ``mode="auto"`` uses native reasoning on the Responses surface and the
    two-step think-then-classify path everywhere else."""

    model_config = ConfigDict(extra="forbid")

    effort: ReasoningEffort | None = None
    summary: ReasoningSummary | None = None
    context: ReasoningContext | None = None
    mode: Literal["auto", "native", "two_step"] = "auto"
    budget_tokens: int | None = None
    """The thinking budget for the Messages API's ``thinking`` field, where the provider wants one.

    Anthropic's Messages API is the only surface with an explicit budget: it requires at least 1024
    and strictly less than ``max_tokens``, and answers a violation with a 400. ``effort`` is not
    translated into a budget — the mapping between a name and a token count is the caller's, not
    jevper's — so a request without this field sends no ``thinking`` at all and the model's own
    default applies. Chat Completions and Responses ignore it: they carry ``reasoning_effort`` and
    ``reasoning`` instead.
    """

    @field_validator("budget_tokens")
    @classmethod
    def _positive_budget(cls, value: int | None) -> int | None:
        # Anthropic's own floor is 1024 and its ceiling is ``max_tokens``; those are its rules, not
        # every server's, so only the meaningless value is refused here and the provider's own answer
        # travels back if it disagrees.
        if value is not None and value < 1:
            raise ValueError(f"budget_tokens must be a positive number of tokens, got {value}")
        return value


def reasoning_text(parts: Sequence[ReasoningContentPart]) -> str:
    """Joined summary texts, else joined content texts, else the empty string."""
    summaries = [part.text for item in parts for part in item.summary if part.text]
    if summaries:
        return "\n\n".join(summaries)
    contents = [part.text for item in parts for part in item.content if part.text]
    return "\n\n".join(contents)


def resolve_reasoning_mode(config: ReasoningConfig | None, surface: str) -> ReasoningMode:
    if config is None:
        return "off"
    if config.mode == "auto":
        return "native" if surface == "responses" else "two_step"
    return config.mode
