"""Exception hierarchy for jevper."""

from __future__ import annotations

from typing import Any


class JevperError(Exception):
    """Base class for every error raised by jevper."""


class InvalidQuestionError(JevperError):
    """A question (or one of its few-shot examples) is locally invalid."""


class UnsupportedMethodError(JevperError):
    """The requested method is not available for the selected surface or client."""


class ClientCapabilityError(JevperError):
    """The supplied client does not expose the attribute a surface requires."""


class LabelReadoutError(JevperError):
    """The first answer token was not a label, or no logprobs were returned."""


class MalformedAnswerError(JevperError):
    """The model's answer had an unusable shape after corrective retries."""


class ProviderError(JevperError):
    """A provider call failed after transient retries were exhausted."""

    def __init__(self, message: str, *, attempts: list[dict[str, Any]] | None = None) -> None:
        super().__init__(message)
        self.attempts: list[dict[str, Any]] = attempts or []
