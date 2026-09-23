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


class _LogprobsUnavailable(LabelReadoutError):
    """The provider cannot supply the logprobs a label readout needs.

    Three ways in: the provider rejects the logprob fields, it returns none at all, or it returns a
    logprob for the answer token with no alternatives — a single logprob is not a distribution.

    ``capability`` marks evidence about the provider itself, which ``method="auto"`` remembers for
    the rest of the client's life. A transient failure on a logprob request raises it with
    ``capability=False``: the question is answered with a logprob-free method, but the next call
    tries logprobs again rather than treating one bad minute as a permanent verdict.
    """

    def __init__(self, message: str, *, capability: bool = True) -> None:
        super().__init__(message)
        self.capability = capability


class MalformedAnswerError(JevperError):
    """The model's answer had an unusable shape after corrective retries."""


class ProviderError(JevperError):
    """A provider call failed after transient retries were exhausted."""

    def __init__(self, message: str, *, attempts: list[dict[str, Any]] | None = None) -> None:
        super().__init__(message)
        self.attempts: list[dict[str, Any]] = attempts or []
