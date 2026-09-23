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
    """The supplied client does not expose the attribute a surface requires.

    Also raised when a chat response carried no choices at all: the surface answered, but nothing in
    it can be read.
    """


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

    ``evidence`` says where the verdict came from. ``"provider"`` is the provider itself rejecting or
    refusing the logprob fields — an explicit statement about what it can do, remembered at once.
    ``"readout"`` is an answer whose logprobs could not be read — one anomalous response says very
    little, so auto waits for a second one before giving up on the provider.
    """

    def __init__(self, message: str, *, capability: bool = True, evidence: str = "provider") -> None:
        super().__init__(message)
        self.capability = capability
        self.evidence = evidence


class MalformedAnswerError(JevperError):
    """The model's answer had an unusable shape after corrective retries."""


class ProviderError(JevperError):
    """A provider call failed after transient retries were exhausted.

    ``status_code`` is the HTTP status the provider reported, when it reported one — including a
    status carried inside the body of a ``200``, which is how OpenRouter reports an upstream failure.
    ``attempts`` holds one record per call attempt, the same records as ``debug["llm_attempts"]``.
    """

    def __init__(
        self,
        message: str,
        *,
        attempts: list[dict[str, Any]] | None = None,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.attempts: list[dict[str, Any]] = attempts or []
        self.status_code = status_code
