"""jevper — the Jev (System One) interface over any OpenAI-compatible client.

Independent implementation of the documented System One wire format; not affiliated with TypeSafe.
"""

from __future__ import annotations

from .client import AsyncSystemOneClient, Examples, RetryPolicy, SystemOneClient
from .errors import (
    ClientCapabilityError,
    IncompleteAnswerError,
    InvalidQuestionError,
    JevperError,
    LabelReadoutError,
    MalformedAnswerError,
    ModelRefusalError,
    ProviderError,
    UnsupportedMethodError,
)
from .methods import Readout
from .reasoning import (
    ReasoningConfig,
    ReasoningContentPart,
    ReasoningSummaryPart,
    ReasoningTextPart,
    reasoning_text,
)
from .types import (
    Answer,
    Api,
    Choice,
    ChoiceAnswer,
    Example,
    JSONContent,
    Method,
    MethodSelection,
    ModelMetadata,
    Noul,
    NoulAnswer,
    NoulCriteria,
    Question,
    Score,
    ScoreAnswer,
    SystemOneResponse,
    Usage,
)

__version__ = "0.7.4"

__all__ = [
    "Answer",
    "Api",
    "AsyncSystemOneClient",
    "Choice",
    "ChoiceAnswer",
    "ClientCapabilityError",
    "Example",
    "Examples",
    "IncompleteAnswerError",
    "InvalidQuestionError",
    "JSONContent",
    "JevperError",
    "LabelReadoutError",
    "MalformedAnswerError",
    "Method",
    "MethodSelection",
    "ModelMetadata",
    "ModelRefusalError",
    "Noul",
    "NoulAnswer",
    "NoulCriteria",
    "ProviderError",
    "Question",
    "Readout",
    "ReasoningConfig",
    "ReasoningContentPart",
    "ReasoningSummaryPart",
    "ReasoningTextPart",
    "RetryPolicy",
    "Score",
    "ScoreAnswer",
    "SystemOneClient",
    "SystemOneResponse",
    "UnsupportedMethodError",
    "Usage",
    "__version__",
    "reasoning_text",
]
