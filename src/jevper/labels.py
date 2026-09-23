"""Label allocation and label -> answer-key mapping.

Single-letter labels only: multi-letter labels break first-token logprob readout (the first token of
``"AA"`` is ``"A"``), so every method shares the 26-option cap.
"""

from __future__ import annotations

import string
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from .errors import InvalidQuestionError

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .types import Question

LABELS = tuple(string.ascii_uppercase)
MAX_LABEL_OPTIONS = len(LABELS)


def labels_for(count: int) -> tuple[str, ...]:
    """The first ``count`` labels, or ``InvalidQuestionError`` when they do not exist."""
    if count > MAX_LABEL_OPTIONS:
        raise InvalidQuestionError(
            f"{count} options exceed the {MAX_LABEL_OPTIONS}-label cap; split the question into smaller ones"
        )
    return LABELS[:count]


def label_to_key(question: Question, labels: Sequence[str]) -> dict[str, Any]:
    """Map each label to the value the answer must report for that option.

    ``choice`` -> criteria key in criteria order, ``score`` -> zero-based level index,
    ``noul`` -> ``True`` for the first label and ``False`` for the second.
    """
    if question.type == "choice":
        return {label: key for label, key in zip(labels, question.criteria)}
    if question.type == "score":
        return {label: index for index, label in enumerate(labels)}
    return {labels[0]: True, labels[1]: False}
