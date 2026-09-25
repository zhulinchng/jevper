"""Label allocation and label -> answer-key mapping.

Labels are single letters ``A``..``Z`` for up to 26 options; past that they become two letters
(``AA``, ``AB``, ...), which only methods that never read a label *token* can use: a first-token
logprob readout cannot tell ``"AA"`` from ``"A"``, so ``logprobs`` and ``grammar`` stay capped at
``MAX_LABEL_OPTIONS`` (enforced by ``methods.require_label_readout``).
"""

from __future__ import annotations

import string
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from .errors import InvalidQuestionError

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .types import Question

LABELS = tuple(string.ascii_uppercase)
MAX_LABEL_OPTIONS = len(LABELS)  # what one label token can distinguish
MAX_CHOICE_OPTIONS = 255  # Jev API limit


def labels_for(count: int) -> tuple[str, ...]:
    """Labels for ``count`` options: ``A``..``Z`` up to 26, then two letters from ``AA`` onwards.

    Past 26 options *every* label is two letters — 27 options are ``AA``..``AZ`` and ``BA``, never
    ``A``..``Z`` plus ``AA`` — so a label readout cannot be confused by a prefix.
    """
    if not 0 <= count <= MAX_LABEL_OPTIONS**2:
        raise InvalidQuestionError(
            f"labels_for supports 0..{MAX_LABEL_OPTIONS**2} options, got {count}"
        )
    if count <= MAX_LABEL_OPTIONS:
        return LABELS[:count]
    return tuple(
        LABELS[index // MAX_LABEL_OPTIONS] + LABELS[index % MAX_LABEL_OPTIONS] for index in range(count)
    )



def ascii_upper(text: str) -> str:
    """Upper-case the ASCII letters only, which is the only case a label can be written in.

    The labels are the ASCII alphabet, so folding case is a courtesy to a model that wrote ``a`` for
    ``A`` — and nothing wider. ``str.upper`` is not: it maps ``ı`` (dotless i) to ``I``, ``ſ`` (long
    s) to ``S`` and the Kelvin sign to ``K``, so a token that is not a label at all would be read as
    one and the caller handed an option nobody sampled. Folding the 26 ASCII letters leaves every
    other code point exactly as the provider sent it.
    """
    return "".join(chr(ord(char) - 32) if "a" <= char <= "z" else char for char in text)


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
