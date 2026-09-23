"""Confidence and probability math, copied from the TypeSafe reference adapter
(``system_one_adapter/_utils/confidence_metrics.py``) so numbers match the hosted API."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import TypeVar

PROBABILITY_TOLERANCE = 1e-6

K = TypeVar("K")


def _normalize(probabilities: Sequence[float]) -> list[float]:
    """Normalize confidence inputs, using uniform probabilities for zero totals."""
    total = math.fsum(probabilities)
    if total == 0:
        return [1.0 / len(probabilities)] * len(probabilities)
    return [probability / total for probability in probabilities]


def rescale(probabilities: Mapping[K, float]) -> dict[K, float]:
    total = math.fsum(probabilities.values())
    if total == 0:
        return {key: 1.0 / len(probabilities) for key in probabilities}
    return {key: value / total for key, value in probabilities.items()}


def choice_confidence(probabilities: Sequence[float]) -> float:
    """Scale peak choice probability from uniform to certainty."""
    if len(probabilities) <= 1:
        return 1.0
    normalized = _normalize(probabilities)
    uniform_probability = 1.0 / len(normalized)
    return (max(normalized) - uniform_probability) / (1.0 - uniform_probability)


def score_confidence(probabilities: Sequence[float]) -> float:
    """Measure score concentration around its modal score."""
    if len(probabilities) <= 1:
        return 1.0
    normalized = _normalize(probabilities)
    mode_index = max(range(len(normalized)), key=normalized.__getitem__)
    distance_from_mode = math.fsum(
        probability * abs(index - mode_index) for index, probability in enumerate(normalized)
    )
    uniform_center = (len(normalized) - 1) / 2
    uniform_mean_absolute_deviation = (
        math.fsum(abs(index - uniform_center) for index in range(len(normalized))) / len(normalized)
    )
    return max(0.0, 1.0 - distance_from_mode / uniform_mean_absolute_deviation)


def probability_error(probabilities: Mapping[K, float]) -> float:
    """``abs(sum - 1)`` for a structured-mode distribution."""
    return abs(math.fsum(probabilities.values()) - 1.0)
