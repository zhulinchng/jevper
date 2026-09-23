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
    try:
        total = math.fsum(probabilities)
    except OverflowError:
        return _normalize_scaled(probabilities)
    if total == 0:
        return [1.0 / len(probabilities)] * len(probabilities)
    return [probability / total for probability in probabilities]


def _normalize_scaled(probabilities: Sequence[float]) -> list[float]:
    """Normalize by the largest value first, for inputs whose sum leaves the float range.

    A model that answers with values near ``1e308`` overflows ``math.fsum``, which raises
    ``OverflowError`` — a bare Python error escaping a readout. Dividing by the largest magnitude
    first keeps every ratio exact and the running total inside the float range.
    """
    largest = max(probabilities, default=0.0)
    if largest <= 0:
        return [1.0 / len(probabilities)] * len(probabilities)
    scaled = [probability / largest for probability in probabilities]
    total = math.fsum(scaled)
    return [probability / total for probability in scaled]


def rescale(probabilities: Mapping[K, float]) -> dict[K, float]:
    try:
        total = math.fsum(probabilities.values())
    except OverflowError:
        scaled = _normalize_scaled(list(probabilities.values()))
        return {key: value for key, value in zip(probabilities, scaled)}
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
    """``abs(sum - 1)`` for a structured-mode distribution, without raising on huge values."""
    try:
        total = math.fsum(probabilities.values())
    except OverflowError:
        # The values themselves are near the float limit; the caller rescales them, which
        # `_normalize_scaled` can do. Reporting an infinite error is what triggers that.
        return math.inf
    return abs(total - 1.0)
