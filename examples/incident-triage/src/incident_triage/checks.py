"""The invariants a consumer is entitled to hold, checked after every call.

These are the library's documented contract, not aspirations: the answer carries exactly the
question's own keys, the probabilities are a distribution, ``score`` is the expectation over
it, ``confidence`` is a share, the call counters are counts, and the debug record has the
shape its documentation promises. A service that trusts those does not need to re-derive
them per provider; a service that finds one broken needs to know which one.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from typing import Any

from jevper import ChoiceAnswer, NoulAnswer, ScoreAnswer, SystemOneResponse

__all__ = [
    "ALWAYS_PRESENT",
    "CONDITIONAL",
    "Gap",
    "as_specs",
    "check",
    "check_answer",
    "check_debug",
    "check_json",
    "check_response",
    "prompt_text",
]


TOLERANCE = 1e-6
"""The rescale tolerance the library itself documents for a ``structured`` distribution."""

ALWAYS_PRESENT = (
    "method",
    "api",
    "reasoning_mode",
    "llm_attempts",
    "retry_reasons",
    "probability_errors",
    "original_probabilities",
    "labels_missing",
)
"""The debug keys the API reference says are always there. The conditional ones are read with
``.get``: a key that is absent says its condition did not hold, and that absence is the news."""

CONDITIONAL = ("methods", "apis", "reasoning_modes", "server_limits", "server_limits_by_api")
"""The debug keys that appear only when their condition holds — ``method="auto"``, a multi-surface
call, or a server that has refused a capability field."""


def as_specs(questions: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """The questions as plain mappings, whatever form they were passed in.

    A question model dumps to exactly the Jev wire keys — ``examples`` is excluded by design —
    and a raw mapping is already that, so one set of checks serves typed and raw questions.
    """
    specs: dict[str, dict[str, Any]] = {}
    for question_id, question in questions.items():
        if hasattr(question, "model_dump"):
            specs[question_id] = question.model_dump(mode="json")
        else:
            specs[question_id] = dict(question)
    return specs


class Gap(Exception):
    """A documented invariant the response broke — a finding, not a crash."""


def check(condition: object, message: str) -> None:
    if not condition:
        raise Gap(message)


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _get(obj: Any, name: str) -> Any:
    return obj.get(name) if isinstance(obj, Mapping) else getattr(obj, name, None)


def _content_text(content: Any) -> list[str]:
    """A message body as text, whether it is a string or a list of content parts."""
    if isinstance(content, str):
        return [content]
    if isinstance(content, list):
        return [
            part["text"]
            for part in content
            if isinstance(part, Mapping) and isinstance(part.get("text"), str)
        ]
    return []


def prompt_text(response: SystemOneResponse) -> str:
    """The prompt as the provider saw it, from the recorded request of the last attempt.

    Chat Completions records ``messages`` and the Responses surface records ``input`` items whose
    content is a list of parts, so both shapes are read here.
    """
    attempts = response.debug.get("llm_attempts") or []
    if not attempts:
        return ""
    request = attempts[-1].get("request") or {}
    chunks: list[str] = []
    for message in request.get("messages") or []:
        chunks.extend(_content_text(_get(message, "content")))
    for item in request.get("input") or []:
        chunks.extend(_content_text(_get(item, "content")))
    return "\n".join(chunk for chunk in chunks if chunk)


def check_answer(answer: Any, question: Mapping[str, Any], *, normalized: bool = True) -> None:
    """One answer against the question that asked for it."""
    kind = question["type"]
    check(answer.type == kind, f"answer type {answer.type!r} for a {kind} question")

    if isinstance(answer, NoulAnswer):
        check(_finite(answer.noul), f"noul is not a finite number: {answer.noul!r}")
        check(0.0 <= answer.noul <= 1.0, f"noul outside [0, 1]: {answer.noul!r}")
        return

    if isinstance(answer, ChoiceAnswer):
        keys = list(question["criteria"])
        check(
            list(answer.probabilities) == keys,
            f"choice probabilities keyed {list(answer.probabilities)}, expected {keys}",
        )
        for key, value in answer.probabilities.items():
            check(_finite(value), f"probability for {key!r} is not finite: {value!r}")
            if normalized:
                # With normalize_probabilities=False these are the model's own numbers, and the
                # documented contract is to pass them through — a model that answered 5.0 is
                # reported as 5.0, not silently repaired.
                check(0.0 <= value <= 1.0, f"probability for {key!r} outside [0, 1]: {value!r}")
        if normalized:
            total = math.fsum(float(value) for value in answer.probabilities.values())
            check(
                abs(total - 1.0) <= TOLERANCE,
                f"choice probabilities sum to {total!r}, not 1 (tolerance {TOLERANCE})",
            )
        best = max(keys, key=lambda key: float(answer.probabilities[key]))
        check(
            answer.choice == best,
            f"choice is {answer.choice!r} but the highest probability is {best!r} "
            f"({dict(answer.probabilities)})",
        )
        check(0.0 <= answer.confidence <= 1.0, f"confidence outside [0, 1]: {answer.confidence!r}")
        return

    if isinstance(answer, ScoreAnswer):
        levels = list(range(len(question["criteria"])))
        check(
            sorted(answer.probabilities) == levels,
            f"score probabilities keyed {sorted(answer.probabilities)}, expected {levels}",
        )
        for level, value in answer.probabilities.items():
            check(_finite(value), f"probability for level {level} is not finite: {value!r}")
            if normalized:
                check(0.0 <= value <= 1.0, f"probability for level {level} outside [0, 1]: {value!r}")
        # The score is read off the distribution rescaled to sum 1, and a zero total falls back to
        # uniform — which is what the library documents and what a model answering all zeros gets.
        # So the expectation is checked over the normalized distribution, not the raw one.
        weights = [float(value) for value in answer.probabilities.values()]
        total = math.fsum(weights)
        if total == 0.0:
            weights = [1.0 / len(weights)] * len(weights)
        else:
            weights = [weight / total for weight in weights]
        expected = math.fsum(level * weight for level, weight in enumerate(weights))
        check(
            abs(answer.score - expected) <= 1e-6,
            f"score {answer.score!r} is not the expectation over the distribution ({expected!r})",
        )
        check(
            {index: text for index, text in answer.legend.items()} == dict(enumerate(question["criteria"])),
            f"legend {answer.legend} does not map every level to its criteria entry",
        )
        check(0.0 <= answer.confidence <= 1.0, f"confidence outside [0, 1]: {answer.confidence!r}")
        return

    raise Gap(f"answer of an undocumented type: {type(answer).__name__}")


def check_debug(response: SystemOneResponse) -> dict[str, Any]:
    """The debug record has the documented shape and can be serialized."""
    debug = response.debug
    check(isinstance(debug, dict), f"debug is {type(debug).__name__}, not a mapping")
    for key in ALWAYS_PRESENT:
        check(key in debug, f"debug has no {key!r} key, which the reference says is always there")
    for key in CONDITIONAL:
        if key in debug:
            check(isinstance(debug[key], dict), f"debug[{key!r}] is not a mapping")
    check(isinstance(debug["llm_attempts"], list) and debug["llm_attempts"], "debug carries no attempt record")
    json.dumps(debug)  # a record a service logs must serialize
    return debug


def check_json(response: SystemOneResponse, questions: Mapping[str, Any]) -> None:
    """``model_dump_json`` produces the Jev answer shape and round-trips."""
    payload = json.loads(response.model_dump_json())
    check(payload.get("model") == response.model, f"serialized model is {payload.get('model')!r}")
    answers = payload.get("answers")
    check(isinstance(answers, dict), f"serialized answers is {type(answers).__name__}")
    check(set(answers) == set(questions), f"serialized answers keyed {sorted(answers)}, expected {sorted(questions)}")
    for question_id, answer in answers.items():
        expected = questions[question_id]["type"]
        check(answer.get("type") == expected, f"serialized {question_id} type is {answer.get('type')!r}")
        if expected == "noul":
            check(set(answer) == {"type", "noul"}, f"serialized noul keys are {sorted(answer)}")
        elif expected == "choice":
            check(
                set(answer) == {"type", "choice", "probabilities", "confidence"},
                f"serialized choice keys are {sorted(answer)}",
            )
        else:
            check(
                set(answer) == {"type", "score", "legend", "probabilities", "confidence"},
                f"serialized score keys are {sorted(answer)}",
            )
            check(
                all(isinstance(key, str) for key in answer["legend"]),
                "serialized score legend keys are not strings",
            )
            check(
                all(isinstance(key, str) for key in answer["probabilities"]),
                "serialized score probability keys are not strings",
            )


def check_usage(response: SystemOneResponse, *, min_calls: int = 1) -> None:
    usage = response.usage
    check(usage.n_calls >= min_calls, f"n_calls {usage.n_calls} below the {min_calls} expected")
    check(usage.n_retries >= 0, f"n_retries is {usage.n_retries}")
    check(usage.latency > 0.0, f"latency is {usage.latency!r}")
    for name in ("input_tokens", "output_tokens", "reasoning_tokens", "cached_tokens"):
        value = getattr(usage, name)
        check(value is None or (isinstance(value, int) and value >= 0), f"{name} is {value!r}")


def check_response(
    response: SystemOneResponse,
    questions: Mapping[str, Any],
    *,
    model: str | None = None,
    normalized: bool = True,
    min_calls: int | None = None,
) -> None:
    """Every documented invariant of one ``system_one`` call."""
    if model is not None:
        check(response.model == model, f"response model is {response.model!r}, asked {model!r}")
    check(
        list(response.answers) == list(questions),
        f"answers keyed {list(response.answers)}, expected {list(questions)} in order",
    )
    for question_id, question in questions.items():
        check(question_id in response.answers, f"no answer for {question_id!r}")
        if question_id in response.answers:
            check_answer(response.answers[question_id], question, normalized=normalized)
    check_usage(response, min_calls=len(questions) if min_calls is None else min_calls)
    check_debug(response)
    check_json(response, questions)
    covered = len(response.nouls) + len(response.choices) + len(response.scores)
    check(covered == len(response.answers), f"the filtered views cover {covered} of {len(response.answers)} answers")
    for view, expected_type in ((response.nouls, NoulAnswer), (response.choices, ChoiceAnswer), (response.scores, ScoreAnswer)):
        for key, answer in view.items():
            check(
                isinstance(answer, expected_type),
                f"{key!r} is in the wrong filtered view ({type(answer).__name__})",
            )
