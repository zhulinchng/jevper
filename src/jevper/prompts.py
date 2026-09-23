"""Prompt rendering: state turns, question blocks, few-shot turns, correction messages.

Every method and both reasoning passes go through the builders here, so a few-shot demonstration is
rendered in the same format as the answer the method expects.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .errors import InvalidQuestionError, JevperError
from .labels import label_to_key
from .types import Example, JSONContent, Method, Question

SYSTEM_PROMPT = (
    "You are a precise classification engine. Answer the question by selecting exactly one of the "
    "provided labels. Reply with the label only: no punctuation, no explanation, no other text."
)
STRUCTURED_SYSTEM_PROMPT = (
    "You are a precise classification engine. Answer the question by returning a JSON object that "
    "matches the provided schema exactly. Return JSON only: no explanation, no markdown fences."
)
ANALYSIS_SYSTEM_PROMPT = (
    "You are a precise analyst. Work through the state and the question carefully. Do not state a "
    "final label; explain your considerations and the trade-offs between the options."
)
ANSWER_CUE = "Now reply with the label only."
STRUCTURED_ANSWER_CUE = "Now reply with the JSON object only."

_STATE_ROLES = ("system", "user", "assistant", "developer")


def render_content(value: JSONContent) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise JevperError(f"content must be JSON-serializable with finite numbers: {exc}") from exc


def render_state_messages(state: Any) -> list[dict[str, str]]:
    """``str`` -> one user turn; chat-message list (or ``{"messages": [...]}``) -> verbatim turns;
    anything else -> one user turn holding pretty-printed JSON."""
    if isinstance(state, str):
        return [{"role": "user", "content": state}]
    messages: Any = None
    if isinstance(state, Mapping) and set(state) == {"messages"}:
        messages = state["messages"]
    elif isinstance(state, list):
        messages = state
    if messages is not None:
        if not isinstance(messages, (list, tuple)) or not messages:
            raise JevperError("state messages must be a non-empty list of {'role': ..., 'content': ...} dicts")
        rendered = []
        for message in messages:
            if not isinstance(message, Mapping) or set(message) != {"role", "content"}:
                raise JevperError(
                    "state messages must be dicts with exactly the keys 'role' and 'content', "
                    f"got {message!r}"
                )
            role, content = message["role"], message["content"]
            if role not in _STATE_ROLES:
                raise JevperError(f"state message role must be one of {_STATE_ROLES!r}, got {role!r}")
            if not isinstance(content, str):
                raise JevperError(f"state message content must be a string, got {type(content).__name__}")
            rendered.append({"role": role, "content": content})
        return rendered
    return [{"role": "user", "content": render_content(state)}]


def render_question_block(question: Question, labels: Sequence[str]) -> str:
    sections: list[str] = []
    if question.instructions is not None:
        sections.append("Question:\n" + render_content(question.instructions))
    options = ["Options:"]
    if question.type == "choice":
        for label, key in zip(labels, question.criteria):
            description = question.criteria[key]
            options.append(f"{label}: {key}" if description is None else f"{label}: {key} — {render_content(description)}")
    elif question.type == "score":
        for level, (label, description) in enumerate(zip(labels, question.criteria)):
            options.append(
                f"{label}: {level}"
                if description is None
                else f"{label}: {level} — {render_content(description)}"
            )
    else:
        options.append(f"{labels[0]}: Yes")
        options.append(f"{labels[1]}: No")
        criteria = question.criteria
        if criteria is not None:
            if criteria.true is not None:
                options.append(f"Yes means: {render_content(criteria.true)}")
            if criteria.false is not None:
                options.append(f"No means: {render_content(criteria.false)}")
    sections.append("\n".join(options))
    return "\n\n".join(sections)


def render_question_turn(state: Any, question: Question, labels: Sequence[str]) -> str:
    """Collapse a state into a single user turn: state message contents joined with the question block."""
    contents = [message["content"] for message in render_state_messages(state)]
    contents.append(render_question_block(question, labels))
    return "\n\n".join(contents)


def example_answer_label(question: Question, labels: Sequence[str], answer: Any, index: int) -> str:
    """Resolve a few-shot answer to a label: label, ``Choice`` key, ``Score`` level index or bool."""
    problem = InvalidQuestionError(
        f"example {index}: answer {answer!r} does not match any option of this question"
    )
    if isinstance(answer, bool):
        if question.type == "noul":
            return labels[0] if answer else labels[1]
        raise problem
    if isinstance(answer, str):
        if question.type == "choice":
            # An exact criteria key wins over a label: with criteria {"b", "a"}, answer "a" names the
            # option keyed "a", not the first label "A" — the two readings disagree, so the explicit
            # one is the only safe choice.
            for label, key in zip(labels, question.criteria):
                if key == answer:
                    return label
        candidate = answer.strip().upper()
        if candidate in labels:
            return candidate
        raise problem
    if isinstance(answer, int) and question.type == "score" and 0 <= answer < len(question.criteria):
        return labels[answer]
    raise problem


def validate_example_probabilities(
    question: Question, probabilities: Mapping[Any, float], index: int
) -> None:
    """Check an example's own numbers against its question, whatever method answers the call.

    Only ``method="structured"`` renders them — no other answer shape carries a distribution — but a
    caller mistake fails here, before any provider call, instead of silently disappearing for a method
    that cannot show it. ``Example`` itself already rejects non-finite numbers; what is left is the
    shape: the keys, a non-negative weight, and a ``Noul`` weight within [0, 1].
    """
    if question.type == "noul":
        for name, value in probabilities.items():
            if name not in (True, False) and str(name).lower() not in ("true", "false"):
                raise InvalidQuestionError(
                    f"example {index}: probabilities must have a True or False key, "
                    f"got {sorted(str(name) for name in probabilities)}"
                )
            if not 0.0 <= float(value) <= 1.0:
                raise InvalidQuestionError(
                    f"example {index}: noul probability must be in [0, 1], got {value!r}"
                )
        return
    expected = (
        [str(level) for level in range(len(question.criteria))]
        if question.type == "score"
        else list(question.criteria)
    )
    given = {str(name): float(value) for name, value in probabilities.items()}
    if set(given) != set(expected):
        raise InvalidQuestionError(
            f"example {index}: probabilities must have exactly the keys {expected}, got {sorted(given)}"
        )
    for name, value in given.items():
        if value < 0:
            raise InvalidQuestionError(
                f"example {index}: probability for {name!r} must be >= 0, got {value!r}"
            )


def _example_probabilities(
    question: Question, probabilities: Mapping[Any, float], index: int
) -> dict[str, float]:
    """An example's own distribution, keyed the way the answer schema is."""
    validate_example_probabilities(question, probabilities, index)
    expected = (
        [str(level) for level in range(len(question.criteria))]
        if question.type == "score"
        else list(question.criteria)
    )
    given = {str(name): float(value) for name, value in probabilities.items()}
    return {name: given[name] for name in expected}


def _structured_example_answer(
    question: Question, labels: Sequence[str], example: Example, index: int, keys: Mapping[str, Any]
) -> str:
    label = example_answer_label(question, labels, example.answer, index)
    key = keys[label]
    probabilities = example.probabilities
    if question.type == "noul":
        # The key set and the [0, 1] range were checked by ``validate_example_probabilities``.
        if probabilities is None:
            value = 1.0 if key is True else 0.0
        elif _present(probabilities, True):
            value = float(_lookup(probabilities, True))
        else:
            value = 1.0 - float(_lookup(probabilities, False))
        payload: dict[str, Any] = {"noul": value}
    elif probabilities is not None:
        payload = {"probabilities": _example_probabilities(question, probabilities, index)}
    elif question.type == "choice":
        payload = {"probabilities": {name: (1.0 if name == key else 0.0) for name in question.criteria}}
    else:
        payload = {
            "probabilities": {
                str(level): (1.0 if level == key else 0.0) for level in range(len(question.criteria))
            }
        }
    return json.dumps(payload, ensure_ascii=False)


def _discrete_example_answer(
    question: Question, labels: Sequence[str], example: Example, index: int, keys: Mapping[str, Any]
) -> str:
    """The one-hot answer ``method="discrete"`` reads: one label, one level index or one boolean.

    That schema cannot carry a distribution, so ``Example.probabilities`` is validated in
    ``_resolve_examples`` and never rendered here.
    """
    label = example_answer_label(question, labels, example.answer, index)
    if question.type == "choice":
        payload: dict[str, Any] = {"choice": label}
    elif question.type == "noul":
        payload = {"noul": bool(keys[label])}
    else:
        payload = {"score": int(keys[label])}
    return json.dumps(payload, ensure_ascii=False)


def _present(probabilities: Mapping[Any, float], key: Any) -> bool:
    return key in probabilities or str(key).lower() in probabilities


def _lookup(probabilities: Mapping[Any, float], key: Any) -> float:
    if key in probabilities:
        return probabilities[key]
    return probabilities[str(key).lower()]


def render_examples(
    examples: Iterable[Example], question: Question, labels: Sequence[str], *, method: Method
) -> list[dict[str, str]]:
    """One user turn (example state + question block) and one assistant turn (expected answer) per example."""
    turns: list[dict[str, str]] = []
    keys = label_to_key(question, labels) if method in ("structured", "discrete") else {}
    for index, example in enumerate(examples):
        try:
            turn = render_question_turn(example.state, question, labels)
        except JevperError as exc:
            raise JevperError(f"example {index}: {exc}") from exc
        turns.append({"role": "user", "content": turn})
        if method == "structured":
            content = _structured_example_answer(question, labels, example, index, keys)
        elif method == "discrete":
            content = _discrete_example_answer(question, labels, example, index, keys)
        else:
            content = example_answer_label(question, labels, example.answer, index)
        turns.append({"role": "assistant", "content": content})
    return turns


@dataclass(frozen=True)
class PromptParts:
    """One question's prompt, rendered once and assembled per pass.

    ``state_messages`` are the call's state turns: identical for every question, so they are rendered
    once per ``system_one`` call and shared.
    """

    state_messages: Sequence[dict[str, str]]
    example_turns: Sequence[dict[str, str]]
    question_block: str


def system_prompt(method: Method) -> str:
    """The answer-pass system prompt for a method: the JSON one whenever the answer is a JSON object."""
    return STRUCTURED_SYSTEM_PROMPT if method in ("structured", "discrete") else SYSTEM_PROMPT


def answer_cue(method: Method) -> str:
    """The two-step cue for the answer pass, phrased for the shape that pass has to produce."""
    return STRUCTURED_ANSWER_CUE if method in ("structured", "discrete") else ANSWER_CUE


def build_parts(
    state_messages: Sequence[dict[str, str]],
    question: Question,
    labels: Sequence[str],
    examples: Iterable[Example] = (),
    *,
    method: Method = "logprobs",
) -> PromptParts:
    """Render everything a question's prompt needs, once, for both reasoning passes."""
    return PromptParts(
        state_messages=state_messages,
        example_turns=render_examples(examples, question, labels, method=method),
        question_block=render_question_block(question, labels),
    )


def assemble(parts: PromptParts, *, system: str) -> list[dict[str, str]]:
    """The message list for one pass: system prompt, state turns, few-shot turns, question block.

    The caller's state turns are preserved as-is and the question block is the final user turn, so the
    state is never repeated. Each pass gets its own message dicts; the rendered strings are shared.
    """
    return [
        {"role": "system", "content": system},
        *(dict(message) for message in parts.state_messages),
        *(dict(message) for message in parts.example_turns),
        {"role": "user", "content": parts.question_block},
    ]


def correction_message(reason: str, labels: Sequence[str]) -> str:
    """The corrective turn for a label readout; a JSON answer uses ``structured_correction_message``."""
    return (
        f"Your previous reply was invalid: {reason}. Reply with exactly one of these labels and nothing "
        f"else: {', '.join(labels)}."
    )


def structured_correction_message(reason: str) -> str:
    return f"Your previous reply was invalid: {reason}. Return only a JSON object matching the schema."
