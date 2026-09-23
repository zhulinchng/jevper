"""Prompt rendering: state turns, question blocks, few-shot turns, correction messages.

Every method and both reasoning passes go through the builders here, so a few-shot demonstration is
rendered in the same format as the answer the method expects.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from .errors import InvalidQuestionError, JevperError
from .labels import MAX_LABEL_OPTIONS, label_to_key
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

_STATE_ROLES = ("system", "user", "assistant", "developer")


def render_content(value: JSONContent) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2)


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
        candidate = answer.strip().upper()
        if candidate in labels:
            return candidate
        if question.type == "choice":
            for label, key in zip(labels, question.criteria):
                if key == answer:
                    return label
        raise problem
    if isinstance(answer, int) and question.type == "score" and 0 <= answer < len(question.criteria):
        return labels[answer]
    raise problem


def _structured_example_answer(question: Question, labels: Sequence[str], example: Example, index: int) -> str:
    label = example_answer_label(question, labels, example.answer, index)
    key = label_to_key(question, labels)[label]
    probabilities = example.probabilities
    if question.type == "noul":
        if probabilities is not None and _present(probabilities, True):
            value = float(_lookup(probabilities, True))
        elif probabilities is not None and _present(probabilities, False):
            value = 1.0 - float(_lookup(probabilities, False))
        else:
            value = 1.0 if key is True else 0.0
        payload: dict[str, Any] = {"noul": value}
    elif probabilities is not None:
        payload = {"probabilities": {str(name): float(value) for name, value in probabilities.items()}}
    elif question.type == "choice":
        payload = {"probabilities": {name: (1.0 if name == key else 0.0) for name in question.criteria}}
    else:
        payload = {
            "probabilities": {
                str(level): (1.0 if level == key else 0.0) for level in range(len(question.criteria))
            }
        }
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
    for index, example in enumerate(examples):
        turns.append({"role": "user", "content": render_question_turn(example.state, question, labels)})
        if method == "structured":
            content = _structured_example_answer(question, labels, example, index)
        else:
            content = example_answer_label(question, labels, example.answer, index)
        turns.append({"role": "assistant", "content": content})
    return turns


def build_messages(
    state: Any,
    question: Question,
    labels: Sequence[str],
    *,
    examples: Iterable[Example] = (),
    method: Method = "logprobs",
) -> list[dict[str, str]]:
    """The single message builder for every method and both reasoning passes.

    The caller's state turns are preserved as-is and the question block is the final user turn, so the
    state is never repeated.
    """
    system = STRUCTURED_SYSTEM_PROMPT if method == "structured" else SYSTEM_PROMPT
    return (
        [{"role": "system", "content": system}]
        + render_state_messages(state)
        + render_examples(examples, question, labels, method=method)
        + [{"role": "user", "content": render_question_block(question, labels)}]
    )


def build_analysis_messages(
    state: Any,
    question: Question,
    labels: Sequence[str],
    examples: Iterable[Example] = (),
    *,
    method: Method = "logprobs",
) -> list[dict[str, str]]:
    """The two-step analysis pass: same turns as ``build_messages`` with the analysis system prompt."""
    return (
        [{"role": "system", "content": ANALYSIS_SYSTEM_PROMPT}]
        + render_state_messages(state)
        + render_examples(examples, question, labels, method=method)
        + [{"role": "user", "content": render_question_block(question, labels)}]
    )


def correction_message(reason: str, labels: Sequence[str]) -> str:
    if len(labels) > MAX_LABEL_OPTIONS:
        # Listing 100+ labels would dwarf the question; the labels are already in the options block.
        return (
            f"Your previous reply was invalid: {reason}. Reply with exactly one of the labels listed "
            f"with the options above, and nothing else."
        )
    return (
        f"Your previous reply was invalid: {reason}. Reply with exactly one of these labels and nothing "
        f"else: {', '.join(labels)}."
    )


def structured_correction_message(reason: str) -> str:
    return f"Your previous reply was invalid: {reason}. Return only a JSON object matching the schema."
