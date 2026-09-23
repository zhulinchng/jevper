"""The four elicitation methods: request shape and answer readout.

``logprobs`` and ``grammar`` read the model's own distribution over the single label token;
``structured`` asks for a schema-constrained JSON distribution; ``discrete`` asks for one label and
reports a one-hot distribution.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from .errors import (
    InvalidQuestionError,
    LabelReadoutError,
    MalformedAnswerError,
    UnsupportedMethodError,
    _LogprobsUnavailable,
)
from .labels import MAX_CHOICE_OPTIONS, MAX_LABEL_OPTIONS, label_to_key
from .reasoning import ReasoningConfig
from .transport import CallResult, CallSpec, TokenLogprob
from .types import Method, Question

GRAMMAR_SURFACE_HINT = (
    "grammar requires a Chat Completions surface that accepts a `grammar` field (llama-cpp-python and "
    "similar servers); pass api='chat_completions'"
)


def require_label_readout(method: Method, question: Question, question_id: str) -> None:
    """``logprobs``/``grammar`` read one label token, so they cannot go past the single-letter alphabet.

    ``structured`` and ``discrete`` answer in JSON, where a label is just a string, so they carry the
    full Jev API range and two-letter labels.
    """
    if method not in ("logprobs", "grammar") or question.type != "choice":
        return
    count = len(question.criteria)
    if count > MAX_LABEL_OPTIONS:
        raise InvalidQuestionError(
            f"question {question_id!r}: {count} options exceed the {MAX_LABEL_OPTIONS} labels a "
            f"single-token {method} readout can distinguish (the first token of 'AA' is 'A'); use "
            f"method='structured' or method='discrete', which handle up to {MAX_CHOICE_OPTIONS}"
        )


@dataclass(frozen=True)
class Readout:
    probabilities: dict[Any, float]
    source: Literal["logprobs", "grammar", "structured", "discrete"]
    missing_labels: tuple[str, ...] = ()
    observed_text: str | None = None


def labels_grammar(labels: Sequence[str]) -> str:
    return "root ::= " + " | ".join(f'"{label}"' for label in labels) + "\n"


def _object_schema(properties: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def choice_schema(keys: Sequence[str]) -> dict[str, Any]:
    # The bounds are the readout's: a schema-valid answer is always a readable one, so a negative
    # number costs no corrective retry.
    return _object_schema(
        {
            "probabilities": {
                "type": "object",
                "properties": {key: {"type": "number", "minimum": 0} for key in keys},
                "required": list(keys),
                "additionalProperties": False,
            }
        }
    )


def noul_schema() -> dict[str, Any]:
    return _object_schema({"noul": {"type": "number", "minimum": 0, "maximum": 1}})


def score_schema(level_count: int) -> dict[str, Any]:
    return choice_schema([str(level) for level in range(level_count)])


def choice_discrete_schema(labels: Sequence[str]) -> dict[str, Any]:
    return _object_schema({"choice": {"type": "string", "enum": list(labels)}})


def noul_discrete_schema() -> dict[str, Any]:
    return _object_schema({"noul": {"type": "boolean"}})


def score_discrete_schema(level_count: int) -> dict[str, Any]:
    return _object_schema({"score": {"type": "integer", "enum": list(range(level_count))}})


def probabilities_schema(question: Question) -> tuple[str, dict[str, Any]]:
    if question.type == "choice":
        return "jevper_choice", choice_schema(list(question.criteria))
    if question.type == "noul":
        return "jevper_noul", noul_schema()
    return "jevper_score", score_schema(len(question.criteria))


def discrete_schema(question: Question, labels: Sequence[str]) -> tuple[str, dict[str, Any]]:
    if question.type == "choice":
        return "jevper_choice", choice_discrete_schema(labels)
    if question.type == "noul":
        return "jevper_noul", noul_discrete_schema()
    return "jevper_score", score_discrete_schema(len(question.criteria))


def require_grammar_surface(surface: str) -> None:
    if surface != "chat_completions":
        raise UnsupportedMethodError(GRAMMAR_SURFACE_HINT)


def build_spec(
    method: Method,
    messages: list[dict[str, str]],
    question: Question,
    labels: Sequence[str],
    *,
    top_logprobs: int = 20,
    temperature: float | None = None,
    reasoning: ReasoningConfig | None = None,
) -> CallSpec:
    """The request shape for one method. Only ``logprobs``/``grammar`` ask for logprobs."""
    if method == "structured":
        name, schema = probabilities_schema(question)
        return CallSpec(
            messages=messages,
            json_schema=schema,
            schema_name=name,
            temperature=temperature,
            reasoning=reasoning,
        )
    if method == "discrete":
        name, schema = discrete_schema(question, labels)
        return CallSpec(
            messages=messages,
            json_schema=schema,
            schema_name=name,
            temperature=temperature,
            reasoning=reasoning,
        )
    return CallSpec(
        messages=messages,
        logprobs=True,
        top_logprobs=top_logprobs,
        grammar=labels_grammar(labels) if method == "grammar" else None,
        temperature=temperature,
        reasoning=reasoning,
    )


def softmax_over_labels(values: Mapping[str, float]) -> dict[str, float]:
    """Softmax over the supplied labels only; ``-inf`` entries become exactly ``0.0``.

    A grammar masks logits but never renormalizes them, so renormalizing the pre-mask distribution
    over the label set equals the post-mask distribution: one code path is correct for both.
    """
    for key, value in values.items():
        if value != float("-inf") and (math.isnan(value) or value == float("inf")):
            raise LabelReadoutError(f"logprob for {key!r} must be finite or -inf, got {value!r}")
    best = max(values.values(), default=float("-inf"))
    if best == float("-inf"):
        raise LabelReadoutError("no probability mass on any label")
    weights = {key: math.exp(value - best) for key, value in values.items() if value != float("-inf")}
    total = math.fsum(weights.values())
    if total == 0.0:
        raise LabelReadoutError("no probability mass on any label")
    return {key: weights.get(key, 0.0) / total for key in values}


def _answer_tokens(result: CallResult) -> tuple[TokenLogprob, ...]:
    """The tokens of the answer itself, when the provider separated its reasoning from the answer.

    A reasoning server reports logprobs for every generated token — vLLM, SGLang and ollama include
    the thinking span, while ``message.content`` holds only the answer — so the first token of the
    stream is the first token of the *thinking*, which is never a label. The answer text is the
    anchor: when it is exactly the tail of the token stream, the token covering its first character
    is the answer's first token. The check is strict on purpose — no separated trace, or no exact
    tail, means nothing is skipped and the stream is read as it arrives.

    Two servers append their own end-of-turn token to the stream (vLLM and SGLang report
    ``<|im_end|>`` after the answer), which would hide the tail. A trailing token that cannot be part
    of the answer — one whose text does not occur in it — is therefore dropped before the tail is
    tested, at most two of them, so the anchor stays exact about everything it keeps.
    """
    tokens = result.token_logprobs
    content = result.text
    if not result.reasoning or not content.strip():
        return tokens
    kept = list(tokens)
    for _ in range(2):
        if "".join(token.token for token in kept).endswith(content):
            break
        if len(kept) == 1 or (kept[-1].token and kept[-1].token in content):
            return tokens
        kept.pop()
    stream = "".join(token.token for token in kept)
    if len(content) >= len(stream) or not stream.endswith(content):
        return tokens
    tokens = tuple(kept)
    start = len(stream) - len(content)
    offset = 0
    for index, token in enumerate(tokens):
        offset += len(token.token)
        if offset > start:
            return tokens[index:]
    return tokens


def _stop_note(result: CallResult) -> str:
    """Why the provider stopped, when it stopped before writing an answer.

    A reasoning model can spend the entire output budget thinking: vLLM and SGLang answer with
    ``status: "incomplete"`` and an empty message, llama.cpp and ollama with ``finish_reason:
    "length"`` and nothing but reasoning tokens. "No non-whitespace token in the response" is true
    and useless; the budget is the actionable fact.
    """
    if result.stop is None:
        return ""
    if result.stop in ("length", "max_output_tokens"):
        return (
            f" — the provider ran out of output tokens before the answer was complete "
            f"({result.stop!r}); raise the limit, for example extra_body={{'max_tokens': 2048}}"
        )
    return f" — the provider reported {result.stop!r}"


def first_answer_token(result: CallResult, labels: Sequence[str], *, method: Method) -> TokenLogprob:
    """The first non-whitespace token of the answer, which must be one of the labels."""
    if not result.token_logprobs:
        raise _LogprobsUnavailable(
            f"no logprobs returned for the answer token (method={method!r}); this provider does not "
            f"report them — use method='structured' for the model's own probabilities, or "
            f"method='discrete' for one label, neither of which needs logprobs{_stop_note(result)}",
            evidence="readout",
        )
    for token in _answer_tokens(result):
        if not token.token.strip():
            continue
        if token.token.strip().upper() in labels:
            return token
        raise LabelReadoutError(
            f"first non-whitespace token {token.token!r} is not one of the labels {list(labels)!r}"
            f"{_stop_note(result)}"
        )
    raise LabelReadoutError(
        f"no non-whitespace token in the response (method={method!r}){_stop_note(result)}"
    )


def _logprob_readout(
    result: CallResult,
    question: Question,
    labels: Sequence[str],
    source: Literal["logprobs", "grammar"],
    method: Method,
) -> Readout:
    token = first_answer_token(result, labels, method=method)
    if token.logprob is None:
        raise LabelReadoutError(
            f"the provider returned no logprob for the answer token {token.token!r} (method={method!r})"
        )
    if token.reported_alternatives < 2:
        # Nothing but the sampled token came back, so there is no distribution to read: the answer
        # would be a one-hot built from absence of data. A provider that reports two or more entries
        # but nulls for some of them is a different case — those labels are missing, not unknown,
        # and land in debug["labels_missing"].
        raise _LogprobsUnavailable(
            f"the provider returned {token.reported_alternatives} top_logprobs for the answer token "
            f"{token.token!r} (method={method!r}), which is not a distribution over the options; use "
            f"method='structured' for the model's own probabilities, or method='discrete' for one label",
            evidence="readout",
        )
    answer_label = token.token.strip().upper()
    logprobs: dict[str, float] = {label: float("-inf") for label in labels}
    logprobs[answer_label] = token.logprob
    for top_token, top_logprob in token.top_logprobs:
        candidate = top_token.strip().upper()
        if candidate in logprobs and candidate != answer_label:
            logprobs[candidate] = top_logprob
    missing = tuple(label for label in labels if logprobs[label] == float("-inf"))
    normalized = softmax_over_labels(logprobs)
    keys = label_to_key(question, labels)
    return Readout(
        probabilities={keys[label]: value for label, value in normalized.items()},
        source=source,
        missing_labels=missing,
        observed_text=result.text,
    )


def readout_logprobs(result: CallResult, question: Question, labels: Sequence[str]) -> Readout:
    return _logprob_readout(result, question, labels, "logprobs", "logprobs")


def readout_grammar(result: CallResult, question: Question, labels: Sequence[str]) -> Readout:
    require_grammar_surface(result.surface)
    if not result.token_logprobs:
        # A provider-side fact, so no corrective retry is spent on it — another turn cannot change
        # what the server reports. Same message, same public class (a LabelReadoutError subclass).
        raise _LogprobsUnavailable(
            "grammar mode needs logprobs in the response; pass method='discrete' to skip probabilities",
            evidence="readout",
        )
    return _logprob_readout(result, question, labels, "grammar", "grammar")


def parse_json_object(text: str, note: str = "") -> dict[str, Any]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError as first_error:
        start = text.find("{")
        if start < 0:
            raise MalformedAnswerError(
                f"no JSON object in the answer ({first_error}){note}"
            ) from first_error
        try:
            value, _ = json.JSONDecoder().raw_decode(text[start:])
        except ValueError as exc:
            raise MalformedAnswerError(f"could not parse a JSON object from the answer ({exc})") from exc
    if not isinstance(value, dict):
        raise MalformedAnswerError(f"expected a JSON object, got {type(value).__name__}")
    return value


def _number(value: Any, *, where: str, upper: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise MalformedAnswerError(f"{where} must be a finite number, got {value!r}")
    number = float(value)
    if number < 0 or (upper is not None and number > upper):
        bound = ">= 0" if upper is None else f"in [0, {upper}]"
        raise MalformedAnswerError(f"{where} must be {bound}, got {number!r}")
    return number


def readout_structured(result: CallResult, question: Question) -> Readout:
    payload = parse_json_object(result.text, _stop_note(result))
    if question.type == "choice":
        keys = list(question.criteria)
        raw = payload.get("probabilities")
        if not isinstance(raw, Mapping) or set(raw) != set(keys):
            raise MalformedAnswerError(
                f"'probabilities' must have exactly the option keys {sorted(keys)}, got {payload!r}"
            )
        return Readout(
            probabilities={key: _number(raw[key], where=f"probability for {key!r}") for key in keys},
            source="structured",
            observed_text=result.text,
        )
    if question.type == "noul":
        value = _number(payload.get("noul"), where="'noul'", upper=1.0)
        return Readout(
            probabilities={True: value, False: 1.0 - value}, source="structured", observed_text=result.text
        )
    expected = [str(level) for level in range(len(question.criteria))]
    raw = payload.get("probabilities")
    if not isinstance(raw, Mapping) or set(raw) != set(expected):
        raise MalformedAnswerError(
            f"'probabilities' must have exactly the level keys {expected}, got {payload!r}"
        )
    return Readout(
        probabilities={
            int(level): _number(raw[level], where=f"probability for level {level}") for level in expected
        },
        source="structured",
        observed_text=result.text,
    )


def _level_index(value: Any, levels: Sequence[int]) -> int | None:
    """A level index given as an int, an integral float or an integral number in a string."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        index = value
    else:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        if not number.is_integer():
            return None
        index = int(number)
    return index if index in levels else None


def _boolean(value: Any) -> bool | None:
    """A JSON boolean, or its string form; ``None`` when it is neither."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ("true", "false"):
            return text == "true"
    return None


def readout_discrete(result: CallResult, question: Question, labels: Sequence[str]) -> Readout:
    payload = parse_json_object(result.text, _stop_note(result))
    if question.type == "choice":
        keys = list(question.criteria)
        raw = payload.get("choice")
        label = None
        if isinstance(raw, str):
            candidate = raw.strip()
            # An exact option key wins over a label, the same way an example's answer does: a key that
            # is also a label ("a" and "A") must not be read as the first option by accident.
            if candidate in keys:
                label = labels[keys.index(candidate)]
            elif candidate.upper() in labels:
                label = candidate.upper()
        if label is None:
            raise MalformedAnswerError(
                f"'choice' must be one of the labels {list(labels)!r} or the option keys {keys!r}, got {raw!r}"
            )
        chosen = label_to_key(question, labels)[label]
        return Readout(
            probabilities={key: (1.0 if key == chosen else 0.0) for key in keys},
            source="discrete",
            observed_text=result.text,
        )
    if question.type == "noul":
        raw = payload.get("noul")
        flag = _boolean(raw)
        if flag is None:
            raise MalformedAnswerError(f"'noul' must be a boolean, got {raw!r}")
        return Readout(
            probabilities={True: 1.0 if flag else 0.0, False: 0.0 if flag else 1.0},
            source="discrete",
            observed_text=result.text,
        )
    levels = list(range(len(question.criteria)))
    raw = payload.get("score")
    level = _level_index(raw, levels)
    if level is None:
        raise MalformedAnswerError(f"'score' must be one of the level indexes {levels!r}, got {raw!r}")
    return Readout(
        probabilities={candidate: (1.0 if candidate == level else 0.0) for candidate in levels},
        source="discrete",
        observed_text=result.text,
    )


def readout(
    method: Method, result: CallResult, question: Question, labels: Sequence[str]
) -> Readout:
    if method == "logprobs":
        return readout_logprobs(result, question, labels)
    if method == "grammar":
        return readout_grammar(result, question, labels)
    if method == "structured":
        return readout_structured(result, question)
    return readout_discrete(result, question, labels)
