"""The four elicitation methods: request shape and answer readout.

``logprobs`` and ``grammar`` read the model's own distribution over the single label token;
``structured`` asks for a schema-constrained JSON distribution; ``discrete`` asks for one label and
reports a one-hot distribution.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Literal

from .errors import (
    IncompleteAnswerError,
    InvalidQuestionError,
    LabelReadoutError,
    MalformedAnswerError,
    ModelRefusalError,
    ProviderError,
    UnsupportedMethodError,
    _LogprobsUnavailable,
)
from .labels import MAX_CHOICE_OPTIONS, MAX_LABEL_OPTIONS, ascii_upper, label_to_key
from .reasoning import ReasoningConfig
from .transport import CallResult, CallSpec, TokenLogprob
from .types import Method, Question, bounded_text

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
    prompt_cache_key: str | None = None,
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
            prompt_cache_key=prompt_cache_key,
        )
    if method == "discrete":
        name, schema = discrete_schema(question, labels)
        return CallSpec(
            messages=messages,
            json_schema=schema,
            schema_name=name,
            temperature=temperature,
            reasoning=reasoning,
            prompt_cache_key=prompt_cache_key,
        )
    return CallSpec(
        messages=messages,
        logprobs=True,
        top_logprobs=top_logprobs,
        grammar=labels_grammar(labels) if method == "grammar" else None,
        temperature=temperature,
        reasoning=reasoning,
        prompt_cache_key=prompt_cache_key,
    )


def softmax_over_labels(values: Mapping[str, float]) -> dict[str, float]:
    """Softmax over the supplied labels only; ``-inf`` entries become exactly ``0.0``.

    A grammar masks logits but never renormalizes them, so renormalizing the pre-mask distribution
    over the label set equals the post-mask distribution: one code path is correct for both.

    A log probability is never positive, so a value above zero is not a distribution to normalize
    but a different quantity sent in its place — a gateway that passes probabilities through as
    logprobs, or a server with the sign flipped. Exponentiating those would turn ``0.9`` and
    ``-0.1`` into a confident-looking 73/27 split that no model ever reported, so they are refused
    with the number that gave them away.
    """
    for key, value in values.items():
        if value != float("-inf") and (math.isnan(value) or value == float("inf")):
            raise LabelReadoutError(f"logprob for {key!r} must be finite or -inf, got {value!r}")
        if value > 0.0:
            raise LabelReadoutError(
                f"logprob for {key!r} is positive ({value!r}), which no log probability can be — the "
                f"provider sent something other than logprobs"
            )
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


_TRUNCATED_STOPS = frozenset(
    {"length", "max_tokens", "max_output_tokens", "model_context_window_exceeded"}
)
"""What each surface calls "the budget ran out": Chat Completions ``length``, the Messages API
``max_tokens`` and ``model_context_window_exceeded``, the Responses surface ``max_output_tokens``."""

_CONTEXT_STOPS = frozenset({"model_context_window_exceeded"})
"""A spent *context window* is terminal in the same way a spent output budget is, but the remedy is
the opposite one: the request is already too long to answer, so raising ``max_tokens`` makes it longer.
Anthropic names the two differently for that reason, and the message says which one happened."""


_REFUSAL_STOPS = frozenset({"refusal", "content_filter"})
"""What each surface calls "the model did not answer": the Messages API reports a declined turn as
``stop_reason: "refusal"``, and OpenAI reports a safety-filtered generation as
``finish_reason: "content_filter"`` on Chat Completions and ``incomplete_details.reason:
"content_filter"`` on the Responses surface. A filter is a refusal in everything but the word: the
content was withheld on purpose, so a second attempt spends a call to be filtered the same way, and
the error says so instead of reporting a generation that stopped early."""


def _truncation_message(stop: str, surface: str) -> str:
    """The error text for a generation that ran out of room, naming the room and the knob that opens it.

    The output-budget field is the surface's own, because they do not share a name. Advice naming
    the wrong field is advice that cannot be followed, and the names have moved: OpenAI's Chat
    Completions route now takes ``max_completion_tokens`` (and refuses ``max_tokens`` for its
    reasoning models), while most local servers still take only ``max_tokens`` — so the chat advice
    names both, and the Responses and Messages names are the only ones there is.
    """
    if stop in _CONTEXT_STOPS:
        return (
            f"the provider's context window ran out before the answer was complete ({stop!r}); "
            f"shorten the state or the examples, or use a model with a larger context"
        )
    knob, alternative = {
        "responses": ("max_output_tokens", ""),
        "chat_completions": (
            "max_completion_tokens",
            " (or 'max_tokens', which is what most local servers take)",
        ),
    }.get(surface, ("max_tokens", ""))
    return (
        f"the provider ran out of output tokens before the answer was complete ({stop!r}); "
        f"raise the limit, for example extra_body={{{knob!r}: 2048}}{alternative}"
    )


def _stop_note(result: CallResult) -> str:
    """Why the provider stopped, when it stopped before writing an answer.

    A reasoning model can spend the entire output budget thinking: vLLM and SGLang answer with
    ``status: "incomplete"`` and an empty message, llama.cpp and ollama with ``finish_reason:
    "length"`` and nothing but reasoning tokens, and the Messages API with ``stop_reason: "max_tokens"``.
    "No non-whitespace token in the response" is true and useless; the budget is the actionable fact.

    A refusal is the other reason an answer never arrives: OpenAI reports it in a ``refusal`` sibling of
    ``content``, the Messages API as ``stop_reason: "refusal"``. Both read as a missing JSON object
    otherwise, which sends a caller looking for a parsing bug that is not there.
    """
    if result.stop in _TRUNCATED_STOPS:
        return f" — {_truncation_message(result.stop, result.surface)}"
    note = f" — the provider reported {result.stop!r}" if result.stop is not None else ""
    if result.refusal:
        note += f" — the model refused to answer: {result.refusal[:200]!r}"
    elif result.stop in _REFUSAL_STOPS:
        note += " — the model refused to answer"
    if not result.text.strip() and result.reasoning:
        # A reasoning parser can put the whole generation in the reasoning channel and send no answer
        # at all: vLLM and SGLang do exactly that whenever thinking is on, which is a deployment
        # setting rather than something another attempt could fix.
        note += (
            " — the answer was empty and the response carried reasoning only: this server separates "
            "reasoning from the answer, and its thinking may be on"
        )
    return note


_COMPLETE_STOPS: Mapping[str, frozenset[str]] = {
    "chat_completions": frozenset({"stop"}),
    "responses": frozenset(),
    "messages": frozenset({"end_turn", "stop_sequence"}),
}
"""What each surface calls a finished generation. The Responses surface needs no entry: it reports
``status`` separately and jevper only fills ``stop`` there when the response did not complete."""


def answer_failure(result: CallResult) -> ProviderError | None:
    """The reason this response carries no answer, when the provider gave one.

    A generation that was cut short, filtered, or refused is not a malformed answer to be corrected:
    reading it would turn a truncated or declined generation into a typed decision, and another
    attempt spends a call to be cut short or refused the same way. The TypeSafe reference adapter
    rejects the same three cases for the same reason.
    """
    if result.refusal or result.stop in _REFUSAL_STOPS:
        message = "the model refused to answer"
        if result.refusal:
            message += f": {result.refusal[:200]!r}"
        elif result.stop == "content_filter":
            message += ": the provider filtered the content for safety"
        if result.stop:
            message += f" — the provider reported {result.stop!r}"
        return ModelRefusalError(message)
    if result.stop in _TRUNCATED_STOPS:
        return IncompleteAnswerError(_truncation_message(result.stop, result.surface))
    if result.stop is not None and result.stop not in _COMPLETE_STOPS.get(result.surface, frozenset()):
        return IncompleteAnswerError(
            f"the provider stopped before the answer was complete ({result.stop!r})"
        )
    return None


_LABEL_SHAPED = re.compile(r"^([A-Z]+)(?=$|[\s.,:;!?)}\]])")
"""A label at the very start of an answer, up to its first letter run and then nothing but a
separator. The punctuation matters as much as the letters: a model that answers ``A.`` or ``A)``
names label ``A``, and a check that stopped at the first non-letter would read the answer as prose
and let a sampled token that contradicts it through."""



def _agree_with_answer_text(sampled: str, text: str, labels: Sequence[str]) -> None:
    """Refuse a sampled token that contradicts an answer text that names a different label.

    The label readout reads the sampled token; the answer text is the same generation seen through
    another channel. When both are label-shaped and they disagree, one of the two is not this
    answer — a proxy stitching two responses, a server whose logprobs belong to another request — and
    reporting either as the answer would be a coin flip presented as a decision. The bounded
    correction path gets its turn instead.

    Both sides are read the same way — the leading label run of each — so a server whose sampled
    token carries the sentence's punctuation (``A.``) agrees with a text that says ``A``, and only
    a genuine disagreement (``B`` against ``A.``) is refused.
    """
    stripped = text.strip()
    if not stripped:
        return
    named = _LABEL_SHAPED.match(ascii_upper(stripped))
    if named is None or named.group(1) not in labels:
        return
    token = _LABEL_SHAPED.match(ascii_upper(sampled.strip()))
    if token is None or token.group(1) == named.group(1):
        return
    raise LabelReadoutError(
        f"the sampled token {sampled!r} contradicts the answer text, which starts with the label "
        f"{named.group(1)!r}; the provider's logprobs and its text are not from the same generation"
    )


def first_answer_token(result: CallResult, labels: Sequence[str], *, method: Method) -> TokenLogprob:
    """The first non-whitespace token of the answer, which must be one of the labels."""
    if not result.token_logprobs:
        raise _LogprobsUnavailable(
            f"no logprobs returned for the answer token (method={method!r}); this provider does not "
            f"report them — use method='structured' for the model's own probabilities, or "
            f"method='discrete' for one label, neither of which needs logprobs{_stop_note(result)}",
            evidence="readout",
            surface=result.surface,
        )
    for token in _answer_tokens(result):
        if not token.token.strip():
            continue
        if ascii_upper(token.token.strip()) in labels:
            _agree_with_answer_text(token.token, result.text, labels)
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
    if len(labels) == 1:
        # A one-option question has no distribution to read: the sampled token is the answer and the
        # only label holds the whole of it. This is the case the one-option minimum makes reachable —
        # the Jev API documents a maximum of 255 options and no minimum, and the reference adapter
        # reports a one-option Choice with confidence 1.0 — so a pinned label readout answers it
        # rather than refusing a distribution that cannot exist.
        return Readout(
            probabilities={label_to_key(question, labels)[labels[0]]: 1.0},
            source=source,
            missing_labels=(),
            observed_text=result.text,
        )
    if token.reported_alternatives < 2 or not token.top_logprobs:
        # Nothing but the sampled token came back, so there is no distribution to read: the answer
        # would be a one-hot built from absence of data. A provider that reports entries but nulls
        # every one of them is the same absence in a different shape — those labels are missing
        # rather than unknown, and with no rival at all there is nothing to normalize over.
        unusable = "" if token.reported_alternatives < 2 else ", none of which carried a logprob"
        raise _LogprobsUnavailable(
            f"the provider returned {token.reported_alternatives} top_logprobs{unusable} for the answer "
            f"token {token.token!r} (method={method!r}), which is not a distribution over the options; "
            f"use method='structured' for the model's own probabilities, or method='discrete' for one "
            f"label",
            evidence="readout",
            surface=result.surface,
        )
    answer_label = ascii_upper(token.token.strip())
    logprobs: dict[str, float] = {label: float("-inf") for label in labels}
    logprobs[answer_label] = token.logprob
    rivals = 0
    for top_token, top_logprob in token.top_logprobs:
        # A log probability is never positive and never NaN, whatever token it belongs to: a broken
        # number beside the answer is a fact about the response, not a rival to drop quietly. The
        # check is the same one ``softmax_over_labels`` applies, run before the label filter so a
        # non-option token cannot smuggle an impossible value through.
        if math.isnan(top_logprob) or top_logprob == float("inf"):
            raise LabelReadoutError(
                f"logprob for the alternative token {top_token!r} must be finite, got {top_logprob!r}"
            )
        if top_logprob > 0.0:
            raise LabelReadoutError(
                f"logprob for the alternative token {top_token!r} is positive ({top_logprob!r}), "
                "which no log probability can be — the provider sent something other than logprobs"
            )
        candidate = ascii_upper(top_token.strip())
        if candidate in logprobs and candidate != answer_label:
            rivals += 1
            logprobs[candidate] = top_logprob
    if not rivals:
        # The provider listed several alternatives and every one of them was the sampled token again
        # or a token outside the option set: what is left is the sampled token's own logprob, and
        # normalizing over it alone would report certainty the provider never expressed.
        raise _LogprobsUnavailable(
            f"the provider returned {token.reported_alternatives} top_logprobs for the answer token "
            f"{token.token!r} (method={method!r}), none of which was an alternative among the "
            f"options {list(labels)!r}; use method='structured' for the model's own probabilities, "
            f"or method='discrete' for one label",
            evidence="readout",
            surface=result.surface,
        )
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
            surface=result.surface,
        )
    return _logprob_readout(result, question, labels, "grammar", "grammar")


def parse_json_object(text: str, note: str = "") -> dict[str, Any]:
    """The one JSON object in an answer, with the reason it stopped travelling on every failure.

    ``note`` says why the provider stopped, and it belongs on all three failure paths: an answer cut off
    mid-object — which is what a spent output budget looks like, and the ``{`` is there but the rest is
    not — reads as a parse bug otherwise, when the actionable fact is the budget.

    Prose around the object is tolerated, because a model that says "here you go: {...}" has answered.
    Every ``{`` the text holds is tried in turn, so a brace in prose (``the shape is {example}``) is
    stepped over rather than taken for the answer, and a later real object is still found. A
    *second* object that decodes is not tolerated: two answers in one response is a generation that
    contradicted itself, and reading the first would report one of the two as the answer without ever
    saying the other existed. The bounded correction path gets a turn at it instead.
    """
    try:
        value = json.loads(text)
    except RecursionError as deep:
        # CPython's decoder refuses to nest past its own limit, and the answer is a generation the
        # provider chose: a thousand opening braces is not an object jevper can read, and a
        # RecursionError from the decoder is not an exception this contract has. The bounded
        # correction path gets a turn at it instead.
        raise MalformedAnswerError(
            f"the answer's JSON is nested too deeply to parse{note}"
        ) from deep
    except ValueError as first_error:  # JSONDecodeError, and the int-conversion limit it shares
        index = text.find("{")
        if index < 0:
            raise MalformedAnswerError(
                f"no JSON object in the answer ({first_error}){note}"
            ) from first_error
        decoder = json.JSONDecoder()
        found: dict[str, Any] | None = None
        first_failure: ValueError | None = None
        while index >= 0:
            try:
                candidate, end = decoder.raw_decode(text[index:])
            except RecursionError as deep:
                raise MalformedAnswerError(
                    f"the answer's JSON is nested too deeply to parse{note}"
                ) from deep
            except ValueError as exc:
                if first_failure is None:
                    first_failure = exc
                index = text.find("{", index + 1)
                continue
            if found is None:
                found = candidate
                index = text.find("{", index + end)
                continue
            raise MalformedAnswerError(f"the answer carries more than one JSON object{note}")
        if found is None:
            raise MalformedAnswerError(
                f"could not parse a JSON object from the answer ({first_failure}){note}"
            ) from first_failure
        value = found
    if not isinstance(value, dict):
        raise MalformedAnswerError(f"expected a JSON object, got {type(value).__name__}{note}")
    return value


def _number(value: Any, *, where: str, upper: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MalformedAnswerError(f"{where} must be a finite number, got {value!r}")
    try:
        number = float(value)
    except OverflowError:
        # An integer with more digits than a float holds: json parses it happily, and both
        # ``math.isfinite`` and ``float`` raise OverflowError on it. It is not a number this contract
        # can carry, and the answer is as malformed as one that is out of range.
        raise MalformedAnswerError(
            f"{where} is too large to be a number, got an integer of "
            f"{len(str(abs(value)))} digits"
        ) from None
    if not math.isfinite(number):
        raise MalformedAnswerError(f"{where} must be a finite number, got {value!r}")
    if number < 0 or (upper is not None and number > upper):
        bound = ">= 0" if upper is None else f"in [0, {upper}]"
        raise MalformedAnswerError(f"{where} must be {bound}, got {number!r}")
    return number


def _require_root_keys(payload: Mapping[str, Any], expected: str, *, method: str) -> None:
    """The answer object must carry exactly the field this method asks for, and nothing else.

    The generated schema says ``additionalProperties: false`` and the docs call a missing or extra
    key malformed, so a body that breaks the contract is not read with the offending keys quietly
    dropped: ``{"probabilities": {...}, "choice": "technical"}`` is a model contradicting itself,
    and the field jevper happened to read is not the one it can vouch for. The bounded correction
    path gets a turn at it first.

    The message names the keys and not the values: a provider that pads the answer with a megabyte
    of its own text would otherwise put that megabyte into every error message, every attempt record
    and every log line that quotes the failure. The same holds for the key *names*, which are the
    provider's text too — one three-megabyte key name would take the place of the megabyte of values.
    Each name is escaped to printable UTF-8 and cut, and the check itself is on the untouched keys.
    """
    keys = [str(key) for key in payload]
    if sorted(keys) == [expected]:
        return
    shown = sorted(bounded_text(key, 200) for key in keys)
    raise MalformedAnswerError(
        f"the {method} answer must be an object with exactly {expected!r}, got keys {shown}"
    )


def readout_structured(result: CallResult, question: Question) -> Readout:
    payload = parse_json_object(result.text, _stop_note(result))
    _require_root_keys(payload, "probabilities" if question.type != "noul" else "noul", method="structured")
    if question.type == "choice":
        keys = list(question.criteria)
        raw = payload.get("probabilities")
        if not isinstance(raw, Mapping) or set(raw) != set(keys):
            raise MalformedAnswerError(
                f"'probabilities' must have exactly the option keys {sorted(keys)}, got keys "
                f"{sorted(str(key) for key in raw) if isinstance(raw, Mapping) else raw!r}"
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
            f"'probabilities' must have exactly the level keys {expected}, got keys "
            f"{sorted(str(key) for key in raw) if isinstance(raw, Mapping) else raw!r}"
        )
    return Readout(
        probabilities={
            int(level): _number(raw[level], where=f"probability for level {level}") for level in expected
        },
        source="structured",
        observed_text=result.text,
    )


def _level_index(value: Any, levels: Sequence[int]) -> int | None:
    """A level index given as an int, an integral float or an integral number in a string.

    "Integral in a string" is decided on the string, not on the float it converts to: ``"2"``,
    ``"2.0"`` and ``"2.000"`` are level 2, while ``"2.0000000000000000000001"`` is a number with a
    fractional part that a binary float rounds away. Accepting it would answer a question the model
    did not answer with a level it did not give.

    Membership is decided on the decimal, before any conversion to ``int``. ``"1e999999999"`` is a
    perfectly finite decimal, and converting it would try to build an integer with a billion digits
    out of nineteen characters of answer — the provider's text deciding how much memory the caller's
    process uses. The levels of a question are a handful of small integers, so comparing the decimal
    with them answers the question without ever materialising one.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value in levels else None
    if isinstance(value, float):
        # An integral float is exact up to 2**53, far above any question's level count, and a float
        # that is not integral is refused rather than rounded into one.
        if not value.is_integer():
            return None
        return int(value) if int(value) in levels else None
    if isinstance(value, str):
        try:
            number = Decimal(value.strip())
        except InvalidOperation:
            return None
        if not number.is_finite() or number != number.to_integral_value():
            return None
        if not any(number == Decimal(level) for level in levels):
            return None
        return int(number)
    return None


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
    expected = {"choice": "choice", "noul": "noul", "score": "score"}[question.type]
    _require_root_keys(payload, expected, method="discrete")
    if question.type == "choice":
        keys = list(question.criteria)
        raw = payload.get("choice")
        label = None
        if isinstance(raw, str):
            # An exact option key wins over a label, the same way an example's answer does: a key that
            # is also a label ("a" and "A") must not be read as the first option by accident, and a
            # key that is itself spelled with surrounding spaces (" billing ") is that key, not a
            # misspelling of it — so the exact match is tried on the answer as it arrived.
            if raw in keys:
                label = labels[keys.index(raw)]
            else:
                candidate = raw.strip()
                if candidate in keys:
                    label = labels[keys.index(candidate)]
                elif ascii_upper(candidate) in labels:
                    label = ascii_upper(candidate)
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
