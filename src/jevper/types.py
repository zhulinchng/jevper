"""Wire-shaped types for the Jev (TypeSafe System One) surface.

Answer field names and JSON keys match ``POST /v1/systemone`` exactly, because ``model_dump_json()``
is the compatibility contract with the hosted API.
"""

from __future__ import annotations

import functools
from collections.abc import Mapping, Sequence
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationError,
    field_validator,
    model_validator,
)

from .errors import InvalidQuestionError, JevperError
from .labels import MAX_CHOICE_OPTIONS
from .reasoning import ReasoningContentPart

JSONContent = str | int | float | bool | None | dict[str, Any] | list[Any]
Probability = Annotated[float, Field(allow_inf_nan=False)]

CHOICE_MIN_OPTIONS = 1
# The Jev API documents a maximum of 255 Choice options and no minimum, and its own reference adapter
# answers a one-option Choice with confidence 1.0, so refusing one here would reject a request the
# wire format accepts. A Score keeps its documented 2..10 levels: a one-level rubric has no ordering
# to read, which is why the API asks for two there and says nothing about Choice.
CHOICE_MAX_OPTIONS = MAX_CHOICE_OPTIONS
SCORE_MIN_LEVELS = 2
SCORE_MAX_LEVELS = 10  # Jev API limit


def _validation_message(exc: ValidationError) -> str:
    """Every problem pydantic found, as ``field.path: what is wrong``."""
    return "; ".join(
        f"{'.'.join(str(part) for part in error['loc']) or 'body'}: {error['msg']}"
        for error in exc.errors(include_url=False)
    )


class _CallerModel(BaseModel):
    """A model the caller builds, so an invalid one fails as ``InvalidQuestionError``.

    Pydantic's ``ValidationError`` is the right error for a pydantic model, but a caller of this
    library catches ``JevperError`` — and these models are the library's own types, so what
    pydantic rejects here is exactly what the library documents it rejects: a question that is
    locally invalid. A question passed as a mapping already fails this way (``parse_question``),
    so this is what makes the two documented paths — building a question, and handing one over
    as data — fail alike, with the same fields named.
    """

    def __init__(self, **data: Any) -> None:
        try:
            super().__init__(**data)
        except ValidationError as exc:
            raise InvalidQuestionError(f"{type(self).__name__}: {_validation_message(exc)}") from None


class Example(_CallerModel):
    """One few-shot demonstration, rendered as a user/assistant turn pair.

    ``answer`` is the option key (choice), the level index (score) or a bool (noul); a label such as
    ``"B"`` is also accepted. ``probabilities`` is only read by ``method="structured"`` and defaults
    to a one-hot distribution over ``answer``.
    """

    model_config = ConfigDict(extra="forbid")

    state: Any
    answer: str | int | bool
    probabilities: Mapping[str | int, Probability] | None = None


class NoulCriteria(BaseModel):
    model_config = ConfigDict(extra="forbid")

    true: JSONContent | None = None
    false: JSONContent | None = None


class Noul(_CallerModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["noul"] = "noul"
    instructions: JSONContent | None = None
    criteria: NoulCriteria | None = None
    examples: tuple[Example, ...] = Field(default=(), exclude=True)

    @field_validator("criteria", mode="before")
    @classmethod
    def _check_criteria(cls, value: Any) -> Any:
        if isinstance(value, Mapping):
            unexpected = sorted(set(value) - {"true", "false"})
            if unexpected:
                raise InvalidQuestionError(
                    f"noul criteria keys must be 'true'/'false', got unexpected {unexpected!r}"
                )
        return value

    @model_validator(mode="after")
    def _check_question(self) -> Noul:
        validate_question(self)
        return self


class Choice(_CallerModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["choice"] = "choice"
    instructions: JSONContent | None = None
    criteria: Mapping[str, JSONContent | None]
    examples: tuple[Example, ...] = Field(default=(), exclude=True)

    @model_validator(mode="after")
    def _check_criteria(self) -> Choice:
        validate_question(self)
        return self


class Score(_CallerModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["score"] = "score"
    instructions: JSONContent | None = None
    criteria: Sequence[JSONContent]
    examples: tuple[Example, ...] = Field(default=(), exclude=True)

    @model_validator(mode="after")
    def _check_criteria(self) -> Score:
        validate_question(self)
        return self


def bounded_text(value: str, limit: int = 500) -> str:
    """A provider's own words, printable and no longer than ``limit``.

    Two things go wrong when provider text reaches a caller as-is. A body can carry an escaped lone
    surrogate, which decodes into a string no UTF-8 encoder accepts, and an error message the caller
    cannot print is a second failure on top of the first — ``backslashreplace`` keeps the characters
    visible as the escapes the wire carried. And a gateway that answers with a megabyte of HTML makes
    every message, attempt record and log line that quotes it a megabyte too; the first few hundred
    characters carry the status, the field and the reason, which is all a reader uses.

    Every path that quotes a provider — an exception's message, an embedded error, a retry reason —
    goes through this one function, so the bound cannot be forgotten on one surface.
    """
    text = value.encode("utf-8", "backslashreplace").decode("utf-8")
    return text if len(text) <= limit else text[:limit] + f"… (+{len(text) - limit} chars)"


def ensure_encodable(value: Any, *, where: str, error: type[JevperError] = JevperError) -> None:
    """Refuse text a request could never carry, before anything tries to send it.

    Python strings may hold unpaired surrogates — the code points a broken decoder leaves behind —
    and JSON can carry them as ``\\udXXX`` escapes, so a state read from a file, or an answer echoed
    back into the next question, can hold one. Nothing downstream can encode such a string: the
    SDK's serializer raises ``UnicodeEncodeError`` from inside the provider call, where the only
    description on offer is a provider failure, and hashing one for the prompt-cache key raises the
    same error before a request is even built. Both are local mistakes, so they are reported as
    local ones, with the field named and nothing sent.

    The walk keeps its own stack instead of recursing — a caller can hand over a structure nested
    deeper than the interpreter's recursion limit, and a ``RecursionError`` from a local validation
    pass is the one failure this function exists to prevent — and it remembers the containers it has
    entered. A structure that contains itself has no finite encoding either, and following it would
    never end: it is refused here, with the field named, rather than left to the serializer or to the
    recursion limit.
    """
    pending: list[Any] = [value]
    entered: set[int] = set()
    while pending:
        item = pending.pop()
        if isinstance(item, str):
            try:
                item.encode("utf-8")
            except UnicodeEncodeError as exc:
                raise error(
                    f"{where} contains a character that cannot be encoded as UTF-8 ({exc.reason} at "
                    f"position {exc.start}); replace it before handing it to jevper"
                ) from None
        elif isinstance(item, (Mapping, list, tuple)):
            if id(item) in entered:
                raise error(
                    f"{where} contains a structure that refers to itself, which no request can "
                    "carry; break the cycle before handing it to jevper"
                ) from None
            entered.add(id(item))
            if isinstance(item, Mapping):
                pending.extend(item.keys())
                pending.extend(item.values())
            else:
                pending.extend(item)


Question = Noul | Choice | Score

Method = Literal["logprobs", "grammar", "structured", "discrete"]
# What you may pass as `method`: a concrete method, or "auto" to resolve one by observation.
MethodSelection = Literal["auto", "logprobs", "grammar", "structured", "discrete"]
Api = Literal["auto", "chat_completions", "responses", "messages"]


def validate_question(question: Question, question_id: str | None = None) -> None:
    """Question-intrinsic limits. Also called from the model validators, so direct construction
    fails as fast as a dict passed to ``system_one``."""
    where = f"question {question_id!r}" if question_id is not None else f"{question.type} question"
    if isinstance(question, Choice):
        count = len(question.criteria)
        if not CHOICE_MIN_OPTIONS <= count <= CHOICE_MAX_OPTIONS:
            raise InvalidQuestionError(
                f"{where}: choice needs {CHOICE_MIN_OPTIONS}..{CHOICE_MAX_OPTIONS} options "
                f"({CHOICE_MAX_OPTIONS} is the Jev API limit; split the question to exceed it), got {count}"
            )
    elif isinstance(question, Score):
        count = len(question.criteria)
        if not SCORE_MIN_LEVELS <= count <= SCORE_MAX_LEVELS:
            raise InvalidQuestionError(
                f"{where}: score needs {SCORE_MIN_LEVELS}..{SCORE_MAX_LEVELS} levels, got {count}"
            )
    # Every string the question puts on the wire — its instructions, its option keys and their
    # descriptions — has to be encodable, and the cache key derived from them is hashed before the
    # first request. A lone surrogate in any of them is a question no request could carry.
    try:
        ensure_encodable(
            question.model_dump(mode="python"), where=where, error=InvalidQuestionError
        )
    except JevperError as exc:
        raise InvalidQuestionError(str(exc)) from None
    _check_examples(question, where)


def _check_examples(question: Question, where: str) -> None:
    """The examples a question carries, checked against that question, wherever it was built.

    An example's answer and its own numbers only mean something next to the question they
    demonstrate, which is why the check needs both — and why it belongs here, where the
    question is available however the caller spelled it. ``prompts`` imports this module, so the
    helpers are imported here rather than at the top.
    """
    from .labels import labels_for
    from .prompts import example_answer_label, validate_example_probabilities

    count = 2 if question.type == "noul" else len(question.criteria)
    labels = labels_for(count)
    for index, example in enumerate(question.examples):
        try:
            example_answer_label(question, labels, example.answer, index)
            if example.probabilities is not None:
                validate_example_probabilities(question, example.probabilities, index)
        except JevperError as exc:
            raise InvalidQuestionError(f"{where}: {exc}") from None


QuestionAdapter = TypeAdapter(Annotated[Noul | Choice | Score, Field(discriminator="type")])


def parse_question(question_id: str, raw: Question | Mapping[str, Any]) -> Question:
    """Coerce a question instance or raw mapping into a validated ``Question``."""
    if not isinstance(question_id, str):
        # The answer mapping is keyed by str, so a non-string id would fail only after the provider
        # call had been paid for. Fail with the library's own error, before anything is sent.
        raise InvalidQuestionError(
            f"question id must be a string, got {type(question_id).__name__} {question_id!r}"
        )
    if isinstance(raw, (Noul, Choice, Score)):
        validate_question(raw, question_id)
        return raw
    try:
        question = QuestionAdapter.validate_python(raw)
    except InvalidQuestionError as exc:
        # The question's own validators run inside the adapter, before this function can name the
        # question it came from — and a service with a rubric of thirty questions needs to be told
        # which one it is.
        raise InvalidQuestionError(f"question {question_id!r} is invalid: {exc}") from None
    except ValidationError as exc:
        raise InvalidQuestionError(
            f"question {question_id!r} is invalid: {_validation_message(exc)}"
        ) from exc
    validate_question(question, question_id)
    return question


class NoulAnswer(BaseModel):
    type: Literal["noul"] = "noul"
    noul: float
    """A regression output, so any real number is a score; only a non-number is not an answer."""


class _Probability(BaseModel):
    """The one field both answers compute: a confidence, which is a share of certainty in ``[0, 1]``.

    The bound belongs here because jevper computed the number — the readouts' confidence formula is
    bounded by construction, so a value outside ``[0, 1]`` was assembled by hand and is not something
    jevper would have produced. The probabilities beside it are the provider's own numbers, and with
    ``normalize_probabilities=False`` they are deliberately passed through as they arrived: bounding
    them would turn a distribution the caller asked to see for themselves into a validation error.
    """

    probabilities: dict[Any, float]
    confidence: float = Field(ge=0.0, le=1.0)


class ChoiceAnswer(_Probability):
    type: Literal["choice"] = "choice"
    choice: str

class ScoreAnswer(_Probability):
    type: Literal["score"] = "score"
    score: float
    """The rubric's own number — a level index, or whatever scale the criteria ask for."""
    legend: dict[int, JSONContent]


Answer = Annotated[NoulAnswer | ChoiceAnswer | ScoreAnswer, Field(discriminator="type")]


class Usage(BaseModel):
    """Aggregated over every provider call and retry of one ``system_one`` call.

    A count is ``None`` when any constituent call omitted it; reported zeros are preserved.
    """

    input_tokens: int | None = None
    output_tokens: int | None = None
    reasoning_tokens: int | None = None
    n_calls: int = Field(default=0, ge=0)
    n_retries: int = Field(default=0, ge=0)
    latency: float = Field(default=0.0, ge=0.0, allow_inf_nan=False)
    cached_tokens: int | None = None
    """Prompt tokens the provider read from its prompt cache. Reported by OpenAI, OpenRouter, vLLM,
    SGLang, llama.cpp and ollama; ``None`` when the provider said nothing, which is not the same as a
    reported ``0`` — that is a provider whose prefix cache is cold or off."""


class SystemOneResponse(BaseModel):
    model: str
    answers: dict[str, Answer]
    usage: Usage = Usage()
    reasoning: tuple[ReasoningContentPart, ...] = ()
    debug: dict[str, Any] = Field(default_factory=dict)

    @functools.cached_property
    def nouls(self) -> dict[str, NoulAnswer]:
        return {key: answer for key, answer in self.answers.items() if isinstance(answer, NoulAnswer)}

    @functools.cached_property
    def choices(self) -> dict[str, ChoiceAnswer]:
        return {key: answer for key, answer in self.answers.items() if isinstance(answer, ChoiceAnswer)}

    @functools.cached_property
    def scores(self) -> dict[str, ScoreAnswer]:
        return {key: answer for key, answer in self.answers.items() if isinstance(answer, ScoreAnswer)}
