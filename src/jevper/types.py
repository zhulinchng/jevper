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

from .errors import InvalidQuestionError
from .labels import MAX_LABEL_OPTIONS
from .reasoning import ReasoningContentPart

JSONContent = str | dict[str, Any] | list[Any]

CHOICE_MIN_OPTIONS = 2
CHOICE_MAX_OPTIONS = MAX_LABEL_OPTIONS
SCORE_MIN_LEVELS = 2
SCORE_MAX_LEVELS = 10  # Jev API limit


class Example(BaseModel):
    """One few-shot demonstration, rendered as a user/assistant turn pair.

    ``answer`` is the option key (choice), the level index (score) or a bool (noul); a label such as
    ``"B"`` is also accepted. ``probabilities`` is only read by ``method="structured"`` and defaults
    to a one-hot distribution over ``answer``.
    """

    model_config = ConfigDict(extra="forbid")

    state: Any
    answer: str | int | bool
    probabilities: Mapping[str | int, float] | None = None


class NoulCriteria(BaseModel):
    model_config = ConfigDict(extra="forbid")

    true: JSONContent | None = None
    false: JSONContent | None = None


class Noul(BaseModel):
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


class Choice(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["choice"] = "choice"
    instructions: JSONContent | None = None
    criteria: Mapping[str, JSONContent | None]
    examples: tuple[Example, ...] = Field(default=(), exclude=True)

    @model_validator(mode="after")
    def _check_criteria(self) -> Choice:
        validate_question(self)
        return self


class Score(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["score"] = "score"
    instructions: JSONContent | None = None
    criteria: Sequence[JSONContent]
    examples: tuple[Example, ...] = Field(default=(), exclude=True)

    @model_validator(mode="after")
    def _check_criteria(self) -> Score:
        validate_question(self)
        return self


Question = Noul | Choice | Score

Method = Literal["logprobs", "grammar", "structured", "discrete"]
Api = Literal["auto", "chat_completions", "responses"]


def validate_question(question: Question, question_id: str | None = None) -> None:
    """Question-intrinsic limits. Also called from the model validators, so direct construction
    fails as fast as a dict passed to ``system_one``."""
    where = f"question {question_id!r}" if question_id is not None else f"{question.type} question"
    if isinstance(question, Choice):
        count = len(question.criteria)
        if not CHOICE_MIN_OPTIONS <= count <= CHOICE_MAX_OPTIONS:
            raise InvalidQuestionError(
                f"{where}: choice needs {CHOICE_MIN_OPTIONS}..{CHOICE_MAX_OPTIONS} options "
                f"({CHOICE_MAX_OPTIONS} is the label cap; split the question to exceed it), got {count}"
            )
    elif isinstance(question, Score):
        count = len(question.criteria)
        if not SCORE_MIN_LEVELS <= count <= SCORE_MAX_LEVELS:
            raise InvalidQuestionError(
                f"{where}: score needs {SCORE_MIN_LEVELS}..{SCORE_MAX_LEVELS} levels, got {count}"
            )


QuestionAdapter = TypeAdapter(Annotated[Noul | Choice | Score, Field(discriminator="type")])


def parse_question(question_id: str, raw: Question | Mapping[str, Any]) -> Question:
    """Coerce a question instance or raw mapping into a validated ``Question``."""
    if isinstance(raw, (Noul, Choice, Score)):
        validate_question(raw, question_id)
        return raw
    try:
        question = QuestionAdapter.validate_python(raw)
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(part) for part in error['loc']) or 'body'}: {error['msg']}"
            for error in exc.errors(include_url=False)
        )
        raise InvalidQuestionError(f"question {question_id!r} is invalid: {problems}") from exc
    validate_question(question, question_id)
    return question


class NoulAnswer(BaseModel):
    type: Literal["noul"] = "noul"
    noul: float


class ChoiceAnswer(BaseModel):
    type: Literal["choice"] = "choice"
    choice: str
    probabilities: dict[str, float]
    confidence: float


class ScoreAnswer(BaseModel):
    type: Literal["score"] = "score"
    score: float
    legend: dict[int, str | dict[str, Any] | list[Any]]
    probabilities: dict[int, float]
    confidence: float


Answer = Annotated[NoulAnswer | ChoiceAnswer | ScoreAnswer, Field(discriminator="type")]


class Usage(BaseModel):
    """Aggregated over every provider call and retry of one ``system_one`` call.

    A count is ``None`` when any constituent call omitted it; reported zeros are preserved.
    """

    input_tokens: int | None = None
    output_tokens: int | None = None
    reasoning_tokens: int | None = None
    n_calls: int = 0
    n_retries: int = 0
    latency: float = 0.0


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
