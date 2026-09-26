"""Wire-shaped types for the Jev (TypeSafe System One) surface.

Answer field names and JSON keys match ``POST /v1/systemone`` exactly, because ``model_dump_json()``
is the compatibility contract with the hosted API.
"""

from __future__ import annotations

import functools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
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

from .errors import InvalidQuestionError, JevperError, MalformedAnswerError
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


@dataclass(frozen=True)
class SystemOnePayload:
    """One System One request: the state, and the questions asked of it.

    The state is the caller's value as they passed it, not a rendered prompt: this wire format takes
    structured state, so a dict or a list arrives as itself rather than as text describing itself.
    """

    state: Any
    questions: Mapping[str, Question]


def question_on_wire(question: Question) -> dict[str, Any]:
    """A question as ``POST /v1/systemone`` takes it.

    An optional field the caller did not set is left off the body, as the service's own client does,
    while a ``null`` *inside* ``criteria`` is kept: ``{"billing": null}`` is how a caller says this
    option needs no description, and dropping the key would drop the option with it. Few-shot
    examples are jevper's own field and have no place on this wire, which is why a question carrying
    them is refused before a request rather than quietly sent without them.
    """
    wire: dict[str, Any] = {"type": question.type}
    if question.instructions is not None:
        wire["instructions"] = question.instructions
    criteria = question.criteria
    if criteria is None:
        return wire
    if isinstance(criteria, NoulCriteria):
        sides = criteria.model_dump(exclude_none=True)
        # An object with nothing in it says no more than leaving the key off, and the service's own
        # answer to a noul with no criteria is the same 400 either way — so the key is left off.
        if sides:
            wire["criteria"] = sides
        return wire
    if isinstance(criteria, Mapping):
        wire["criteria"] = dict(criteria)
    else:
        wire["criteria"] = list(criteria)
    return wire

Method = Literal["logprobs", "grammar", "structured", "discrete", "systemone"]
# What you may pass as `method`: a concrete method, "auto" to resolve one by observation, or nothing
# to let the surface decide. "systemone" is not among them: it is what the System One endpoint's own
# method is reported as, never something a caller can ask for on another surface.
MethodSelection = Literal["auto", "logprobs", "grammar", "structured", "discrete"]
Api = Literal["auto", "chat_completions", "responses", "messages", "systemone"]


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
    # A noul carrying no instructions and no criteria is refused on the System One surface, in
    # `_prepare_systemone`, rather than here: the service answers 400 for one, but on a prompt
    # surface jevper can render the question from a few-shot example, which `Noul(examples=…)`
    # is.
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


class LayaExtras(BaseModel):
    """Ollaya's own per-answer readout, added by ``extras=["laya"]`` on its native endpoint.

    Namespaced under ``laya`` so it cannot be read as the ``confidence`` beside it: the two are
    different numbers from different heads — this one is 1 − H(p)/ln K for a choice or a score and
    max(p, 1 − p) for a noul. ``None`` on an answer means the request did not ask for it, which is
    what every other surface sees, so nothing that reads a ``confidence`` changes meaning.
    """

    confidence: float = Field(ge=0.0, le=1.0)
    act_probability: float | None = Field(default=None, ge=0.0, le=1.0)
    """Whether acting on this answer is appropriate, or ``None`` for a model with no act head — the
    service reports the absence as null rather than omitting the number, and jevper says ``None``."""


class Routing(BaseModel):
    """Which checkpoint a router model chose, as ollaya's native endpoint reports it.

    ``route`` is the stable key to branch on. ``reason`` is the router's own prose, which the service
    documents as informative and free to change in any release, so it is carried but never parsed.
    """

    router: str
    model: str
    route: str
    reason: str = ""

    @field_validator("reason", mode="before")
    @classmethod
    def _no_prose_is_no_reason(cls, value: Any) -> Any:
        """``null`` reads as the absent key does.

        Every other optional native field is spelled ``null`` rather than omitted — that is what
        ``act_probability`` is declared ``float | None`` for — so a service saying "no prose here"
        that way is saying what an omitted key says, not sending something unreadable.
        """
        return "" if value is None else value


class NoulAnswer(BaseModel):
    type: Literal["noul"] = "noul"
    noul: float = Field(allow_inf_nan=False)
    """A regression output, so any real number is a score; only a non-number is not an answer."""
    laya: LayaExtras | None = None
    """Ollaya's own confidence, when its native endpoint was asked for it. ``None`` everywhere else."""


class _Probability(BaseModel):
    """The one field both answers compute: a confidence, which is a share of certainty in ``[0, 1]``.

    The bound belongs here because jevper computed the number — the readouts' confidence formula is
    bounded by construction, so a value outside ``[0, 1]`` was assembled by hand and is not something
    jevper would have produced. The probabilities beside it are the provider's own numbers, and with
    ``normalize_probabilities=False`` they are deliberately passed through as they arrived: bounding
    them would turn a distribution the caller asked to see for themselves into a validation error.
    Finite is not a bound, though — ``NaN`` and ``Infinity`` are not numbers to reason about, and
    ``json.loads`` reads both literals without complaint, so they are refused here.
    """

    probabilities: dict[Any, Probability]
    confidence: float = Field(ge=0.0, le=1.0)
    laya: LayaExtras | None = None
    """Ollaya's own confidence, when its native endpoint was asked for it. ``None`` everywhere else."""


class ChoiceAnswer(_Probability):
    type: Literal["choice"] = "choice"
    choice: str

class ScoreAnswer(_Probability):
    type: Literal["score"] = "score"
    score: float = Field(allow_inf_nan=False)
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


class NativeSystemOneResponse(SystemOneResponse):
    """A System One response from ollaya's native endpoint, which reports more than TypeSafe's.

    Every field the native endpoint adds is additive, so the answers and the usage are the ones above
    and this is that same response with the endpoint's own report beside them. ``model`` above stays
    the name the caller asked for: on the wire it is the checkpoint that answered, and reading that
    here instead would make one field mean two things across jevper's surfaces. The checkpoint is
    ``routing.model``.

    Durations are the service's own nanoseconds, passed through as sent rather than converted, so a
    number read here is one that can be compared against the service's own log. ``None`` means the
    endpoint reported nothing, which is what every other surface says about a field it has no value
    for.
    """

    routing: Routing | None = None
    """The router's decision, or ``None`` when a model named directly answered."""
    state_truncated: bool = False
    """Whether part of the state was dropped to fit the model's context."""
    done_reason: str = "decide"
    created_at: str = ""
    """When the endpoint produced the response, in the service's own timestamp format."""
    total_duration: int | None = None
    load_duration: int | None = None
    eval_duration: int | None = None


class ModelMetadata(BaseModel):
    """One model the System One endpoint offers, as ``GET /v1/models`` describes it."""

    name: str
    description: str = ""
    release_date: str | None = None
    """``YYYY-MM-DD`` when the service reports one; ``None`` when it does not."""


_ANSWER_TYPES: dict[str, type[Answer]] = {
    "noul": NoulAnswer,
    "choice": ChoiceAnswer,
    "score": ScoreAnswer,
}


def parse_answer(question_id: str, question: Question, payload: Any) -> Answer:
    """One answer as the System One endpoint sent it, as jevper's own answer type.

    The other three surfaces hand a distribution to ``methods.readout`` and jevper computes the
    score, the confidence and the choice from it. This one is the reverse: the service computed all
    three from a model trained for the decision, and recomputing them here would replace its numbers
    with jevper's — which agree to within 0.015 (see ``docs/jev-comparison.md``) but are not the
    service's. So the answer is read as it arrived, with three checks the type system cannot make:
    the answer is for a question that was asked, its ``type`` matches the question's, and the keys of
    a score's distribution are its own levels.
    """
    if not isinstance(payload, Mapping):
        raise MalformedAnswerError(
            f"question {question_id!r}: the System One answer must be an object, got "
            f"{type(payload).__name__}"
        )
    kind = payload.get("type")
    if kind != question.type:
        raise MalformedAnswerError(
            f"question {question_id!r}: asked a {question.type} and the service answered "
            f"{kind!r}"
        )
    if question.type == "score":
        levels = _level_texts(question)
        raw = payload.get("probabilities", {})
        if not isinstance(raw, Mapping):
            # Read before pydantic gets it, so a null here is a jevper error naming the question
            # rather than the ``TypeError`` an unguarded iteration of it would raise.
            raise MalformedAnswerError(
                f"question {question_id!r}: the service's score distribution must be an object of "
                f"level probabilities, got {type(raw).__name__}"
            )
        # Both directions, because a decision has to be about the question that was asked. A level the
        # rubric does not have is an answer to something else; a rubric level the service left out is
        # a decision the caller cannot read, since ``answer.probabilities[level]`` is how the
        # documented arithmetic reaches it. The prompt surfaces make the same exact-set check
        # (``methods.readout_structured``), so a service and a model answer under one rule.
        answered = {str(key) for key in raw}
        if answered != levels:
            raise MalformedAnswerError(
                f"question {question_id!r}: the service's score distribution must carry exactly "
                f"this rubric's levels {sorted(levels)}, got {sorted(answered)}"
            )
        # The levels are this rubric's own indices, and JSON object keys are always text. ``legend``
        # beside it is typed ``dict[int, JSONContent]`` and the prompt surfaces build these keys with
        # ``int(level)``, so leaving them as text here would make one answer type mean two things and
        # turn the documented ``answer.probabilities[0]`` into a KeyError on this surface alone.
        payload = {**payload, "probabilities": {int(key): value for key, value in raw.items()}}
    if question.type == "choice":
        # The same check the score branch makes, for the same reason: a choice naming an option the
        # caller never offered is a decision about a question that was not asked, and the caller
        # indexes their own criteria with it.
        options = {str(option) for option in question.criteria}
        probabilities = payload.get("probabilities")
        unknown = sorted(
            str(key)
            for key in (probabilities if isinstance(probabilities, Mapping) else {})
            if str(key) not in options
        )
        chosen = payload.get("choice")
        if unknown or (chosen is not None and str(chosen) not in options):
            raise MalformedAnswerError(
                f"question {question_id!r}: the service answered with "
                f"{unknown or [chosen]!r}, which are not this question's {sorted(options)}"
            )
    try:
        return _ANSWER_TYPES[kind].model_validate(payload)
    except ValidationError as exc:
        raise MalformedAnswerError(
            f"question {question_id!r}: {_validation_message(exc)}"
        ) from None


def noul_carries_a_question(question: Noul) -> bool:
    """Whether a noul says what it is asking, which the System One wire format requires.

    Measured against the live service on 2026-09-26: it answers 400 for a noul with no
    ``instructions``, with ``instructions=""``, with ``criteria={}`` and with
    ``criteria={"true": null, "false": null}`` — "Noul question must have criteria or
    instructions" — and 200 for either side carrying any value. So the test is whether a value is
    there, not whether the key is, and a ``NoulCriteria`` with both sides unset does not count.
    """
    if question.instructions:
        return True
    criteria = question.criteria
    return criteria is not None and (criteria.true is not None or criteria.false is not None)


def _level_texts(question: Question) -> set[str]:
    """A score rubric's level indices as the wire spells them."""
    return {str(level) for level in range(len(question.criteria))}


def parse_models(payload: Any) -> list[ModelMetadata]:
    """``GET /v1/models`` as the OpenAPI schema describes it: ``{"models": [...]}``.

    Only that shape is read. A gateway that answers the path with its own model list — an OpenAI-style
    ``{"data": [{"id": ...}]}``, which is what opencode Zen serves on the same URL — is a different
    API wearing this path, and saying so is more use than guessing which of its fields was meant to
    be a release date.
    """
    if not isinstance(payload, Mapping) or not isinstance(payload.get("models"), list):
        # What arrived is the point of this message: the reader is deciding whether a gateway
        # answered a different API, a proxy dropped a field, or the service is not what it claims.
        raise MalformedAnswerError(
            "the model list must be an object with a 'models' array, as GET /v1/models documents; "
            f"got {type(payload).__name__} {payload!r:.100}"
        )
    models: list[ModelMetadata] = []
    for index, entry in enumerate(payload["models"]):
        try:
            models.append(ModelMetadata.model_validate(entry))
        except ValidationError as exc:
            raise MalformedAnswerError(f"model {index}: {_validation_message(exc)}") from None
    return models
