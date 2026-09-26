"""The surface ``docs/api.md`` publishes: the error taxonomy, the constants, and the exported names.

Everything here is asserted against the module the documentation names, so a value that drifts, a
class that leaves its base, a name that stops being exported or a default the endpoint did not send
is caught where a caller would hit it rather than in a form that cannot fail.
"""

from __future__ import annotations

from types import ModuleType
from typing import Any

import pytest
from fakes import openai_client, openai_root_client
from pydantic import ValidationError

import jevper
import jevper.client
import jevper.labels
import jevper.normalize
import jevper.types
from jevper import (
    Choice,
    ClientCapabilityError,
    IncompleteAnswerError,
    InvalidQuestionError,
    JevperError,
    LabelReadoutError,
    MalformedAnswerError,
    ModelMetadata,
    ModelRefusalError,
    NativeSystemOneResponse,
    Noul,
    ProviderError,
    Routing,
    SystemOneClient,
    UnsupportedMethodError,
)

CRITERIA = {"billing": None, "technical": None, "sales": None}


# -- a provider failure is told apart from a missing route --------------------------------------


def test_a_404_from_the_status_line_is_not_an_embedded_error(stub_server):
    """A body's ``404`` is not a missing route; a status line's is. Only the second may rotate.

    ``embedded`` is exactly that distinction, so an ordinary 404 must say ``False``.
    """
    stub = stub_server(
        chat=lambda _: (404, {"error": {"message": "no such route", "type": "invalid_request_error"}})
    )
    client = SystemOneClient(openai_client(stub), model="m", api="chat_completions")

    with pytest.raises(ProviderError) as error:
        client.system_one(state="The charge appeared twice", questions={"q": Choice(criteria=CRITERIA)})

    assert error.value.status_code == 404
    assert error.value.embedded is False


# -- the error taxonomy --------------------------------------------------------------------------


EXPORTED_ERRORS = [
    JevperError,
    InvalidQuestionError,
    UnsupportedMethodError,
    ClientCapabilityError,
    LabelReadoutError,
    MalformedAnswerError,
    ProviderError,
    IncompleteAnswerError,
    ModelRefusalError,
]


@pytest.mark.parametrize("error", EXPORTED_ERRORS, ids=lambda error: error.__name__)
def test_every_exported_error_inherits_from_the_library_error(error):
    """The documented ``except JevperError`` catches every one of them."""
    assert issubclass(error, JevperError)


def test_the_two_named_provider_errors_are_caught_as_provider_errors():
    """``IncompleteAnswerError`` and ``ModelRefusalError`` are ``ProviderError`` subclasses."""
    for error in (IncompleteAnswerError, ModelRefusalError):
        assert issubclass(error, ProviderError)
        with pytest.raises(ProviderError):
            raise error("the generation did not complete")


# -- the documented constants --------------------------------------------------------------------

DOCUMENTED_CONSTANTS: list[tuple[ModuleType, str, Any]] = [
    (jevper.client, "METHODS", ("logprobs", "grammar", "structured", "discrete")),
    (jevper.client, "METHOD_SELECTIONS", ("auto", "logprobs", "grammar", "structured", "discrete")),
    (jevper.client, "APIS", ("auto", "chat_completions", "responses", "messages", "systemone")),
    (jevper.client, "AUTO_METHOD", "logprobs"),
    (jevper.client, "FALLBACK_METHOD", "structured"),
    (jevper.client, "TRANSIENT_STATUS_CODES", frozenset({408, 409, 429})),
    (jevper.client, "MAX_TOP_LOGPROBS", 20),
    (jevper.client, "MAX_PROMPT_CACHE_KEY", 256),
    (jevper.labels, "MAX_LABEL_OPTIONS", 26),
    (jevper.labels, "MAX_CHOICE_OPTIONS", 255),
    (jevper.types, "CHOICE_MAX_OPTIONS", 255),
    (jevper.types, "CHOICE_MIN_OPTIONS", 1),
    (jevper.types, "SCORE_MIN_LEVELS", 2),
    (jevper.types, "SCORE_MAX_LEVELS", 10),
    (jevper.normalize, "PROBABILITY_TOLERANCE", 1e-6),
]


@pytest.mark.parametrize(
    ("module", "name", "value"),
    DOCUMENTED_CONSTANTS,
    ids=[f"{module.__name__}.{name}" for module, name, _ in DOCUMENTED_CONSTANTS],
)
def test_a_documented_constant_keeps_its_published_value(module, name, value):
    """Every constant ``docs/api.md`` names, at the value it publishes for callers to validate against."""
    assert getattr(module, name) == value


# -- the exported names --------------------------------------------------------------------------

EXPORTED_NAMES = [
    "Answer",
    "Api",
    "AsyncSystemOneClient",
    "Choice",
    "ChoiceAnswer",
    "ClientCapabilityError",
    "Example",
    "Examples",
    "IncompleteAnswerError",
    "InvalidQuestionError",
    "JSONContent",
    "JevperError",
    "LabelReadoutError",
    "LayaExtras",
    "MalformedAnswerError",
    "Method",
    "MethodSelection",
    "ModelMetadata",
    "ModelRefusalError",
    "NativeSystemOneResponse",
    "Noul",
    "NoulAnswer",
    "NoulCriteria",
    "ProviderError",
    "Question",
    "Readout",
    "ReasoningConfig",
    "ReasoningContentPart",
    "ReasoningSummaryPart",
    "ReasoningTextPart",
    "RetryPolicy",
    "Routing",
    "Score",
    "ScoreAnswer",
    "SystemOneClient",
    "SystemOneResponse",
    "UnsupportedMethodError",
    "Usage",
    "__version__",
    "reasoning_text",
]


@pytest.mark.parametrize("name", EXPORTED_NAMES)
def test_an_exported_name_is_importable_from_jevper_and_listed(name):
    """``__all__`` is the contract ``from jevper import *`` reads: a name in it must exist."""
    assert name in jevper.__all__
    assert getattr(jevper, name) is not None


# -- defaults a response type owns ---------------------------------------------------------------


def test_a_model_description_defaults_to_empty():
    """``GET /v1/models`` may describe a model not at all, and ``""`` is how that is said."""
    assert ModelMetadata(name="stub").description == ""
    assert ModelMetadata(name="stub", description="a model").description == "a model"


def test_a_routing_report_must_name_the_route_key():
    """``route`` is the stable key to branch on, so a report without one is not this type."""
    with pytest.raises(ValidationError):
        Routing(router="laya:latest", model="laya:en")

    assert Routing(router="laya:latest", model="laya:en", route="english").route == "english"


# -- the native endpoint's own fields ------------------------------------------------------------


def test_the_native_endpoints_omitted_fields_keep_their_documented_defaults(stub_server):
    """``done_reason`` and ``created_at`` come from the body, not from a value jevper invented.

    The endpoint reports both; a body that leaves them out has not made the claim, so the type's own
    defaults are what a reader gets rather than a plausible-looking substitute.
    """
    body = {
        "model": "laya:en",
        "answers": {"refund": {"type": "noul", "noul": 0.5}},
        "usage": {"input_tokens": 3},
    }
    stub = stub_server(decide=lambda _: (200, body))
    client = SystemOneClient(openai_root_client(stub), model="laya:en", api="systemone", native=True)

    response = client.system_one(
        state="The charge appeared twice", questions={"refund": Noul(instructions="Refund?")}
    )

    assert type(response) is NativeSystemOneResponse
    assert response.done_reason == "decide"
    assert response.created_at == ""
