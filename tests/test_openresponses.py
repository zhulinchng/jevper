"""The Responses surface read against both dialects it now has to survive.

OpenAI's Responses API and the OpenResponses specification (https://www.openresponses.org, served at
the same ``/v1/responses`` path by LM Studio since 0.3.39, llama.cpp, vLLM and SGLang) agree on the
shape of an answer and spell several things differently around it: an input turn carries its item
type, a message may arrive in phases, reasoning text sits under five different part types, every logprob
token comes with its bytes, and a response object can be an event stream. Each test here replays one
such body through the public client and asserts what a caller gets: the answer, or a named failure —
never a confident decision read from a body that says the generation did not finish.
"""

from __future__ import annotations

import json
import math

import pytest
from fakes import async_openai_client, chat_body, openai_client, responses_body

from jevper import (
    Choice,
    ClientCapabilityError,
    IncompleteAnswerError,
    ProviderError,
    ReasoningConfig,
    SystemOneClient,
)

CRITERIA = {"billing": None, "technical": None, "sales": None}
STRUCTURED = json.dumps({"probabilities": {"billing": 0.7, "technical": 0.2, "sales": 0.1}})


def question() -> Choice:
    return Choice(instructions="Pick the intent.", criteria=CRITERIA)


def client_for(stub, **kwargs) -> SystemOneClient:
    return SystemOneClient(
        openai_client(stub), model="stub", api="responses", method="structured",
        n_retry_malformed=0, **kwargs
    )


# ------------------------------------------------------------------ request: the portable form


def test_every_input_turn_carries_its_item_type(stub_server):
    """The OpenResponses input union discriminates on ``type`` and lists it as required.

    OpenAI accepts the ``{"role": ..., "content": ...}`` shorthand, but a server that validates
    against the spec refuses every turn without the type, and naming it costs nothing on OpenAI.
    """
    stub = stub_server(responses=lambda _: (200, responses_body(text=STRUCTURED)))
    client_for(stub).system_one(state="s", questions={"q": question()})

    sent = stub.bodies("/responses")[0]
    assert [turn["type"] for turn in sent["input"]] == ["message"] * len(sent["input"])
    assert {turn["role"] for turn in sent["input"]} == {"system", "user"}


def test_the_schema_in_the_prompt_keeps_the_item_types(stub_server):
    """The structured path rewrites the input to carry the schema; the types come with it."""
    stub = stub_server(responses=lambda _: (200, responses_body(text=STRUCTURED)))
    client_for(stub, structured_outputs=False).system_one(state="s", questions={"q": question()})

    sent = stub.bodies("/responses")[0]
    assert all(turn["type"] == "message" for turn in sent["input"])
    assert "validates against this JSON Schema" in sent["input"][0]["content"]


def test_a_caller_who_owns_the_input_keeps_it(stub_server):
    """A caller's own ``input`` is the value that reaches the wire, so the recorded request says so.

    The SDK merges ``extra_body`` after the typed parameters; jevper therefore does not also set the
    typed key, and ``debug`` — which records the kwargs — shows the body the provider received.
    """
    seen: list[dict] = []

    def script(body):
        seen.append(body)
        return 200, responses_body(text=STRUCTURED)

    stub = stub_server(responses=script)
    client = SystemOneClient(
        openai_client(stub),
        model="stub",
        api="responses",
        method="structured",
        extra_body={"input": [{"type": "message", "role": "user", "content": "mine"}], "store": True},
        n_retry_malformed=0,
    )
    response = client.system_one(state="s", questions={"q": question()})

    assert seen[0]["input"] == [{"type": "message", "role": "user", "content": "mine"}]
    assert seen[0]["store"] is True
    recorded = response.debug["llm_attempts"][0]["request"]
    assert "input" not in recorded and "store" not in recorded
    assert recorded["extra_body"]["input"] == seen[0]["input"]


def test_a_caller_who_turns_logprobs_off_gets_no_logprob_fields(stub_server):
    """``logprobs: false`` is Chat Completions' switch; this surface honours it the same way."""
    from jevper.transport import CallSpec, build_responses_kwargs

    spec = CallSpec(messages=[{"role": "user", "content": "x"}], logprobs=True, top_logprobs=5)
    kwargs = build_responses_kwargs(spec, model="stub", extra_body={"logprobs": False})
    assert "top_logprobs" not in kwargs
    assert "include" not in kwargs
    kwargs = build_responses_kwargs(spec, model="stub")
    assert kwargs["top_logprobs"] == 5
    assert kwargs["include"] == ["message.output_text.logprobs"]


def test_caller_owned_top_logprobs_are_not_duplicated_by_a_typed_one(stub_server):
    from jevper.transport import CallSpec, build_responses_kwargs

    spec = CallSpec(messages=[{"role": "user", "content": "x"}], logprobs=True, top_logprobs=20)
    kwargs = build_responses_kwargs(spec, model="stub", extra_body={"top_logprobs": 3})
    assert "top_logprobs" not in kwargs
    assert kwargs["extra_body"]["top_logprobs"] == 3


# ------------------------------------------------------------------ response: items and phases


def test_a_message_item_marked_incomplete_is_not_an_answer(stub_server):
    """OpenResponses gives every item its own lifecycle; ``incomplete`` means the budget ran out.

    A cut-off JSON object that happens to parse is exactly what a model produces when it runs out
    of room mid-write, so the partial text is reported as the truncation it is — with the knob that
    opens it on this surface.
    """
    body = responses_body(text=STRUCTURED)
    body["output"][-1]["status"] = "incomplete"
    body["incomplete_details"] = {"reason": "max_output_tokens"}
    stub = stub_server(responses=lambda _: (200, body))

    with pytest.raises(IncompleteAnswerError) as error:
        client_for(stub).system_one(state="s", questions={"q": question()})

    assert "max_output_tokens" in str(error.value)


def test_an_incomplete_item_without_details_is_still_an_incomplete_answer(stub_server):
    body = responses_body(text=STRUCTURED)
    body["output"][-1]["status"] = "incomplete"
    stub = stub_server(responses=lambda _: (200, body))

    with pytest.raises(IncompleteAnswerError):
        client_for(stub).system_one(state="s", questions={"q": question()})


def test_a_message_item_still_in_progress_is_a_provider_failure(stub_server):
    body = responses_body(text=STRUCTURED)
    body["output"][-1]["status"] = "in_progress"
    stub = stub_server(responses=lambda _: (200, body))

    with pytest.raises(ProviderError) as error:
        client_for(stub).system_one(state="s", questions={"q": question()})

    assert "in_progress" in str(error.value)


def test_a_completed_message_item_is_read_as_before(stub_server):
    body = responses_body(text=STRUCTURED)
    body["output"][-1]["status"] = "completed"
    stub = stub_server(responses=lambda _: (200, body))

    response = client_for(stub).system_one(state="s", questions={"q": question()})

    assert response.answers["q"].choice == "billing"


def test_only_the_final_answer_phase_is_the_answer(stub_server):
    """The 2026-04-24 spec lets a model answer in commentary and then in a final message.

    Reading both as one answer would concatenate the commentary with the answer — and a label
    readout would sample the commentary's token.
    """
    body = responses_body(text=STRUCTURED)
    message = body["output"][-1]
    commentary = {
        "type": "message",
        "id": "msg_commentary",
        "status": "completed",
        "role": "assistant",
        "phase": "commentary",
        "content": [{"type": "output_text", "text": "thinking out loud", "annotations": []}],
    }
    body["output"] = [commentary, {**message, "phase": "final_answer"}]
    stub = stub_server(responses=lambda _: (200, body))

    response = client_for(stub).system_one(state="s", questions={"q": question()})

    assert response.answers["q"].choice == "billing"


def test_a_response_of_commentary_alone_has_no_answer(stub_server):
    body = responses_body(text=STRUCTURED)
    body["output"][-1]["phase"] = "commentary"
    stub = stub_server(responses=lambda _: (200, body))

    with pytest.raises(Exception) as error:
        client_for(stub).system_one(state="s", questions={"q": question()})

    assert "no JSON object" in str(error.value) or "empty" in str(error.value)


@pytest.mark.parametrize("part_type", ["output_text", "text", "input_text"])
def test_every_text_bearing_content_part_is_read(stub_server, part_type):
    """OpenAI names the answer part ``output_text``; OpenResponses also allows ``text``/``input_text``."""
    body = responses_body(text=STRUCTURED)
    body["output"][-1]["content"] = [{"type": part_type, "text": STRUCTURED}]
    stub = stub_server(responses=lambda _: (200, body))

    response = client_for(stub).system_one(state="s", questions={"q": question()})

    assert response.answers["q"].choice == "billing"


def test_reasoning_text_is_read_from_every_part_type_the_specs_use(stub_server):
    item = {
        "type": "reasoning",
        "id": "rs_1",
        "status": "completed",
        "summary": [{"type": "text", "text": "a summary"}],
        "content": [{"type": "output_text", "text": "the trace"}],
    }
    stub = stub_server(responses=lambda _: (200, responses_body(text=STRUCTURED, reasoning=[item])))

    response = client_for(stub).system_one(state="s", questions={"q": question()})

    assert response.reasoning[0].summary[0].text == "a summary"
    assert response.reasoning[0].content[0].text == "the trace"


def test_logprob_tokens_are_read_from_their_bytes(stub_server):
    """A byte-level token (``ĠA``) is the same text as the bytes beside it: " A"."""
    body = responses_body(text="A")
    body["output"][-1]["content"][0]["logprobs"] = [
        {
            "token": "ĠA",
            "logprob": -0.1,
            "bytes": [32, 65],
            "top_logprobs": [
                {"token": "ĠA", "logprob": -0.1, "bytes": [32, 65]},
                {"token": "ĠB", "logprob": -2.0, "bytes": [32, 66]},
            ],
        }
    ]
    stub = stub_server(responses=lambda _: (200, body))
    client = SystemOneClient(
        openai_client(stub), model="stub", api="responses", method="logprobs", n_retry_malformed=0
    )

    response = client.system_one(
        state="s", questions={"q": Choice(criteria={"billing": None, "sales": None})}
    )

    probabilities = response.answers["q"].probabilities
    assert probabilities["billing"] / probabilities["sales"] == pytest.approx(math.exp(1.9))


def test_an_event_stream_where_a_response_object_belongs_is_named(stub_server):
    """A server that streams an answer nobody asked for gets a named error, not a malformed answer."""
    event = (
        "event: response.completed\n"
        "data: " + json.dumps(responses_body(text=STRUCTURED)) + "\n\n"
    )
    stub = stub_server(responses=lambda _: (200, event))

    with pytest.raises(ProviderError) as error:
        client_for(stub).system_one(state="s", questions={"q": question()})

    assert "streaming event stream" in str(error.value)


def test_a_null_output_is_an_empty_answer_not_a_type_error(stub_server):
    """The SDK's ``output_text`` property walks ``output``; a null one must not reach the caller raw."""
    body = responses_body(text=STRUCTURED)
    body["output"] = None
    stub = stub_server(responses=lambda _: (200, body))

    with pytest.raises(Exception) as error:
        client_for(stub).system_one(state="s", questions={"q": question()})

    assert "TypeError" not in str(error.value)
    assert "not iterable" not in str(error.value)


def test_a_refusal_part_is_still_a_refusal(stub_server):
    body = responses_body(text="", refusal="I cannot help with that")
    stub = stub_server(responses=lambda _: (200, body))

    from jevper import ModelRefusalError

    with pytest.raises(ModelRefusalError) as error:
        client_for(stub).system_one(state="s", questions={"q": question()})

    assert "I cannot help" in str(error.value)


# ------------------------------------------------------------------ capability: include entries


INCLUDE_REFUSAL_OF_THE_REASONING_ENTRY = {
    "error": {
        "message": (
            "Invalid value: 'reasoning.encrypted_content' at 'include[1]'. Supported values are: "
            "'message.output_text.logprobs'."
        ),
        "param": "include",
        "code": "invalid_request_error",
    }
}


def test_a_refusal_of_the_reasoning_include_keeps_the_label_readout(stub_server):
    """A server that lists the logprob entry among the values it accepts is refusing the other one.

    Dropping only the reasoning entry keeps the method the caller asked for; reading the logprob word
    in that sentence as a refusal of the carrier would answer a different method instead.
    """
    def script(body):
        if len(body.get("include") or ()) > 1:
            return 400, INCLUDE_REFUSAL_OF_THE_REASONING_ENTRY
        return 200, responses_body(
            text="A", logprobs=[("A", -0.1), ("B", -2.0)],
        )

    stub = stub_server(responses=script)
    client = SystemOneClient(
        openai_client(stub),
        model="stub",
        api="responses",
        method="logprobs",
        reasoning=ReasoningConfig(effort="medium"),
        n_retry_malformed=0,
    )

    response = client.system_one(
        state="s", questions={"q": Choice(criteria={"billing": None, "sales": None})}
    )

    sent = stub.bodies("/responses")
    assert sent[0]["include"] == ["message.output_text.logprobs", "reasoning.encrypted_content"]
    assert sent[1]["include"] == ["message.output_text.logprobs"]
    assert response.debug["method"] == "logprobs"
    assert response.answers["q"].choice == "billing"


# ------------------------------------------------------------------ provider strings


def test_an_escaped_surrogate_in_the_body_never_reaches_the_public_response(stub_server):
    """A body can carry ``\\udXXX``; the SDK decodes it, and the response must still serialize."""
    raw = json.dumps(
        {
            **responses_body(text=STRUCTURED),
            "output": [
                {
                    "type": "reasoning",
                    "id": "rs_1",
                    "status": "completed",
                    "summary": [{"type": "summary_text", "text": "x\ud800y"}],
                }
            ]
            + responses_body(text=STRUCTURED)["output"],
        }
    ).encode("utf-8", "surrogatepass")
    stub = stub_server(responses=lambda _: (200, raw.decode("utf-8", "surrogatepass")))

    response = client_for(stub).system_one(state="s", questions={"q": question()})

    # The reasoning is decoration and cannot be encoded, so it is dropped; the answer stands.
    assert response.answers["q"].choice == "billing"
    assert response.reasoning == ()
    payload = response.model_dump_json()
    assert "billing" in payload


def test_a_surrogate_in_debug_data_is_escaped_not_carried(stub_server):
    raw = json.dumps(
        {
            **responses_body(text=STRUCTURED),
            "service_tier": "x\ud800y",
        }
    ).encode("utf-8", "surrogatepass")
    stub = stub_server(responses=lambda _: (200, raw.decode("utf-8", "surrogatepass")))

    response = client_for(stub).system_one(state="s", questions={"q": question()})

    payload = response.model_dump_json()
    assert "\\ud800" in payload


def test_the_async_client_reads_the_same_dialects(stub_server):
    import asyncio

    body = responses_body(text=STRUCTURED)
    body["output"][-1]["content"] = [{"type": "text", "text": STRUCTURED}]
    stub = stub_server(responses=lambda _: (200, body))

    from jevper import AsyncSystemOneClient

    async def run():
        client = AsyncSystemOneClient(
            async_openai_client(stub), model="stub", api="responses", method="structured"
        )
        return await client.system_one(state="s", questions={"q": question()})

    response = asyncio.run(run())

    assert response.answers["q"].choice == "billing"


def test_a_chat_surface_still_answers_when_a_responses_client_is_passed(stub_server):
    """Sanity: the two surfaces stay independent through the real SDK client."""
    stub = stub_server(chat=lambda _: (200, chat_body(content=STRUCTURED)))
    client = SystemOneClient(
        openai_client(stub), model="stub", api="chat_completions", method="structured",
        n_retry_malformed=0,
    )

    response = client.system_one(state="s", questions={"q": question()})

    assert response.debug["api"] == "chat_completions"
    assert response.answers["q"].choice == "billing"


def test_a_client_without_the_responses_route_still_reports_it(stub_server):
    stub = stub_server(responses=lambda _: (404, {"error": {"message": "Unknown request URL"}}))
    client = SystemOneClient(
        openai_client(stub), model="stub", api="responses", method="structured",
        n_retry_malformed=0,
    )

    with pytest.raises(ProviderError) as error:
        client.system_one(state="s", questions={"q": question()})

    assert "404" in str(error.value)


def test_the_capability_error_names_the_surface_a_client_cannot_speak(stub_server):
    class ChatOnly:
        def __init__(self) -> None:
            class Completions:
                def create(self, **kwargs):
                    return chat_body(content=STRUCTURED)

            class Chat:
                completions = Completions()

            self.chat = Chat()

    client = SystemOneClient(ChatOnly(), model="stub", api="responses", method="structured")

    with pytest.raises(ClientCapabilityError) as error:
        client.system_one(state="s", questions={"q": question()})

    assert "responses.create" in str(error.value)
