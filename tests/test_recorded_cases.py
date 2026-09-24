"""Recorded captures the surface suite does not replay.

``tests/test_provider_surfaces.py`` replays the bodies that pin the shapes a live server sends on the
paths jevper takes most often. The fixtures hold more than that: the ``n=2`` bodies two servers
answered, the one-top and no-tops logprob streams, the Responses cache pair, the Messages thinking
blocks, and the four servers' disagreement about a model id. Replaying those here keeps the other
half of the capture set in front of every future change instead of letting it rot.

The loading helpers and the capture list come from the other module, so both suites read the same
files the same way.
"""

from __future__ import annotations

import json

import pytest
from fakes import openai_client
from test_provider_surfaces import (
    NO_RETRIES,
    QUESTIONS,
    SERVERS,
    STATE,
    client_for,
    recorded,
    replay,
    served,
)

from jevper import (
    LabelReadoutError,
    MalformedAnswerError,
    ProviderError,
    SystemOneClient,
    reasoning_text,
)
from jevper.errors import _LogprobsUnavailable
from jevper.methods import readout_logprobs
from jevper.transport import SURFACES, CallSpec, build_messages_kwargs

CHAT = SURFACES["chat_completions"][1]
RESPONSES = SURFACES["responses"][1]
MESSAGES = SURFACES["messages"][1]

INTENT = QUESTIONS["intent"]
LABELS = ("A", "B", "C")


# --- n=2: which choice is the answer ------------------------------------------------------------


@pytest.mark.parametrize("server", ("ollama", "vllm", "sglang"))
def test_a_recorded_two_choice_capture_carries_the_first_choice(server):
    """vLLM and SGLang answered ``n=2`` with two choices, ollama with one, and only the first is read.

    The later choices are relabelled, so a normalizer that took the last one — or joined them — would
    carry that text instead of the recorded empty answer and the first choice's thinking span.
    """
    body = json.loads(json.dumps(recorded(server)["chat-n-two"]["body"]))
    for choice in body["choices"][1:]:
        choice["message"]["content"] = "the second choice"

    result = CHAT(body, {})

    assert result.text == ""  # the recorded first choice was still thinking when it hit the budget
    assert "second choice" not in result.text
    assert reasoning_text(result.reasoning).strip() == "Thinking Process:"
    assert result.stop == "length"


# --- one top, no tops, twenty-one tops ----------------------------------------------------------


@pytest.mark.parametrize(
    ("case", "server", "counts"),
    [
        ("chat-logprobs-1-top", "ollama", [0, 0, 1, 0]),
        ("chat-logprobs-1-top", "llamacpp", [0, 0, 1, 0]),
        ("chat-logprobs-1-top", "vllm", [0, 0, 1, 0]),
        ("chat-logprobs-1-top", "sglang", [0, 0, 1, 0]),
        ("chat-logprobs-no-tops", "ollama", [0, 0, 0, 0]),
        ("chat-logprobs-no-tops", "llamacpp", [0, 0, 20, 0]),
        ("chat-logprobs-no-tops", "vllm", [0, 0, 0, 0]),
        ("chat-logprobs-no-tops", "sglang", [0, 0, 0, 0]),
        ("chat-logprobs-21-tops", "llamacpp", [0, 0, 21, 0]),
        ("chat-logprobs-21-tops", "sglang", [0, 0, 21, 0]),
    ],
)
def test_the_recorded_alternatives_count_survives(case, server, counts):
    """``top_logprobs=1`` is one alternative, ``=20`` twenty, and ``=0`` sends none at all.

    The count the provider put in the body is what decides whether a distribution exists at all — one
    entry is the sampled token alone — so it is counted as reported, and every recorded entry is kept.
    """
    result = CHAT(recorded(server)[case]["body"], {})

    assert [token.reported_alternatives for token in result.token_logprobs] == counts
    assert [len(token.top_logprobs) for token in result.token_logprobs] == counts
    assert result.text == ""  # every one of these captures ran out of budget while thinking
    assert result.stop == "length"


@pytest.mark.parametrize("server", SERVERS)
def test_a_recorded_single_alternative_is_refused_rather_than_read_as_certainty(server):
    """A one-top capture is not a distribution, and a one-hot built from it would be a lie.

    The single alternative is the one ``chat-logprobs-1-top`` recorded — a token and its logprob —
    moved onto the answer token of the recorded ``chat-thinking-off-logprobs`` body, because the
    one-top capture itself spent its whole budget thinking. What is pinned is the rule: the sampled
    token came back with one alternative, so there is nothing to read a distribution out of.
    """
    answer = json.loads(json.dumps(replay(server, "chat-thinking-off-logprobs")[1]))
    single = next(
        entry["top_logprobs"]
        for entry in recorded(server)["chat-logprobs-1-top"]["body"]["choices"][0]["logprobs"]["content"]
        if entry.get("top_logprobs")
    )
    assert len(single) == 1
    answer["choices"][0]["logprobs"]["content"][0]["top_logprobs"] = single

    result = CHAT(answer, {})

    with pytest.raises(_LogprobsUnavailable) as error:
        readout_logprobs(result, INTENT, LABELS)

    assert "1 top_logprobs for the answer token 'A'" in str(error.value)
    assert "not a distribution over the options" in str(error.value)
    assert error.value.evidence == "readout"  # one anomalous answer is not a provider verdict


@pytest.mark.parametrize(
    ("server", "words"),
    [
        ("ollama", "top_logprobs must be between 0 and 20"),
        ("vllm", "Requested sample logprobs of 21, which is greater than max allowed: 20"),
    ],
)
def test_a_recorded_top_logprobs_cap_reaches_the_caller(stub_server, server, words):
    """A 21-top request is over both servers' cap, and each says so in its own words."""
    stub = served(stub_server, (server, "chat-logprobs-21-tops"))
    client = client_for(stub, api="chat_completions", method="logprobs", retry=NO_RETRIES)

    with pytest.raises(ProviderError) as error:
        client.system_one(state=STATE, questions=QUESTIONS)

    assert error.value.status_code == 400
    assert words in str(error.value)


# --- logprobs asked for without the flag, and answered without any -------------------------------


@pytest.mark.parametrize(
    ("server", "words"),
    [
        ("llamacpp", "top_logprobs requires logprobs to be set to true"),
        ("vllm", "when using `top_logprobs`, `logprobs` must be set to true."),
    ],
)
def test_a_recorded_refusal_of_tops_without_logprobs_reaches_the_caller(stub_server, server, words):
    """llama.cpp and vLLM answer a ``top_logprobs``-without-``logprobs`` request with their own 400."""
    stub = served(stub_server, (server, "chat-tops-without-logprobs"))
    client = client_for(stub, api="chat_completions", method="logprobs", retry=NO_RETRIES)

    with pytest.raises(ProviderError) as error:
        client.system_one(state=STATE, questions=QUESTIONS)

    assert error.value.status_code == 400
    assert words in str(error.value)


@pytest.mark.parametrize("server", ("ollama", "sglang"))
def test_a_recorded_200_without_logprobs_is_not_a_distribution(stub_server, server):
    """ollama and SGLang answer the same request 200 and send no logprobs at all.

    The verdict has to be jevper's "this provider does not report them" — the one ``method="auto"``
    remembers — and it has to name the budget, because the recorded answer never got past thinking.
    """
    stub = served(stub_server, (server, "chat-tops-without-logprobs"))
    client = client_for(stub, api="chat_completions", method="logprobs", retry=NO_RETRIES)

    with pytest.raises(LabelReadoutError) as error:
        client.system_one(state=STATE, questions=QUESTIONS)

    assert isinstance(error.value, _LogprobsUnavailable)
    assert "does not report them" in str(error.value)
    assert "output tokens" in str(error.value)


# --- the Responses surface: a completed logprob stream, and the cache pair ------------------------


@pytest.mark.parametrize("server", ("vllm", "sglang"))
def test_a_recorded_responses_logprob_stream_reads_the_answer_token(stub_server, server):
    """The Responses surface carries logprobs too, in a completed body rather than a truncated one.

    Both servers answered 'A' with five alternatives for the answer token and a trailing
    ``<|im_end|>`` entry beside it, so the readout has to find the label before the end-of-turn token.
    """
    stub = served(
        stub_server,
        (server, "chat-thinking-off-logprobs"),
        (server, "responses-thinking-off-logprobs"),
    )
    client = client_for(stub, api="responses", method="logprobs", n_retry_malformed=0)

    response = client.system_one(state=STATE, questions=QUESTIONS)
    answer = response.answers["intent"]

    assert response.debug["api"] == "responses"
    assert answer.choice == "billing"  # 'A' is the first option
    assert set(answer.probabilities) == set(INTENT.criteria)
    assert sum(answer.probabilities.values()) == pytest.approx(1.0)
    assert response.usage.n_calls == 1


@pytest.mark.parametrize(
    ("server", "cold_text", "warm_text", "cached"),
    [("ollama", "", "", 1985), ("vllm", "D", "B", 1584), ("sglang", "D", "D", 1984)],
)
def test_a_recorded_responses_cache_pair_is_read(server, cold_text, warm_text, cached):
    """The Responses pair for one prompt: the answer text is what its two halves differ in.

    vLLM answered 'D' then 'B' while reporting the same 1584 cached tokens both times, so a reader
    that took the text from anywhere but ``output`` is caught here. ollama's half is reasoning-only
    with an empty answer, and even the "cold" captures reported reuse under
    ``usage.input_tokens_details`` — the field this surface reports it in, not the chat one.
    """
    cold = RESPONSES(recorded(server)["responses-cache-cold"]["body"], {})
    warm = RESPONSES(recorded(server)["responses-cache-warm"]["body"], {})

    assert (cold.text, warm.text) == (cold_text, warm_text)
    assert cold.cached_tokens == warm.cached_tokens == cached
    assert cold.input_tokens is not None
    assert cold.stop is None  # 'completed': the provider never said the budget ran out


@pytest.mark.parametrize(
    ("server", "text", "cached"),
    [("ollama", "D", 0), ("llamacpp", "D", 40), ("vllm", "B", 0), ("sglang", "D", None)],
)
def test_a_recorded_cold_cache_is_not_reported_as_silence(server, text, cached):
    """The same prompt on each server with nothing cached yet.

    ollama and vLLM report a plain ``0``, llama.cpp a partial 40, and SGLang sends
    ``prompt_tokens_details: null``. Silence has to stay ``None``, and a reported zero has to stay a
    zero: "the cache is off or cold" is a different fact from "this server does not report at all".
    """
    result = CHAT(recorded(server)["chat-cache-key"]["body"], {})

    assert result.cached_tokens == cached
    assert result.text == text
    if cached == 0:
        assert result.cached_tokens is not None


# --- a model id the servers disagree about ------------------------------------------------------


@pytest.mark.parametrize("server", ("ollama", "vllm"))
def test_a_recorded_bad_model_404_names_the_model(stub_server, server):
    """Two servers answer a bad model id with a 404 that names it, and both facts must reach the caller."""
    stub = served(stub_server, (server, "chat-bad-model"))
    client = SystemOneClient(
        openai_client(stub),
        model="definitely-not-a-model",
        api="chat_completions",
        method="structured",
        retry=NO_RETRIES,
    )

    with pytest.raises(ProviderError) as error:
        client.system_one(state=STATE, questions=QUESTIONS)

    assert error.value.status_code == 404
    assert "definitely-not-a-model" in str(error.value)
    assert len(stub.paths) == 1


@pytest.mark.parametrize("server", ("llamacpp", "sglang"))
def test_a_recorded_bad_model_that_was_served_anyway_is_an_answer(stub_server, server):
    """llama.cpp serves its loaded model and SGLang echoes the id back: a 200, and no model error.

    The recorded body is reasoning-only, so the failure is jevper's own answer-shape error naming the
    budget — not a provider error, which would blame an id the server never looked at.
    """
    stub = served(stub_server, (server, "chat-bad-model"))
    client = SystemOneClient(
        openai_client(stub),
        model="definitely-not-a-model",
        api="chat_completions",
        method="structured",
        retry=NO_RETRIES,
        n_retry_malformed=0,
    )

    with pytest.raises(MalformedAnswerError) as error:
        client.system_one(state=STATE, questions=QUESTIONS)

    assert "output tokens" in str(error.value)
    assert len(stub.paths) == 1


def test_a_recorded_responses_bad_model_that_was_served_anyway_is_read():
    """llama.cpp's Responses shim answers the bad id with its loaded model's reasoning and no text."""
    result = RESPONSES(recorded("llamacpp")["responses-bad-model"]["body"], {})

    assert result.text == ""
    assert "Thinking Process" in reasoning_text(result.reasoning)
    assert result.stop is None  # 'completed': the provider never said the budget ran out


# --- the developer role -------------------------------------------------------------------------


def test_a_recorded_role_refusal_reaches_the_caller_in_the_providers_words(stub_server):
    """SGLang answers a ``developer`` message with a 400 whose envelope is not OpenAI's.

    The body is ``{"code": 400, "message": ...}`` with no ``error`` object, and the caller still has to
    get the status and the sentence rather than an empty message.
    """
    stub = served(stub_server, ("sglang", "chat-developer-role"))
    client = client_for(stub, api="chat_completions", method="structured", retry=NO_RETRIES)

    with pytest.raises(ProviderError) as error:
        client.system_one(state=STATE, questions=QUESTIONS)

    assert error.value.status_code == 400
    assert "Unexpected message role." in str(error.value)


@pytest.mark.parametrize("server", ("ollama", "llamacpp", "vllm"))
def test_a_server_that_ignores_a_developer_role_still_answers(server):
    """Three servers took the ``developer`` role without comment and answered with reasoning only."""
    result = CHAT(recorded(server)["chat-developer-role"]["body"], {})

    assert result.text == ""
    assert reasoning_text(result.reasoning).strip()
    assert result.stop == "length"


# --- the Messages route: thinking blocks, and the system turn ------------------------------------


@pytest.mark.parametrize(
    ("server", "text", "thinking", "signature"),
    [
        ("ollama", '{"classification": "B"}', None, None),
        ("llamacpp", '{"classification": "B"}', None, None),
        ("lmstudio", '{"classification": "B"}', None, None),
        ("vllm", "", "We are given a question to classify", "a6007ec7e0484950be81d0a79bb23d9a"),
        ("sglang", "", "We are given a question to classify", None),
    ],
)
def test_a_recorded_thinking_block_survives_with_its_signature(server, text, thinking, signature):
    """The Messages route's thinking blocks, one shape per server.

    vLLM's block carries a signature and SGLang's does not, and both have to land in the reasoning
    part's extra fields: a caller replaying the turn has to be able to send back what the provider
    asked for. The two non-thinking servers answer with a text block and no reasoning at all.
    """
    result = MESSAGES(recorded(server)["messages-thinking"]["body"], {})

    assert result.text == text
    assert result.token_logprobs == ()  # no server implements logprobs through this API
    if thinking is None:
        assert result.reasoning == ()
        assert result.stop == "end_turn"
    else:
        assert result.reasoning[0].type == "reasoning"
        assert reasoning_text(result.reasoning).startswith(thinking)
        assert (result.reasoning[0].model_extra or {}).get("signature") == signature
        assert result.stop == "max_tokens"  # the recorded thinking spent the whole budget


@pytest.mark.parametrize(
    ("server", "text"),
    [
        ("ollama", '{"classification": "B"}'),
        ("llamacpp", '{"classification": "B"}'),
        ("lmstudio", '{"primary_subject": "B"}'),
        ("vllm", ""),
        ("sglang", ""),
    ],
)
def test_a_recorded_system_turn_answer_is_read_and_the_system_turn_is_not_a_turn(server, text):
    """The capture that put a system message in front of a few-shot prompt.

    The recorded answer reads like any other Messages answer. The recorded *request* pins the other
    half: the system message is hoisted to the API's top-level ``system`` field, and neither stays a
    turn nor is folded into one — which is where llama.cpp's template and SGLang refuse it.
    """
    request = recorded(server)["messages-system-turn"]["request"]
    result = MESSAGES(recorded(server)["messages-system-turn"]["body"], {})

    assert result.text == text
    if not text:
        assert reasoning_text(result.reasoning).startswith("We are given")

    kwargs = build_messages_kwargs(CallSpec(messages=request["messages"]), model=request["model"])

    assert "precise classification engine" in kwargs["system"]
    assert [message["role"] for message in kwargs["messages"]] == [
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
        "user",
    ]
    assert all(
        "precise classification engine" not in (message.get("content") or "")
        for message in kwargs["messages"]
    )
