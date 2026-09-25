"""The live matrix: every documented feature, asked of a real server, one scenario at a time.

Each scenario is something a deployment does — classify with the rubric, pin a method, read
a distribution from logprobs, ask for a 255-way decision, turn reasoning on, override the
cache key, send a header, hand the library a state it has never seen, or make it refuse
before it spends a request. A scenario states which failures are legitimate for the surface
it runs on; anything else, and any broken invariant, comes back as a **gap** with the
request that produced it.

The matrix runs unchanged against ollama, llama.cpp, vLLM, SGLang and LM Studio: the only
per-server facts are the profile (base URL, model, how thinking is turned off), and those
are configuration.
"""

from __future__ import annotations

import contextlib
import json
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from jevper import (
    AsyncSystemOneClient,
    Choice,
    Example,
    Noul,
    ReasoningConfig,
    RetryPolicy,
    SystemOneResponse,
    reasoning_text,
)
from jevper.errors import JevperError

from . import tickets
from .checks import Gap, as_specs, check, check_response, prompt_text
from .duck import DuckClient
from .profiles import RequestCounter, ServerProfile
from .rubrics import (
    CATALOG_OPTIONS,
    PRODUCT_AREAS,
    build_rubric,
    catalog_choice,
    raw_rubric,
    wide_26,
)
from .service import Triage, TriageReport, build_async_client, build_client

__all__ = ["SCENARIOS", "Ctx", "Scenario", "run_matrix", "summarize"]

OPENAI_APIS = ("chat_completions", "responses")
ALL_APIS = ("chat_completions", "responses", "messages")
RUBRIC = build_rubric()
PLAIN = build_rubric(few_shot=False)


@dataclass
class Ctx:
    """The bound client for one attempt, plus the profile it was built from."""

    profile: ServerProfile
    counter: RequestCounter = field(default_factory=RequestCounter)
    client: Any = None
    api: str = "chat_completions"
    method: str = "auto"

    def options(self, **extra: Any) -> dict[str, Any]:
        """Client options for this surface: the profile's thinking-off field, then the caller's."""
        body = self.profile.body_for(self.api)
        body.update(extra.pop("extra_body", None) or {})
        options: dict[str, Any] = {"method": self.method, "api": self.api, **extra}
        if body:
            options["extra_body"] = body
        return options

    def service(self, *, questions: Mapping[str, Any] | None = None, model: str | None = None, **extra: Any) -> Triage:
        return Triage(
            client=self.client,
            model=model or self.profile.model,
            options=self.options(**extra),
            rubric=dict(questions or RUBRIC),
        )


@dataclass(frozen=True)
class Scenario:
    """One feature, on the surfaces it can run on."""

    name: str
    run: Callable[[Ctx], dict[str, Any]]
    apis: tuple[str, ...] = OPENAI_APIS
    methods: tuple[str, ...] = ("auto",)
    allow: tuple[str, ...] = ()
    check_answers: bool = True
    note: str = ""


# --------------------------------------------------------------------------------------
# the rubric itself
# --------------------------------------------------------------------------------------


def core(ctx: Ctx) -> dict[str, Any]:
    """The rubric every deployment runs: intent, urgency and the human handoff."""
    service = ctx.service()
    response = service.classify(dict(tickets.SHORT_TICKET))
    check_response(response, as_specs(service.rubric), model=ctx.profile.model)
    return {
        "report": TriageReport.from_response(response, "INC-1041").to_json(),
        "method": response.debug.get("method"),
        "methods": response.debug.get("methods"),
        "server_limits": response.debug.get("server_limits"),
        "usage": response.usage.model_dump(mode="json"),
    }


def without_few_shot(ctx: Ctx) -> dict[str, Any]:
    """The same questions with no examples, so the two runs are comparable."""
    service = ctx.service(questions=PLAIN)
    response = service.classify(dict(tickets.SHORT_TICKET))
    check_response(response, as_specs(PLAIN), model=ctx.profile.model)
    return {"answers": {key: value.model_dump(mode="json") for key, value in response.answers.items()}}


def raw_mapping_questions(ctx: Ctx) -> dict[str, Any]:
    """Questions passed as raw mappings, which the library validates identically."""
    questions = raw_rubric()
    service = ctx.service(questions=questions)
    response = service.classify(dict(tickets.SHORT_TICKET))
    check_response(response, as_specs(questions), model=ctx.profile.model)
    return {"answers": {key: value.model_dump(mode="json") for key, value in response.answers.items()}}


def noul_without_criteria(ctx: Ctx) -> dict[str, Any]:
    """A ``Noul`` with no criteria: one probability, no options."""
    questions = {"is_outage": Noul(instructions="Is this an outage?")}
    service = ctx.service(questions=questions)
    response = service.classify(dict(tickets.SHORT_TICKET))
    check_response(response, as_specs(questions), model=ctx.profile.model)
    return {"answer": response.answers["is_outage"].model_dump(mode="json")}


# --------------------------------------------------------------------------------------
# methods
# --------------------------------------------------------------------------------------


def structured(ctx: Ctx) -> dict[str, Any]:
    service = ctx.service()
    response = service.classify(dict(tickets.SHORT_TICKET))
    check_response(response, as_specs(service.rubric), model=ctx.profile.model)
    return {
        "probability_errors": response.debug.get("probability_errors"),
        "original_probabilities": list(response.debug.get("original_probabilities") or {}),
        "readouts": [
            (attempt.get("readout") or {}).get("source")
            for attempt in response.debug.get("llm_attempts", [])
        ],
    }


def discrete(ctx: Ctx) -> dict[str, Any]:
    service = ctx.service()
    response = service.classify(dict(tickets.SHORT_TICKET))
    check_response(response, as_specs(service.rubric), model=ctx.profile.model)
    answers = {key: value.model_dump(mode="json") for key, value in response.answers.items()}
    for answer in answers.values():
        if answer["type"] != "noul":
            distribution = answer.get("probabilities") or {}
            check(
                max(distribution.values()) == 1.0 if distribution else True,
                f"discrete distribution is not one-hot: {distribution}",
            )
    return {"answers": answers}


def logprobs(ctx: Ctx) -> dict[str, Any]:
    service = ctx.service()
    response = service.classify(dict(tickets.SHORT_TICKET))
    check_response(response, as_specs(service.rubric), model=ctx.profile.model)
    intent = response.answers["intent"]
    return {
        "readout": [attempt.get("readout", {}).get("source") for attempt in response.debug.get("llm_attempts", [])],
        "labels_missing": response.debug.get("labels_missing"),
        "intent": intent.model_dump(mode="json"),
    }


def grammar(ctx: Ctx) -> dict[str, Any]:
    service = ctx.service()
    response = service.classify(dict(tickets.SHORT_TICKET))
    check_response(response, as_specs(service.rubric), model=ctx.profile.model)
    grammar_sent = [
        "grammar" in (attempt.get("request") or {}).get("extra_body", {}) for attempt in response.debug["llm_attempts"]
    ]
    check(any(grammar_sent), "no request carried the grammar field")
    return {"intent": response.answers["intent"].model_dump(mode="json")}


def wide_26_logprobs(ctx: Ctx) -> dict[str, Any]:
    """Twenty-six options: the widest a one-token label readout can be asked for."""
    questions = {"queue": wide_26()}
    service = ctx.service(questions=questions, method="logprobs")
    response = service.classify("The export is stuck and finance needs it before the board meeting.")
    check_response(response, as_specs(questions), model=ctx.profile.model)
    answer = response.answers["queue"]
    return {
        "options": len(answer.probabilities),
        "choice": answer.choice,
        "labels_missing": response.debug.get("labels_missing"),
    }


def wide_30_structured(ctx: Ctx) -> dict[str, Any]:
    """Thirty options: past 26 the labels are two letters, which only the JSON methods use."""
    questions = {"area": PRODUCT_AREAS}
    service = ctx.service(questions=questions, method="structured")
    response = service.classify("The CSV export from the analytics area times out for large tenants.")
    check_response(response, as_specs(questions), model=ctx.profile.model)
    answer = response.answers["area"]
    check(len(answer.probabilities) == 30, f"30 options came back as {len(answer.probabilities)}")
    return {"options": len(answer.probabilities), "choice": answer.choice}


def catalog_255_structured(ctx: Ctx) -> dict[str, Any]:
    """The Jev API's documented maximum for a Choice, asked as one question."""
    questions = {"sku": catalog_choice()}
    service = ctx.service(questions=questions, method="structured")
    response = service.classify("Our invoice lists catalogue entry 0042 twice this month.")
    check_response(response, as_specs(questions), model=ctx.profile.model)
    answer = response.answers["sku"]
    check(len(answer.probabilities) == CATALOG_OPTIONS, f"255 options came back as {len(answer.probabilities)}")
    return {"options": len(answer.probabilities), "choice": answer.choice}


def examples_with_probabilities(ctx: Ctx) -> dict[str, Any]:
    """Few-shot examples that carry their own distribution, the ``structured`` way."""
    question = Choice(
        instructions="Pick the severity.",
        criteria={"sev1": "production down", "sev2": "degraded", "sev3": "cosmetic"},
        examples=(
            Example(
                state="All logins fail for every user.",
                answer="sev1",
                probabilities={"sev1": 0.95, "sev2": 0.04, "sev3": 0.01},
            ),
        ),
    )
    questions = {"severity": question}
    service = ctx.service(questions=questions, method="structured")
    response = service.classify("The dashboard icon is the wrong shade of blue.")
    check_response(response, as_specs(questions), model=ctx.profile.model)
    prompt = prompt_text(response)
    check("0.95" in prompt, "the example's own probabilities are not in the prompt")
    return {"answer": response.answers["severity"].model_dump(mode="json")}


def few_shot_precedence(ctx: Ctx) -> dict[str, Any]:
    """Question-level examples beat the call's, which beat the constructor's."""
    question = Choice(
        instructions="Pick the topic.",
        criteria={"alpha": "topic A", "beta": "topic B"},
        examples=(Example(state="a question about A", answer="alpha"),),
    )
    service = ctx.service(
        questions={"topic": question},
        method="logprobs",
        examples=(Example(state="a call-level question about B", answer="beta"),),
    )
    response = service.classify("Tell me about A.")
    check_response(response, as_specs({"topic": question}), model=ctx.profile.model)
    prompt = prompt_text(response)
    check("a question about A" in prompt, "the question's own example is not in the prompt")
    check("a call-level question about B" not in prompt, "the call's example overrode the question's")
    return {"prompt_chars": len(prompt)}


# --------------------------------------------------------------------------------------
# state forms
# --------------------------------------------------------------------------------------


def _one_question(ctx: Ctx, state: Any, *, method: str | None = None) -> SystemOneResponse:
    questions = {"intent": Choice(criteria={"billing": "money", "technical": "errors", "other": "anything else"})}
    service = ctx.service(questions=questions, **({"method": method} if method else {}))
    response = service.classify(state)
    check_response(response, as_specs(questions), model=ctx.profile.model)
    return response


def state_text(ctx: Ctx) -> dict[str, Any]:
    response = _one_question(ctx, "I was charged twice for the same subscription this month.")
    return {"intent": response.answers["intent"].choice}


def state_json(ctx: Ctx) -> dict[str, Any]:
    response = _one_question(ctx, dict(tickets.SHORT_TICKET))
    return {"intent": response.answers["intent"].choice}


def state_chat_list(ctx: Ctx) -> dict[str, Any]:
    response = _one_question(ctx, [dict(turn) for turn in tickets.CONVERSATION])
    return {"intent": response.answers["intent"].choice}


def state_messages_mapping(ctx: Ctx) -> dict[str, Any]:
    response = _one_question(ctx, {"messages": [dict(turn) for turn in tickets.CONVERSATION]})
    return {"intent": response.answers["intent"].choice}


def state_assistant_last(ctx: Ctx) -> dict[str, Any]:
    """A conversation whose last turn is the assistant's: the question has to follow it."""
    turns = [dict(turn) for turn in tickets.CONVERSATION] + [
        {"role": "assistant", "content": "I have escalated this to the identity team."}
    ]
    response = _one_question(ctx, turns)
    return {"intent": response.answers["intent"].choice}


def state_developer_role(ctx: Ctx) -> dict[str, Any]:
    """A ``developer`` turn, which one server in the fleet refuses outright."""
    response = _one_question(
        ctx,
        [
            {"role": "developer", "content": "Answer from the knowledge base only."},
            {"role": "user", "content": "How do I rotate the SSO certificate?"},
        ],
    )
    return {"intent": response.answers["intent"].choice}


# --------------------------------------------------------------------------------------
# options
# --------------------------------------------------------------------------------------


def temperature_zero(ctx: Ctx) -> dict[str, Any]:
    """A temperature of 0.0 reaches the wire on every surface — in the body on Messages, where the
    current SDK has no typed parameter for it."""
    service = ctx.service(questions=PLAIN, temperature=0.0)
    response = service.classify(dict(tickets.SHORT_TICKET), method="structured")
    check_response(response, as_specs(PLAIN), model=ctx.profile.model)
    request = response.debug["llm_attempts"][-1]["request"]
    sent = request.get("temperature", (request.get("extra_body") or {}).get("temperature"))
    check(sent == 0.0, f"temperature on the wire is {sent!r}")
    return {"temperature": sent, "in_body": "temperature" not in request}


def prompt_cache_key_override(ctx: Ctx) -> dict[str, Any]:
    """The caller's own key reaches the wire on both OpenAI surfaces.

    The Messages API has no such field, so nothing is sent there and the caller's key is simply
    not in play on that route — the library's own derived key is all that API can use.
    """
    service = ctx.service(questions=PLAIN, prompt_cache_key="tenant-42")
    response = service.classify(dict(tickets.SHORT_TICKET), method="structured")
    body = response.debug["llm_attempts"][-1]["request"].get("extra_body") or {}
    if ctx.api == "messages":
        return {"prompt_cache_key": None, "route": "messages carries no cache-key field"}
    check(body.get("prompt_cache_key") == "tenant-42", f"cache key on the wire is {body.get('prompt_cache_key')!r}")
    return {"prompt_cache_key": body.get("prompt_cache_key")}


def prompt_cache_key_72(ctx: Ctx) -> dict[str, Any]:
    """A key longer than OpenAI's 64-character cap: the provider's to refuse, not jevper's."""
    key = "t" * 72
    service = ctx.service(questions=PLAIN, prompt_cache_key=key)
    response = service.classify(dict(tickets.SHORT_TICKET), method="structured")
    body = response.debug["llm_attempts"][-1]["request"].get("extra_body") or {}
    if ctx.api == "messages":
        return {"prompt_cache_key_chars": 0, "route": "messages carries no cache-key field"}
    check(body.get("prompt_cache_key") == key, "the 72-character key did not reach the wire unchanged")
    return {"prompt_cache_key_chars": len(body.get("prompt_cache_key") or "")}


def structured_outputs_off(ctx: Ctx) -> dict[str, Any]:
    """``structured_outputs=False`` puts the schema in the prompt and a plain JSON object in the
    request — a `json_object` format on the OpenAI surfaces, and nothing on Responses, which has no
    such fallback beyond the prompt."""
    service = ctx.service(questions=PLAIN, structured_outputs=False)
    response = service.classify(dict(tickets.SHORT_TICKET), method="structured")
    check_response(response, as_specs(PLAIN), model=ctx.profile.model)
    request = response.debug["llm_attempts"][-1]["request"]
    sent = request.get("response_format") or (request.get("text") or {}).get("format")
    check(
        not (isinstance(sent, dict) and sent.get("type") == "json_schema"),
        f"a json_schema was sent although structured output was turned off: {sent!r}",
    )
    prompt = prompt_text(response)
    check("probabilities" in prompt, "the schema is neither in the request nor in the prompt")
    return {"format": sent, "schema_in_prompt": True}


def normalize_off(ctx: Ctx) -> dict[str, Any]:
    """``normalize_probabilities=False`` hands back the model's own numbers."""
    service = ctx.service(questions=PLAIN, normalize_probabilities=False)
    response = service.classify(dict(tickets.SHORT_TICKET), method="structured")
    check_response(response, as_specs(PLAIN), model=ctx.profile.model, normalized=False)
    return {
        "probability_errors": response.debug.get("probability_errors"),
        "rescaled": list(response.debug.get("original_probabilities") or {}),
    }


def top_logprobs(ctx: Ctx) -> dict[str, Any]:
    service = ctx.service(questions=PLAIN, top_logprobs=5, method="logprobs")
    response = service.classify(dict(tickets.SHORT_TICKET))
    check_response(response, as_specs(PLAIN), model=ctx.profile.model)
    request = response.debug["llm_attempts"][-1]["request"]
    check(request.get("top_logprobs") == 5, f"top_logprobs on the wire is {request.get('top_logprobs')!r}")
    return {"top_logprobs": request.get("top_logprobs")}


def max_concurrency(ctx: Ctx) -> dict[str, Any]:
    questions = {
        f"q{index}": Choice(criteria={"yes": "yes", "no": "no"}) for index in range(4)
    }
    service = ctx.service(questions=questions, max_concurrency=2, method="structured")
    response = service.classify(dict(tickets.SHORT_TICKET))
    check_response(response, as_specs(questions), model=ctx.profile.model)
    return {"calls": response.usage.n_calls, "latency": round(response.usage.latency, 3)}


def retry_settings(ctx: Ctx) -> dict[str, Any]:
    service = ctx.service(
        questions=PLAIN,
        n_retry_malformed=0,
        retry=RetryPolicy(n_retries=1, base_delay=0.1, max_delay=0.5),
    )
    response = service.classify(dict(tickets.SHORT_TICKET), method="structured")
    check_response(response, as_specs(PLAIN), model=ctx.profile.model)
    return {"calls": response.usage.n_calls, "retries": response.usage.n_retries}


def extra_headers(ctx: Ctx) -> dict[str, Any]:
    """A caller's own header, and a credential one that must never reach the record."""
    secret = "sekret-triage-key-9f3a"
    service = ctx.service(
        questions=PLAIN,
        extra_headers={"X-Triage-Tenant": "t-42", "authorization": secret},
        method="structured",
    )
    response = service.classify(dict(tickets.SHORT_TICKET))
    check_response(response, as_specs(PLAIN), model=ctx.profile.model)
    record = json.dumps(response.debug)
    check(secret not in record, "the credential appears in the debug record")
    check("<redacted>" in record, "the credential header was not recorded as redacted")
    serialized = response.model_dump_json()
    check(secret not in serialized, "the credential appears in the serialized response")
    headers = response.debug["llm_attempts"][-1]["request"].get("extra_headers") or {}
    return {"redacted": True, "headers_seen": sorted(headers)}


def extra_body_output_budget(ctx: Ctx) -> dict[str, Any]:
    """The caller's own output budget, and one small enough to be spent mid-answer."""
    service = ctx.service(questions=PLAIN, extra_body={"max_tokens": 512}, method="structured")
    response = service.classify(dict(tickets.SHORT_TICKET))
    check_response(response, as_specs(PLAIN), model=ctx.profile.model)
    body = response.debug["llm_attempts"][-1]["request"].get("extra_body") or {}
    return {"max_tokens": body.get("max_tokens")}


def tiny_output_budget(ctx: Ctx) -> dict[str, Any]:
    questions = {"intent": Choice(criteria={"billing": "money", "technical": "errors", "other": "anything else"})}
    service = ctx.service(questions=questions, extra_body={"max_tokens": 1}, method="structured")
    service.classify("I was charged twice for the same subscription this month.")
    return {"note": "a spent budget should be reported, not answered around"}


def stream_zero(ctx: Ctx) -> dict[str, Any]:
    """``stream: 0`` is a typed refusal on three of the five servers."""
    service = ctx.service(questions=PLAIN, extra_body={"stream": 0}, method="structured")
    response = service.classify(dict(tickets.SHORT_TICKET))
    check_response(response, as_specs(PLAIN), model=ctx.profile.model)
    return {"ok": True}


# --------------------------------------------------------------------------------------
# reasoning
# --------------------------------------------------------------------------------------


def reasoning_two_step(ctx: Ctx) -> dict[str, Any]:
    """Chat Completions has no native reasoning, so jevper analyses and then answers."""
    service = ctx.service(questions=PLAIN, reasoning=ReasoningConfig(effort="low", mode="two_step"))
    response = service.classify(dict(tickets.SHORT_TICKET), method="structured")
    check_response(response, as_specs(PLAIN), model=ctx.profile.model, min_calls=2)
    check(
        response.debug.get("reasoning_mode") == "two_step",
        f"reasoning_mode is {response.debug.get('reasoning_mode')!r}",
    )
    return {
        "reasoning_mode": response.debug.get("reasoning_mode"),
        "reasoning_chars": len(reasoning_text(response.reasoning)),
        "calls": response.usage.n_calls,
    }


def reasoning_native(ctx: Ctx) -> dict[str, Any]:
    service = ctx.service(questions=PLAIN, reasoning=ReasoningConfig(effort="none", mode="native"))
    response = service.classify(dict(tickets.SHORT_TICKET), method="structured")
    check_response(response, as_specs(PLAIN), model=ctx.profile.model)
    check(
        response.debug.get("reasoning_mode") == "native",
        f"reasoning_mode is {response.debug.get('reasoning_mode')!r}",
    )
    return {
        "reasoning_mode": response.debug.get("reasoning_mode"),
        "parts": [part.type for part in response.reasoning],
    }


def reasoning_thinking_budget(ctx: Ctx) -> dict[str, Any]:
    """The Messages surface's own thinking: asked for with a budget, not an effort name.

    The budget has to sit below the output budget, and the profile's budget is the output one — so
    the call names a larger ceiling here, which is exactly the arithmetic the library refuses
    locally when a caller's own numbers do not fit.
    """
    budget = 1024
    service = ctx.service(
        questions=PLAIN,
        reasoning=ReasoningConfig(budget_tokens=budget),
        extra_body={"max_tokens": ctx.profile.messages_max_tokens + 2048},
    )
    response = service.classify(dict(tickets.SHORT_TICKET), method="structured")
    check_response(response, as_specs(PLAIN), model=ctx.profile.model)
    body = response.debug["llm_attempts"][-1]["request"].get("extra_body") or {}
    check((body.get("thinking") or {}).get("budget_tokens") == 1024, f"thinking on the wire is {body.get('thinking')!r}")
    return {
        "thinking": body.get("thinking"),
        "max_tokens": body.get("max_tokens"),
        "reasoning_chars": len(reasoning_text(response.reasoning)),
    }


# --------------------------------------------------------------------------------------
# surface discovery, the async facade and the SDK-free client
# --------------------------------------------------------------------------------------


def api_auto(ctx: Ctx) -> dict[str, Any]:
    """``api="auto"`` discovers the surface once and remembers it."""
    service = ctx.service(questions=PLAIN, api="auto", method="auto")
    first = service.classify(dict(tickets.SHORT_TICKET))
    check_response(first, as_specs(PLAIN), model=ctx.profile.model)
    second = service.classify(dict(tickets.LONG_TICKET))
    check_response(second, as_specs(PLAIN), model=ctx.profile.model)
    return {
        "api": first.debug.get("api"),
        "api_again": second.debug.get("api"),
        "methods": first.debug.get("methods"),
        "server_limits": first.debug.get("server_limits"),
    }


def async_parity(ctx: Ctx) -> dict[str, Any]:
    """The async facade, same rubric, same invariants — on a client of its own, as it must be.

    The caller's client stays the caller's, so the scenario closes it — inside the loop that made
    the calls, which is the only loop its connection pool belongs to.
    """
    import asyncio

    questions = PLAIN
    client = build_async_client(ctx.profile, messages=ctx.api == "messages")

    async def one() -> SystemOneResponse:
        try:
            facade = AsyncSystemOneClient(client, model=ctx.profile.model, **ctx.options())
            return await facade.system_one(state=dict(tickets.SHORT_TICKET), questions=questions)
        finally:
            await client.close()

    response = asyncio.run(one())
    check_response(response, as_specs(questions), model=ctx.profile.model)
    return {"answers": {key: value.model_dump(mode="json") for key, value in response.answers.items()}}


def async_batch(ctx: Ctx) -> dict[str, Any]:
    """Four tickets in flight, two at a time, one semaphore."""
    import asyncio

    questions = PLAIN
    client = build_async_client(ctx.profile, messages=ctx.api == "messages")

    async def one(item: Mapping[str, Any]) -> SystemOneResponse:
        facade = AsyncSystemOneClient(client, model=ctx.profile.model, max_concurrency=2, **ctx.options())
        return await facade.system_one(state=dict(item), questions=questions)

    async def batch() -> list[SystemOneResponse]:
        try:
            return list(await asyncio.gather(*(one(item) for item in tickets.TICKETS * 2)))
        finally:
            await client.close()

    responses = asyncio.run(batch())
    for response in responses:
        check_response(response, as_specs(questions), model=ctx.profile.model)
    return {"responses": len(responses), "calls": sum(item.usage.n_calls for item in responses)}


def duck_client_chat(ctx: Ctx) -> dict[str, Any]:
    """The same call with no SDK anywhere: a hand-rolled client, the server's own JSON."""
    duck = DuckClient(ctx.profile.base_url, api_key=ctx.profile.api_key)
    try:
        questions = PLAIN
        service = Triage(
            client=duck,
            model=ctx.profile.model,
            options={"method": "structured", "api": "chat_completions", **({"extra_body": ctx.profile.extra_body} if ctx.profile.extra_body else {})},
            rubric=questions,
        )
        response = service.classify(dict(tickets.SHORT_TICKET))
        check_response(response, as_specs(questions), model=ctx.profile.model)
        return {
            "requests": duck.calls,
            "routes": [request["route"] for request in duck.requests],
            "cache_key": (duck.requests[0]["body"].get("prompt_cache_key") or "")[:16],
            "intent": response.answers["intent"].choice,
        }
    finally:
        duck.close()


# --------------------------------------------------------------------------------------
# provider failure paths, against a real server
# --------------------------------------------------------------------------------------


def unknown_model(ctx: Ctx) -> dict[str, Any]:
    service = ctx.service(questions=PLAIN, model="model-that-does-not-exist")
    response = service.classify(dict(tickets.SHORT_TICKET), method="structured")
    return {"model": response.model, "answered_anyway": True}


def cache_measurement(ctx: Ctx) -> dict[str, Any]:
    """The same rubric twice with different states: the prefix is what the cache can reuse."""
    service = ctx.service(questions=RUBRIC, method="structured")
    first = service.classify(dict(tickets.SHORT_TICKET))
    second = service.classify(dict(tickets.LONG_TICKET))
    return {
        "first": first.usage.cached_tokens,
        "second": second.usage.cached_tokens,
        "input_tokens": second.usage.input_tokens,
    }


SCENARIOS: tuple[Scenario, ...] = (
    Scenario("core", core, ALL_APIS, ("auto", "structured", "discrete")),
    Scenario("without-few-shot", without_few_shot, ALL_APIS, ("structured",)),
    Scenario("raw-mapping-questions", raw_mapping_questions, ALL_APIS, ("structured",)),
    Scenario("noul-without-criteria", noul_without_criteria, ALL_APIS, ("structured",)),
    Scenario("logprobs", logprobs, ALL_APIS, ("logprobs",), allow=("LabelReadoutError", "UnsupportedMethodError", "ClientCapabilityError", "MalformedAnswerError", "ProviderError"), note="messages carries no logprobs at all"),
    Scenario("grammar", grammar, ("chat_completions",), ("grammar",), allow=("LabelReadoutError", "MalformedAnswerError", "ProviderError"), note="only llama.cpp takes the field; the others answer unconstrained"),
    Scenario("wide-26-logprobs", wide_26_logprobs, ("chat_completions",), ("logprobs",), allow=("LabelReadoutError", "MalformedAnswerError", "ProviderError"), note="a 4B model answers the widest label prompt with prose"),
    Scenario("wide-30-structured", wide_30_structured, ("chat_completions",), ("structured",), allow=("MalformedAnswerError", "ProviderError")),
    Scenario("catalog-255-structured", catalog_255_structured, ("chat_completions",), ("structured",), allow=("MalformedAnswerError", "ProviderError"), note="the Jev maximum for one Choice"),
    Scenario("examples-with-probabilities", examples_with_probabilities, ("chat_completions",), ("structured",), allow=("MalformedAnswerError", "ProviderError")),
    Scenario("few-shot-precedence", few_shot_precedence, ("chat_completions",), ("logprobs",), allow=("LabelReadoutError", "ProviderError")),
    Scenario("state-text", state_text, ALL_APIS, ("structured",), allow=("MalformedAnswerError", "ProviderError")),
    Scenario("state-json", state_json, ALL_APIS, ("structured",), allow=("MalformedAnswerError", "ProviderError")),
    Scenario("state-chat-list", state_chat_list, ALL_APIS, ("structured",), allow=("MalformedAnswerError", "ProviderError")),
    Scenario("state-messages-mapping", state_messages_mapping, ALL_APIS, ("structured",), allow=("MalformedAnswerError", "ProviderError")),
    Scenario("state-assistant-last", state_assistant_last, ALL_APIS, ("structured",), allow=("MalformedAnswerError", "ProviderError")),
    Scenario("state-developer-role", state_developer_role, ALL_APIS, ("structured",), allow=("ProviderError",), note="SGLang refuses a developer turn"),
    Scenario("temperature-zero", temperature_zero, ALL_APIS, ("structured",), allow=("MalformedAnswerError", "ProviderError")),
    Scenario("prompt-cache-key-override", prompt_cache_key_override, ALL_APIS, ("structured",), allow=("MalformedAnswerError", "ProviderError")),
    Scenario("prompt-cache-key-72", prompt_cache_key_72, ALL_APIS, ("structured",), allow=("MalformedAnswerError", "ProviderError"), note="longer than OpenAI's cap; the provider's to refuse"),
    Scenario("structured-outputs-off", structured_outputs_off, OPENAI_APIS, ("structured",), allow=("MalformedAnswerError", "ProviderError")),
    Scenario("normalize-off", normalize_off, ALL_APIS, ("structured",), allow=("MalformedAnswerError", "ProviderError")),
    Scenario("top-logprobs", top_logprobs, OPENAI_APIS, ("logprobs",), allow=("LabelReadoutError", "ProviderError")),
    Scenario("max-concurrency", max_concurrency, ALL_APIS, ("structured",), allow=("MalformedAnswerError", "ProviderError")),
    Scenario("retry-settings", retry_settings, ALL_APIS, ("structured",), allow=("MalformedAnswerError", "ProviderError")),
    Scenario("extra-headers", extra_headers, ALL_APIS, ("structured",), allow=("MalformedAnswerError", "ProviderError")),
    Scenario("extra-body-output-budget", extra_body_output_budget, OPENAI_APIS, ("structured",), allow=("MalformedAnswerError", "ProviderError")),
    Scenario("tiny-output-budget", tiny_output_budget, OPENAI_APIS + ("messages",), ("structured",), allow=("IncompleteAnswerError", "MalformedAnswerError", "ProviderError", "JevperError")),
    Scenario("stream-zero", stream_zero, OPENAI_APIS, ("structured",), allow=("ProviderError",), note="a typed refusal on three of the five"),
    Scenario("reasoning-two-step", reasoning_two_step, ("chat_completions",), ("structured",), allow=("MalformedAnswerError", "ProviderError")),
    Scenario("reasoning-native", reasoning_native, ("responses",), ("structured",), allow=("MalformedAnswerError", "ProviderError")),
    Scenario("reasoning-thinking-budget", reasoning_thinking_budget, ("messages",), ("structured",), allow=("MalformedAnswerError", "ProviderError", "JevperError"), note="some servers spend the budget and answer nothing"),
    Scenario("api-auto", api_auto, ("auto",), ("auto",)),
    Scenario("async-parity", async_parity, OPENAI_APIS, ("structured",), allow=("MalformedAnswerError", "ProviderError")),
    Scenario("async-batch", async_batch, OPENAI_APIS, ("structured",), allow=("MalformedAnswerError", "ProviderError")),
    Scenario("duck-client-chat", duck_client_chat, ("chat_completions",), ("structured",), allow=("MalformedAnswerError", "ProviderError")),
    Scenario("unknown-model", unknown_model, ALL_APIS, ("structured",), allow=("ProviderError", "JevperError"), note="some servers ignore an unknown model id"),
    Scenario("cache-measurement", cache_measurement, OPENAI_APIS, ("structured",), allow=("MalformedAnswerError", "ProviderError")),
)


# --------------------------------------------------------------------------------------
# the runner
# --------------------------------------------------------------------------------------


_TRANSPORT_FAILURES = frozenset(
    {
        "APIConnectionError",
        "APITimeoutError",
        "ConnectError",
        "ConnectTimeout",
        "ConnectionError",
        "ReadTimeout",
        "RemoteProtocolError",
        "TimeoutError",
    }
)
"""A server that cannot be reached says nothing about what it supports."""


def _is_unreachable(exc: BaseException) -> bool:
    """Whether this failure is the network rather than the provider."""
    pending = [exc]
    seen: list[BaseException] = []
    while pending:
        error = pending.pop()
        if error in seen:
            continue
        seen.append(error)
        for link in (error.__cause__, error.__context__):
            if isinstance(link, BaseException):
                pending.append(link)
    return any(type(error).__name__ in _TRANSPORT_FAILURES for error in seen)



def _allowed(exc: BaseException, allow: tuple[str, ...]) -> bool:
    """Whether a failure is one this scenario declared legitimate, by class or by any base class.

    The library raises private subclasses for the provider-side cases its public errors cover — a
    logprob carrier the provider cannot fill, for one — so a scenario that declared
    ``LabelReadoutError`` means the case, not the spelling.
    """
    return any(base.__name__ in allow for base in type(exc).__mro__)


def _attempt(scenario: Scenario, ctx: Ctx, api: str, method: str) -> dict[str, Any]:
    ctx.api = api
    ctx.method = method
    before = ctx.counter.total
    started = time.perf_counter()
    record: dict[str, Any] = {"scenario": scenario.name, "api": api, "method": method}
    try:
        record["observation"] = scenario.run(ctx)
        record["status"] = "ok"
    except Gap as exc:
        record.update(status="gap", error=f"invariant: {exc}")
    except JevperError as exc:
        name = type(exc).__name__
        if _is_unreachable(exc):
            record.update(status="unreachable", error=f"{name}: {exc}")
        elif (
            ctx.api == "responses"
            and method in ("structured", "discrete")
            and not ctx.profile.responses_json_schema
            and _allowed(exc, ("MalformedAnswerError",))
        ):
            # The server took `text.format` and ignored it, so the model answered in the prompt's
            # own shape. Measured and documented per server; the service routes around it with
            # api="chat_completions", where the same schema is enforced.
            record.update(
                status="documented",
                error=f"{name}: {ctx.profile.name} ignores text.format on /v1/responses",
            )
        else:
            record.update(
                status="documented" if _allowed(exc, scenario.allow) else "gap",
                error=f"{name}: {exc}",
            )
    except Exception as exc:  # noqa: BLE001 - a raw exception from a consumer is a finding
        record.update(status="gap", error=f"raw {type(exc).__name__}: {exc}")
    record["seconds"] = round(time.perf_counter() - started, 3)
    record["requests"] = ctx.counter.total - before
    return record

def run_matrix(
    profile: ServerProfile,
    *,
    only: tuple[str, ...] = (),
    methods: tuple[str, ...] = (),
    apis: tuple[str, ...] = (),
    scenarios: tuple[Scenario, ...] = SCENARIOS,
) -> list[dict[str, Any]]:
    """Every scenario on every surface it declares, against one server."""
    ctx = Ctx(profile=profile)
    report: list[dict[str, Any]] = []
    for scenario in scenarios:
        if only and scenario.name not in only:
            continue
        for api in scenario.apis:
            if apis and api not in apis:
                continue
            for method in (methods or scenario.methods):
                if method not in scenario.methods:
                    continue
                client = build_client(profile, messages=api == "messages", counter=ctx.counter)
                ctx.client = client
                try:
                    report.append(_attempt(scenario, ctx, api, method))
                finally:
                    with contextlib.suppress(Exception):
                        client.close()
                    ctx.client = None
    return report


def summarize(report: list[dict[str, Any]]) -> dict[str, Any]:
    """Counts by status, the gaps, and what the run cost."""
    counts: dict[str, int] = {}
    for record in report:
        counts[record["status"]] = counts.get(record["status"], 0) + 1
    return {
        "counts": counts,
        "requests": sum(record.get("requests", 0) for record in report),
        "seconds": round(sum(record.get("seconds", 0.0) for record in report), 1),
        "gaps": [
            {"scenario": record["scenario"], "api": record["api"], "method": record["method"], "error": record.get("error")}
            for record in report
            if record["status"] in ("gap", "unreachable")
        ],
    }
