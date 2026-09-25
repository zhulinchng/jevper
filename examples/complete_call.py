"""One jevper call with every public option set explicitly.

This file is the program the "Complete example" page embeds, and ``tests/test_docs.py`` runs it
against a local stub server, so the page and a program that works cannot drift apart.

Run it against any OpenAI-compatible server:

    JEVPER_API_KEY=sk-... JEVPER_MODEL=gpt-5.6-terra python examples/complete_call.py

A local server ignores the key but the SDK still wants one, and the base URL carries the ``/v1``:

    JEVPER_API_KEY=local JEVPER_BASE_URL=http://127.0.0.1:11434/v1 \\
        JEVPER_MODEL=qwen3.5:9b python examples/complete_call.py
"""

import json
import os

from openai import OpenAI

from jevper import (
    Choice,
    Example,
    Noul,
    ReasoningConfig,
    RetryPolicy,
    Score,
    SystemOneClient,
)

BASE_URL = os.environ.get("JEVPER_BASE_URL", "https://api.openai.com/v1")
API_KEY = os.environ.get("JEVPER_API_KEY", "replace-me")
MODEL = os.environ.get("JEVPER_MODEL", "gpt-5.6-terra")

# Few-shot demonstrations are a state and the answer it earned. They can be attached at three
# levels, and all three are used below: on the client (``examples``), on the call (``examples``), and
# on the question itself (``Choice(examples=...)``). They add up rather than replace each other, in
# the order question, call, client — so the client's demonstrations are the ones closest to the
# question. A mapping is keyed by question id; a sequence applies to every question, which only
# works when they are all the same type.
CLIENT_EXAMPLES = {
    "intent": [
        Example(
            state="The app signs me out every time I open it.",
            answer="technical",
            probabilities={"billing": 0.02, "technical": 0.94, "sales": 0.04},
        )
    ],
    "sentiment": [
        Example(
            state="Three outages this week. I am done paying for this.",
            answer=2,
            probabilities={0: 0.0, 1: 0.1, 2: 0.9},
        )
    ],
}

QUESTIONS = {
    # Choice picks one of your keys and reports a probability for each. One to 255 keys; the key
    # order is the answer order, and it decides a tie.
    "intent": Choice(
        instructions="Pick the intent of the message.",
        criteria={
            "billing": "money, invoices, refunds or charges",
            "technical": "errors, crashes, login or performance problems",
            "sales": "pricing, plans, purchasing or an upgrade",
        },
        examples=[
            Example(
                state="Where do I download the invoice for last month?",
                answer="billing",
                probabilities={"billing": 0.91, "technical": 0.06, "sales": 0.03},
            )
        ],
    ),
    # Noul answers with one probability: 1.0 is true, 0.0 is false. The criteria are optional
    # descriptions of each end, not the answer.
    "needs_human": Noul(
        instructions="Does this need a person to answer it today?",
        criteria={
            "true": "the customer is waiting on an answer only a person can give",
            "false": "the documented answer is enough",
        },
    ),
    # Score rates on an ordered scale of 2 to 10 levels and reports a probability per level. The
    # answer is the probability-weighted level index, levels counted from zero.
    "sentiment": Score(
        instructions="Rate how angry the customer is.",
        criteria=["calm", "frustrated", "angry"],
    ),
}


def main() -> None:
    # The provider client is yours. jevper never creates, closes or reconfigures one: it calls the
    # object you hand it, through a copy whose own retry loop is off, so the ``RetryPolicy`` below is
    # the only retry loop and ``usage.n_retries`` counts every request the provider saw. ``timeout``
    # is the SDK's own, and it applies to each request the SDK makes.
    provider = OpenAI(base_url=BASE_URL, api_key=API_KEY, timeout=60.0)
    try:
        with SystemOneClient(
            provider,
            # Required. The model id every call sends; also the key jevper remembers capability
            # verdicts under, so a second model on one client is judged on its own first refusals.
            model=MODEL,
            # How the decision is elicited: "auto" (the default) asks for logprobs and falls back to
            # JSON where the provider has none, or one of "logprobs", "grammar", "structured",
            # "discrete". Pinned here so the program behaves the same on every server. "logprobs" and
            # "grammar" read a one-token label and are Chat-Completions only; "structured" asks for
            # a JSON distribution and works everywhere; "discrete" asks for one option and reports
            # it as one-hot.
            method="structured",
            # The wire surface: "auto" (the default) prefers responses, then chat_completions, then
            # messages, and falls back when a route is missing; or pin one. The choice decides which
            # provider fields exist, which is why the output budget below is named per surface.
            api="chat_completions",
            # Reasoning. mode="native" is one call carrying the provider's own reasoning fields,
            # "two_step" is an analysis call and then the answer call, "auto" picks native on
            # responses and two_step elsewhere, and "off" is the same as reasoning=None. effort and
            # summary are the Responses parameters, context is llama.cpp's, and budget_tokens is
            # the Messages surface's ``thinking`` budget — the one field of this block Chat
            # Completions does not use.
            reasoning=ReasoningConfig(
                mode="native",
                effort="low",
                summary="auto",
                context="auto",
                budget_tokens=1024,
            ),
            # Default few-shot examples for every call this client makes.
            examples=CLIENT_EXAMPLES,
            # Send a strict JSON schema for the answer (``response_format`` here, ``text.format`` on
            # the Responses surface, ``output_config.format`` on Messages). False leaves the schema
            # in the prompt and asks for plain JSON; a server that refuses the strict schema gets
            # that same fallback on its own, whichever way this is set.
            structured_outputs=True,
            # Rescale a structured distribution that misses 1 by more than 1e-6, keeping the model's
            # numbers in ``debug["original_probabilities"]``. False reports them verbatim, sum and
            # all — and an all-zero distribution then has no argmax, so it comes back as the
            # uniform expectation rather than a raise.
            normalize_probabilities=True,
            # Alternatives requested for the ``logprobs`` and ``grammar`` readouts, 0 to 20. A label
            # readout needs at least two: one alternative to compare the sampled token against.
            # Ignored by the structured readouts this program uses.
            top_logprobs=20,
            # Questions answered at once. Questions are independent, so this is a thread pool here
            # and an asyncio semaphore in AsyncSystemOneClient.
            max_concurrency=8,
            # Corrective retries when an answer cannot be read: the client quotes the model its own
            # bad reply and asks again. Separate from the transient-failure retries below.
            n_retry_malformed=1,
            # Transient-failure retries per provider call: HTTP 408, 409, 429 and any 5xx, plus
            # connection and timeout errors, with exponential backoff that honours Retry-After when
            # the server sends one. These are the defaults, written out.
            retry=RetryPolicy(
                n_retries=2,
                base_delay=0.5,
                max_delay=8.0,
                respect_retry_after=True,
            ),
            # Sampling temperature, sent only when set. 0.0 is what a classification wants: the
            # distribution should be the model's belief, not a sample from it. The OpenAI surfaces
            # take it as a typed field; on Messages it travels in the body and is left out entirely
            # when a thinking budget is on, which that API refuses alongside a non-default
            # temperature.
            temperature=0.0,
            # Provider request fields jevper has no parameter for, merged into every request body.
            # This is where the output budget lives, and its name is the surface's own:
            # max_completion_tokens here, max_output_tokens on responses, max_tokens on messages,
            # where it is also required and defaults to 1024. A field named here is the value that
            # reaches the wire, so naming response_format, text or output_config also moves the
            # schema into the prompt.
            extra_body={"max_completion_tokens": 2048},
            # Headers sent with every request. A name spelled the way the client spells its own
            # replaces the default instead of joining it. Credential headers are redacted in
            # ``debug``.
            extra_headers={"x-jevper-example": "complete"},
            # The provider's cache-routing key. Left unset, jevper derives a stable one per question
            # from the parts of the prompt that do not change between calls, so a rubric's requests
            # share a cached prefix. Set it to group requests your own way — and pass your own when
            # the derived key is a fingerprint of your rubric at the provider.
            prompt_cache_key="complete-example",
        ) as client:
            # The state under judgement. A string, a list of chat turns, {"messages": [...]}, or any
            # JSON value; it is rendered last, after the system prompt, the demonstrations and the
            # question block, so every state classified with this rubric shares the cached prefix.
            state = [
                {"role": "system", "content": "Support inbox for a subscription product."},
                {
                    "role": "user",
                    "content": (
                        "I was charged twice this month and the second charge is not on my card "
                        "statement. I need this fixed today."
                    ),
                },
            ]

            # Per-call options override the client's for this call only. The values repeat the
            # client's here on purpose: what this shows is that the override exists and where it
            # sits, not that the two should differ.
            response = client.system_one(
                state=state,
                questions=QUESTIONS,
                examples={
                    "intent": [
                        Example(
                            state="Why is my invoice higher than the plan I signed up for?",
                            answer="billing",
                        )
                    ]
                },
                model=MODEL,
                method="structured",
                api="chat_completions",
                reasoning=ReasoningConfig(mode="native", effort="low"),
                temperature=0.0,
                prompt_cache_key="complete-example",
            )

            # One typed answer per question, in the order the questions were given. NoulAnswer
            # carries ``noul``, ChoiceAnswer ``choice``, ``probabilities`` and ``confidence``,
            # ScoreAnswer ``score`` and ``legend`` on top of the same two.
            for question_id, answer in response.answers.items():
                print(question_id, answer.type, json.dumps(answer.model_dump(mode="json")))

            # Every provider call this response cost, summed: the token counts, the call count, the
            # transient retries, the wall-clock latency, and the prompt tokens the provider read
            # from its own cache. A count is None when the provider reported none, which is not the
            # same as a reported 0.
            print("usage:", response.usage)

            # What the call actually did. These four keys are always there; the rest are conditional
            # and are read with .get(). Each attempt record holds the request kwargs sent (with
            # credential headers redacted), the provider's response, the error if there was one, and
            # the parsed readout.
            debug = response.debug
            print(
                "debug:",
                debug["method"],
                debug["api"],
                debug["reasoning_mode"],
                len(debug["llm_attempts"]),
                "attempt(s)",
            )
            if debug.get("server_limits"):
                print(
                    "the server refused these fields, and jevper stopped sending them:",
                    debug["server_limits"],
                )

            # The whole response is the Jev wire shape, so it serializes to what the hosted API
            # returns.
            print(response.model_dump_json(exclude_none=True))
    finally:
        # Leaving the ``with`` block closed jevper's own thread pool. The provider client is yours
        # to close; jevper never closes it, on either facade.
        provider.close()


if __name__ == "__main__":
    main()
