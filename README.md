# jevper

The [Jev](https://docs.typesafe.ai) interface — `state` in, typed `questions` (`noul`, `choice`, `score`) out,
answers carrying probabilities and confidence — on top of any OpenAI-compatible model.

Same call as `typesafe-sdk`, different backend: point `jevper` at a hosted LLM or a self-hosted llama.cpp
server and code written for Jev keeps working, unchanged.

It does not call the hosted TypeSafe API and does not depend on `typesafe-sdk` or `openai` at runtime — the
client object is duck-typed. Any object exposing `responses.create` or `chat.completions.create` works,
including a self-hosted llama.cpp server.

> `jevper` is an independent implementation of the documented System One wire format. It is not affiliated
> with, endorsed by, or supported by TypeSafe AI — questions about the API itself belong in
> [their docs](https://docs.typesafe.ai).

```python
from openai import OpenAI
from jevper import Choice, SystemOneClient

client = SystemOneClient(OpenAI(), model="gpt-5.6-terra")

response = client.system_one(
    state="I was charged twice for the same subscription this month.",
    questions={
        "intent": Choice(
            instructions="Pick the intent of the message.",
            criteria={
                "billing": "money, invoices, refunds, charges",
                "technical": "errors, crashes, login or performance problems",
                "sales": "pricing, plans, purchasing, upgrades",
            },
        )
    },
)

answer = response.answers["intent"]
answer.choice        # "billing"
answer.probabilities # {"billing": 0.88, "technical": 0.08, "sales": 0.03}
answer.confidence    # 0.83
```

`method` defaults to `auto`: it asks for logprobs where the provider has them and answers in JSON where it
does not, remembering the verdict per model and surface. `gpt-5.6-terra` is a reasoning model and returns
none, so the probabilities above arrive as JSON. Point `method="logprobs"` at a provider that does return
them — a local ollama, llama.cpp or vLLM server, or a non-reasoning OpenAI model — to read the model's
real distribution instead of its self-report.

## Install

```sh
pip install jevper
```

Python 3.10+. The only runtime dependency is `pydantic>=2.7`.

For development:

```sh
git clone https://github.com/zhulinchng/jevper && cd jevper
uv venv && uv pip install -e '.[test]'
pytest -q
```

## What one call does

```mermaid
flowchart LR
    A["state + questions"] --> B["build_parts + assemble: system prompt, few-shot turns, question block, state turns"]
    B --> C{"method (auto resolves first)"}
    C -->|logprobs| D["logprobs=true, top_logprobs=20"]
    C -->|grammar| E["+ GBNF grammar in extra_body"]
    C -->|structured| F["strict JSON schema: probabilities"]
    C -->|discrete| G["strict JSON schema: one label"]
    D --> H["first label token -> softmax over the labels"]
    E --> H
    F --> I["probability dict from JSON"]
    G --> J["one-hot from the chosen label"]
    H --> K["Answer: choice / noul / score"]
    I --> K
    J --> K
```

Each question becomes its own provider call, so questions are independent and run concurrently
(`max_concurrency`, default 8). Answers come back keyed by your question ids, in insertion order.

## Questions

Three types, mirroring the Jev API — `Noul` answers yes/no with one probability, `Choice` picks one of your
labelled options, `Score` rates on an ordered scale:

| Type | Criteria | Answer |
| --- | --- | --- |
| `Noul(instructions=..., criteria={"true": ..., "false": ...})` | optional | `{"type": "noul", "noul": 0.93}` |
| `Choice(instructions=..., criteria={"billing": "...", ...})` | 1–255 keys | `{"type": "choice", "choice": "billing", "probabilities": {...}, "confidence": 0.83}` |
| `Score(instructions=..., criteria=["Calm", "Frustrated", "Very angry"])` | 2–10 levels | `{"type": "score", "score": 1.05, "legend": {...}, "probabilities": {...}, "confidence": 0.92}` |

`Score.score` is the probability-weighted level index (`Σ i·pᵢ`, levels zero-based), as in the Jev API —
read off the distribution rescaled to sum 1, so with `normalize_probabilities=False` the reported
`probabilities` stay the model's own numbers while the score stays on the `0..N-1` line.
`Choice` takes up to 255 options, the Jev API limit, and the API documents no minimum.
The two methods that read a label *token* — `logprobs` and `grammar` — stop at 26, because the first token
of `"AA"` is `"A"`; past 26 options they
raise `InvalidQuestionError` pointing at `structured` and `discrete`, which answer in JSON and use
two-letter labels. The default `method="auto"` never hits that error: it answers a wide `Choice` in JSON.

Questions can also be passed as raw mappings (`{"type": "choice", "criteria": {...}}`) and are validated the
same way.

## Methods

`method=` decides how the decision is elicited. All four share the same label→option mapping, so switching
methods does not change your types; only the label alphabet differs (`logprobs` and `grammar` need
single-letter labels, so they cap at 26 options).

| Method | Request | Readout | Needs |
| --- | --- | --- | --- |
| `auto` (default) | `logprobs`, or `structured` where the provider cannot return logprobs | whichever method it resolved to | a provider that returns logprobs, or JSON-schema structured output |
| `logprobs` | `logprobs=true, top_logprobs=20` | softmax over the labels' logprobs of the first answer token | a provider that returns chat logprobs, or the Responses surface with `include` logprobs. A surface that refuses the carrier hands the readout to the other OpenAI surface, method intact; with nowhere left to go, the refusal is reported |
| `grammar` | the same plus a GBNF `grammar` in `extra_body` | same as `logprobs` | a Chat Completions server that accepts `grammar` (llama.cpp and friends) |
| `structured` | strict JSON schema, model returns a probability per option | the model's own numbers, rescaled to sum 1 when off by more than `1e-6` | JSON-schema structured output |
| `discrete` | strict JSON schema, model returns one option | one-hot distribution | JSON-schema structured output |

`auto` is the default because logprobs are not universal: OpenAI's reasoning models — the GPT-5.6 family
included — do not offer them, Anthropic and Gemini's OpenAI-compatibility endpoints never had them, and a
model that returns a logprob with no alternatives gives you no distribution at all. A gateway in front of
one says so in as many words (`logprobs are not supported with reasoning models.`), and so does a
Responses endpoint that refuses the `include` list the carrier travels in.
`auto` reads the logprobs where they exist — one short call, and the model's real distribution rather than a
self-report — and answers in JSON where they do not, remembering the verdict per model and surface. See
[docs/methods.md](https://github.com/zhulinchng/jevper/blob/main/docs/methods.md#auto) for the
provider table, the exact request bodies, the readout rules and the failure modes.

The same holds for the request fields jevper adds: a server that refuses structured output, the reasoning
parameters, the Responses `include` list, the cache key or the Messages `output_config` gets that field
dropped and the call re-asked, so a partially implemented server answers instead of failing.
`debug["server_limits"]` reports what it refused.

Hosted providers need no adapter of their own. Gemini speaks the OpenAI API at
`https://generativelanguage.googleapis.com/v1beta/openai/`, so `OpenAI(base_url=…, api_key=…)` is the whole
integration; what it lacks in logprobs, `auto` answers around.

## Reasoning

Pass `reasoning=ReasoningConfig(...)` to make the model think before it classifies:

```python
from jevper import ReasoningConfig, reasoning_text

client = SystemOneClient(OpenAI(), model="gpt-5.6-terra", reasoning=ReasoningConfig(effort="medium"))

response = client.system_one(state=..., questions=...)
reasoning_text(response.reasoning)   # the trace, as text
```

`mode="auto"` (the default) uses native provider reasoning on the Responses surface and a two-step
think-then-classify path on Chat Completions, where the analysis text is replayed as an assistant turn before
the answer. The trace always lands on `response.reasoning`, and the two-step analysis call's usage is counted
in `response.usage`. See [docs/reasoning.md](https://github.com/zhulinchng/jevper/blob/main/docs/reasoning.md).

## Few-shot examples

Examples are chat turns (question block + example state, then the expected answer), so the demonstration is
always in the format the active method expects. They can be attached at three levels:

```python
from jevper import Choice, Example, SystemOneClient

question = Choice(
    criteria={"billing": "...", "technical": "..."},
    examples=[Example(state="Charged twice for one order", answer="billing")],
)

client = SystemOneClient(OpenAI(), model="gpt-5.6-terra",
                         examples=[Example(state="Login fails", answer="technical")])  # fallback for every question

client.system_one(state=..., questions={"intent": question},
                  examples={"intent": [...]})  # or a bare sequence for all questions
```

Precedence is question → per call → constructor, and the first non-empty level wins. `examples` is excluded
from `model_dump()`, so question dumps keep exactly the Jev wire keys. See
[docs/few-shot.md](https://github.com/zhulinchng/jevper/blob/main/docs/few-shot.md).

## Response

```python
response.model                 # the model id jevper asked for
response.answers               # {"intent": ChoiceAnswer(...)}
response.nouls / .choices / .scores   # filtered views
response.usage                 # input_tokens, output_tokens, reasoning_tokens, cached_tokens, n_calls, n_retries, latency
response.reasoning             # tuple[ReasoningContentPart, ...]
response.debug                 # per-attempt requests/responses, retry reasons, normalization notes
```

`response.model_dump_json()` serializes to the Jev answer shape — the answer field names and JSON keys match
`POST /v1/systemone`. Token counts are `None` when any constituent call omitted them; `n_calls` counts the
provider calls that returned a result, including analysis passes and corrective retries, while `n_retries`
counts transient-failure retries only. A failed attempt appears in `debug["llm_attempts"]` but not in `usage`.
Confidence is a share in `[0, 1]` and the call counters are counts, because jevper computes them; the
probabilities beside them are the model's own numbers, passed through verbatim when
`normalize_probabilities=False`. See [docs/api.md](https://github.com/zhulinchng/jevper/blob/main/docs/api.md)
for the full reference.

## Prompt caching

Every provider that serves these calls caches the *prefix* of a prompt and reuses it for the next request
that starts the same way, and jevper is shaped for it: the state comes last, so the system prompt, the
few-shot examples and the question block are identical across every state classified with one rubric.

```python
client = SystemOneClient(OpenAI(), model="gpt-5.6")

client.system_one(state=record_a, questions=rubric)
client.system_one(state=record_b, questions=rubric)   # the shared prefix is reused
```

Two things make it steerable and observable:

- **`prompt_cache_key`** is sent with every request, derived per question from the parts of the prompt that do
  not change between calls — model, method, examples, question block — so a rubric's requests are routed
  together, and a `logprobs` request is not routed with a `structured` one whose prefix differs. Pass your own
  to group or account for them your way, on the client (`prompt_cache_key="tenant-42"`) or per call. A server
  that refuses the field gets it dropped and the call re-asked, like the other optional fields.
- **`usage.cached_tokens`** is the prompt tokens the provider read from its cache, summed over the call.
  `None` means the provider said nothing — vLLM needs `--enable-prompt-tokens-details`, and SGLang's Chat
  Completions route needs `--enable-cache-report` — while a reported `0` means a cold or disabled cache.

Measured on one 2388-token prompt carrying two examples, second call differing only in the state: reused
tokens went from 40 — the system prompt alone — to 1010 on llama.cpp, 528 on vLLM and 896 on SGLang once the
state moved to the end. Per-server flags, what each server accepts or ignores, and how to isolate a cache with
`cache_salt`: [docs/local-servers.md](https://github.com/zhulinchng/jevper/blob/main/docs/local-servers.md#prompt-caching).

## Anthropic-compatible servers

Every server in the local fleet serves the Anthropic Messages API at `/v1/messages` as well as the OpenAI
ones. jevper speaks it with `api="messages"`: point the `anthropic` client at the server and pass it in place
of the OpenAI one.

```python
from anthropic import Anthropic

from jevper import SystemOneClient

client = SystemOneClient(Anthropic(base_url="http://127.0.0.1:1234"), model="qwen3-4b-instruct")

client.system_one(state=record, questions=rubric, api="messages", method="structured")
```

Five things differ from the OpenAI surfaces, and all five come from the protocol rather than from any
server:

- **No logprobs exist in it.** Not withheld by some servers — absent from the API. `method="logprobs"` and
  `method="grammar"` raise `UnsupportedMethodError` before a request is sent, and `method="auto"` answers in
  JSON without spending a call to find out. `structured` and `discrete` work exactly as they do elsewhere: the
  prompt already asks for one JSON object.
- **Its schema field is Anthropic's own**, `output_config.format`, the counterpart of `response_format`:
  `structured`/`discrete` send it wherever the server takes it, and a server that refuses it gets it dropped
  and the call re-asked, reported in `debug["server_limits"]["output_config"]`. The JSON Schema also stays
  in the system prompt, because a server can take that field and discard it without a word — vLLM's
  Messages request model drops what it does not model — and on none of the local servers is the answer's
  shape constrained by the request.
- **`max_tokens` has no server-side default.** jevper sends `1024` — or `1024` plus the caller's thinking
  budget, because Anthropic requires the budget to be strictly *below* `max_tokens` and would otherwise refuse
  the 1024 its own docs call the floor. `extra_body={"max_tokens": n}` overrides both, and a value that cannot
  hold the budget you asked for raises `JevperError` locally, naming both numbers, rather than being sent to
  earn the 400.
- **`temperature` is not a typed parameter of the current SDK** and is left out of a request that enables
  thinking, which the API refuses alongside a non-default one. On the other requests it travels in the request
  body, so a local server still reads it.
- **Thinking is asked for with a budget, not an effort name.** `ReasoningConfig(budget_tokens=2048)` sends
  `thinking={"type": "enabled", "budget_tokens": 2048}`: a budget is the only reason to ask for this surface's
  own thinking, so it selects it even under `mode="auto"`. A server that refuses the field gets it dropped and
  the call re-asked, reported in `debug["server_limits"]["thinking"]`; a server that refuses the *value* —
  SGLang answers `budget_tokens: must be at least 1024` — gets its own error back instead, because a bad
  number is not a missing field.

Thinking blocks come back as ordinary `response.reasoning` parts with the block's `signature` kept, and
`usage.cached_tokens` is read from `cache_read_input_tokens`. Which servers implement the route, and since
which version: [docs/local-servers.md](https://github.com/zhulinchng/jevper/blob/main/docs/local-servers.md#the-messages-route).

## Failures

Local problems fail before any request is sent: an invalid question, an empty `questions` mapping, an
unusable `state`, a `model` that is not a non-empty string, a count option that is not an integer, or
`grammar` on a surface that cannot carry a grammar.

| Error | Raised when |
| --- | --- |
| `InvalidQuestionError` | question or few-shot example is locally invalid |
| `UnsupportedMethodError` | `method="grammar"` on the Responses surface, or `method="logprobs"`/`"grammar"` on the Messages surface — that API has no logprobs at all |
| `ClientCapabilityError` | the client lacks the attribute the chosen surface needs, or the response carried no choices and no explanation of why |
| `LabelReadoutError` | the first answer token is not a label, or the provider returned no logprobs (or no alternatives, or no logprob for that token). The provider-side cases are not corrective-retried, and `method="auto"` answers them with `structured` |
| `MalformedAnswerError` | the JSON answer had an unusable shape after corrective retries |
| `IncompleteAnswerError` | the provider stopped generating before the answer was complete — `finish_reason: "length"`, `stop_reason: "max_tokens"`, a Responses `status: "incomplete"`, or a filtered answer. A `ProviderError` subclass, and terminal: a cut-off generation is not something a corrective retry can fix |
| `ModelRefusalError` | the model declined to answer and the provider said so. Also a `ProviderError` subclass, and terminal — a refusal is complete, not broken |
| `ProviderError` | a provider call failed; `.attempts` carries the attempt history and `.status_code` the status the provider reported — including one carried inside a `200` body, which is how OpenRouter reports an upstream failure |
| `JevperError` | constructor misuse, a bad `state` message, or content that is not JSON-serializable |

Transient failures (HTTP 408/429/500/502/503/504/529, connection and timeout errors — including the `httpx`
transport errors whose class names carry neither word) are retried per call with
`RetryPolicy(n_retries=2, base_delay=0.5, max_delay=8.0, respect_retry_after=True)`. The wait is the
provider's own instruction when it sent one: a `Retry-After` (seconds or an HTTP date) or the millisecond
`retry-after-ms` replaces the exponential backoff `min(base_delay · 3ⁿ, max_delay)`, which is what the
TypeSafe clients do — coming back sooner than a rate limit asked only extends it. `max_delay` caps jevper's
curve, not the server's number; `respect_retry_after=False` goes back to the curve alone. Unreadable answers
get one corrective retry (`n_retry_malformed`) with the failure appended to the conversation. `ProviderError`
propagates after all questions have settled, in question insertion order.

## Tracing

Every provider call goes through the SDK client you hand in, so `mlflow.openai.autolog()`,
`mlflow.anthropic.autolog()` or OpenTelemetry instrumentation sees them without extra wiring — including the
attempts made before settling on a request shape the provider accepts. The per-question worker threads run a
copy of your context, so a span you open around `system_one` parents every one of them: one trace per call,
one child span per question, however many questions it carried. Span names, attributes and the hosting
recipes: [docs/mlflow.md](https://github.com/zhulinchng/jevper/blob/main/docs/mlflow.md).

## Verification

```sh
pytest -q                      # the whole suite runs against a local stub HTTP server; no network, no API keys
ruff check src tests           # clean except three PYI034 hints (see docs/internals.md)
```

The suite drives a real `openai` SDK client at a stdlib `ThreadingHTTPServer` stub, so the SDK's own
serialization path is exercised; see [docs/internals.md](https://github.com/zhulinchng/jevper/blob/main/docs/internals.md#testing).

Optional live check, skipped unless both variables are set:

```sh
LLM_MODEL=gpt-5.6-terra OPENAI_API_KEY=... pytest -q tests/test_live.py
```

Optional MLflow check, skipped unless MLflow is installed — tracing, hosting jevper as a model, the AI
Gateway, and `mlflow.genai.evaluate` (see [docs/mlflow.md](https://github.com/zhulinchng/jevper/blob/main/docs/mlflow.md)):

```sh
uv pip install -e '.[test,mlflow]' && pytest -q tests/test_mlflow.py
```

## Docs

- [docs/api.md](https://github.com/zhulinchng/jevper/blob/main/docs/api.md) — constructor and `system_one` parameters, answer/usage/debug shapes, errors
- [docs/methods.md](https://github.com/zhulinchng/jevper/blob/main/docs/methods.md) — the four methods, request bodies, readout rules, surface selection
- [docs/local-servers.md](https://github.com/zhulinchng/jevper/blob/main/docs/local-servers.md) — ollama, llama.cpp, vLLM, SGLang and LM Studio: what to pass, turning thinking off, what fits a small GPU, and what each one ignores or refuses
- [docs/reasoning.md](https://github.com/zhulinchng/jevper/blob/main/docs/reasoning.md) — native vs two-step reasoning, traces, encrypted content
- [docs/few-shot.md](https://github.com/zhulinchng/jevper/blob/main/docs/few-shot.md) — example levels, precedence, rendering, structured examples
- [docs/internals.md](https://github.com/zhulinchng/jevper/blob/main/docs/internals.md) — module map, call flow, concurrency, retries, testing
- [docs/mlflow.md](https://github.com/zhulinchng/jevper/blob/main/docs/mlflow.md) — MLflow 3.16.1: autolog tracing of jevper's calls, hosting jevper as a model, the AI Gateway, `mlflow.genai.evaluate`

## License

Apache-2.0 — see [LICENSE](https://github.com/zhulinchng/jevper/blob/main/LICENSE).
