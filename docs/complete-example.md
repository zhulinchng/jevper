# Complete example

*Independent implementation of the documented System One wire format — not affiliated with TypeSafe.*

This is the one page that sets **every public option at once**: the program below is
[`examples/complete_call.py`](https://github.com/zhulinchng/jevper/blob/main/examples/complete_call.py),
embedded here as written rather than copied, and the test suite runs it against a local stub server —
so if a new option is added to the library without appearing in this program, the suite fails.

[Getting started](getting-started.md) is still the page to read first: it is one question, one call,
one answer. This page is for when you want to see the whole surface and what each part of it does.

## Before you run it

```sh
pip install jevper openai     # openai is the client used here; anthropic works the same way
```

The program reads three environment variables, so no secret is ever written down:

| Variable | Default | Meaning |
| --- | --- | --- |
| `JEVPER_BASE_URL` | `https://api.openai.com/v1` | Where the provider client points. A local server's URL carries the `/v1` |
| `JEVPER_API_KEY` | `replace-me` | The SDK's key. Local servers ignore it; the SDK still wants one |
| `JEVPER_MODEL` | `gpt-5.6-terra` | The model id sent with every call |

```sh
JEVPER_API_KEY=sk-... JEVPER_MODEL=gpt-5.6-terra python examples/complete_call.py
JEVPER_API_KEY=local JEVPER_BASE_URL=http://127.0.0.1:11434/v1 JEVPER_MODEL=qwen3.5:9b \
    python examples/complete_call.py
```

Two choices in the program are deliberate, and neither is the default. `method="structured"` is
pinned so the program answers the same way on a server with logprobs and one without, and
`api="chat_completions"` is pinned so the request fields named below are the ones that go on the
wire. `method="auto"` and `api="auto"` — the defaults — try to find the best surface the provider
has, and the tables further down say what each one resolves to.

--8<-- "examples/complete_call.py"

It prints one line per answer, the aggregated `usage`, what the call resolved to, and the whole
response as the Jev wire shape. Every comment in it says what an option *does*, not what it is
called; the reference is where names are defined.

## The output budget, per surface

jevper has no `max_tokens` parameter: the three surfaces do not share a name for it, so the field
belongs to the provider and travels in `extra_body`. What each one calls it, confirmed against the
current SDKs and each server measured in [local-servers.md](local-servers.md):

| Surface | Field in `extra_body` | Notes |
| --- | --- | --- |
| `chat_completions` | `{"max_completion_tokens": 2048}` | OpenAI's current name, and it counts reasoning tokens; `max_tokens` is the older field, still what most local servers take, and refused by OpenAI's reasoning models |
| `responses` | `{"max_output_tokens": 2048}` | Also counts reasoning tokens; this route refuses a `max_tokens` it does not know |
| `messages` | `{"max_tokens": 2048}` | Required by the API. jevper sends 1024 when you do not set one, or 1024 plus your `budget_tokens`, because Anthropic refuses a thinking budget that is not strictly below `max_tokens` |

The budget is a ceiling, not a target: a distribution is a label or a small object, so 2048 is room
for a reasoning trace to finish. When a provider runs out of it, jevper raises
`IncompleteAnswerError` naming the surface's own field, rather than reporting a missing answer.

## Which option reaches which surface

Not every option applies to every combination, and jevper says so by leaving a field out rather
than sending one the surface does not have:

| Option | `chat_completions` | `responses` | `messages` |
| --- | --- | --- | --- |
| `method="logprobs"`, `"grammar"` | `logprobs=true`, `top_logprobs`, plus `grammar` in the body | `top_logprobs` and `include=["message.output_text.logprobs"]` | refused: the API returns no logprobs |
| `method="structured"`, `"discrete"` | `response_format` with a strict schema | `text.format` with the same schema | `output_config.format` through the body |
| `structured_outputs=False` | `response_format={"type": "json_object"}`, schema in the prompt | `text={"format": {"type": "json_object"}}` | schema in the prompt |
| `reasoning` `effort` | `reasoning_effort` | `reasoning.effort` | no equivalent field; not sent |
| `reasoning` `summary`, `context` | no equivalent field; not sent | `reasoning.summary`, `reasoning.context` | not sent |
| `reasoning` `budget_tokens` | not sent | not sent | `thinking={"type": "enabled", "budget_tokens": n}` |
| `temperature` | typed field | typed field | body, and left out when a thinking budget is on |
| `top_logprobs` | typed field | typed field, with the `include` entry | not sent |
| `examples` | chat turns before the question block | same | same |
| `prompt_cache_key` | request body | request body | the API has no such field, so the option is not sent — an `extra_body` key of that name still would be |
| `extra_headers` | every request | every request | every request |
| `max_concurrency`, `n_retry_malformed`, `retry` | client-side; no wire field | client-side | client-side |

Two `extra_body` keys are refused before any request: `model`, which would disagree with the model
id jevper keys its cache and its capability memory on, and a truthy `stream`, because jevper reads
the answer from one whole response. A key named in `extra_body` wins over the typed field for the
same name, so naming `response_format`, `text` or `output_config` there also moves the schema into
the prompt.

## Switching surface

The program above is Chat Completions. The other two are three lines each — and the client changes
with the surface, because the Messages API is Anthropic's:

```python
# Responses: the budget field is max_output_tokens, and effort/summary travel natively
with SystemOneClient(provider, model=MODEL, method="structured", api="responses",
                     reasoning=ReasoningConfig(mode="native", effort="low", summary="auto"),
                     extra_body={"max_output_tokens": 2048}) as client:
    ...

# Messages: the anthropic SDK, the budget field is max_tokens, and only the thinking budget applies
from anthropic import Anthropic

provider = Anthropic(base_url=BASE_URL, api_key=API_KEY)
with SystemOneClient(provider, model=MODEL, method="structured", api="messages",
                     reasoning=ReasoningConfig(mode="native", budget_tokens=1024),
                     extra_body={"max_tokens": 2048}) as client:
    ...
```

`method="logprobs"` and `"grammar"` cannot be pinned to `messages` — the API has no logprobs — and
`grammar` needs a Chat Completions server that accepts a GBNF `grammar`. `api="auto"` needs no
choice: it prefers `responses`, falls back to `chat_completions`, then `messages`, and remembers a
route that answered 404. What each server implements is measured per server in
[local-servers.md](local-servers.md).

## Async is a delta, not a second program

Same constructor, same `system_one`, one `await`. The async client holds an asyncio semaphore
instead of a thread pool, and `aclose()` is a no-op because it owns nothing to close:

```python
from anthropic import AsyncAnthropic
from jevper import AsyncSystemOneClient

async with AsyncSystemOneClient(AsyncAnthropic(base_url=BASE_URL, api_key=API_KEY),
                                model=MODEL) as client:
    response = await client.system_one(state=state, questions=QUESTIONS)
```

The provider client is still yours to close, on both facades: jevper never closes the object it was
handed. `close()` shuts down jevper's own thread pool and nothing else.

## A client that is not an SDK

jevper imports neither `openai` nor `anthropic`. It calls whatever object it is handed, reading each
field as an attribute *or* a mapping key, so the whole structural contract is small: one
`chat.completions.create(**kwargs)` (or `responses.create`, or `messages.create`) returning
`choices[0].message.content`, `choices[0].finish_reason` and the `usage` counts. `api="auto"`
follows what the object exposes, so a chat-only object is never asked for a Responses route.

--8<-- "examples/duck_client.py"

The next level up is the same contract with the parts a real client adds — streaming, retries, auth,
connection pooling — which is what both official SDKs give you, and what
[`examples/incident-triage`](https://github.com/zhulinchng/jevper/tree/main/examples/incident-triage)
is built on: a support-ticket triage service that installs jevper as a dependency, uses nothing
but its public API, and runs against five local servers.

## Where each option is defined

| What | Where |
| --- | --- |
| Every constructor option, with defaults | [API reference](api.md#systemoneclient) |
| Every per-call option | [API reference](api.md#system_one) |
| `RetryPolicy` | [API reference](api.md#retrypolicy) |
| `ReasoningConfig` and mode resolution | [Reasoning](reasoning.md), [API reference](api.md#reasoningconfig) |
| The `debug` record and its conditional keys | [API reference](api.md#debug) |
| Every error and what raises it | [Troubleshooting](troubleshooting.md), [API reference](api.md#errors) |
| Choosing a method or a surface | [Methods](methods.md) |
| Few-shot examples | [Few-shot examples](few-shot.md) |
| Per-server behaviour, measured | [Local servers](local-servers.md) |
| Tracing and hosting | [MLflow](mlflow.md) |
