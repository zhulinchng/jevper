# Troubleshooting

*Independent implementation of the documented System One wire format — not affiliated with TypeSafe.*

What went wrong, what it cost, and what to change. Errors raised by jevper inherit from
`JevperError`, so one `except` around a call covers those errors; SDK construction errors such as
`TypeError` do not. The table below is for when you want the reason. The
[API reference](api.md#errors) defines each error precisely, and
[local-servers.md](local-servers.md) measures what each server does with the fields jevper sends.

[Architecture](architecture.md#when-a-server-refuses-a-field) explains the decisions behind the
refusals below: what jevper drops, in what order, and what it remembers.

## Start with the debug record

For provider attempts, the request that produced the failure is in the response's `debug` or on the
error's `attempts` — the request kwargs sent (credential headers redacted), the provider's response
as data, the error text, and the parsed readout. Local validation and client-capability errors can
have neither:

```python
from jevper import JevperError, SystemOneClient

client = SystemOneClient(provider, model=MODEL)
try:
    response = client.system_one(state=state, questions=questions)
    print(response.debug["method"], response.debug["api"], response.debug["llm_attempts"][-1])
except JevperError as error:
    attempts = getattr(error, "attempts", None) or []
    print(type(error).__name__, error)
    if attempts:
        print(attempts[-1]["request"], attempts[-1]["error"])
```

`method`, `api`, `reasoning_mode`, `llm_attempts`, `retry_reasons`, `probability_errors`,
`original_probabilities` and `labels_missing` are always there; the rest are conditional and read
with `.get()`. A server that has refused a capability field is remembered per model and surface, and
`debug["server_limits"]` is where you see limits for the surface the call ended on; its absence means
that surface has no recorded non-default limits, not that no other surface or model has refused a field.

## Refused before any request

These cost nothing: jevper validates the whole call — questions, examples, `state`, options — before
the first provider request, so a rubric mistake is a `JevperError` with no bill attached.

| What you see | What it means | What to change |
| --- | --- | --- |
| `InvalidQuestionError` naming a field | The question or one of its examples is invalid — a duplicate probability key, an extra `weight` field (the question models have no `weight`), an example whose answer is not one of the options, or a selected `examples` value that is not a sequence of `Example` objects. The public `examples` option may be a sequence or a mapping keyed by question id | Fix the rubric. It fails the same way whether the question was built in Python or handed over as a mapping, so one `except JevperError` covers both |
| `InvalidQuestionError` naming 26 labels | `method="logprobs"`/`"grammar"` with more than 26 options: a single-token readout cannot tell `AA` from `AB` | `method="structured"`, which handles up to 255 options |
| `UnsupportedMethodError` | `grammar` on the Responses surface, or `logprobs`/`grammar` pinned to Messages, which returns no logprobs | `method="structured"`, or `api="chat_completions"` |
| `JevperError` about `state` | An empty chat-turn list, a wrong turn shape, an unknown role, or non-`str` content. Other JSON values, including non-string documents, are accepted and rendered as a document | The accepted forms are in [api.md](api.md#system_one) |
| `JevperError` about encodable text | An unpaired surrogate, or content `json` cannot write — in the state, a criterion, the cache key, `extra_body` or the model id | The field is named; replace the text |
| `JevperError` "content is nested too deeply" | The state nests past what the interpreter's JSON encoder can write, which differs by Python version | Flatten the state, or classify it in two calls |
| `JevperError` about `max_tokens` vs `budget_tokens` | Your `max_tokens` cannot hold the thinking budget, and Anthropic requires the budget to be strictly below it | Raise `extra_body={"max_tokens": n}`, or lower `budget_tokens` |
| `ClientCapabilityError` | The client object has no `responses`/`chat.completions`/`messages` for the surface you pinned | Unpin `api="auto"`, or hand in a client that has the surface. The async facade refuses a blocking client, and the blocking one refuses an async client, naming the class to use |
| `TypeError: Invalid http_client argument` | `anthropic` 1.8 is built on `httpx2` and refuses an `httpx.Client` (the `openai` 3.19 SDK accepts one) | Build the client from the module the SDK expects — `httpx2.Client()` for `anthropic`, or let the SDK make its own by not passing one |

## The answer came back unusable

| What you see | What it means | What to change |
| --- | --- | --- |
| `LabelReadoutError` with `method="logprobs"` | The first answer token was not a label, or there were no logprobs to compare it against: a reasoning model thinking first, a server that does not return logprobs, or `top_logprobs` so small the labels are not in the alternatives (ollama returned five alternatives, none of them a label) | `method="auto"` falls back to JSON; or `method="structured"` to ask for the distribution directly; `top_logprobs=20`; and `reasoning=None` so the first token is the answer |
| `MalformedAnswerError` | The JSON answer had a wrong key, an extra key, two objects, or a number out of range, after the corrective retries | `n_retry_malformed=2`, `temperature=0.0`, and `structured_outputs=True`. The correction turn describes the failure and asks for a conforming JSON object; it does not quote the previous reply |
| `IncompleteAnswerError` naming an output budget | The provider ran out of output tokens — typically a reasoning model spending the budget thinking | Raise the surface's own field in `extra_body` (see the [table](complete-example.md#the-output-budget-per-surface)); on Messages, a 4096-token thinking budget makes jevper send `max_tokens: 5120` by default, but whether that finishes a particular 9B trace is provider- and deployment-dependent |
| `IncompleteAnswerError` naming `model_context_window_exceeded` | The request itself is longer than the model's context — raising the output budget makes it worse | Shorten the state, the demonstrations, or the question block; or use a larger context |
| `ModelRefusalError` | The model declined, or a safety filter withheld the content | Terminal by design: a refusal is complete, not broken. Rephrase, or route the content elsewhere |

## The provider or the route failed

| What you see | What it means | What to change |
| --- | --- | --- |
| `ProviderError` with 404 under `api="auto"` | The server has no route for that surface, so jevper moved to the next one and remembered the verdict | Nothing; check `debug["api"]` to see where the answer came from. A 404 that names the *model* is the model, and `api=` pinned is never overridden |
| `ProviderError` with 401/403 | The key was refused | The SDK's own auth, not jevper's. `extra_headers` naming `authorization` replaces the SDK's `Authorization` header — spell it the way the client does |
| `ProviderError` with 400 naming a field jevper knows | The server does not implement that capability — structured outputs, reasoning, the Responses `include` list, the cache key, the Messages `thinking` field | Automatic: the field is dropped, the call re-asked, and the limit remembered per model and surface. `debug["server_limits"]` lists them. A server that refuses a field's *value* — `budget_tokens: must be at least 1024` — gets its own error, because a bad number is not a missing field |
| `ProviderError` after retries | A transient failure that outlived the retry budget: 408, 409, 429, any 5xx, or a connection/timeout error | See the retry table below; `RetryPolicy(n_retries=…, base_delay=…, max_delay=…)` is yours to set, and `respect_retry_after=True` honours the server's own `Retry-After` |
| `ProviderError` on an event stream | The server streamed although nothing asked it to | A provider's choice, not a client capability: catch `ProviderError`. Do not put a truthy `stream` in `extra_body` — jevper refuses that locally, while `stream: false` is allowed and forwarded |
| `ProviderError` with 404 and an HTML body | The `base_url` you passed already ends in `/systemone`, so the path was doubled (`/v1/systemone/systemone`) and a front-end proxy answered instead of the API. jevper appends the path itself — the base URL is the host | Drop `/systemone` from `base_url`. The same variable is often written as the full endpoint for `curl`, which posts to whatever path it is given; the two are not interchangeable |
| `ProviderError` with 400 and `"Model is unavailable"` | The model name is not one the deployment serves, and a gateway may report that as a 400 carrying `{"error": {"type": "server_error", ...}}` — a `server_error` at a 400 status | The model name, not jevper. Where the deployment's own listing omits the decision models, the name cannot be discovered through `list_models()`; see [Jev comparison](jev-comparison.md#jevper-can-call-this-endpoint) |

## Three retry loops, and which one spent what

| Loop | Counted by | What it retries | Configure with |
| --- | --- | --- | --- |
| Corrective | successful corrective results in `usage.n_calls` | An answer that cannot be read: the client describes the failure and asks for a conforming answer; it does not quote the previous reply | `n_retry_malformed` (default 1) |
| Transient | `usage.n_retries` | 408, 409, 429, any 5xx, connection and timeout errors, with exponential backoff and `Retry-After` when sent | `retry=RetryPolicy(...)` (default 2 retries, 0.5 s base, 8 s cap) |
| The SDK's own | — | The same transient set, twice, before jevper sees anything | Already off: jevper calls the SDK through a copy with `max_retries=0`, so `usage.n_retries` is the true count. Your own client keeps its setting |

Timeout is the SDK's, not jevper's: `OpenAI(..., timeout=60.0)` bounds each request the SDK makes.
There is no jevper option for it, because the client object is yours.

## How many calls will this cost?

One question, one call, by default — with three ways that changes. The table counts provider
requests visible in `debug["llm_attempts"]`, not `usage.n_calls`; that counter includes successful
provider results only:

| Configuration | Provider requests per question |
| --- | --- |
| `method="structured"`, reasoning off | 1 |
| `method="auto"` on a server that returns no logprobs | 2: one logprob attempt, then the JSON answer — measured against a stub that answers without logprobs |
| `reasoning=ReasoningConfig(mode="two_step")` | 2: an analysis call, then the answer call with the analysis replayed |
| A malformed answer, with `n_retry_malformed=1` | 2: the original, then the correction |

`usage.n_calls` is 1 for the auto fallback because only the successful JSON result is counted. It
does count successful analysis and corrective results when those are successful.

Questions are independent, so a rubric of five questions is five of these in parallel, capped by
`max_concurrency`.

## When a local server is the problem

Per-server behaviour is measured rather than guessed, in [local-servers.md](local-servers.md). The
findings that most often explain a surprise:

- **Thinking cannot be switched off on every server.** ollama's Messages route ignores the
  OpenAI-style thinking-off field, so a thinking model is paid for out of the output budget there;
  llama.cpp and vLLM expose their own switch. Check the per-server table before pinning `api`.
- **LM Studio's Responses route ignores `text.format`**, so `structured_outputs=True` is accepted
  and dropped on that surface; the schema in the prompt still carries the answer.
- **SGLang's Messages route requires a thinking budget of at least 1024** and reports a bad value as
  such — the error names the number, not the field.
- **`cached_tokens` is `None` until you ask for it**: vLLM needs `--enable-prompt-tokens-details` and
  SGLang `--enable-cache-report` for its Chat Completions route. `None` is "the provider said
  nothing", which is not the same as a reported `0`.
- **A conversation that ends on the assistant's turn** is refused by the llama.cpp engines
  (`400 Failed to initialize samplers`). jevper moves the question after such a state; if you build
  the prompt yourself, do the same.

## In a bug report

The model id, the resolved `method` and `api`, the `debug` record with credential headers already
redacted, and the server with its version and launch flags. That is enough to reproduce a
classification difference without a key or a private state.
