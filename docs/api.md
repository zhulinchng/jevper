# API reference

*Independent implementation of the documented System One wire format — not affiliated with TypeSafe.*

Everything below is importable from the package root:

```python
from jevper import SystemOneClient, AsyncSystemOneClient, Choice, Noul, Score, Example, ReasoningConfig
```

`__all__` also contains `Answer`, `Api`, `ChoiceAnswer`, `Example`, `Examples`, `JSONContent`, `Method`,
`MethodSelection`, `NoulAnswer`, `NoulCriteria`, `ProviderError`, `Question`, `Readout`,
`ReasoningContentPart`, `ReasoningSummaryPart`, `ReasoningTextPart`, `RetryPolicy`, `ScoreAnswer`,
`SystemOneResponse`, `Usage`, `reasoning_text`, the seven other error classes, and `__version__`.

## `SystemOneClient`

```python
SystemOneClient(client, *, model, method="auto", api="auto", reasoning=None, examples=(),
                structured_outputs=True, normalize_probabilities=True, top_logprobs=20,
                max_concurrency=8, n_retry_malformed=1, retry=None, temperature=None,
                extra_body=None, extra_headers=None, prompt_cache_key=None,
                native=False, noul_requires_question=True)
```

Only `client` is positional; every option is keyword-only, and an unknown keyword is a `TypeError`
from the constructor rather than an option.

`client` is any object exposing `responses.create`, `chat.completions.create` and/or `messages.create` (the
Anthropic SDK's, pointed at any server that implements the Messages API); it stays owned by the caller
(`close()` only shuts down jevper's own thread pool). With `api="systemone"` the object needs `post`
instead, which an `openai` client has: `OpenAI(base_url="https://api.typesafe.ai/v1", api_key=…)`
reaches the Jev service, and jevper posts the Jev request body itself. See
[jev-comparison.md](jev-comparison.md).

The `responses` surface speaks both OpenAI's Responses API and the [OpenResponses](https://www.openresponses.org)
specification, and both are served at the same `/v1/responses` path. The OpenResponses site lists LM Studio
among the ecosystem's implementers, and [vLLM says its route "aligns with" the spec](https://github.com/vllm-project/vllm/issues/32850)
while extending it with fields of its own (`top_k`, `request_id`, `priority`); the others serve
OpenAI's own dialect at that path.
So the request is the portable form both accept: every input turn
carries the item `type` the spec's union requires, with its content as a plain string, which that union
also allows. The reader answers in either dialect — OpenAI's own shapes and the spec's content parts,
reasoning part types, per-item statuses and message phases alike. What each server does with the two is
measured per server in [local-servers.md](local-servers.md).

Two rules about the client object itself. A client with a retry loop of its own — the official SDKs retry
twice by default — is used through a copy with that loop off, so `RetryPolicy` is the only retry loop and
`usage.n_retries` counts every request the provider saw; the caller's own client keeps its setting. And the
two facades do not mix: `SystemOneClient` refuses an asynchronous client and `AsyncSystemOneClient`
refuses a blocking one, before any request, naming the class to use instead.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `model` | required | Model id sent with every call; overridable per `system_one` call |
| `method` | `"auto"` | `"auto"`, `"logprobs"`, `"grammar"`, `"structured"` or `"discrete"`; `"auto"` resolves per model and surface — see [methods.md](methods.md#auto) |
| `api` | `"auto"` | `"auto"`, `"chat_completions"`, `"responses"` or `"messages"` (the Anthropic Messages API); `"auto"` prefers `responses`, then `chat_completions`, then `messages`, falling back when the server answers 404 for a route — see [methods.md](methods.md#auto) |
| `reasoning` | `None` | A `ReasoningConfig`; `None` disables reasoning entirely. `mode="auto"` resolves to `native` on the Responses surface and on Messages when `budget_tokens` is set, and to `two_step` everywhere else; `budget_tokens` is what the Messages surface sends as its `thinking` field — see [reasoning.md](reasoning.md) |
| `examples` | `()` | Default few-shot examples: a sequence for all questions, or a mapping keyed by question id |
| `structured_outputs` | `True` | Send a strict `json_schema` response format — `response_format` on Chat Completions, `text.format` on Responses, and the Messages API's own `output_config.format`; `False` leaves the schema in the prompt instead (`{"type": "json_object"}` on the OpenAI surfaces). A server that refuses the strict schema gets the same fallback automatically |
| `normalize_probabilities` | `True` | Rescale `structured` distributions that are off by more than `1e-6`; `False` returns the model's numbers verbatim |
| `top_logprobs` | `20` | Requested alternatives for `logprobs`/`grammar`; must be in `[0, 20]`, and at least 2 when the method is pinned to `logprobs`/`grammar` — the sampled token alone is not a distribution |
| `max_concurrency` | `8` | Questions in flight at once (thread pool, or asyncio semaphore) |
| `n_retry_malformed` | `1` | Corrective retries when an answer cannot be read |
| `retry` | `None` | Transient-failure retries; `RetryPolicy()` (2 retries, 0.5s base, 8s cap) when unset |
| `temperature` | `None` | Not sent unless set. `0.0` is recommended for `structured`/`discrete`; `logprobs` needs no setting. On Messages it travels in the request body (the current `anthropic` SDK has no `temperature` parameter) and is left out whenever a `thinking` budget is sent; the newest Claude models refuse *any* non-default value with a `400`, thinking or not, so set it only for a provider that still reads it |
| `prompt_cache_key` | `None` | The provider's cache-routing key. Unset, jevper derives one per question from the parts of the prompt that do not change between calls, so a rubric's requests are routed together; set it to group (or account for) requests your own way. It travels in the request *body* on both OpenAI surfaces: the field belongs to the API, and only newer SDK releases type it — `openai` grew `prompt_cache_key` on Chat Completions and Responses after 1.92, the floor this package declares — while the body is identical either way, since the SDK merges `extra_body` into the same JSON. A keyword a supported release does not type is a `TypeError` from inside the call, with the caller's remedy nowhere in the message |
| `extra_body` | `None` | Merged into every request body (the grammar field is merged here too). A key it names is the value that reaches the wire — the SDK merges `extra_body` *after* the typed parameters — so jevper leaves that field alone rather than sending a typed value the caller's own key would override. `response_format`/`text`/`output_config` named here therefore also puts the JSON Schema in the prompt, since no schema of jevper's is in the request. On the Messages surface, `max_tokens` comes from here — jevper always sends one there, defaulting to `DEFAULT_MAX_TOKENS` (1024), or 1024 plus `ReasoningConfig(budget_tokens=…)`; a `max_tokens` that cannot hold the thinking budget asked for raises `JevperError` locally, naming both numbers, rather than earning the API's own refusal. Every other key travels on all three surfaces, whether or not a temperature was set |
| `extra_headers` | `None` | Sent with every request. A name spelled differently from the client's own (`authorization` against the SDK's `Authorization`) is renamed to the client's spelling, so it replaces the default header instead of joining it on the wire. Names and values must be what a header can carry — an HTTP token name and a printable-ASCII value, horizontal tabs allowed — and a name or value that is not is refused here, not by the SDK's encoder from inside the request |
| `native` | `False` | Post System One requests to [Ollaya](local-servers.md#ollaya-the-decision-server)'s native `POST /api/decide` rather than the TypeSafe `POST /v1/systemone`, and return a `NativeSystemOneResponse` carrying that endpoint's report. A caller's choice rather than one jevper probes for: both routes are one server on one port, so nothing a call returns says which was meant. The two sit at different depths, so this needs a client whose `base_url` is the server root — and `api="systemone"`, since `api="auto"` never selects that surface; a client left on `auto` with `native=True` is refused before any request rather than posting to a prompt route |
| `noul_requires_question` | `True` | Refuse a noul carrying neither instructions nor criteria, as the hosted Jev service does with a 400. `False` sends one anyway, for a server that answers it by reading the question id in their place — [Ollaya](local-servers.md#ollaya-the-decision-server) and [CLM](local-servers.md#clm-the-contrastive-decision-model) both do, measured. Unrelated to `api`, and ignored by a prompt call |

The [OpenResponses schema](https://github.com/openresponses/openresponses/blob/main/public/openapi/openapi.json)
documents a 64-character maximum for `prompt_cache_key`; OpenAI's own API reference states no length
limit, so on that surface a longer key is the server's to refuse. OpenRouter does not refuse one
(measured 2026-09-25: a 72-character key answered normally).

Two `extra_body` keys are refused outright, before any request: `model` (the wire model would disagree with
the cache key, the learned verdicts and the public response — pass `model=` to the constructor or to
`system_one()` instead) and a truthy `stream` (jevper reads the answer from one non-streaming response, and
a server that streamed anyway would hand the SDK an event stream no reader here understands). A
`logprobs: false` here switches jevper's own Responses logprob fields off — `top_logprobs` and the
`include` entry — the way it does on Chat Completions; the key itself still reaches the body, because
the Responses API has no `logprobs` field.

Constructor validation is eager: an unknown `method`/`api`, `top_logprobs` outside `[0, 20]` or below 2 with a
pinned `logprobs`/`grammar`, `max_concurrency < 1`, a negative `n_retry_malformed`, a non-`ReasoningConfig`
`reasoning`, a `retry` that is not a `RetryPolicy` or has a negative or non-finite field, a `prompt_cache_key`
that is not a non-empty string of at most `MAX_PROMPT_CACHE_KEY` characters, and an `extra_body` or
`extra_headers` that is not a mapping all raise `JevperError`.

Text that cannot be encoded as UTF-8 is refused the same way, with the field named: an unpaired surrogate —
which JSON can carry as a `\udXXX` escape, and which Python's `json` module decodes into a string no
encoder accepts — in the `state`, a question's instructions or option keys, `prompt_cache_key`,
`extra_body` or the model id raises before any request, rather than from inside the SDK's serializer where
the only description on offer would be a provider failure. A header value is held to the stricter rule the
SDKs impose on it: the official clients encode headers as ASCII, so a value that is not printable ASCII (or
a horizontal tab) — a newline, a NUL, a `é`, a surrogate — is refused with the header named, and so is a name
that is not an HTTP token. A CRLF in a value is the one that could otherwise become a second header on a
client that forwards it.

The default `prompt_cache_key` is a stable digest: the model, the method, the few-shot examples and the
question block, and nothing about the state. The same rubric therefore sends the same key on every call —
which is the point, and also means the key is a fingerprint of that material at the provider. Pass your own
key when the examples or the question text are sensitive and that link is not wanted.

### `system_one`

```python
system_one(*, state, questions, examples=(), model=None, method=None, api=None, reasoning=None,
           temperature=None, prompt_cache_key=None, extras=(), keep_alive=None)
    -> SystemOneResponse
```

| Parameter | Default | Meaning |
| --- | --- | --- |
| `state` | required | `str`, a chat-message list, `{"messages": [...]}`, or any JSON value (see below) |
| `questions` | required | Mapping of question id to a `Noul`/`Choice`/`Score` or an equivalent raw mapping; at least one |
| `examples` | `()` | Per-call examples: sequence for all questions, or mapping keyed by question id |
| `model` | `None` | Overrides the constructor model |
| `method` | `None` | Overrides the constructor method |
| `api` | `None` | Overrides the constructor api: `auto`, `chat_completions`, `responses`, `messages` or `systemone` |
| `reasoning` | `None` | Overrides the constructor reasoning |
| `temperature` | `None` | Overrides the constructor temperature |
| `prompt_cache_key` | `None` | Overrides the constructor cache key; `None` keeps the constructor's (or the derived one) |
| `extras` | `()` | Ollaya's own model-family outputs for its native endpoint, passed through as given. The closed set today is `("laya",)`, which adds a `laya` object to every answer; the set is the endpoint's to widen, so an unknown value is its refusal to report rather than one made here. A bare `str` is refused before any request — it type-checks as a sequence of them and would otherwise be sent one character per name |
| `keep_alive` | `None` | Ollama's model lifecycle control, in that server's own units; `0` unloads the model. `None` sends no field, so the service's own `OLLAYA_KEEP_ALIVE` applies |

On `api="systemone"` every question goes in one request — that is the shape the service is built to be
asked — and four of the options above have no field on that wire, so they are refused by name in one
error before anything is sent: a `method` other than `auto`, a `reasoning`, `examples`, a
`temperature` and a `prompt_cache_key`. A noul carrying neither instructions nor criteria is refused
there too, because the hosted service answers 400 for one; `noul_requires_question=False` lifts that,
for a server that answers one by reading the question id in its place. The service's own `score`,
`confidence`, `choice` and `legend` are read as they arrived rather than recomputed, and the request's
usage is counted once for the whole call however many questions it answered.

`extras` and `keep_alive` belong to `native=True` alone, and are refused on the TypeSafe route by name
before anything is sent, since the fix there is the other route rather than dropping the field. On a
prompt surface (`chat_completions`, `responses`, `messages`) neither is refused — that route has no such
field, so both are ignored, the same treatment `top_logprobs` gets on the System One wire.

The service's own `score`, `confidence`, `choice`, `legend` and `probabilities` are read as they arrived
rather than recomputed, with one check the type system cannot make: a score's distribution has to carry
exactly its rubric's levels, in both directions, because `answer.probabilities[level]` is how the
documented arithmetic reaches it and a level outside the rubric answers a question that was not asked.

Per-call values win over constructor defaults. Everything is resolved and validated before the first provider
call, so a bad question, an empty `questions` mapping, an unusable `state`, or `grammar` on the Responses
surface costs zero requests. The same holds for a per-call `reasoning` that is not a `ReasoningConfig` and a
per-call `method` of `logprobs`/`grammar` with `top_logprobs` below 2, and for an example whose `answer` or
`probabilities` do not fit its question.

`state` forms:

| `state` | Rendered as |
| --- | --- |
| `"text"` | one `user` turn holding `<document>…</document>` |
| `[{"role": "user", "content": "..."}, ...]` | those turns verbatim (roles `system`, `user`, `assistant`, `developer`; `content` must be `str`) |
| `{"messages": [...]}` | same as above |
| any other JSON value | one `user` turn holding `<document>` around `json.dumps(state, ensure_ascii=False, sort_keys=True, indent=2)` |

A list is read as chat turns when its elements are all dicts — a conversation by intent, so a missing
`content`, an unknown role or a typo in a key is a `JevperError` rather than a quoted blob, and an empty
list is refused for the same reason. A list of anything else (`[1, 2]`, `["a", "b"]`) is not a
conversation and is quoted like any other JSON value. A `system`/`developer` turn inside a chat list is
moved into the leading system message, because no server here accepts one in a later position — and it
is quoted on the way in, since moving it is a position fix and not a statement that the caller's state
is trusted.

A state handed over as one value is quoted, and every angle bracket inside it is written as its JSON
escape, so the document cannot close its own quote and carry on as prompt text: the state is the
content under judgement, and content under judgement is what an attacker would try to steer an
answer with. Every system prompt says so in as many words. A state handed over as chat turns keeps its
roles instead — the turns are already its boundary, and folding them into a document would destroy the
conversation they are.

Whichever form it takes, the state is rendered into the **last** messages of the prompt — after the system
prompt, the few-shot turns and the question block — so that every state classified with one rubric shares the
same prefix and the provider can reuse its cache. A chat-list state's own `system`/`developer` turns are
folded into that system prompt — every server here refuses a `system` turn that is not first, jevper's own
prompt leads, and the content arrives quoted — while the rest of the state stays verbatim and goes last.
See [local-servers.md](local-servers.md#message-order-is-what-decides-the-reuse).

The one exception is a chat-list state whose last turn is the **assistant's**: the question turn then follows
the state instead, so the conversation still ends on a question. Ending it on the assistant's turn is not a
question at all — the llama.cpp engines refuse it outright (`400 Failed to initialize samplers`, from both
ollama and LM Studio) and a server that reads it as a prefill continues that turn rather than answering. The
cost is that this one shape cannot reuse the question block as a cached prefix.

### `list_models()` / `alist_models()`

```python
list_models() -> list[ModelMetadata]
```

The models the deployment offers, from the System One endpoint's `GET /v1/models`, read through the
same client object and parsed as the service's own OpenAPI declares it —
`{"models": [{"name", "description", "release_date"}]}`. A gateway answering that path with a list of
its own is reported as the wrong shape rather than half-read. `ModelMetadata.description` defaults to
`""` and `release_date` to `None` when the service omits them. `alist_models()` is the async twin.
No `api=` is needed: the path is the service's, and the client is the caller's own. It is jevper's
request, so it follows the `retry` policy and arrives as a `ProviderError` carrying the provider's
status, exactly as `system_one` reports the same failure; a client whose `get` is a coroutine (or is
not) is refused before the request, naming the class to use instead. The SDK's own retry loop is
switched off for it, as it is for every other request jevper makes, so one call is one policy rather
than two nested ones.

### `close()` / `aclose()`

`close()` shuts down the internal thread pool and is called by `with SystemOneClient(...) as client:`.
`AsyncSystemOneClient.aclose()` is a no-op — the async client holds no resources — and `async with` calls it.
Neither closes the client you passed in.

## `AsyncSystemOneClient`

Identical constructor and `system_one` signature, with `async def system_one(...)`. The async client uses
`asyncio.Semaphore(max_concurrency)` instead of a thread pool, and has no thread pool to close.

## Questions

```python
Noul(instructions=None, criteria={"true": "the message is a complaint", "false": "it is not"} | None = None,
     examples=())
Choice(instructions=None, criteria={"key": "description" | None}, examples=())
Score(instructions=None, criteria=["level description"], examples=())
```

- `instructions` and each criteria description may be a `str` or any JSON value; non-strings are rendered as
  pretty-printed JSON.
- `criteria` is required for `Choice` and `Score`, optional for `Noul`. `Noul` criteria keys must be `true`
  and/or `false`; anything else raises `InvalidQuestionError`.
- Limits: `Choice` 1–255 options (the Jev API documents the 255 maximum and no minimum), `Score` 2–10
  levels. `logprobs` and `grammar` cap a `Choice` at 26 options, because they read one label token;
  `structured` and `discrete` carry the full range with two-letter labels past `Z`. Unknown fields are
  rejected (`extra="forbid"`).
- `examples` is a tuple of `Example` and is excluded from `model_dump()`, so dumps keep exactly the Jev wire
  keys `{"type", "instructions", "criteria"}`.
- A question is validated where it is built *and* again in `system_one`, and both paths fail the same
  way: building one raises `InvalidQuestionError` — pydantic's own `ValidationError` is reported as the
  library's error, with the offending field named — and a mapping handed to `system_one` is parsed with a
  discriminated union and refused the same way, with the question id in the message. One `except
  JevperError` around a rubric therefore covers a rubric written in Python and one loaded from data.
- A question's own `examples` are checked against that question where the question is built: the answer
  has to resolve to one of its options, and the probabilities have to carry exactly its keys. `examples`
  passed to the constructor or to `system_one` have no question to be checked against until the call
  pairs them with one, so those are checked there — still before any request.

```python
Example(state=..., answer="billing" | "B" | 2 | True, probabilities={"billing": 0.6, ...} | None = None)
```

`answer` may be a label (`"B"`), a `Choice` criteria key, a `Score` level index, or a bool for `Noul`. An
exact criteria key is tried first and a label second, so an option key that is itself a label (`"a"` beside
`"A"`) is read as the key the caller wrote. `probabilities` is only read by `method="structured"` and defaults
to a one-hot distribution over `answer`.

When given, `probabilities` is validated, because the example is replayed into the prompt as the answer the
model is asked to imitate:

- it must carry exactly the option keys of a `Choice`, the level indexes of a `Score`, or a `True`/`False`
  (`"true"`/`"false"`) key for a `Noul` — a wrong key set raises `InvalidQuestionError` naming the example
  index, and so does a `Noul` mapping with no key at all, or two keys that name the same answer
  (`{True: 0.2, "true": 0.8}` for a `Noul`, `{1: 0.9, "1": 0.1}` for a `Score`), because the rendered
  demonstration can carry only one of them;
- values must be finite (refused by `Example` itself) and `>= 0`, and a `Noul` value must be in `[0, 1]`.

A non-finite number or a non-JSON-serializable `state` is refused rather than rendered: `NaN`/`Infinity` are
not valid JSON, so they would put an unparseable example in front of the model.

## Answers

Answer models carry the same field names and JSON keys as `POST /v1/systemone`.

```python
NoulAnswer(type="noul", noul=0.93)
ChoiceAnswer(type="choice", choice="billing", probabilities={"billing": 0.88, "technical": 0.08, "sales": 0.03},
             confidence=0.83)
ScoreAnswer(type="score", score=1.05, legend={0: "Calm", 1: "Frustrated", 2: "Very angry"},
            probabilities={0: 0.0, 1: 0.95, 2: 0.05}, confidence=0.92)
```

- `choice` is the highest-probability option, ties resolved by criteria order; `probabilities` is keyed in
  criteria order.
- `noul` answers carry no `confidence` — the Jev API omits it for `noul`.
- `score` is `Σ i·pᵢ` over zero-based levels, read off the distribution rescaled to sum to one; `legend`
  maps level index to the criteria entry. Both are keyed by that integer index on every surface,
  including the System One one, where the wire spells a JSON object's keys as text. The rescale matters
  only with `normalize_probabilities=False`, where the reported `probabilities` stay the model's own
  numbers: a score carried from an unnormalized distribution would leave the `0..N-1` line the Jev answer
  schema documents. This is the arithmetic the TypeSafe reference adapter uses.
- `noul`, `score` and every `probabilities` value have to be finite. `NaN` and `Infinity` are literals a
  JSON reader accepts and no arithmetic survives, so they arrive as `MalformedAnswerError` rather than as
  numbers a caller would go on to compare.
- `confidence` for `choice` is `(max(p) − 1/n) / (1 − 1/n)`, i.e. the peak probability scaled from uniform
  (0) to certainty (1). For `score` it is `max(0, 1 − MAD(p) / MAD_uniform)`, where `MAD` is the mean absolute
  deviation from the modal level and `MAD_uniform` is that quantity for a uniform distribution. Both are
  computed after normalizing the inputs, with a uniform fallback when the total is zero — the same arithmetic
  as the TypeSafe reference adapter.
- Integer keys (`score`, `legend`, `probabilities`) serialize to JSON object keys as strings.

## `SystemOneResponse`

| Field | Type | Meaning |
| --- | --- | --- |
| `model` | `str` | The model the call asked for. On Ollaya's native endpoint this is the *alias* the caller named — a router's `laya`, not the checkpoint that answered — so one field means one thing on every surface; the checkpoint is `NativeSystemOneResponse.routing.model` |
| `answers` | `dict[str, Answer]` | Keyed by question id, in insertion order |
| `usage` | `Usage` | Aggregated over every provider call and retry |
| `reasoning` | `tuple[ReasoningContentPart, ...]` | Trace, in question order; see [reasoning.md](reasoning.md) |
| `debug` | `dict[str, Any]` | Attempts and normalization notes; see below |

`nouls`, `choices` and `scores` are cached properties returning `answers` filtered by answer type.

### `NativeSystemOneResponse`

Returned instead when the client was built with `native=True`, so the answers and usage above are the
same response carrying Ollaya's report on the one request that answered it. It is a subclass, so code
that reads a `SystemOneResponse` reads this one unchanged.

| Field | Type | Meaning |
| --- | --- | --- |
| `routing` | `Routing` or `None` | A router's choice of checkpoint, or `None` when a model named directly answered |
| `state_truncated` | `bool` | Whether part of the `state` was dropped to fit the model's context |
| `done_reason` | `str` | `"decide"`, or `"load"`/`"unload"` |
| `created_at` | `str` | When the endpoint produced the response, in the service's own timestamp format |
| `total_duration` | `int` or `None` | Request to response, queueing included, in nanoseconds |
| `load_duration` | `int` or `None` | Time spent waiting for the model to load; `0` when it was warm |
| `eval_duration` | `int` or `None` | Tokenization, forward pass and calibration, in nanoseconds |

Durations are the service's own nanoseconds, passed through as sent rather than converted, so a number
read here can be compared with the service's own log. A reported `0` is kept as the measurement it is
and only an absent field reads as `None`.

```python
Routing(router="laya:latest", model="laya:en", route="english", reason="English Latin text")
LayaExtras(confidence=0.615, act_probability=1.0)
```

`route` is the stable key to branch on. `reason` is the router's own prose, which the service documents
as informative and free to change in any release, so it is carried but never parsed. `LayaExtras` is the
`laya` object `extras=["laya"]` puts on every answer: the model's own confidence, a different number
computed by a different head from the `confidence` beside it, and `act_probability`, which is `None`
for a model with no act head. It is `None` on every answer on every other surface.

`routing` is `null` when a model named directly answered, which is the one value that reads as "no
router": any other value that is not an object is a `MalformedAnswerError`, since reading it as absence
would assert something the service never said. An explicit `null` `reason` reads as the empty string, as
an omitted key does, and `act_probability` is bounded to `[0, 1]` like the `confidence` beside it.

```python
Usage(input_tokens=None, output_tokens=None, reasoning_tokens=None, cached_tokens=None, n_calls=0, n_retries=0, latency=0.0)
```

A token count is `None` when any constituent call omitted it (a reported `0` is preserved). `n_calls` counts
analysis passes and corrective retries; `n_retries` counts transient-failure retries only. `latency` is
wall-clock seconds for the whole `system_one` call.

`cached_tokens` is the prompt tokens the provider read from its own prompt cache, read from
`usage.prompt_tokens_details.cached_tokens` on Chat Completions, `usage.input_tokens_details.cached_tokens`
on the Responses surface, and `usage.cache_read_input_tokens` on Messages, which reports no
`*_tokens_details` object at all. It is the one number that says caching worked; a provider that reports
nothing leaves it `None`, which is not the same as a reported `0` — that is a server whose prefix cache is
cold, or off. Not every server reports it at all, and two need a flag to: vLLM's `--enable-prompt-tokens-details`
and SGLang's `--enable-cache-report` for its Chat Completions route. See
[local-servers.md](local-servers.md#prompt-caching).

### `debug`

| Key | Content |
| --- | --- |
| `method` | The call-level method: what `method="auto"` resolved to for this call, or the method you pinned. A question that skips the probe — a `Choice` past 26 options is answered in JSON without ever asking for logprobs — reports its own method in `methods`, so read that when they can differ |
| `methods` | `{question_id: method}` — only for `method="auto"`, since the method is then chosen per question |
| `api` | Surface actually used (`"chat_completions"`, `"responses"` or `"messages"`) |
| `apis` | Only when the questions were answered on more than one surface — which concurrent questions on a shared context can arrange, one worker's 404 moving the client while another is still answering: `{question_id: surface}`. `api` is the surface the call ended on |
| `reasoning_mode` | `"off"`, `"native"` or `"two_step"` — the mode of the surface the call ended on; a call answered on more than one surface also gets `reasoning_modes`, `{question_id: mode}`, derived from each question's own last attempt the way `apis` is |
| `llm_attempts` | One record per provider call: `question_id`, `surface`, `request`, `response`, `error`, `readout` |
| `retry_reasons` | Corrective-retry messages, in order |
| `probability_errors` | `{question_id: abs(sum − 1)}` for `structured` distributions outside `1e-6` |
| `original_probabilities` | The model's raw distribution, only for questions that were rescaled |
| `server_limits` | Only when the server refused a capability field: `structured` (`"schema"`/`"object"`/`"none"`), `output_config` (the Messages API's schema field), `reasoning`, `include`, `cache_key` and `thinking` as it last accepted them |
| `server_limits_by_api` | The same per-surface limits as `server_limits`, for a call whose questions used more than one surface. A limit is remembered per (model, surface): a refusal of a request field is usually about the model that earned it, so a second model on the same client is still sent the field it asked for |
| `labels_missing` | Labels the provider did not report a logprob for, per question |

`method`, `api`, `reasoning_mode`, `llm_attempts`, `retry_reasons`, `probability_errors`,
`original_probabilities` and `labels_missing` are always present, the last three as empty mappings
when nothing applies. The rest are conditional and should be read with `.get()`: `methods` only for
`method="auto"`; `apis`, `server_limits_by_api` and `reasoning_modes` only for a call answered on
more than one surface; and `server_limits` only once a server has refused a capability field, so
its absence means the server has refused nothing yet. `request` holds the
kwargs sent to the provider — for a failed call, the kwargs that were about to be sent, so the shape is the
same either way — `response` holds the provider object dumped with `model_dump(mode="json")` when available,
`error` is a `"Type: message"` string, and `readout` is the parsed readout: `source`, `probabilities` (string
keys), `missing_labels` and `observed_text`. `debug["llm_attempts"][-1]["readout"]` is set for the attempt
that produced the final answer. `ProviderError.attempts` is the same list of records.

The record is debugging evidence, not a data channel, and it is held to three rules so that logging a
response can neither leak a credential nor fail to serialize:

- A header whose name carries a credential — `authorization`, or anything with `token`, `api-key`, `secret`,
  `cookie`, `credential`, `password` or `signature` in it — is recorded as `<redacted>`, name kept, value
  dropped. The wire is untouched: this is the record, not the request. The same applies to
  `ProviderError.attempts`, and the client's own auth headers are never recorded at all when the caller
  passed no `extra_headers`.
- Every string is escaped to printable UTF-8 (a lone surrogate is shown as the escape the wire carried) and
  bounded to 64 KiB, with the length it had noted; mappings are walked 24 levels deep and anything deeper
  is replaced by a marker. A provider object that cannot be dumped at all is recorded as
  `{"undumpable": "<ChatCompletion is not data jevper can keep for debug>"}` — the class name is in the
  text — rather than kept as the object itself, which
  `model_dump_json()` would then have to serialize.
- Provider text quoted in an error is bounded the same way, and the caller's own credential values are
  removed from it: an `error` carried in a `200`, a status failure's detail and an exception's own message all
  pass through one formatter — and a gateway that quotes the key it rejected does not get to put it in every
  message, attempt record and log line that quotes one. A value shorter than eight characters is left alone,
  because replacing `1` or `test` everywhere would corrupt every message that contains those letters.

An event stream where a whole response belongs is a `ProviderError` on all three surfaces, whether or not it
carries a failure frame — on Chat Completions that replaces the `ClientCapabilityError` of 0.7.0 and on Messages
the `MalformedAnswerError`, because a server that streamed when nothing asked it to is a provider's choice
rather than a capability the client lacks. Catch `JevperError` (or `ProviderError`) for it, not those two.

An interrupt is answered at once. A `KeyboardInterrupt` or `SystemExit` that reaches one question of a batch
cancels the questions still queued for it and is raised without waiting for the ones already running on a
worker thread, which cannot be cancelled and may still finish; `close()` joins them. An ordinary per-question
failure is the opposite: every question runs, and the first failure in insertion order is the one raised.

## `RetryPolicy`

```python
RetryPolicy(n_retries=2, base_delay=0.5, max_delay=8.0, respect_retry_after=True)
```

Applies per provider call. A failure is transient when its HTTP status is 408, 409, 429 or any 5xx — the
set both official SDKs retry, read as an int, an `http.HTTPStatus` or a digit string, from the exception or
from `exc.response.status_code` (which is also read when the exception's own attribute is missing or
unreadable, and whether that response is an object or a plain mapping) — unless the response carries
`x-should-retry: false`, which outranks the status in OpenAI's and Anthropic's SDKs and here too, the way
`x-should-retry: true` makes a 400 worth repeating. It is also transient when the exception's class, or any
class in its MRO, is one of the transport and timeout types the SDKs and the standard library raise
(`TransportError`, `ConnectError`, `ReadError`, `RemoteProtocolError`, `APIConnectionError`,
`APITimeoutError`, `URLError`, …) or a builtin `TimeoutError`/`ConnectionError`. Those names are matched
whole, so a caller's own class named after one of them is a programming error and is not repeated, and a
client-side `httpx.LocalProtocolError` is not retried either.

The wait is whatever the provider asked for when it said: a `Retry-After` — delta-seconds, which are read
as the `1*DIGIT` the HTTP grammar defines, or an HTTP date — or the millisecond `retry-after-ms` some
providers send instead replaces the computed backoff, which is what the TypeSafe clients do by default:
coming back sooner than a rate limit asked is one way to extend it. Header names are matched
case-insensitively, as HTTP requires, and headers may be a mapping, an `httpx.Headers`, or a sequence of
`(name, value)` pairs — a hand-rolled client has no `.get` to ask — so a client that hands its exceptions
a plain `{"RETRY-AFTER": "120"}` dict, or `[("Retry-After", "120")]`, is heard the same as one that carries
real header objects, and a date already past means come back now, a wait of zero. `max_delay` caps jevper's
own curve, not the server's instruction: a header asking for hours is waited out in hours, up to
`jevper.client.MAX_RETRY_AFTER` (24 hours — longer than any provider asks for, including a gateway that
means "come back when the quota resets", and short enough that a header nobody sane sends cannot park a
call for a century). Past the ceiling the header is not an instruction any client should carry out, so
the curve answers. `respect_retry_after=False` goes back to the curve alone, and `n_retries=0` fails on the
first rate-limited response. A header jevper cannot read — not delta-seconds, not a date, not a number
within the grammar — is the backoff's business, never a reason to skip the wait. Only an exception the SDK
raised for a failed status carries headers to read: a provider failure carried in the body of a `200`
(OpenRouter's overloaded-upstream answer) is a plain model object with no headers on it, so that case
waits out the curve too.

Without a readable header the delay before retry `n` is `min(base_delay · 3ⁿ, max_delay)` (so 0.5s, 1.5s, …
by default). Anything else — and a transient failure with the retries exhausted — is raised as
`ProviderError` carrying `.attempts` and `.status_code`. A `RetryPolicy` with a negative or non-finite
field (`NaN`, `inf`), or a `retry` that is not a `RetryPolicy`, raises `JevperError` at construction.

A provider failure carried in the body of a `200` arrives as a `ProviderError` with `.embedded` true and
`.status_code` set from the body's own code when it has one. That flag is what keeps `api="auto"` from
treating a body's `404` as a missing route: the status came from inside a successful response, not from the
status line.

## `ReasoningConfig`

```python
ReasoningConfig(effort=None, summary=None, context=None, mode="auto", budget_tokens=None)
```

- `effort`: `none`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max`
- `summary`: `auto`, `concise`, `detailed`
- `context`: `auto`, `current_turn`, `all_turns`
- `mode`: `auto`, `native`, `two_step`
- `budget_tokens`: the Messages surface's own thinking budget, and the only reason to select that
  surface's native mode under `mode="auto"`. jevper sizes `max_tokens` above it (the API requires the
  budget to be strictly below `max_tokens`) and refuses locally when a caller's own `max_tokens` cannot
  hold it. Chat Completions and Responses ignore it — they carry `reasoning_effort` instead.

See [reasoning.md](reasoning.md) for how the mode resolves per surface.

## Errors

All inherit from `JevperError`.

| Error | Raised when |
| --- | --- |
| `InvalidQuestionError` | question or example is locally invalid — including an `examples` container or element that is not a sequence of `Example`, and probability keys that name the same answer key twice (`{1: 0.9, "1": 0.1}`); also raised when `logprobs`/`grammar` get a `Choice` with more than 26 options |
| `UnsupportedMethodError` | `method="grammar"` on a surface that is not Chat Completions, and `method="logprobs"`/`"grammar"` pinned to the Messages surface, which returns no logprobs at all |
| `ClientCapabilityError` | the client lacks the attribute a surface needs (or raises while being asked), or a response carried no choices and no explanation of why |
| `LabelReadoutError` | no logprobs at all, no usable alternatives for the answer token, no non-whitespace token, a first token that is not a label, a logprob that is not finite or is positive (which no log probability can be), a sampled token that contradicts an answer text naming another label, or no probability mass on any label. The first two are the provider's doing, so they are not corrective-retried and `method="auto"` answers with `structured` instead |
| `MalformedAnswerError` | JSON answer missing/extra keys, more than one JSON object in the answer, a number too large to be a float, a non-finite or out-of-range number, an unknown label, a score that is not a level index |
| `IncompleteAnswerError` | the provider stopped generating before the answer was complete — `finish_reason: "length"`, `stop_reason: "max_tokens"`, `model_context_window_exceeded`, a Responses `status: "incomplete"`, or any stop reason that is not one the surface documents. The message names the surface's own budget field (`max_output_tokens` on the Responses surface, `max_completion_tokens` on Chat Completions — with the `max_tokens` a local server takes named beside it — and `max_tokens` on Messages). A `ProviderError` subclass, and terminal: a cut-off generation is not a malformed answer to correct, because another attempt spends a call to be cut off the same way |
| `ModelRefusalError` | the model declined to answer and the provider said so — OpenAI's `refusal` field or content part, `stop_reason: "refusal"`, or a safety filter (`finish_reason: "content_filter"`, a Responses `incomplete_details.reason` of the same). A `ProviderError` subclass, and terminal: a refusal is complete, not broken, so a corrective retry would only be refused again |
| `ProviderError` | provider failure: a transient one after its `RetryPolicy` retries are exhausted, or a non-transient one at once. `.attempts` holds the attempt records and `.status_code` the status the provider reported, including one carried inside a `200` body — which wins over any answer the same body carries, and whose `code` is read as a number or as a digit string. Also raised when every surface `api="auto"` could try answered `404` (the route is missing, so the failure is the provider's, not a private verdict's), and when a Responses call reports a `status` that is neither `completed` nor `incomplete` — `failed`, `cancelled` — which is a generation the provider did not finish, not a malformed answer to correct |
| `JevperError` | base class, and the type used for constructor misuse, bad `state` messages, and content that is not JSON-serializable or contains a non-finite number |

The provider-side logprob failures — a rejected logprob request, no logprobs at all, no alternatives for the
answer token — are raised as `_LogprobsUnavailable`, a private `LabelReadoutError` subclass, and only while
`method="auto"` or `api="auto"` is in play: with both pinned, the provider's own refusal is what reaches you,
as a `ProviderError`. The verdict's `capability` attribute records whether the failure is evidence about the
provider (`True`, which is remembered) or a bad minute (`False`, which is not). Catch the public
`LabelReadoutError`. `capability=True` from a rejection is remembered at once; the same verdict read out of an
answer that carried no usable logprobs needs a second one, because a single anomalous response is not evidence
about the provider — [`internals.md`](internals.md#invariants) has the rule.

The surface has its own private verdict. An `openai` client object exposes `responses.create` whether or not the
server behind it implements the route — every local server (ollama, llama.cpp, SGLang, vLLM without the route)
does not — so under `api="auto"` a 404 that does not name the model is read as *this server has no Responses
route*: the call is re-issued on `chat_completions` and the verdict is remembered for the rest of the client's
life. A 404 that names the model is the model, and `api="responses"` asked for explicitly is never overridden.

A gateway can also say the protocol is the problem while naming the model in the same breath, which is why the
rule reads the evidence before the model markers. Measured on opencode Zen, 2026-09-26:
`muse-spark-1.3-contributor` answers every Chat Completions request with
`400 {"type": "ModelProtocolUnsupported", "message": "Model does not support this protocol."}` and answers the
same question on Responses, so under `api="auto"` the call moves and is answered, and afterwards goes straight
there. The marker has to name a protocol or a route — a 400 that says only `unsupported` is about a field, and
that is the field-downgrade path rather than a change of surface.

A surface that cannot deliver a distribution — it answered without logprobs, or refused the logprob fields —
is the same kind of verdict: under `api="auto"` the readout moves to the other surface once, and the surface
that failed is marked so later calls for that model start where the distribution is. `reasoning="native"` keeps
the surface it implies, and a provider failure that survived its retries moves nothing.

## Constants

`Method`, `MethodSelection` and `Api` are `Literal` aliases; the runtime tuples are `jevper.client.METHODS`
(the four concrete methods), `jevper.client.METHOD_SELECTIONS` (those plus `"auto"`) and `jevper.client.APIS`.
`jevper.client.AUTO_METHOD` (`"logprobs"`) and `jevper.client.FALLBACK_METHOD` (`"structured"`) are what
`method="auto"` tries and falls back to.
`jevper.client.TRANSIENT_STATUS_CODES`, `jevper.client.MAX_TOP_LOGPROBS` (`20`),
`jevper.client.MAX_PROMPT_CACHE_KEY` (`256`, the cap jevper enforces on a caller's cache key; OpenAI and
the OpenResponses schema cap the field itself at 64 characters, so a longer key is the server's to refuse),
`jevper.labels.MAX_LABEL_OPTIONS` (`26`, the single-token alphabet), `jevper.labels.MAX_CHOICE_OPTIONS` and
`jevper.types.CHOICE_MAX_OPTIONS` (`255`), `jevper.types.CHOICE_MIN_OPTIONS` (`1`),
`jevper.types.SCORE_MIN_LEVELS` (`2`), `jevper.types.SCORE_MAX_LEVELS` (`10`) and
`jevper.normalize.PROBABILITY_TOLERANCE` (`1e-6`) are available for callers that need to validate their own
inputs before constructing a question.
