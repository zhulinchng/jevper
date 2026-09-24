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
SystemOneClient(client, **options)
```

`client` is any object exposing `responses.create`, `chat.completions.create` and/or `messages.create` (the
Anthropic SDK's, pointed at any server that implements the Messages API); it stays owned by the caller
(`close()` only shuts down jevper's own thread pool).

| Parameter | Default | Meaning |
| --- | --- | --- |
| `model` | required | Model id sent with every call; overridable per `system_one` call |
| `method` | `"auto"` | `"auto"`, `"logprobs"`, `"grammar"`, `"structured"` or `"discrete"`; `"auto"` resolves per model and surface — see [methods.md](methods.md#auto) |
| `api` | `"auto"` | `"auto"`, `"chat_completions"`, `"responses"` or `"messages"` (the Anthropic Messages API); `"auto"` prefers `responses`, then `chat_completions`, then `messages`, falling back when the server answers 404 for a route — see [methods.md](methods.md#auto) |
| `reasoning` | `None` | A `ReasoningConfig`; `None` disables reasoning entirely. `mode="auto"` resolves to `native` on the Responses surface and `two_step` elsewhere, and `budget_tokens` is what the Messages surface sends as its `thinking` field — see [reasoning.md](reasoning.md) |
| `examples` | `()` | Default few-shot examples: a sequence for all questions, or a mapping keyed by question id |
| `structured_outputs` | `True` | Send a strict `json_schema` response format — `response_format` on Chat Completions, `text.format` on Responses, and the Messages API's own `output_config.format`; `False` leaves the schema in the prompt instead (`{"type": "json_object"}` on the OpenAI surfaces). A server that refuses the strict schema gets the same fallback automatically |
| `normalize_probabilities` | `True` | Rescale `structured` distributions that are off by more than `1e-6`; `False` returns the model's numbers verbatim |
| `top_logprobs` | `20` | Requested alternatives for `logprobs`/`grammar`; must be in `[0, 20]`, and at least 2 when the method is pinned to `logprobs`/`grammar` — the sampled token alone is not a distribution |
| `max_concurrency` | `8` | Questions in flight at once (thread pool, or asyncio semaphore) |
| `n_retry_malformed` | `1` | Corrective retries when an answer cannot be read |
| `retry` | `None` | Transient-failure retries; `RetryPolicy()` (2 retries, 0.5s base, 8s cap) when unset |
| `temperature` | `None` | Not sent unless set. `0.0` is recommended for `structured`/`discrete`; `logprobs` needs no setting. Left out of a Messages request that enables `thinking`, which the API refuses alongside a non-default temperature |
| `prompt_cache_key` | `None` | The provider's cache-routing key. Unset, jevper derives one per question from the parts of the prompt that do not change between calls, so a rubric's requests are routed together; set it to group (or account for) requests your own way |
| `extra_body` | `None` | Merged into every request body (the grammar field is merged here too). A key it names is the value that reaches the wire — the SDK merges `extra_body` *after* the typed parameters — so jevper leaves that field alone rather than sending a typed value the caller's own key would override. `response_format`/`text`/`output_config` named here therefore also puts the JSON Schema in the prompt, since no schema of jevper's is in the request. On the Messages surface, `max_tokens` comes from here — jevper always sends one there, defaulting to `DEFAULT_MAX_TOKENS` (1024), or 1024 plus `ReasoningConfig(budget_tokens=…)`; a `max_tokens` that cannot hold the thinking budget asked for raises `JevperError` locally, naming both numbers, rather than earning the API's own refusal. Every other key travels on all three surfaces, whether or not a temperature was set |
| `extra_headers` | `None` | Sent with every request |

Constructor validation is eager: an unknown `method`/`api`, `top_logprobs` outside `[0, 20]` or below 2 with a
pinned `logprobs`/`grammar`, `max_concurrency < 1`, a negative `n_retry_malformed`, a non-`ReasoningConfig`
`reasoning`, a `retry` that is not a `RetryPolicy` or has a negative or non-finite field, a `prompt_cache_key`
that is not a non-empty string of at most `MAX_PROMPT_CACHE_KEY` characters, and an `extra_body` or
`extra_headers` that is not a mapping all raise `JevperError`.

### `system_one`

```python
system_one(*, state, questions, examples=(), model=None, method=None, api=None, reasoning=None, temperature=None, prompt_cache_key=None)
    -> SystemOneResponse
```

| Parameter | Default | Meaning |
| --- | --- | --- |
| `state` | required | `str`, a chat-message list, `{"messages": [...]}`, or any JSON value (see below) |
| `questions` | required | Mapping of question id to a `Noul`/`Choice`/`Score` or an equivalent raw mapping; at least one |
| `examples` | `()` | Per-call examples: sequence for all questions, or mapping keyed by question id |
| `model` | `None` | Overrides the constructor model |
| `method` | `None` | Overrides the constructor method |
| `api` | `None` | Overrides the constructor api |
| `reasoning` | `None` | Overrides the constructor reasoning |
| `temperature` | `None` | Overrides the constructor temperature |
| `prompt_cache_key` | `None` | Overrides the constructor cache key; `None` keeps the constructor's (or the derived one) |

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
- Validation runs on construction and again in `system_one`, so both paths fail identically. A mapping passed
  as a question is parsed with a discriminated union; a malformed mapping raises `InvalidQuestionError` with
  the field path of every problem.

```python
Example(state=..., answer="billing" | "B" | 2 | True, probabilities={"billing": 0.6, ...} | None = None)
```

`answer` may be a label (`"B"`), a `Choice` criteria key, a `Score` level index, or a bool for `Noul`. The
label is tried first, so an option key that is itself a label is read as the label. `probabilities` is only
read by `method="structured"` and defaults to a one-hot distribution over `answer`.

When given, `probabilities` is validated, because the example is replayed into the prompt as the answer the
model is asked to imitate:

- it must carry exactly the option keys of a `Choice`, the level indexes of a `Score`, or a `True`/`False`
  (`"true"`/`"false"`) key for a `Noul` — a wrong key set raises `InvalidQuestionError` naming the example
  index;
- values must be finite (rejected by the `Example` model itself) and `>= 0`, and a `Noul` value must be in
  `[0, 1]`.

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
  maps level index to the criteria entry. The rescale matters only with `normalize_probabilities=False`,
  where the reported `probabilities` stay the model's own numbers: a score carried from an unnormalized
  distribution would leave the `0..N-1` line the Jev answer schema documents. This is the arithmetic
  the TypeSafe reference adapter uses.
- `confidence` for `choice` is `(max(p) − 1/n) / (1 − 1/n)`, i.e. the peak probability scaled from uniform
  (0) to certainty (1). For `score` it is `max(0, 1 − MAD(p) / MAD_uniform)`, where `MAD` is the mean absolute
  deviation from the modal level and `MAD_uniform` is that quantity for a uniform distribution. Both are
  computed after normalizing the inputs, with a uniform fallback when the total is zero — the same arithmetic
  as the TypeSafe reference adapter.
- Integer keys (`score`, `legend`, `probabilities`) serialize to JSON object keys as strings.

## `SystemOneResponse`

| Field | Type | Meaning |
| --- | --- | --- |
| `model` | `str` | Effective model |
| `answers` | `dict[str, Answer]` | Keyed by question id, in insertion order |
| `usage` | `Usage` | Aggregated over every provider call and retry |
| `reasoning` | `tuple[ReasoningContentPart, ...]` | Trace, in question order; see [reasoning.md](reasoning.md) |
| `debug` | `dict[str, Any]` | Attempts and normalization notes; see below |

`nouls`, `choices` and `scores` are cached properties returning `answers` filtered by answer type.

```python
Usage(input_tokens=None, output_tokens=None, reasoning_tokens=None, cached_tokens=None, n_calls=0, n_retries=0, latency=0.0)
```

A token count is `None` when any constituent call omitted it (a reported `0` is preserved). `n_calls` counts
analysis passes and corrective retries; `n_retries` counts transient-failure retries only. `latency` is
wall-clock seconds for the whole `system_one` call.

`cached_tokens` is the prompt tokens the provider read from its own prompt cache, read from
`usage.prompt_tokens_details.cached_tokens` on Chat Completions and `usage.input_tokens_details.cached_tokens`
on the Responses surface. It is the one number that says caching worked; a provider that reports nothing
leaves it `None`, which is not the same as a reported `0` — that is a server whose prefix cache is cold, or
off. Not every server reports it at all, and two need a flag to: vLLM's `--enable-prompt-tokens-details` and
SGLang's `--enable-cache-report` for its Chat Completions route. See
[local-servers.md](local-servers.md#prompt-caching).

### `debug`

| Key | Content |
| --- | --- |
| `method` | The call-level method: what `method="auto"` resolved to for this call, or the method you pinned. A question that skips the probe — a `Choice` past 26 options is answered in JSON without ever asking for logprobs — reports its own method in `methods`, so read that when they can differ |
| `methods` | `{question_id: method}` — only for `method="auto"`, since the method is then chosen per question |
| `api` | Surface actually used (`"chat_completions"`, `"responses"` or `"messages"`) |
| `reasoning_mode` | `"off"`, `"native"` or `"two_step"` |
| `llm_attempts` | One record per provider call: `question_id`, `surface`, `request`, `response`, `error`, `readout` |
| `retry_reasons` | Corrective-retry messages, in order |
| `probability_errors` | `{question_id: abs(sum − 1)}` for `structured` distributions outside `1e-6` |
| `original_probabilities` | The model's raw distribution, only for questions that were rescaled |
| `server_limits` | Only when the server refused a capability field: `structured` (`"schema"`/`"object"`/`"none"`), `output_config` (the Messages API's schema field), `reasoning`, `include`, `cache_key` and `thinking` as it last accepted them |
| `labels_missing` | Labels the provider did not report a logprob for, per question |

Every key is always present — except `methods`, which only `method="auto"` adds — and the last three are empty
mappings when nothing applies. `request` holds the exact
kwargs sent to the provider — for a failed call, the kwargs that were about to be sent, so the shape is the
same either way — `response` holds the provider object dumped with `model_dump(mode="json")` when available,
`error` is a `"Type: message"` string, and `readout` is the parsed readout: `source`, `probabilities` (string
keys), `missing_labels` and `observed_text`. `debug["llm_attempts"][-1]["readout"]` is set for the attempt
that produced the final answer.

## `RetryPolicy`

```python
RetryPolicy(n_retries=2, base_delay=0.5, max_delay=8.0, respect_retry_after=True)
```

Applies per provider call. A failure is transient when the exception exposes a `status_code` reading as one of
`{408, 429, 500, 502, 503, 504, 529}` (an int, an `http.HTTPStatus`, or a digit string, taken from the
exception or from `exc.response.status_code` when only the response carries it), or when its class — or any
class in its MRO — contains `Connection` or
`Timeout`, or is an `httpx`-family transport failure (`TransportError`, `TimeoutException`). That last clause
is what covers `httpx.ConnectError`, `ReadError` and `RemoteProtocolError`, whose names carry neither marker;
a client-side `LocalProtocolError` is not retried.

The wait is whatever the provider asked for when it said: a `Retry-After` (delta-seconds or an HTTP date)
or the millisecond `retry-after-ms` some providers send instead replaces the computed backoff, which is
what the TypeSafe clients do by default — coming back sooner than a rate limit asked is one way to extend
it. Header names are matched case-insensitively, as HTTP requires, so a client that hands its exceptions a
plain `{"RETRY-AFTER": "120"}` dict is heard the same as one that carries an `httpx.Headers`, and a date
already past means come back now — a wait of zero. `max_delay` caps jevper's own curve, not the server's
instruction, so a header asking for minutes is waited out in minutes, up to `jevper.client.MAX_RETRY_AFTER` (2^53
seconds: the largest integer a float holds exactly, and no real provider's idea of a wait). Past that the
header is not an instruction any runtime can carry out, so the curve answers rather than `time.sleep`
raising `OverflowError`. `respect_retry_after=False` goes back to the curve alone, and `n_retries=0` fails
on the first rate-limited response. A header jevper cannot read — not a number, not a date, negative — is
the backoff's business, never a reason to skip the wait. Only an exception the SDK raised for a failed
status carries headers to read: a provider failure carried in the body of a `200` (OpenRouter's
overloaded-upstream answer) is a plain model object with no headers on it, so that case waits out the
curve too.

Without a readable header the delay before retry `n` is `min(base_delay · 3ⁿ, max_delay)` (so 0.5s, 1.5s, …
by default). Anything else — and a transient failure with the retries exhausted — is raised as
`ProviderError` carrying `.attempts` and `.status_code`. A `RetryPolicy` with a negative or non-finite
field (`NaN`, `inf`), or a `retry` that is not a `RetryPolicy`, raises `JevperError` at construction.

## `ReasoningConfig`

```python
ReasoningConfig(effort=None, summary=None, context=None, mode="auto")
```

- `effort`: `none`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max`
- `summary`: `auto`, `concise`, `detailed`
- `context`: `auto`, `current_turn`, `all_turns`
- `mode`: `auto`, `native`, `two_step`

See [reasoning.md](reasoning.md) for how the mode resolves per surface.

## Errors

All inherit from `JevperError`.

| Error | Raised when |
| --- | --- |
| `InvalidQuestionError` | question or example is locally invalid; also raised when `logprobs`/`grammar` get a `Choice` with more than 26 options |
| `UnsupportedMethodError` | `method="grammar"` and the selected surface is not Chat Completions |
| `ClientCapabilityError` | the client lacks the attribute a surface needs, or a response carried no choices and no explanation of why |
| `LabelReadoutError` | no logprobs at all, no alternatives for the answer token, no non-whitespace token, a first token that is not a label, no logprob for the answer token, a non-finite logprob, or no probability mass on any label. The first two are the provider's doing, so they are not corrective-retried and `method="auto"` answers with `structured` instead |
| `MalformedAnswerError` | JSON answer missing/extra keys, a non-finite or out-of-range number, an unknown label, a score that is not a level index |
| `IncompleteAnswerError` | the provider stopped generating before the answer was complete — `finish_reason: "length"`, `stop_reason: "max_tokens"`, a Responses `status: "incomplete"`, or a filtered answer. A `ProviderError` subclass, and terminal: a cut-off generation is not a malformed answer to correct, because another attempt spends a call to be cut off the same way |
| `ModelRefusalError` | the model declined to answer and the provider said so — OpenAI's `refusal` field or content part, `stop_reason: "refusal"`. A `ProviderError` subclass, and terminal: a refusal is complete, not broken, so a corrective retry would only be refused again |
| `ProviderError` | provider failure after transient retries; `.attempts` holds the attempt records and `.status_code` the status the provider reported, including one carried inside a `200` body. Also raised when every surface `api="auto"` could try answered `404` (the route is missing, so the failure is the provider's, not a private verdict's), and when a Responses call reports a `status` that is neither `completed` nor `incomplete` — `failed`, `cancelled` — which is a generation the provider did not finish, not a malformed answer to correct |
| `JevperError` | base class, and the type used for constructor misuse, bad `state` messages, and content that is not JSON-serializable or contains a non-finite number |

The provider-side logprob failures — a rejected logprob request, no logprobs at all, no alternatives for the
answer token — are raised as `_LogprobsUnavailable`, a private `LabelReadoutError` subclass. It is private
because `method="auto"` is the only thing that reads it: its `capability` attribute records whether the failure
is evidence about the provider (`True`, which `auto` remembers) or a bad minute (`False`, which it does not).
Catch the public `LabelReadoutError`. `capability=True` from a rejection is remembered at once; the same verdict
read out of an answer that carried no usable logprobs needs a second one, because a single anomalous response is
not evidence about the provider — [`internals.md`](internals.md#invariants) has the rule.

The surface has its own private verdict. An `openai` client object exposes `responses.create` whether or not the
server behind it implements the route — every local server (ollama, llama.cpp, SGLang, vLLM without the route)
does not — so under `api="auto"` a 404 that does not name the model is read as *this server has no Responses
route*: the call is re-issued on `chat_completions` and the verdict is remembered for the rest of the client's
life. A 404 that names the model is the model, and `api="responses"` asked for explicitly is never overridden.

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
`jevper.client.MAX_PROMPT_CACHE_KEY` (`256`, the cap jevper enforces on a caller's cache key),
`jevper.labels.MAX_LABEL_OPTIONS` (`26`, the single-token alphabet), `jevper.labels.MAX_CHOICE_OPTIONS` and
`jevper.types.CHOICE_MAX_OPTIONS` (`255`), `jevper.types.CHOICE_MIN_OPTIONS` (`1`),
`jevper.types.SCORE_MIN_LEVELS` (`2`), `jevper.types.SCORE_MAX_LEVELS` (`10`) and
`jevper.normalize.PROBABILITY_TOLERANCE` (`1e-6`) are available for callers that need to validate their own
inputs before constructing a question.
