# API reference

*Independent implementation of the documented System One wire format — not affiliated with TypeSafe.*

Everything below is importable from the package root:

```python
from jevper import SystemOneClient, AsyncSystemOneClient, Choice, Noul, Score, Example, ReasoningConfig
```

`__all__` also contains `Answer`, `Api`, `ChoiceAnswer`, `Example`, `Examples`, `JSONContent`, `Method`,
`NoulAnswer`, `NoulCriteria`, `ProviderError`, `Question`, `Readout`, `ReasoningContentPart`,
`ReasoningSummaryPart`, `ReasoningTextPart`, `RetryPolicy`, `ScoreAnswer`, `SystemOneResponse`, `Usage`,
`reasoning_text`, the five other error classes, and `__version__`.

## `SystemOneClient`

```python
SystemOneClient(client, **options)
```

`client` is any object exposing `responses.create` and/or `chat.completions.create`; it stays owned by the
caller (`close()` only shuts down jevper's own thread pool).

| Parameter | Default | Meaning |
| --- | --- | --- |
| `model` | required | Model id sent with every call; overridable per `system_one` call |
| `method` | `"auto"` | `"auto"`, `"logprobs"`, `"grammar"`, `"structured"` or `"discrete"`; `"auto"` resolves per model and surface — see [methods.md](methods.md#auto) |
| `api` | `"auto"` | `"auto"`, `"chat_completions"` or `"responses"` |
| `reasoning` | `None` | A `ReasoningConfig`; `None` disables reasoning entirely |
| `examples` | `()` | Default few-shot examples: a sequence for all questions, or a mapping keyed by question id |
| `structured_outputs` | `True` | Send a strict `json_schema` response format; `False` falls back to `{"type": "json_object"}` with the schema left in the prompt |
| `normalize_probabilities` | `True` | Rescale `structured` distributions that are off by more than `1e-6`; `False` returns the model's numbers verbatim |
| `top_logprobs` | `20` | Requested alternatives for `logprobs`/`grammar`; must be in `[0, 20]` |
| `max_concurrency` | `8` | Questions in flight at once (thread pool, or asyncio semaphore) |
| `n_retry_malformed` | `1` | Corrective retries when an answer cannot be read |
| `retry` | `None` | Transient-failure retries; `RetryPolicy()` (2 retries, 0.5s base, 8s cap) when unset |
| `temperature` | `None` | Not sent unless set. `0.0` is recommended for `structured`/`discrete`; `logprobs` needs no setting |
| `extra_body` | `None` | Merged into every request body (the grammar field is merged here too) |
| `extra_headers` | `None` | Sent with every request |

Constructor validation is eager: an unknown `method`/`api`, `top_logprobs` outside `[0, 20]`,
`max_concurrency < 1`, a negative `n_retry_malformed`, a non-`ReasoningConfig` `reasoning`, a `retry` that is
not a `RetryPolicy` or has a negative field raises `JevperError`.

### `system_one`

```python
system_one(*, state, questions, examples=(), model=None, method=None, api=None, reasoning=None, temperature=None)
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

Per-call values win over constructor defaults. Everything is resolved and validated before the first provider
call, so a bad question, an empty `questions` mapping, an unusable `state`, or `grammar` on the Responses
surface costs zero requests.

`state` forms:

| `state` | Rendered as |
| --- | --- |
| `"text"` | one `user` turn |
| `[{"role": "user", "content": "..."}, ...]` | those turns verbatim (roles `system`, `user`, `assistant`, `developer`; `content` must be `str`) |
| `{"messages": [...]}` | same as above |
| any other JSON value | one `user` turn holding `json.dumps(state, ensure_ascii=False, sort_keys=True, indent=2)` |

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
- Limits: `Choice` 2–255 options (the Jev API limit), `Score` 2–10 levels. `logprobs` and `grammar` cap a
  `Choice` at 26 options, because they read one label token; `structured` and `discrete` carry the full range
  with two-letter labels past `Z`. Unknown fields are rejected (`extra="forbid"`).
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
- `score` is `Σ i·pᵢ` over zero-based levels; `legend` maps level index to the criteria entry.
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
Usage(input_tokens=None, output_tokens=None, reasoning_tokens=None, n_calls=0, n_retries=0, latency=0.0)
```

A token count is `None` when any constituent call omitted it (a reported `0` is preserved). `n_calls` counts
analysis passes and corrective retries; `n_retries` counts transient-failure retries only. `latency` is
wall-clock seconds for the whole `system_one` call.

### `debug`

| Key | Content |
| --- | --- |
| `method` | Effective method: what `method="auto"` resolved to, or the method you pinned |
| `methods` | `{question_id: method}` — only for `method="auto"`, since the method is then chosen per question |
| `api` | Surface actually used (`"chat_completions"` or `"responses"`) |
| `reasoning_mode` | `"off"`, `"native"` or `"two_step"` |
| `llm_attempts` | One record per provider call: `question_id`, `surface`, `request`, `response`, `error`, `readout` |
| `retry_reasons` | Corrective-retry messages, in order |
| `probability_errors` | `{question_id: abs(sum − 1)}` for `structured` distributions outside `1e-6` |
| `original_probabilities` | The model's raw distribution, only for questions that were rescaled |
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
RetryPolicy(n_retries=2, base_delay=0.5, max_delay=8.0)
```

Applies per provider call. A failure is transient when the exception exposes a `status_code` reading as one of
`{429, 500, 502, 503, 504, 529}` (an int, an `http.HTTPStatus`, or a digit string), or when its class — or any
class in its MRO — contains `Connection` or
`Timeout`, or is an `httpx`-family transport failure (`TransportError`, `TimeoutException`). That last clause
is what covers `httpx.ConnectError`, `ReadError` and `RemoteProtocolError`, whose names carry neither marker;
a client-side `LocalProtocolError` is not retried. The delay before retry
`n` is `min(base_delay · 3ⁿ, max_delay)` (so 0.5s, 1.5s, … by default). Anything else — and a transient
failure with the retries exhausted — is raised as `ProviderError` carrying `.attempts`. A `RetryPolicy` with a
negative field, or a `retry` that is not a `RetryPolicy`, raises `JevperError` at construction.

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
| `ClientCapabilityError` | the client lacks the attribute a surface needs, or a chat response carried no choices |
| `LabelReadoutError` | no logprobs at all, no alternatives for the answer token, no non-whitespace token, a first token that is not a label, no logprob for the answer token, a non-finite logprob, or no probability mass on any label. The first two are the provider's doing, so they are not corrective-retried and `method="auto"` answers with `structured` instead |
| `MalformedAnswerError` | JSON answer missing/extra keys, a non-finite or out-of-range number, an unknown label, a score that is not a level index |
| `ProviderError` | provider failure after transient retries; `.attempts` holds the attempt records |
| `JevperError` | base class, and the type used for constructor misuse, bad `state` messages, and content that is not JSON-serializable or contains a non-finite number |

## Constants

`Method`, `MethodSelection` and `Api` are `Literal` aliases; the runtime tuples are `jevper.client.METHODS`
(the four concrete methods), `jevper.client.METHOD_SELECTIONS` (those plus `"auto"`) and `jevper.client.APIS`.
`jevper.client.AUTO_METHOD` (`"logprobs"`) and `jevper.client.FALLBACK_METHOD` (`"structured"`) are what
`method="auto"` tries and falls back to.
`jevper.client.TRANSIENT_STATUS_CODES`, `jevper.client.MAX_TOP_LOGPROBS` (`20`),
`jevper.labels.MAX_LABEL_OPTIONS` (`26`, the single-token alphabet), `jevper.labels.MAX_CHOICE_OPTIONS` and
`jevper.types.CHOICE_MAX_OPTIONS` (`255`), `jevper.types.SCORE_MAX_LEVELS` (`10`) and
`jevper.normalize.PROBABILITY_TOLERANCE` (`1e-6`) are available for callers that need to validate their own
inputs before constructing a question.
