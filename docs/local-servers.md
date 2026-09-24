# Local servers

jevper talks to anything that speaks the OpenAI wire format, so a local server is the same client with a
different `base_url`. Three things differ between servers, and all three are decided by what you pass rather
than by the library: **which surface carries logprobs**, **how thinking is turned off**, and **which request
fields the server quietly ignores**. This page is the short version, checked against ollama 0.34, llama.cpp
b11139, vLLM 0.30 and SGLang 0.5.20 serving `Qwen3.5-9B` at 4-bit on one 12 GB card.

```python
from openai import OpenAI
from jevper import Choice, SystemOneClient

client = SystemOneClient(
    OpenAI(base_url="http://127.0.0.1:11434/v1", api_key="local"),
    model="qwen3.5:9b",
    extra_body={"reasoning_effort": "none"},  # thinking off — see below
)

response = client.system_one(
    state="I was charged twice for the same subscription this month.",
    questions={
        "intent": Choice(
            criteria={
                "billing": "money, invoices, refunds, charges",
                "technical": "errors, crashes, login or performance problems",
                "sales": "pricing, plans, purchasing, upgrades",
            }
        )
    },
)

response.answers["intent"].probabilities  # a real distribution, read from the server's logprobs
```

## What to pass per server

| Server | `base_url` | `model` | Thinking off | Notes |
| --- | --- | --- | --- | --- |
| ollama | `http://127.0.0.1:11434/v1` | the tag you pulled, e.g. `qwen3.5:9b` | `extra_body={"reasoning_effort": "none"}` | Chat Completions carries logprobs; the Responses route exists but returns an empty logprob list |
| llama.cpp | `http://127.0.0.1:8080/v1` | the `--alias` value | `extra_body={"chat_template_kwargs": {"enable_thinking": False}}` | serve with `--jinja` for the model's own template; `grammar` is llama.cpp-only |
| vLLM | `http://127.0.0.1:8000/v1` | the `--served-model-name` value | `extra_body={"chat_template_kwargs": {"enable_thinking": False}}` | serve with `--reasoning-parser qwen3`; `top_logprobs` is capped by `--max-logprobs` (20) |
| SGLang | `http://127.0.0.1:30000/v1` | the `--served-model-name` value | `extra_body={"chat_template_kwargs": {"enable_thinking": False}}` | serve with `--reasoning-parser qwen3`; its Responses route needs `top_logprobs` sent explicitly, which jevper always does |
| LM Studio | `http://127.0.0.1:1234/v1` | the id `lms ls` prints, e.g. `qwen3-4b-instruct-2507` | nothing does — load a non-thinking model | all three surfaces on one box, as do the other four now; `logprobs` arrives on both OpenAI surfaces; its Responses route accepts `text.format` and ignores it, so structured answers belong on Chat Completions |

All five answer the Anthropic Messages route as well (`/v1/messages`), so `api="auto"` has all three to
choose from on any of them; ollama, llama.cpp, SGLang and LM Studio were exercised through it directly.

`api="auto"` (the default) works against all five: it prefers the Responses surface, and when that route is
missing — or answers without carrying logprobs through — it re-asks on Chat Completions and remembers the
verdict. Passing `api="chat_completions"` skips the discovery entirely.

## Thinking is the one decision you must make

`logprobs` and `grammar` read a **one-token answer**, and every one of these servers reports logprobs for
*every* generated token. With thinking on, that is the first token of the reasoning, not the label: the
readout raises `LabelReadoutError`, spends one corrective retry, and raises again. `structured` and
`discrete` are unaffected — they read answer text, not a token's probability — but thinking still costs
a whole reasoning pass to produce one JSON object, so turn it off for classification work either way.

| Server | Request field that turns it off | Also available |
| --- | --- | --- |
| ollama | `reasoning_effort: "none"` | `OLLAMA_CONTEXT_LENGTH`, per-model `think` on the native API |
| llama.cpp | `chat_template_kwargs: {"enable_thinking": false}` | `reasoning_effort: "none"` works too, since `--jinja` runs the model's own template; `reasoning_budget: 0`; `--reasoning-format deepseek` splits the trace into `reasoning_content` |
| vLLM | `chat_template_kwargs: {"enable_thinking": false}` | `reasoning_effort` — but only with the values its parser accepts |
| SGLang | `chat_template_kwargs: {"enable_thinking": false}` | `reasoning_effort` — but only with the values its parser accepts |
| LM Studio | none reliably: omitting the field leaves the model's own template in charge, and `reasoning_effort: "none"` is ignored on its chat route (bug-tracker #2413; seen here — the trace still arrived, 1016 characters of it) | load an instruct model instead — `qwen3-4b-instruct-2507` answers without a reasoning trace |

If you keep thinking on, `structured` and `discrete` are unaffected — they read the answer text, and the
trace lands in `response.reasoning` (`message.reasoning` on ollama, `reasoning_content` on the others;
jevper reads all three). A label readout also survives it when the server separates the trace *and* the token
stream ends exactly with the answer text: jevper anchors on that tail and reads the answer's own first token.
That anchor is deliberately strict — anything else is reported rather than guessed — with one allowance:
vLLM and SGLang append their own end-of-turn token (`<|im_end|>`) to the logprob stream after the answer, so
up to two trailing tokens that cannot be part of the answer are dropped before the tail is tested.

## What each server ignores, and what it rejects

Unknown fields are accepted and dropped by all five, so a field that does not apply is not an error:

- `grammar` (jevper's `method="grammar"`) is a llama.cpp convention. ollama, vLLM and SGLang ignore it, so
  the model answers unconstrained and the label readout reports a non-label first token instead of a grammar
  failure. Use `logprobs` or `structured` there.
- `strict: true` inside `json_schema` is ignored by ollama and honoured by vLLM and SGLang.
- `reasoning_effort` reaches the chat template on llama.cpp, ollama and SGLang; vLLM validates it against its
  own enum and answers `400` for a value outside it (`xhigh` and `max` are the usual casualties).
- `max_completion_tokens` is ignored by ollama, which only knows `max_tokens` — bound output with
  `max_tokens`.
- `n` is rejected outright by llama.cpp (`1 <= value <= 1`); the others accept it, and vLLM and SGLang then
  return two choices where jevper reads the first.
- A `developer` message is a `400` (`Unexpected message role.`) on SGLang, so keep `state` to
  `system`/`user`/`assistant` roles. jevper's own turns never use another role.
- LM Studio's Responses route accepts `text.format` with a strict `json_schema` and **ignores it** — structured
  output is a Chat Completions feature there (bug-tracker #2403, #1396). Verified here with a schema whose keys
  the prompt never named: the Responses route answered in the prompt's own shape, the Chat Completions route in
  the schema's. jevper sends the schema in the request and, since it was accepted, does not repeat it in the
  prompt, so `method="structured"` on that surface reads whatever the model invents. Use
  `api="chat_completions"`, or `method="logprobs"`, for structured work there.
- llama.cpp's Responses route accepts `text.format` and **ignores it** as well (its own converter never
  reads the field; issue #21922), while its Chat Completions route converts the schema to a grammar and
  enforces it. Same advice as LM Studio: structured work belongs on `api="chat_completions"` there, and
  `api="auto"` gets there by itself when the logprob readout moves surfaces.
- The output budget is the server's own on both OpenAI surfaces, so a model whose context is smaller than
  the server's default refuses the request before inference: vLLM answers `400 max_tokens=2048 cannot be
  greater than max_model_len=max_total_tokens=1024`. jevper reports that 400 with the numbers in it; bound
  the output with `extra_body={"max_tokens": n}` when a server is serving a small-context model. The
  Messages route is the exception — that API has no default at all, so jevper sends one (1024, plus any
  thinking budget) and refuses a caller's `max_tokens` that cannot hold the budget it asked for.
- vLLM's `/v1/messages` answers 200 with a `thinking` block and no text at all for these probes — a
  three-character trace, `stop_reason: "end_turn"`, `output_tokens: 2` — while the same server answers
  its Chat Completions and Responses routes normally. jevper reports that as an empty answer naming the
  stop reason, which is what happened; the cause is on the server side, so a Messages request there
  wants a model whose template emits text on that route.
- An unknown path is not a `404` on LM Studio: it answers `200` with
  `{"error": "Unexpected endpoint or method. (POST /…)"}` (bug-tracker #618; confirmed here — `POST /v1/nope` and
  `POST /v1/messages/count_tokens` both answer `200` with that body), which jevper reads as an embedded
  provider error rather than a missing route — right for a real endpoint failing, so do not rely on route
  discovery to catch a typo'd path there.
- Usage accounting differs by route: Chat Completions reports no cached-token count at all, Responses reports
  `input_tokens_details.cached_tokens`, and Messages reports `cache_read_input_tokens` — all three observed here
  on the same server.
- `reasoning_effort` is honoured on LM Studio's `/v1/responses` and ignored on `/v1/chat/completions`
  (#2413), so the two OpenAI surfaces disagree about it; `/v1/responses` also ignores `instructions` (#1154).
  The Messages route returns Anthropic-shaped `thinking` blocks when asked with an explicit
  `budget_tokens` (1542 characters of it here), and reports `cache_read_input_tokens`.

## What each server actually answers

Recorded with raw HTTP against each server — 45 requests each, and the responses jevper has to read are kept
in `tests/fixtures/providers/`. The shapes matter more than the verdicts: they are what a client has to
tolerate, and the fixture tests replay them on every change.

| | ollama 0.34.3 | llama.cpp b11139 | vLLM 0.30.1 | SGLang 0.5.20 |
| --- | --- | --- | --- | --- |
| `/v1/responses` route | yes | yes | yes | yes |
| `/v1/messages` route | yes, since 0.14.0 | yes, since Nov 2025 | yes, since 0.12.0 | yes, since 0.5.9 |
| logprobs on Chat Completions | yes | yes | yes | yes |
| logprobs on Responses | **empty list** | **`400`** | yes, with `include` | yes, with `top_logprobs` |
| `top_logprobs` above 20 | `400` (`must be between 0 and 20`) | accepted | `400` | accepted |
| unknown model id | `404` naming it | ignored, `200` | `404` naming it | Chat ignored, Responses `404` |
| `n: 2` | accepted, one choice back | `400` | accepted, two choices | accepted, two choices |
| `developer` role | accepted | accepted | accepted | **`400`** |
| unknown request field | ignored | ignored | ignored | ignored |
| route 404 body | `404 page not found` (text) | `{"error": {...}}` | `{"detail": "Not Found"}` | `{"detail": "Not Found"}` |
| some `400` bodies | JSON object | JSON object | JSON object | **a bare JSON string** |
| reasoning field | `reasoning` | `reasoning_content` | `reasoning` | `reasoning_content` |

Three of these decide behaviour jevper implements rather than documents:

- **A 404 that names the model is about the model.** ollama and vLLM answer a bad model id with a 404 that
  quotes it; switching surfaces would only produce the same 404 again, so jevper reports it as it stands.
- **A 404 that does not name the model is a missing route.** `{"detail": "Not Found"}` and
  `404 page not found` say nothing about the model, so `api="auto"` re-asks on the other surface and
  remembers which one exists.
- **A surface that answers without a distribution is left behind.** ollama's Responses route returns the
  message with `logprobs: []` and llama.cpp's refuses the fields outright, while Chat Completions on both
  carries the full distribution; `api="auto"` moves the label readout there, marks the surface, and later
  calls start where the distribution is.

## Prompt caching

Every one of these servers caches the prefix of a prompt and reuses it for the next request that starts the
same way. None of them needs to be asked: prefix caching is on by default on all five (llama.cpp's
`--cache-prompt`, vLLM's `enable_prefix_caching`, SGLang's RadixAttention, ollama's runner cache), and the
request fields the hosted APIs use to steer it — `prompt_cache_key`, `prompt_cache_retention`, `cache_salt`,
`session_id`, `prompt_cache_options`, `prompt_cache_breakpoint` — are accepted with `200` and ignored by all
four. Sending one costs nothing; expecting it to do anything here does.

What they do *not* agree on is telling you it happened:

| | ollama 0.34.3 | llama.cpp b11139 | vLLM 0.30.1 | SGLang 0.5.20 |
| --- | --- | --- | --- | --- |
| `usage.prompt_tokens_details.cached_tokens` | always | always | needs `--enable-prompt-tokens-details` | needs `--enable-cache-report` |
| `usage.input_tokens_details.cached_tokens` (Responses) | always | always | always | always |
| other fields | `prompt_eval_cached_count` on native `/api/chat` | `timings.cache_n` (same number) | — | — |
| cache inspection | — | `GET /slots` | `/metrics` (`vllm:prefix_cache_hits`) | `/metrics` (`sglang:cache_hit_rate`) |
| cache flush | unload the model | `POST /slots/{id}?action=erase` | `/reset_prefix_cache` (dev mode) | `POST /flush_cache` |

jevper reads both usage paths into `Usage.cached_tokens` and leaves it `None` when the server says nothing —
which is why turning the flag on matters if you want to see the number on vLLM or SGLang. A server that
reports a plain `0` is reporting a cold or disabled cache, and that is preserved as `0`.

### Message order is what decides the reuse

A cached prefix is reused up to the first token that differs, so the part of a jevper prompt that changes
between calls — the state — has to come last for a rubric's calls to share anything. Measured on one 2388-token
prompt (two examples, a ~1300-token state), second call differing only in the state:

| Message order | ollama | llama.cpp | vLLM | SGLang |
| --- | --- | --- | --- | --- |
| `state, examples, question` (jevper ≤ 0.3.0) | 0 | 40 | 0 | — |
| `examples, state, question` | 0 | 40 | 0 | 192 |
| `examples, question, state` (jevper ≥ 0.4.0) | 0 | **1010** | **528** | **896** |
| identical repeat of any of them | 2384 | 2384 | 2112 | 2368 |

Latency agrees with the counts: only in the last row is the state-varied second call faster than the first
(llama.cpp 1.04 s against 1.13 s, vLLM 0.85 s against 1.48 s, SGLang 0.46 s against 0.68 s). ollama reports the
field but credited no part of a state-varied prefix here, though it reported the full 2384 for an identical
repeat; its own cache is per loaded model runner, so `keep_alive` (native API only — its OpenAI route ignores
the field) is what keeps it warm.

Three of these servers also constrain *where* a system message can go: vLLM and SGLang answer
`400 System message must be at the beginning.` when one follows a `user` turn, and llama.cpp's Qwen template
*raises* the same sentence as a 500. jevper's own prompt always leads, so a chat-list state carrying a
`system`/`developer` turn has that content folded into the system prompt instead of being sent late — the same
request has to work on every server.

Two consequences worth knowing:

- **On a hosted API this is the difference between caching at all and not.** OpenAI's minimum cacheable prefix
  is 1024 tokens, so a prompt that reuses only its system message never caches there; with the state last, a
  prompt carrying a few examples crosses the threshold and the derived `prompt_cache_key` has something to
  route.
- **Two-step reasoning keys both passes alike.** The analysis and answer passes have different system prompts,
  so they cannot share a cache entry, but `prompt_cache_key` puts them on the same machine, which is what
  routing is for.

### Isolating a cache

`cache_salt` is the one cache field two of these servers do implement: vLLM and SGLang group reuse by salt, so
requests with the same salt share cached prefixes and requests with different salts cannot see each other's.
It is an isolation control, not a routing one, and jevper does not send it — pass it per deployment when
tenants share a server:

```python
SystemOneClient(client, model="qwen3.5-9b", extra_body={"cache_salt": tenant_id})
```

vLLM caps the salt at 128 characters and rejects `@`, `/`, `\` and NUL; llama.cpp and ollama ignore the field.

## The Messages route

All five servers here also implement the Anthropic Messages API (`POST /v1/messages`), so jevper's
`api="messages"` works against each of them — the `anthropic` SDK pointed at the same host and port as the
OpenAI one. What differs is how much of the protocol each server implements; the versions are the first
release of each project that ships the route.

| Server | Since | `thinking` field | Thinking blocks back | `usage` cache counts |
| --- | --- | --- | --- | --- |
| LM Studio | 0.4.1 | accepted, and the answer is still separated from it | `thinking` blocks when the model thinks | `cache_read_input_tokens`, including a reported `0` on a cold call |
| llama.cpp | b7187 | accepted, and the budget grows `max_tokens` as below | reported | not documented |
| vLLM | 0.11.1 | **accepted and ignored**: its request model has no `thinking` field and pydantic drops extras, so the field reaches nothing and no downgrade fires — the answer comes back with no thinking and no way to tell | `thinking` blocks | yes |
| SGLang | 0.5.9 | **refused**: this version has no `thinking` field at all, so the request is answered `400` and jevper drops the field and re-asks, reporting `debug["server_limits"]["thinking"]` | `thinking` blocks | yes |
| ollama | 0.14.0 | accepted, but `budget_tokens` is accepted and **not enforced** | `thinking` blocks | no cache fields at all |

No server returns logprobs through this route — the API has no field for one — so `method="structured"` or
`"discrete"` is how to use it, and `method="auto"` resolves to `structured` there without spending a request
to find out. `max_tokens` is required by vLLM's and SGLang's implementations and has no default on any of
them, so jevper always sends one: 1024, or 1024 plus the caller's `ReasoningConfig(budget_tokens=…)`, because
this API also requires the thinking budget to be *strictly below* `max_tokens` and would refuse the 1024 its
own documentation calls the floor. Measured on all five: a 1024 budget sends `max_tokens: 2048`, a 2048 budget
sends `3072`, and a caller's own `extra_body={"max_tokens": 4096}` still wins outright.

Anthropic has since added mid-conversation `system` messages, but no server here implements them: they render
a `system` turn positionally into the chat template, so jevper still moves it to the top-level `system` field,
where it cannot be dropped or rejected. Measured here, ollama, llama.cpp, vLLM and SGLang all answer `200` for
a `system` role inside `messages` on this route: the `400 System message must be at the beginning.` that vLLM
and SGLang give belongs to the *OpenAI* surfaces, where a state's own instruction turns would otherwise land
in the middle of the conversation.

The reasoning parsers matter here too. With thinking left on — vLLM's and SGLang's templates default to it —
the parser puts the whole generation into a thinking block and returns no text block at all, so there is no
answer to read and the call raises `IncompleteAnswerError` when the route also reports a spent budget
(`stop_reason: "max_tokens"`), naming the limit to raise. Disable thinking per call, exactly as on the other
surfaces: `extra_body={"chat_template_kwargs": {"enable_thinking": False}}` — with that, every scenario on
vLLM's Messages route answers, on both the non-thinking and the thinking model.

SGLang needs one more decision, on the server side. With `--reasoning-parser qwen3` and a *non-thinking*
model — whose template has no `enable_thinking` to set, so the parser never sees the closing thinking marker
it waits for — the whole generation is classified as reasoning and the route answers with a thinking block and
no text block. `separate_reasoning: false` does not change that; dropping `--reasoning-parser` does, and the
structured scenarios then answer normally. It is a property of serving a model that never emits the marker
with a parser that waits for it, not of the client.

One thing the fleet showed that the protocol did not say then: none of these five servers constrains the
*shape* of the answer on this route. Anthropic's own API has a field for it — `output_config.format` — and
jevper sends it wherever the server takes it, but none of these five does, so the JSON Schema also stays in
the system prompt and the answer is only as good as the model's instruction-following. Without that the
structured system prompt referred to "the provided schema" while nothing provided one, and a 4B model
answered `{"intent": "A"}` — a real answer in the wrong shape — to every structured question. A server that
*refuses* the field costs one request and is reported as `debug["server_limits"]["output_config"]`. vLLM is
the awkward case: its Messages request model has no `thinking` field either and pydantic drops what it does
not model, so a field it ignores produces no error and no verdict — which is exactly why the prompt keeps
the schema on this surface whether the field is sent or not.

## Sizing a 12 GB card

A 9B model at 4-bit is 5.5-8.5 GB of weights, which leaves room for a small KV cache but not much else. What
worked here, one server at a time (they cannot share the card):

| Server | Weights | Serve flags that fit |
| --- | --- | --- |
| ollama | `qwen3.5:9b` (Q4_K_M, 6.1 GiB) | `OLLAMA_CONTEXT_LENGTH=4096`, `OLLAMA_MAX_LOADED_MODELS=1`, `OLLAMA_NUM_PARALLEL=1` |
| llama.cpp | Q4_K_M GGUF (5.75 GiB) | `-ngl 99 --ctx-size 4096 -np 1 --jinja` |
| vLLM | 4-bit compressed-tensors AWQ (8.45 GiB) | `--language-model-only --gpu-memory-utilization 0.80 --max-model-len 1024 --max-num-seqs 1 --enforce-eager` — dropping the vision tower is what makes it fit: without it the engine dies at init, with it the server came up in 43 s and gave 1.88 GiB of KV cache. A **bf16** model (the 4B 2507 pair, 8.1 GiB) needs `VLLM_USE_FLASHINFER_SAMPLER=0`: FlashInfer's top-p/top-k sampler JIT-compiles when the engine starts, and with no CUDA toolkit on the box the engine dies with `Could not find nvcc and default cuda_home='/usr/local/cuda' doesn't exist` before it serves anything — a path the AWQ checkpoint never reaches. Disabled, it starts and says so: *FlashInfer top-p/top-k sampling disabled* |
| SGLang | same 8.45 GiB checkpoint | `--mem-fraction-static 0.85 --context-length 4096 --attention-backend triton --sampling-backend pytorch --disable-cuda-graph` — the weights alone exceed 0.80 of the card (SGLang's own guard says so), and FlashInfer's JIT cannot build here, so SGLang's Triton kernels are required; unlike vLLM it has no `--language-model-only` for this architecture, so the vision tower cannot be dropped |

Neither builder sends `max_tokens`, so a chatty model can generate far more than the answer needs. Bound it
per call with `extra_body={"max_tokens": 512}` — the one field worth setting on a small card. With thinking
left on, that cap is spent on the reasoning first: ollama answered a 512-token cap with an empty `content` and
`finish_reason: "length"`, which reaches the caller as an `IncompleteAnswerError` naming the budget
(`finish_reason: "length"` on Chat Completions, `stop_reason: "max_tokens"` on the Messages route,
`status: "incomplete"` with `incomplete_details.reason: "max_output_tokens"` on the Responses surface).
Disable thinking *and* bound the output.
