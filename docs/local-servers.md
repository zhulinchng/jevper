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

`api="auto"` (the default) works against all four: it prefers the Responses surface, and when that route is
missing — or answers without carrying logprobs through — it re-asks on Chat Completions and remembers the
verdict. Passing `api="chat_completions"` skips the discovery entirely.

## Thinking is the one decision you must make

`logprobs`, `grammar` and `discrete` all read a **one-token answer**, and every one of these servers reports
logprobs for *every* generated token. With thinking on, that is the first token of the reasoning, not the
label: the readout raises `LabelReadoutError`, spends one corrective retry, and raises again. Turn thinking
off for classification work — it costs a whole reasoning pass to choose one letter.

| Server | Request field that turns it off | Also available |
| --- | --- | --- |
| ollama | `reasoning_effort: "none"` | `OLLAMA_CONTEXT_LENGTH`, per-model `think` on the native API |
| llama.cpp | `chat_template_kwargs: {"enable_thinking": false}` | `reasoning_effort: "none"` works too, since `--jinja` runs the model's own template; `reasoning_budget: 0`; `--reasoning-format deepseek` splits the trace into `reasoning_content` |
| vLLM | `chat_template_kwargs: {"enable_thinking": false}` | `reasoning_effort` — but only with the values its parser accepts |
| SGLang | `chat_template_kwargs: {"enable_thinking": false}` | `reasoning_effort` — but only with the values its parser accepts |

If you keep thinking on, `structured` and `discrete` are unaffected — they read the answer text, and the
trace lands in `response.reasoning` (`message.reasoning` on ollama, `reasoning_content` on the others;
jevper reads all three). A label readout also survives it when the server separates the trace *and* the token
stream ends exactly with the answer text: jevper anchors on that tail and reads the answer's own first token.
That anchor is deliberately strict — anything else is reported rather than guessed — with one allowance:
vLLM and SGLang append their own end-of-turn token (`<|im_end|>`) to the logprob stream after the answer, so
up to two trailing tokens that cannot be part of the answer are dropped before the tail is tested.

## What each server ignores, and what it rejects

Unknown fields are accepted and dropped by all four, so a field that does not apply is not an error:

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

## What each server actually answers

Recorded with raw HTTP against each server — 45 requests each, and the responses jevper has to read are kept
in `tests/fixtures/providers/`. The shapes matter more than the verdicts: they are what a client has to
tolerate, and the fixture tests replay them on every change.

| | ollama 0.34.3 | llama.cpp b11139 | vLLM 0.30.1 | SGLang 0.5.20 |
| --- | --- | --- | --- | --- |
| `/v1/responses` route | yes | yes | yes | yes |
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

## Sizing a 12 GB card

A 9B model at 4-bit is 5.5-8.5 GB of weights, which leaves room for a small KV cache but not much else. What
worked here, one server at a time (they cannot share the card):

| Server | Weights | Serve flags that fit |
| --- | --- | --- |
| ollama | `qwen3.5:9b` (Q4_K_M, 6.1 GiB) | `OLLAMA_CONTEXT_LENGTH=4096`, `OLLAMA_MAX_LOADED_MODELS=1`, `OLLAMA_NUM_PARALLEL=1` |
| llama.cpp | Q4_K_M GGUF (5.75 GiB) | `-ngl 99 --ctx-size 4096 -np 1 --jinja` |
| vLLM | 4-bit compressed-tensors AWQ (8.45 GiB) | `--language-model-only --gpu-memory-utilization 0.80 --max-model-len 1024 --max-num-seqs 1 --enforce-eager` — dropping the vision tower is what makes it fit: without it the engine dies at init, with it the server came up in 43 s and gave 1.88 GiB of KV cache |
| SGLang | same 8.45 GiB checkpoint | `--mem-fraction-static 0.85 --context-length 4096 --attention-backend triton --sampling-backend pytorch --disable-cuda-graph` — the weights alone exceed 0.80 of the card (SGLang's own guard says so), and FlashInfer's JIT cannot build here, so SGLang's Triton kernels are required; unlike vLLM it has no `--language-model-only` for this architecture, so the vision tower cannot be dropped |

Neither builder sends `max_tokens`, so a chatty model can generate far more than the answer needs. Bound it
per call with `extra_body={"max_tokens": 512}` — the one field worth setting on a small card. With thinking
left on, that cap is spent on the reasoning first: ollama answered a 512-token cap with an empty `content` and
`finish_reason: "length"`, which reaches the caller as a malformed answer whose message names the budget
(`finish_reason: "length"` on Chat Completions, `status: "incomplete"` with
`incomplete_details.reason: "max_output_tokens"` on the Responses surface). Disable thinking *and* bound the
output.
