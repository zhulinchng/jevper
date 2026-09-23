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
| llama.cpp | `chat_template_kwargs: {"enable_thinking": false}` | `reasoning_budget: 0`; `--reasoning-format deepseek` splits the trace into `reasoning_content` |
| vLLM | `chat_template_kwargs: {"enable_thinking": false}` | `reasoning_effort` |
| SGLang | `chat_template_kwargs: {"enable_thinking": false}` | `reasoning_effort` |

If you keep thinking on, `structured` and `discrete` are unaffected — they read the answer text, and the
trace lands in `response.reasoning` (`message.reasoning` on ollama, `reasoning_content` on the others;
jevper reads all three). A label readout also survives it when the server separates the trace *and* the token
stream ends exactly with the answer text: jevper anchors on that tail and reads the answer's own first token.
That anchor is deliberately strict — anything else is reported rather than guessed.

## What each server ignores

Unknown fields are accepted and dropped by all four, so a field that does not apply is not an error:

- `grammar` (jevper's `method="grammar"`) is a llama.cpp convention. ollama, vLLM and SGLang ignore it, so
  the model answers unconstrained and the label readout reports a non-label first token instead of a grammar
  failure. Use `logprobs` or `structured` there.
- `strict: true` inside `json_schema` is ignored by ollama and honoured by vLLM and SGLang.
- `reasoning_effort` is a no-op on llama.cpp (it has no such field) and is accepted by the others.

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
`finish_reason: "length"`, which reaches the caller as a malformed answer. Disable thinking *and* bound the
output.
