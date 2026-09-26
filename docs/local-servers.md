# Local servers

jevper talks to anything that speaks the OpenAI wire format, so a local server is the same client with a
different `base_url`. Three things differ between servers, and all three are decided by what you pass rather
than by the library: **which surface carries logprobs**, **how thinking is turned off**, and **which request
fields the server quietly ignores**.

Unless a row or paragraph says otherwise, server-behaviour results and measurements on this page come from
one recorded run on **2026-09-25**, against Ollama 0.34.3, llama.cpp b11139, vLLM 0.30.1, SGLang 0.5.20 and
LM Studio 0.4.1. The main sweep served `Qwen3.5-9B` at 4-bit on one remote 12 GB GPU box; prose or a row that
names a 4B model identifies that explicit exception. The consumer harness in `examples/incident-triage/` ran
as `python -m incident_triage.cli sweep --server <server> --full --out /tmp/<server>.json`. Its `sweep`
command defines 38 scenarios and runs each only on the surfaces and methods it declares. These are
version-specific observations, not a promise about a later server build. Statements about what jevper sends,
normalizes or raises are code behaviour; historical "first release" values marked unmeasured were not verified
from this repository. Hosted OpenRouter and OpenAI observations name that endpoint and are not measurements
of the local GPU box.

In a harness response, `usage.n_calls` counts successful provider results. Failed attempts and fallback
probes are recorded in `debug["llm_attempts"]`, so `n_calls` is not a count of every wire request.

```python
from openai import OpenAI
from jevper import Choice, SystemOneClient

client = SystemOneClient(
    OpenAI(base_url="http://127.0.0.1:11434/v1", api_key="local"),
    model="qwen3.5:9b",
    extra_body={"reasoning": {"effort": "none"}},  # Responses thinking off; see below
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

## Ollaya, the decision server

[Ollaya](https://github.com/ollaya-dev/ollaya) (lowercase, no second `l`) is **not** the Ollama in the table
below. It serves open *decision* models — models trained to answer a typed question about a state in one
forward pass, with no generation at all — and its `/v1` API is declared wire-identical to TypeSafe's. That
is the format jevper's `api="systemone"` surface already speaks, so ollaya is reached by base URL like any
other server, and its `GET /v1/models` returns TypeSafe's own `{"models": [...]}` shape, which is the one
`list_models()` reads. A gateway that answers that path OpenAI-style is reported as the wrong shape; ollaya
is not such a gateway.

Install, pull and serve, then point a client at it:

```bash
curl -fsSL https://ollaya.dev/install.sh | sh      # 0.7.0; no root installs to ~/.local
export PATH="$HOME/.local/bin:$PATH"
ollaya pull laya:en
ollaya pull laya:typed-decisions
ollaya serve                                       # 127.0.0.1:11435
```

```python
from openai import OpenAI
from jevper import Noul, SystemOneClient

with SystemOneClient(
    OpenAI(base_url="http://127.0.0.1:11435/v1", api_key="ollama"),
    model="laya:en",
    api="systemone",
) as client:
    response = client.system_one(
        state="I was charged twice for my subscription this month.",
        questions={"refund": Noul(instructions="Does the customer ask for a refund?")},
    )

response.answers["refund"].noul   # 0.8551, measured 2026-09-26
```

The `api_key` is never checked unless the daemon was started with `OLLAYA_API_KEY` set; any non-empty
string works. Three questions of mixed type travel in one request, as on any System One call.

### The two routes sit at different depths

Ollaya serves the TypeSafe routes under `/v1` and its own native endpoint, `/api/decide`, at the server
root. An SDK client appends its path to whichever base URL it was handed, so one client object reaches
exactly one of them — measured 2026-09-26 against ollaya 0.7.0:

| `base_url` | `SYSTEMONE_PATH` → | `MODELS_PATH` → | `/api/decide` → |
| --- | --- | --- | --- |
| `http://127.0.0.1:11435/v1` | `/v1/systemone` 200 | `/v1/models` 200 | `/v1/api/decide` **404** |
| `http://127.0.0.1:11435` | `/systemone` 404 | `/models` 404 | `/api/decide` 200 |

So a caller who wants decisions *and* the model list points two client objects at the one server. The 404
is ollaya's own, reported as `ProviderError`:
`{'error': '/v1/api/decide not found', 'code': 'NOT_FOUND'}`.

### The native endpoint, with `native=True`

`/api/decide` takes the same request body and answers in the same format, then adds a report on the
request that produced it. Build the client with `native=True` and jevper posts there and returns a
`NativeSystemOneResponse`:

```python
with SystemOneClient(
    OpenAI(base_url="http://127.0.0.1:11435", api_key="ollama"),
    model="laya",                       # a router: picks laya:en or laya:multilingual
    api="systemone",
    native=True,
) as client:
    response = client.system_one(state=state, questions=questions)

response.routing.route      # 'english' — the stable key to branch on
response.routing.model      # 'laya:en' — the checkpoint that answered
response.state_truncated    # whether part of the state was dropped to fit the context
response.eval_duration      # 21_954_691 ns, the service's own nanoseconds, passed through as sent
response.done_reason        # 'decide'
```

`response.model` stays the name the caller asked for — `laya` above, not `laya:en`. On the wire ollaya's
`model` is the checkpoint that answered, and reading that instead would make one field mean two things
across jevper's surfaces; `routing.model` carries it. `/v1/systemone` reports no routing at all, so
`native=True` is the only way to see it.

Two more fields are that route's own, and jevper refuses them on the TypeSafe one by name rather than
dropping them in silence:

* `extras=["laya"]` adds a `laya` object to **every** answer, carrying the model's own confidence
  (`1 − H(p)/ln K`, or `max(p, 1 − p)` for a noul) beside TypeSafe's, plus `act_probability`. The two
  confidences are different numbers from different heads and are not interchangeable — measured on
  `laya:en`, 0.615 against 0.8496 for the same choice. `act_probability` is `None` for a model with no
  act head, which the service reports as null. `"laya"` is a closed set that belongs to ollaya, so
  jevper passes the caller's value through and an unknown one is the server's to reject — 400,
  `Input should be 'laya'`, naming the expected value.
* `keep_alive` is Ollama's model lifecycle control in that server's own units. `0` unloads the model and
  is sent like any other value.

### A noul with nothing to judge

jevper refuses a noul carrying neither instructions nor criteria, because the hosted Jev service answers
400 for one. **Ollaya answers one**, reading the question id in place of the missing instruction —
measured 200 on both of its routes. A caller who wants that reach for it:

```python
SystemOneClient(provider, model="laya:en", api="systemone",
                 native=True, noul_requires_question=False)
```

The default is unchanged, so a caller on the hosted service never has to ask for its refusal back.

### What ollaya refuses, and at what size

These are the answering model's own budgets, measured 2026-09-26 against `laya:en`, not jevper's. jevper
surfaces each refusal as the server wrote it:

| Limit | Measured | Refusal |
| --- | --- | --- |
| Options in one question | **127** with a 4-character instruction, 126 with a 100-character one | 422 `TOO_MANY_OPTIONS` — "128 options do not fit the option budget of laya:en". The budget is the whole question, so the instruction spends it too |
| Options in one question, absolute | 255 | 422 `INVALID_REQUEST` — "Dictionary should have at most 255 items" |
| Questions in one request | **256** (11 and 64 answered; 257 refused) | 422 `INVALID_REQUEST` — "Dictionary should have at most 256 items" |
| `state` | **65,536 tokens** (200,000 characters answered; 1,000,000 refused) | 422 `INPUT_TOO_LONG` — "state is 200001 tokens long; the limit is 65536" |

jevper enforces none of these: they are properties of the model that answered, and a rubric sized for one
server is not sized for another.

### What was measured, and on what

All the figures on this section come from one run on **2026-09-26** against **ollaya 0.7.0** on the same
remote box the rest of this page uses: Ubuntu 24.04.5 under WSL2, x86_64, glibc 2.39, driver 617.14, one
RTX 3080 (12 GB). `laya:en` and `laya:typed-decisions` are 854 MB each; the `laya` router is 11 KB and
pulls `laya:multilingual` (684 MB) as its second target. The daemon chose **F16 on `cuda:0`**, which
`ollaya ps` reports and which is worth knowing: ollaya's own model card notes the F16 graph can differ
from F32 when the top two options are within 0.01 of each other, so a near-tied answer is not stable
across precisions. All of it is jevper 0.7.10 driven through the public API, plus raw HTTP for the tables
above.

`laya:en` is a 421M-parameter ModernBERT-large, ONNX, onnxruntime, with `choice` and `score` capabilities;
`laya:typed-decisions` is the same architecture fine-tuned on the typed-decisions workflows, which its own
model list describes as **0.766** accuracy. For the Jev side of that comparison, see
[`jev-comparison.md`](jev-comparison.md).

## What to pass per server

| Server | `base_url` | `model` | Thinking off | Notes |
| --- | --- | --- | --- | --- |
| ollama | `http://127.0.0.1:11434/v1` | the tag you pulled, e.g. `qwen3.5:9b` | `extra_body={"reasoning": {"effort": "none"}}` on Responses | Chat Completions carries logprobs; the Responses route exists but returns an empty logprob list. Use `reasoning_effort` only on Chat Completions; jevper does not translate it to the Responses field |
| llama.cpp | `http://127.0.0.1:8080/v1` | the `--alias` value | `extra_body={"chat_template_kwargs": {"enable_thinking": False}}` | serve with `--jinja` for the model's own template; the only measured server of the five that takes jevper's GBNF `grammar` field, so `method="grammar"` answers there |
| vLLM | `http://127.0.0.1:8000/v1` | the `--served-model-name` value | `extra_body={"chat_template_kwargs": {"enable_thinking": False}}` | serve with `--reasoning-parser qwen3`; its `top_logprobs` limit is configured by `--max-logprobs` |
| SGLang | `http://127.0.0.1:30000/v1` | the `--served-model-name` value | `extra_body={"chat_template_kwargs": {"enable_thinking": False}}` | serve with `--reasoning-parser qwen3`; its Responses logprob readout needed `top_logprobs` explicitly. jevper sends it for `logprobs` and `grammar`, but not for `structured` or `discrete` |
| LM Studio | `http://127.0.0.1:1234/v1` | the id `lms ls` prints, e.g. `qwen3-4b-instruct-2507` | nothing does — load a non-thinking model | all three surfaces on one box, as do the other four now; `logprobs` arrives on both OpenAI surfaces; its Responses route accepts `text.format` and ignores it, so structured answers belong on Chat Completions |

All five answered the Anthropic Messages route (`/v1/messages`) in the recorded run, but that route needs the
`anthropic` client and `api="messages"`; an OpenAI client does not expose it for `api="auto"` discovery.

jevper passes `extra_body={"stream": 0}` through because only a truthy `stream` is refused locally. The
provider's response is reported as `ProviderError`; the exact status and text are server behaviour. Omit the
field, or send `false`.

Both facades have been run against every server here, and `AsyncSystemOneClient` has been run live against
ollama's Responses route and against a hosted OpenAI-compatible endpoint (OpenRouter): a batch of three
question types, two concurrent calls on two surfaces, a 26-option question and a `discrete` answer all come
back the same way the blocking client returns them.

Those hosted observations are endpoint-specific; the local blocking and async results are the versioned
server results covered by the conditions above.

`api="auto"` (the default) prefers Responses when that client surface exists, and when the route is missing
— or answers without carrying logprobs through for a label readout — it re-asks on Chat Completions and
remembers the verdict. Passing `api="chat_completions"` skips discovery.

### What a label readout costs on a wide question

A `Choice` of 26 options is the widest a label readout can be asked for (past 26 the labels are two
letters and `method="auto"` answers in JSON instead). In the recorded 4B-model runs, vLLM and LM Studio
answered the label prompt with prose — the first token came back as `no` — so jevper raised
`LabelReadoutError` after its corrective retry. SGLang answered the same 26-label question with a label,
and jevper read all 26 probabilities from it. `method="structured"` does not need a logprob distribution;
the recorded SGLang and OpenRouter runs returned all 26 keys.

For `Score`, `normalize_probabilities=False` preserves the provider's numbers in the returned
`probabilities`, but jevper still computes `score` from a rescaled distribution. Rescaling is a no-op for
the two displayed distributions, so their expectations remain `0.9` for `{0: 0.1, 1: 0.9}` and `2.9` for
`{2: 0.2, 3: 0.7, 4: 0.1}`.

A structured `Noul` readout requires `noul` in `[0, 1]`; the public `NoulAnswer.noul` field is an
unrestricted float. In the recorded Ollama probe, the model generated `{"noul": 521304}`, and the structured
readout raised `'noul' must be in [0, 1.0], got 521304.0`; the `discrete` follow-up answered `true`.

### A cache key longer than OpenAI's cap

`prompt_cache_key` is capped at 64 characters by OpenAI and by the OpenResponses schema, and jevper
enforces 256 locally. None of these five servers refuses a longer one: vLLM answered with a 64-, a 72-
and a 256-character key (one request each), and OpenRouter — which has no such cap to enforce — answered
a 72-character key too. The field is the provider's to validate; jevper's own bound is there so a typo
cannot become an unbounded string on the wire.

## Thinking is the one decision you must make

`logprobs` and `grammar` read a **one-token answer**, and the measured label and grammar scenarios received
logprobs for generated tokens. With thinking on, a label can start in the reasoning trace rather than at the
label: the readout raises `LabelReadoutError`, spends one corrective retry, and raises again. `structured`
and `discrete` do not need logprobs, but thinking can consume the output budget or leave no answer text, so
turn it off for classification work.

| Server | Request field that turns it off | Also available |
| --- | --- | --- |
| ollama | `reasoning_effort: "none"` on Chat Completions; on the **Responses route** that field is ignored and `reasoning: {"effort": "none"}` is what turns it off — measured, with the wrong one the answers arrive carrying the trace and sometimes with no answer text at all | `OLLAMA_CONTEXT_LENGTH`, per-model `think` on the native API |
| llama.cpp | `chat_template_kwargs: {"enable_thinking": false}` | `reasoning_effort: "none"` works too, since `--jinja` runs the model's own template; `reasoning_budget: 0`; `--reasoning-format deepseek` splits the trace into `reasoning_content` |
| vLLM | `chat_template_kwargs: {"enable_thinking": false}` | `reasoning_effort` — but only with the values its parser accepts |
| SGLang | `chat_template_kwargs: {"enable_thinking": false}` | `reasoning_effort` — but only with the values its parser accepts |
| LM Studio | none reliably: omitting the field leaves the model's own template in charge, and `reasoning_effort: "none"` is ignored on its chat route (bug-tracker #2413; seen here — the trace still arrived, 1016 characters of it) | load an instruct model instead — `qwen3-4b-instruct-2507` answers without a reasoning trace |

If you keep thinking on, `structured` and `discrete` still read answer text rather than a token
probability, but thinking can spend the output budget or leave no answer. When returned, the trace lands in
`response.reasoning`; jevper reads the Chat fields `reasoning_content`, `thinking`, and `reasoning`. A label
readout also survives it when the server separates the trace *and* the token stream ends exactly with the
answer text: jevper anchors on that tail and reads the answer's own first token.
That anchor is deliberately strict — anything else is reported rather than guessed — with one allowance:
vLLM and SGLang append their own end-of-turn token (`<|im_end|>`) to the logprob stream after the answer, so
up to two trailing tokens that cannot be part of the answer are dropped before the tail is tested.

## Re-verified on 2026-09-26

A second run over the same box, after the System One surface landed, to check that nothing about the
prompt surfaces had moved. Every server below was driven through jevper's public API with a
three-question call (a noul, a choice, a score), the method matrix, four `state` shapes, a 30-option
choice with non-ASCII labels, examples, async parity, and the refusals — 25 or 26 of the same 26
checks against each, against jevper 0.7.6. vLLM, SGLang and LM Studio were each run twice — the
second time after a reboot of the box, to check the launch recipes as well as the results — and the
timings agree to within noise.

| Check | Ollama 0.34.3 | llama.cpp | vLLM 0.30.1rc1 | SGLang 0.5.20 | LM Studio 0.0.25 |
| --- | --- | --- | --- | --- | --- |
| `api="auto"` | probes Responses, answers on Chat Completions: `method=logprobs`, 6 calls for 3 questions | probe aborted, see below | `api=responses`, `method=logprobs`, **3 calls**, 1.2 s | `api=responses`, `method=logprobs`, **3 calls**, 1.1 s | `api=responses`, `method=logprobs`, **3 calls**, 0.5 s |
| `api="chat_completions"` | `method=logprobs`, 3 calls, 387 input tokens | `method=logprobs`, 3 calls | `method=logprobs`, 3 calls, 393 input tokens, 0.7 s | `method=logprobs`, 3 calls, 393 input tokens, 1.1 s | `method=logprobs`, 3 calls, 366 input tokens, 0.2 s |
| `api="responses"` | answers with `method=structured` — its logprob list comes back empty | not usable, see below | `api="auto"` selects it and reads logprobs off it; the explicit check did not run (the route's readiness request timed out) | `method=logprobs`, 3 calls | `method=logprobs`, 3 calls |
| `method="grammar"` | reads a distribution | **fails on the 4B, see below** | reads a distribution | reads a distribution | reads a distribution |
| 30-option choice, non-ASCII labels | works | works | sum 1.0000, `team-00` | sum 1.0000, `team-00` | sum 1.0000, `team-00` |
| `list_models()` | not this library's shape: `MalformedAnswerError`, an OpenAI-style `data` list | **parses**: 1 model, the GGUF path | `MalformedAnswerError`, an OpenAI-style `data` list | `MalformedAnswerError`, an OpenAI-style `data` list | `MalformedAnswerError`, an OpenAI-style `data` list |

Three things this run settled that the 2026-09-25 pass could not.

**`grammar` works wherever the server's `top_logprobs` name the alternatives.** It read a real
distribution on ollama, vLLM, SGLang and LM Studio today. llama.cpp is the one server where it failed,
and only on the 4B: `_LogprobsUnavailable: the provider returned 20 top_logprobs for the answer token
'A' (method='grammar'), none of which was an alternative`. The grammar itself was honoured — the answer
was a valid label — but a confident small model does not put the losing label in its top 20, so there is
no distribution left to read. The same server's 9B read one on 2026-09-25, so this is a property of the
model behind the server, not of the server. Use `logprobs` or `structured` where the readout cannot find
the alternatives.

**llama.cpp's `/v1/responses` is not usable for this wire format.** Asked for a structured answer, it
returns a label where the schema wants a probability, and jevper refuses it: `MalformedAnswerError:
'noul' must be a finite number, got 'A'`. Reproduced on two different models (a 9B Q4_K_M and a 4B
Q4_K_M), so it is the shim's answer rather than one model's mistake. Its Chat Completions route
serves the same three questions correctly.

**A thinking trace that reaches the answer is a launch fact, not a reading fact.** Started without a
thinking switch, vLLM's first answer began with the literal token `Thinking`, and jevper refused it as
`LabelReadoutError: first non-whitespace token 'Thinking' is not one of the labels ['A', 'B']`. With
`extra_body={"chat_template_kwargs": {"enable_thinking": false}}` — the recipe above — the same server
answers all three questions in 1.2 s. The refusal is the point: a trace that leaked into the answer is
visible rather than scored.

The `api="auto"` cost on Ollama is the documented one: it serves `/v1/responses` but its logprob list
comes back empty there, so the ladder probes it, finds no distribution to read, and moves to Chat
Completions. Six requests for three questions is three answers plus three probes, and the probes are
in `debug["llm_attempts"]`; `usage.n_calls` counts the results, not the wire. vLLM, SGLang and LM Studio
answer on the Responses route on the first try, so `auto` costs them nothing.

### The output budget is bounded below by the thinking and above by the context window

An 18-scenario edge matrix was run over all five servers at the end of 2026-09-26, against jevper
0.7.8, and the one thing that needed tuning was the probe's own output budget. It is worth writing
down because both ends of it are the server's, not jevper's.

At **512** output tokens the always-thinking Qwen3 templates truncated before answering on nearly
every scenario — `IncompleteAnswerError` on ollama, llama.cpp, vLLM, SGLang and LM Studio alike, which
reads like a library fault and is not one. At **2048** that is gone on ollama and vLLM, and what
remains on llama.cpp, SGLang and LM Studio is the thinking itself: `docs/reasoning.md` records that
those templates ignore `enable_thinking: false`, so a thinking span is charged to the output budget.
jevper's verdict is the useful one either way —
`IncompleteAnswerError: the provider ran out of output tokens before the answer was complete
('max_output_tokens')` — because it names the knob and the server's own reason.

At **4096** the same three servers answer `400 Requested token count exceeds the model's maximum
context length of 4096 tokens. You requested a total of 4237 tokens`. They are all launched with
`--max-model-len 4096`, so a 4096-token *output* request leaves nothing for the prompt. 2048 is
therefore the right budget for a 4096-token window, and the useful rule is the window's own: keep
`max_output_tokens` well under `max_model_len`, and treat an `IncompleteAnswerError` naming
`max_output_tokens` as "raise the budget, if the window allows it" rather than as a defect.

The Messages scenarios `messages-budget-with-room` and `messages-thinking-temperature` refuse on vLLM
and SGLang for the same reason and for one more: a thinking budget is added to the output budget,
because that is Anthropic's rule — the budget must sit strictly below `max_tokens` — so a 2048 budget
on top of a 2048 output request is 4096, which is the whole window. jevper cannot know a server's
window, so the server's 400 arrives as a `ProviderError` carrying its own status and wording.

### Launching these three on a 12 GB card

Measured on the RTX 3080 test box under WSL2, where vLLM's own free-memory query reports **10.84 of
12.0 GiB** and refuses its default `--gpu-memory-utilization 0.92` before it loads anything:

```
ValueError: Free memory on device cuda:0 (10.84/12.0 GiB) on startup is less than desired GPU
memory utilization (0.92, 11.04 GiB).
```

That gap is not stale allocations. `nvidia-smi` reported 12,091 MiB free on a freshly restarted box
with nothing else running, and vLLM still saw 10.84 GiB — the two disagree by about a gigabyte, and
vLLM gates on its own view. Rebooting does not raise the ceiling, so budget from the number in the
error rather than from `nvidia-smi`.

Three settings get there, and dropping any one of them puts the failure back:

```bash
# vLLM 0.30.1rc1 — 7.55 GiB of weights, 14,043-token KV cache; 28 s to first token after a reboot
vllm serve ~/models/Qwen3.5-9B-AWQ-4bit --port 8000 \
  --max-model-len 4096 --gpu-memory-utilization 0.87 --enforce-eager \
  --max-num-batched-tokens 1024 --max-num-seqs 2 --reasoning-parser qwen3 \
  --limit-mm-per-prompt '{"image":0,"video":0}'

# SGLang 0.5.20 — ready in 4 s once the kernel cache exists
python -m sglang.launch_server --model-path ~/models/Qwen3.5-9B-AWQ-4bit --port 30000 \
  --mem-fraction-static 0.85 --context-length 4096 --max-running-requests 1 \
  --max-mamba-cache-size 8 --reasoning-parser qwen3 --disable-cuda-graph \
  --attention-backend triton --sampling-backend pytorch
```

- **`--limit-mm-per-prompt '{"image":0,"video":0}'`** (or `--language-model-only`): the AWQ checkpoint is
  a `Qwen3_5ForConditionalGeneration`, so the vision tower's profiling budget is spent before the KV
  cache is allocated. At `--gpu-memory-utilization 0.85` without it, the engine dies with
  `ValueError: No available memory for the cache blocks.`
- **SGLang's two backend flags**: with the default flashinfer backends the server JIT-builds a CUDA
  kernel through ninja, and that build fails on this box with
  `nvcc warning: incompatible redefinition for option 'compiler-bindir'` followed by
  `RuntimeError: Ninja build failed`. `--attention-backend triton --sampling-backend pytorch` keeps it
  off that path entirely.
- **SGLang's CLI spelling**: this build's console script is `sglang serve`, and
  `sglang launch_server` is refused by argparse (`invalid choice`) before anything starts.
  `python -m sglang.launch_server` is the form that works.

One more readiness trap, in the same family as llama.cpp's: **vLLM's engine can die silently in
WSL2**, seconds after it reports a KV cache size — `RuntimeError: Engine core initialization failed.
Failed core proc(s): {}`, with no traceback and nothing in `dmesg`. It happened twice on the day of
this run, and the identical command served correctly on the next attempt, so a driver that gives up on
the first failure will report a server that works. Retry the launch, keep the server log and the
client-side log in different files so a crash leaves evidence, and wait on a real completion rather
than `/v1/models`.

One more trap specific to vLLM, and it is a trap for a readiness *ping* rather than for jevper: a bare
`POST /v1/responses {"model": …, "input": "hi"}` never answers — it times out at 25 s, twice on this
box. The route itself is fine, because `api="auto"` selects it and reads logprobs off it
(`api=responses method=logprobs`, 3 calls, 1.1 s). So a bare-ping timeout on that route means nothing
about whether jevper can use it, and waiting on one will hang a sweep that would otherwise pass.

### What the probe could not decide

Four or five of the probe's checks reported `ok: false` against servers that were otherwise clean, the
same way every time, so none of them is a server result: a dead-port check that rewrites `base_url` only
for non-loopback hosts (against `127.0.0.1` it aimed at the live server), an "empty questions" case that
never emptied the question map, a one-option `Choice` that this library accepts by design — the Jev API
documents a 255-option maximum and no minimum, and `tests/test_client_behaviour.py` says so — and two
async checks that expect the refusal at a different point than the one it surfaces. They are artefacts of
the throwaway probe, not of the servers and not of jevper.

## What each server ignores, and what it rejects

Unknown fields are accepted and dropped by all five, so a field that does not apply is not an error:

- `grammar` (jevper's `method="grammar"`) is a llama.cpp convention, and llama.cpp honours the GBNF label
  grammar. What differs is whether jevper can read a distribution afterwards, which depends on the
  model's `top_logprobs` naming the other option: measured on a 9B and on a 4B that was refused with
  `_LogprobsUnavailable` (see the 2026-09-26 section). ollama, vLLM, SGLang and LM Studio accept the
  field and read a distribution off it. Prefer `logprobs` or `structured` where the readout reports a
  non-label first token instead.
- `strict: true` inside `json_schema` is ignored by ollama and honoured by vLLM and SGLang.
- `reasoning_effort` reaches the chat template on llama.cpp, ollama and SGLang; vLLM validates it against its
  own enum and answers `400` for a value outside it (`xhigh` and `max` are the usual casualties).
- `max_completion_tokens` is ignored by ollama, which only knows `max_tokens` — bound output with
  `max_tokens`.
- `n` is rejected outright by llama.cpp (`1 <= value <= 1`); the others accept it, and vLLM and SGLang then
  return two choices where jevper reads the first.
- A body field the caller's own types strictly is the caller's own business. In the recorded run,
  `extra_body={"stream": 0}` produced these provider errors: llama.cpp answered `400 Field 'stream': type
  must be boolean, but is number`; Ollama answered `400 invalid stream value: json: cannot unmarshal
  number into Go value of type bool`; LM Studio answered `400 Expected boolean, received number`; and vLLM
  accepted it. jevper passes `0` through and reports the provider failure as `ProviderError`. It refuses
  only a truthy `stream` locally; omit the field, or send `false`.
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
- `top_logprobs: 20` — the documented maximum — is taken by all five and answered with twenty
  alternatives for the answer token, so a label readout does not have to settle for five. A
  *smaller* `top_logprobs` can leave the labels out of the list altogether: ollama asked for 5 on
  a five-option question and returned five alternatives, none of them a label, which jevper
  reports as "none of which was an alternative among the options" rather than reading a label out
  of a token that is not one. The default is 20 for exactly this reason. `n: 2` is a different
  story: vLLM and SGLang return two choices, ollama and LM Studio accept the field and answer
  once, and llama.cpp refuses it outright when it is serving a single slot (`400 n must be between
  1 <= value <= 1`), which the recipe above does.

## What each server actually answers

The following server responses were recorded with raw HTTP under the date, builds, model and hardware
conditions stated above. The pruned fixtures in `tests/fixtures/providers/` preserve the response shapes
that jevper reads and are replayed by fixture tests; they are not a request manifest. The shapes matter more
than the verdicts: they are what a client has to tolerate.

| | ollama 0.34.3 | llama.cpp b11139 | vLLM 0.30.1 | SGLang 0.5.20 |
| --- | --- | --- | --- | --- |
| `/v1/responses` route | yes | yes | yes | yes |
| `/v1/messages` route | yes | yes | yes | yes |
| logprobs on Chat Completions | yes | yes | yes | yes |
| logprobs on Responses | **empty list** | **`400`** | yes, with `include` | yes, with `top_logprobs` |
| `top_logprobs` above 20 | `400` (`must be between 0 and 20`) | accepted | `400` | accepted |
| `top_logprobs: 20` | 20 alternatives | 20 alternatives | 20 alternatives | 20 alternatives |
| unknown model id | `404` naming it | ignored, `200` | `404` naming it | Chat ignored, Responses `404` |
| `n: 2` | accepted, one choice back | `400` (`n` must be 1 — the recipe serves one slot) | accepted, two choices | accepted, two choices |
| `developer` role | accepted | accepted | accepted | **`400`** |
| `output_config` on `/v1/messages` | accepted; the trace spends the budget, so nothing returns to enforce | accepted, ignored | **enforced**, invalid `format.type` a `400` | accepted; the trace spends the budget, so nothing returns to enforce |
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

## The OpenResponses route

`/v1/responses` is not one wire format but two that share a path: OpenAI's Responses API and the
[OpenResponses](https://www.openresponses.org) specification (release `2026-04-24`, as cited by that page).
vLLM describes its route as aligned with the specification; LM Studio's implementation history is unverified
here. jevper's `api="responses"` reads both and sends the portable form: every input turn carries the item
`type` the spec's union requires, with its content as a plain string, which that union also allows.
The table records what each server did with exactly that in the dated raw-HTTP run. The reasoning knobs were
asked for with `reasoning: {"effort": "none"}` and the logprob carrier with
`include: ["message.output_text.logprobs"], top_logprobs: 5`.

| | ollama 0.34.3 | llama.cpp b11139 | LM Studio 0.4.1 | vLLM 0.30.1 | SGLang 0.5.20 |
| --- | --- | --- | --- | --- | --- |
| input items with `type: "message"`, content as a string | 200 | 200 | 200 | 200 | 200 |
| the same with `input_text` content parts | 200 | 200 | 200 | 200, but the model receives the parts as objects and answers from them (their input normalization is catching up, #35508) — the string form is the one that renders | 200 |
| strict `text.format` `json_schema` with `minimum: 0` | 200, `strict` ignored | 200, field ignored | 200, field ignored | 200, **enforced** | 200, **enforced** |
| `include` + `top_logprobs: 5` | `logprobs: []` | **`400 top_logprobs requires logprobs to be set to true`** | real logprobs on the `output_text` part, one entry per token, each with its `bytes` | real logprobs on the part | real logprobs on the part, one entry per generated token — the thinking trace included (1.1 MB for one answer) |
| `max_output_tokens: 8` | `completed`, reasoning only | `completed`, reasoning only, no message | `completed` — a cut-off answer, or one in the wrong shape when jevper's own question is asked (the schema field is ignored there) | `incomplete` / `max_output_tokens` | `incomplete` / `max_output_tokens` |
| unknown model id | `404` naming it | ignored, `200` | ignored, `200` | `404` naming it | `404`, `invalid_request_error`, `code: 404` |
| `reasoning.encrypted_content` in `include` | 200 | 200 | 200 | 200 | 200 |
| reasoning items | `summary` absent, `content: [{"type": "reasoning_text"}]` | the same, plus `status: "completed"` | not returned | `summary: []`, `content: [{"type": "reasoning_text"}]` | `summary` absent, `content: [{"type": "reasoning_text"}]` |

Enforcing a schema is not the same as answering with it, and the difference is visible in the numbers: on
SGLang's Responses route with `qwen3.5-9b`, the shape is enforced and the model answers a **uniform**
distribution over the options — `{billing: 0.333…, technical: 0.333…, sales: 0.333…}` and therefore a
`confidence` of 0.0 — measured identically in the 0.7.0 and 0.7.1 sweeps. jevper reports the distribution the
model sent; the label readout, not the schema, is what puts a judgement in the answer.

Three of these decide what jevper does rather than what it documents:

- **A logprob list is the carrier, and each server carries it differently.** An empty list and a
  refusal are both "no distribution here": `api="auto"` answers that question in JSON and marks the
  surface, while a real list — byte-level tokens and all — is read as one. A server that answers
  `400` for the fields is remembered the same way.
- **A spent budget is only a truncation where the server says so.** vLLM and SGLang report
  `status: "incomplete"` with `incomplete_details.reason: "max_output_tokens"`, and jevper's error
  names that field. The other three report `completed` — with the trace and no answer, or with an
  answer cut off mid-string — so what the caller is told is a malformed answer, which is what the
  body is. There is no signal in those bodies to read, and a caller who needs the distinction should
  set a budget the model cannot reach.
- **Reasoning text is decoration and is read wherever it appears.** The five part names the two
  specs use are normalized to the two the public model has, so a trace is not lost to a spelling,
  and a trace that cannot be encoded as UTF-8 is dropped rather than left in a response the caller
  cannot serialize.

The specification also defines a WebSocket transport and a `/responses/compact` endpoint. jevper
reads one whole response per request and needs neither: its surface is the non-streaming HTTP one.
The specification's own compliance suite (`bun run test:compliance`) covers phases, streaming,
tools, images, WebSocket and compaction, and says nothing about structured outputs, logprobs or
the error envelope — passing it is not evidence for the paths jevper uses, which is why the table
above is measured rather than cited.

## Prompt caching

The recorded four-server run reused prefixes without a cache-control field. Prefix caching was on in the
serve configurations used for Ollama 0.34.3, llama.cpp b11139, vLLM 0.30.1 and SGLang 0.5.20. The hosted-style
request fields `prompt_cache_key`, `prompt_cache_retention`, `session_id`, `prompt_cache_options`, and
`prompt_cache_breakpoint` were accepted with `200` and ignored in that run; `cache_salt` is the exception below.

What those builds did *not* agree on was how to report the hit. Every cell below is an observation from the
same dated run, with the flags shown in the cells; no later build is implied:

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
between calls — the state — has to come last for a rubric's calls to share anything. In the dated cache run,
one 2388-token prompt (two examples, a ~1300-token state) was called twice with only the state changed.
The token counts below are that run's measurements, not a general cache guarantee:

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

In the recorded run, `cache_salt` was the cache field vLLM 0.30.1 and SGLang 0.5.20 implemented: requests
with the same salt shared cached prefixes and different salts did not. jevper does not add the field itself,
but a caller can send it through `extra_body`; it is an isolation control, not a routing one.

```python
SystemOneClient(client, model="qwen3.5-9b", extra_body={"cache_salt": tenant_id})
```

vLLM caps the salt at 128 characters and rejects `@`, `/`, `\` and NUL; llama.cpp and ollama ignore the field.

## The Messages route

In the dated run, all five servers answered the Anthropic Messages API (`POST /v1/messages`), so jevper's
`api="messages"` worked against each with the `anthropic` SDK pointed at the same host and port as the OpenAI
one. Protocol behaviour differs by build. The following table is a measurement under the conditions above;
its `First release` column is unmeasured historical metadata, not a finding from that run.

| Server | First release (unmeasured) | `thinking` field | Thinking blocks back | `usage` cache counts |
| --- | --- | --- | --- | --- |
| LM Studio | 0.4.1 | accepted, and the answer is still separated from it | `thinking` blocks when the model thinks | `cache_read_input_tokens`, including a reported `0` on a cold call |
| llama.cpp | b7187 | accepted, and the budget grows `max_tokens` as below | reported | not documented |
| vLLM | 0.11.1 | **accepted and ignored**: its request model has no `thinking` field and pydantic drops extras, so the field reaches nothing and no downgrade fires — the answer comes back with no thinking and no way to tell | `thinking` blocks | yes |
| SGLang | 0.5.9 | **refused**: this version has no `thinking` field at all, so the request is answered `400` and jevper drops the field and re-asks, reporting `debug["server_limits"]["thinking"]` | `thinking` blocks | yes |
| ollama | 0.14.0 | accepted, but `budget_tokens` is accepted and **not enforced** | `thinking` blocks | no cache fields at all |

The Messages request shape has no logprob carrier, so `method="structured"` or `"discrete"` is how to use it;
`method="auto"` resolves to `structured` on this surface without probing. Independently of what an individual
local implementation requires, jevper always sends `max_tokens`: 1024 by default, or 1024 plus the caller's
thinking budget. A caller-provided `max_tokens` wins. A 1024 thinking budget therefore sends 2048 and a 2048
budget sends 3072, and jevper refuses a caller value that cannot hold the requested thinking budget. jevper
also omits temperature whenever that thinking budget is present.

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

ollama is the one server here whose Messages route does not take the OpenAI-style thinking-off field:
`reasoning_effort: "none"` leaves the trace running there (2368 characters of it on one measured
call), and a thinking model then spends jevper's 1024-token default before the answer begins, which
is reported as `IncompleteAnswerError` naming the limit to raise. Answering there takes
`extra_body={"max_tokens": 4096}` — the one knob that route reads.

SGLang needs one more decision, on the server side. With `--reasoning-parser qwen3` and a *non-thinking*
model — whose template has no `enable_thinking` to set, so the parser never sees the closing thinking marker
it waits for — the whole generation is classified as reasoning and the route answers with a thinking block and
no text block. `separate_reasoning: false` does not change that; dropping `--reasoning-parser` does, and the
structured scenarios then answer normally. It is a property of serving a model that never emits the marker
with a parser that waits for it, not of the client.

`output_config.format` — Anthropic's own schema-constrained output — is **implemented by vLLM** and by
neither of the others that can be asked. Measured with a schema whose only legal answer names a constant the
prompt never mentions: vLLM answered `{"canary": "JEVPER-PROBE"}`, and the same request without the field
answered `{"Name": "Canary"}`, so the shape came from the field. It also *validates* the field — an unknown
`format.type` is a `400` naming `body.output_config.format.type` — which is the rung that lets jevper drop
the field and remember it in `debug["server_limits"]["output_config"]`. llama.cpp and LM Studio accept the
field and ignore it, with no error and no verdict, so on those the JSON Schema has to stay in the system
prompt and the answer is only as good as the model's instruction-following: without it, a 4B model answered
`{"intent": "A"}` — a real answer in the wrong shape — to every structured question, and on LM Studio one
answer came back as the schema itself (`{"minimum": 0, "type": "number"}` in the place of a probability),
which jevper's one corrective retry turned into a real answer. ollama and SGLang accept the field too, but
with a thinking model *nothing comes back to enforce it* on this route: a 1024-token request came back empty
with `stop_reason: "max_tokens"` with the field, with a nonsense `format.type`, and with no field at all.
jevper keeps the schema in the prompt on this surface either way: a server that accepts a field and drops it
is indistinguishable from one that never looked at it.

One detail of the wire schema is Anthropic's, not jevper's: the API rejects *numerical constraints*, and a
400 for one costs the field for the rest of the client's life, so each `minimum`/`maximum` is folded into
the description of the field it bounded before the request goes out. vLLM takes the untransformed schema
too (`{"n": 1}` for a `minimum: 0, maximum: 1` number), so the transform costs nothing locally and is what
makes the field usable on the hosted API. The field travels in the request body rather than as an SDK
keyword, because the oldest Anthropic SDK jevper supports has no `output_config` parameter at all.

The same two servers swallow the thinking-off knob on this route: ollama's `reasoning_effort: "none"` and
SGLang's `chat_template_kwargs: {"enable_thinking": false}` both work on their OpenAI routes and neither
stops the trace on `/v1/messages`. A Messages call against either needs a budget the trace fits inside
(jevper's default 1024 answered on ollama) or a model that does not think — SGLang's own paragraph above
covers the parser that turns the whole generation into reasoning.

## Sizing a 12 GB card

The recorded hardware run loaded one server at a time on the 12 GB GPU box. The weights, serve flags,
startup time, KV-cache size and error text below are measurements from that run, not portable sizing rules:

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
