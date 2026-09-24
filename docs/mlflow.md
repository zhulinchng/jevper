# MLflow

jevper is a plain library that drives a caller-supplied client, so integrating it with MLflow means two
different things: MLflow's *tracing* sees the calls jevper makes, and MLflow's *hosting* surfaces can wrap
jevper as a model. Both are covered here, verified against **MLflow 3.16.1** (2026-09-16), `openai` 3.19.2,
`anthropic` 1.8.0 — offline against a stub server, and live against a local LM Studio server on the probe
box, which is the only local server that answers all three surfaces jevper drives.

Nothing in jevper imports MLflow, and MLflow is not a dependency. The integration suite is in
`tests/test_mlflow.py` and skips unless MLflow is installed:

```sh
uv pip install -e '.[test,mlflow]'    # mlflow[gateway,langchain] + openai + anthropic + pytest
pytest tests/test_mlflow.py -q
```

| Distribution | Version | Notes |
| --- | --- | --- |
| `mlflow` | 3.16.1 | The full package; `mlflow-skinny` and `mlflow-tracing` are pinned alongside it |
| `mlflow[gateway]` | — | FastAPI/uvicorn for `mlflow gateway`; without it the CLI has no `gateway` command at all |
| `mlflow[genai]` | — | `mlflow.genai.evaluate`, scorers and judges (already present in the full package) |
| Python | ≥ 3.10 | jevper supports 3.10–3.14; the suite runs on 3.13 locally and 3.12 on the probe box |

## Tracing

`mlflow.openai.autolog()` and `mlflow.anthropic.autolog()` patch the SDK **resource classes**, not the
client constructors, so every call jevper makes through a caller-supplied *SDK* client is traced — including
the attempts it makes before settling on one the provider accepts. A duck-typed client of your own is not an
SDK client, so autolog cannot see it; wrap those calls in `@mlflow.trace` yourself if you need the span.

```python
import mlflow
from jevper import Choice, SystemOneClient
from openai import OpenAI

mlflow.set_tracking_uri("sqlite:///mlflow.db")     # or an HTTP tracking server
mlflow.set_experiment("triage")
mlflow.openai.autolog()                            # chat, responses, embeddings, legacy completions
mlflow.anthropic.autolog()                         # messages

client = SystemOneClient(OpenAI(base_url="http://127.0.0.1:11434/v1", api_key="x"),
                         model="qwen3:4b-instruct-2507-q4_K_M", method="auto")

@mlflow.trace(name="system_one")                   # one trace per jevper call
def classify(text: str) -> str:
    response = client.system_one(state=text, questions={"intent": Choice(criteria={...})})
    return response.answers["intent"].choice or ""

classify("I was charged twice for the same subscription.")
mlflow.flush_trace_async_logging()                 # trace export is asynchronous
```

| jevper surface | Span name | Traced by |
| --- | --- | --- |
| `api="chat_completions"` (sync) | `Completions` | `mlflow.openai.autolog()` |
| `api="chat_completions"` (async) | `AsyncCompletions` | `mlflow.openai.autolog()` |
| `api="responses"` | `Responses` | `mlflow.openai.autolog()` |
| `api="messages"` | `Messages.create` | `mlflow.anthropic.autolog()` |

Each SDK call is its own **trace** unless it happens inside a span you opened. A single question is answered
on the calling thread, so its span becomes a child of your `@mlflow.trace` span. With several questions,
jevper answers them on a `ThreadPoolExecutor`, and MLflow's tracing context lives in a `ContextVar` that does
not cross threads: each question's call is then a separate root trace, and the wrapper trace holds only the
answer. That is MLflow's documented behaviour, not a jevper bug — see its own `advanced-patterns` reference
("Threads require explicit context propagation"). If you need one trace per call, pass one question per call
or wrap the per-question work yourself.

What lands on a span, all observed on real calls:

| Attribute | Value |
| --- | --- |
| `mlflow.llm.model` | The model jevper sent |
| `mlflow.llm.provider` | `anthropic` on the Messages route (absent on the OpenAI routes) |
| `mlflow.message.format` | `openai` or `anthropic` |
| `mlflow.chat.tokenUsage` | `{"input_tokens", "output_tokens", "total_tokens"}`, plus `cache_read_input_tokens` on the Responses route; all three are `null` when the provider sent no usage |
| `mlflow.spanInputs` / `mlflow.spanOutputs` | The request kwargs jevper built, and the raw provider response |
| `mlflow.spanLogLevel` | `20` on a successful SDK call, `40` on a failed one; a plain `@mlflow.trace` span of your own is `10` |
| request fields | `mlflow.spanInputs` always holds every keyword jevper sent. MLflow *also* promotes some of them to attributes, and which ones depends on the route: `logprobs`, `top_logprobs`, `prompt_cache_key`, `model` on chat; `include`, `store`, `top_logprobs`, `model` on responses; none on the Messages route, where `max_tokens` lives in `span.inputs` |

The request fields are what make the fallback ladder auditable: a server that refuses `response_format`
leaves one error span per rung, each naming the exact field it turned down, followed by the span that
answered. The same is true of a retried transient failure, and of the corrective retry after a malformed
answer — so `response.debug["llm_attempts"]` and the trace tell the same story.

Two failure modes are worth knowing. A request the provider *rejects* — a 400, a 404, a refused field —
leaves a span with an error status, and the call still raises jevper's own error. A response the provider
*answers* that jevper cannot read is different: a refusal or a spent budget arrives as an HTTP 200, so the
span is `OK` while jevper raises `MalformedAnswerError`/`LabelReadoutError`. The span tells you what the
provider said; jevper's error tells you whether an answer came out of it. And an *unwritable* tracking store
does not break the call — MLflow cannot export the span, and the answer comes back anyway.
`mlflow.tracing.disable()` records nothing and changes nothing else.

## Hosting jevper as an MLflow model

MLflow 3.16.1 offers three base classes for a chat model. Two of them are exercised end to end in
`tests/fixtures/`, logged with the models-from-code pattern (`python_model=<path>` — a cloudpickled instance
cannot carry an HTTP client, and the client belongs in `load_context` anyway).

| Base class | Status in 3.16.1 | Contract |
| --- | --- | --- |
| `mlflow.pyfunc.PythonModel` | current | `predict(self, context, model_input, params=None)` |
| `mlflow.pyfunc.ChatModel` | deprecated since 3.0.0 | `predict(self, context, messages: list[ChatMessage], params: ChatParams) -> ChatCompletionResponse` |
| `mlflow.pyfunc.ResponsesAgent` | recommended for new code | `predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse`, plus `predict_stream`, which the base class raises `NotImplementedError` for until you write it |

The fixtures in `tests/fixtures/` are the working examples: `jevper_chat_model.py` (`ChatModel`),
`jevper_responses_agent.py` (`ResponsesAgent`, including `predict_stream` and the `Message`-object input
MLflow actually sends), and `jevper_langchain_model.py` (a `SimpleChatModel` for the LangChain flavour).

```python
# jevper_chat_model.py — logged with python_model="jevper_chat_model.py"
import mlflow
from mlflow.pyfunc import ChatModel
from mlflow.types.llm import ChatChoice, ChatCompletionResponse, ChatMessage
from jevper import Choice, SystemOneClient

class JevperChatModel(ChatModel):
    def load_context(self, context):                     # the client is built here, not pickled
        from openai import OpenAI
        config = context.model_config or {}
        self.client = SystemOneClient(
            OpenAI(base_url=config["base_url"], api_key="x"), model=config["model"],
            api="chat_completions", method="structured",
        )

    def predict(self, context, messages, params):
        state = "\n".join(str(getattr(m, "content", "")) for m in messages)
        answer = self.client.system_one(state=state, questions={"intent": Choice(criteria={...})})
        return ChatCompletionResponse(model="jevper", choices=[ChatChoice(
            index=0, message=ChatMessage(role="assistant", content=answer.answers["intent"].choice or ""),
            finish_reason="stop")])

mlflow.models.set_model(JevperChatModel())
```

`mlflow.pyfunc.log_model` runs the `input_example` through the model while logging, so a model that cannot
answer its own example cannot be logged — a jevper model whose provider will not return the shape the method
needs fails there, not later. Invoke the logged model with `mlflow.pyfunc.load_model(uri).predict(...)`, or
with `mlflow.models.predict(model_uri=..., input_data=..., output_path=..., env_manager="local")`, which
writes the answer to `output_path` (its return value is `None`).

## The AI Gateway

MLflow 3.16.1 still ships the standalone gateway — `mlflow gateway start --config-path config.yaml`, behind
the `gateway` extra — but marks the command deprecated in favour of the server-hosted gateway that current
documentation describes (`mlflow server`, endpoints managed through the UI/API, `/gateway/mlflow/v1/...`
passthrough routes). The two are different contracts, and the differences below are the 3.16.1 standalone
one, measured on the probe box.

```yaml
endpoints:
  - name: local-chat
    endpoint_type: llm/v1/chat
    model:
      provider: openai                 # not "openai-compatible": that provider does not exist
      name: qwen3:4b-instruct-2507-q4_K_M
      config:
        openai_api_key: test
        openai_api_base: http://127.0.0.1:11434/v1
```

```python
client = SystemOneClient(
    OpenAI(base_url="http://127.0.0.1:5000/v1", api_key="dummy"),   # the route's /v1 root
    model="local-chat",                                             # the endpoint name is the model
    method="auto",
)
```

Routes: `GET /health`, `POST /v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, and the legacy
`POST /endpoints/{name}/invocations`. The request's `model` field selects the endpoint, so it has to be the
endpoint *name* (`local-chat` above) rather than the backend model name — a wrong name is refused by the
gateway and never reaches the backend. **There is no `/v1/responses` route**, so `api="auto"` pays one 404 and
answers on chat; an unknown path is FastAPI's `404 {"detail": "Not Found"}`, which jevper reads as "this
server has no such route" rather than a bad request.

Requests pass through faithfully — the gateway's request model allows extra fields and it calls the backend
over HTTP rather than through the OpenAI SDK, so `response_format`, `logprobs`, `top_logprobs`,
`prompt_cache_key`, and vendor knobs in `extra_body` (`chat_template_kwargs`, `thinking`) all arrive at the
backend. Only `model` is replaced with the configured backend model.

Responses do **not** pass through. The answer is re-shaped into MLflow's `ChatCompletionResponse`, which has
no room for several fields jevper knows how to read:

| Field | Survives the gateway | Consequence for jevper |
| --- | --- | --- |
| `choices[].finish_reason`, `choices[].message.refusal` | yes | the truncation note and the refusal readout still work |
| `usage.prompt_tokens` / `completion_tokens` / `total_tokens` / `prompt_tokens_details` / `completion_tokens_details` | yes | `usage.cached_tokens` and `usage.reasoning_tokens` both work |
| `choices[].logprobs` | **no** | `method="logprobs"` raises `LabelReadoutError`; `method="auto"` falls back to `structured` and reports it |
| `choices[].message.reasoning_content` | **no** | the reasoning readout is empty, so `reasoning_text(response.reasoning)` is empty too |

A backend error keeps its status code but is rewrapped as `{"detail": "<provider message>"}`; the message
itself survives, so jevper's field-level fallbacks still fire and the caller still sees the provider's words.

Two client-side mismatches are worth knowing before you build on them: `mlflow.deployments.get_deploy_client`
posts to `/v1/endpoints/{name}/invocations` while the standalone gateway serves `/endpoints/{name}/invocations`
(the raw route works; the client does not), and a judge configured with `gateway:/<route>` posts to
`/gateway/mlflow/v1/chat/completions`, which only the server-hosted gateway mounts.

One deployment gotcha, found the hard way on the probe box: the standalone gateway watches the config file's
**directory** (so a config edit reloads routes). On a systemd host a config under `/tmp` makes that walk trip
over `/tmp/systemd-private-*` with `Permission denied` and the server exits before it ever listens — keep the
config in a directory you own.

## Evaluation

`mlflow.genai.evaluate` calls a `predict_fn` with the dataset's `inputs` as keyword arguments, and expects a
trace per call. Wrap the jevper call in `@mlflow.trace` and let autolog trace what happens inside it:

```python
from mlflow.genai.scorers import Correctness, scorer

@scorer
def label_matches(outputs, expectations) -> bool:
    return outputs == expectations["label"]

@mlflow.trace(name="jevper.system_one")
def predict(text: str) -> str:
    response = client.system_one(state=text, questions={"intent": Choice(criteria={...})})
    return response.answers["intent"].choice or ""

result = mlflow.genai.evaluate(
    data=[{"inputs": {"text": "I was charged twice"}, "expectations": {"label": "billing"}}],
    scorers=[label_matches, Correctness(model="openai:/qwen3:4b-instruct-2507-q4_K_M")],
    predict_fn=predict,
)
result.metrics        # {'label_matches/mean': 1.0, 'correctness/mean': 1.0}
```

One MLflow quirk to know before you build a judge-based evaluation: with a **SQLite** tracking store,
logging an LLM-judge metric fails inside MLflow (`logged_model_metrics` foreign-key error), and the
filesystem store is refused outright unless you set `MLFLOW_ALLOW_FILE_STORE=true`. The suite runs those
evaluations against an opted-in file store for that reason; a real tracking server avoids it entirely.

Three things the suite pins, because they change how you write the wrapper:

* A `predict_fn` that **raises** fails the whole evaluation (`MlflowException: Failed to run the prediction
  function`) — so a model that sometimes answers prose where a label is required needs a total function:
  catch `MalformedAnswerError` (or `LabelReadoutError`) and return a sentinel, and let a scorer judge it.
* A **scorer** that raises is recorded against its row and the run continues.
* `Correctness` (and the other built-in judges) needs `expectations.expected_response` or `expected_facts`;
  without one MLflow drops the scorer with a warning and no metric appears.

A judge is an OpenAI-compatible caller like any other, so it can be pointed at a local server: `model="openai:/<name>"`
plus `OPENAI_API_KEY` and `OPENAI_API_BASE` (verified against a stub and, live, against LM Studio on the probe box), or
`model="ollama:/<name>"`. The `gateway:/<route>` form resolves through `MLFLOW_GATEWAY_URI` (or an HTTP
tracking URI) to the server-hosted gateway's passthrough path, so it does not reach the standalone gateway.

## Flavours

| Flavour | Applies to jevper? | Notes |
| --- | --- | --- |
| `mlflow.pyfunc` | yes | `PythonModel`, `ChatModel` (deprecated), `ResponsesAgent` (recommended); the verified hosting path |
| `mlflow.openai` | partly | `autolog()` traces jevper's calls; `save_model`/`log_model` are deprecated since 3.8.0 (they read `OPENAI_API_BASE`, so they can front the same server, but jevper is not what they wrap) |
| `mlflow.anthropic` | partly | Tracing only — there is no `log_model`/`save_model` in this flavour |
| `mlflow.genai` | yes | `evaluate`, scorers, judges; the verified evaluation path |
| `mlflow.gateway` | yes | An OpenAI-compatible front door over any provider, including a local server; see the response caveats above |
| `mlflow.deployments` | partly | The client posts `/v1/endpoints/{name}/invocations`; the standalone gateway serves `/endpoints/{name}/invocations`, so the client and this gateway do not meet |
| `mlflow.langchain` | yes, via an adapter | LangChain v1 requires models-from-code; a `SimpleChatModel` subclass over jevper logs and predicts (`tests/fixtures/jevper_langchain_model.py`) |
| `mlflow.dspy`, `mlflow.llama_index` | not covered | They host their own framework objects; a jevper-backed adapter for either would be new code, not a jevper integration |
| `mlflow.transformers`, `mlflow.sentence_transformers` | no | They host local model weights; jevper is an API client |
| `sklearn`, `xgboost`, `pytorch`, `spark`, `keras`, `onnx`, … | no | Trained-model flavours; nothing for an LLM client to attach to |

## Limits

* Verified on MLflow 3.16.1 only. The gateway in particular is being reshaped: the standalone command is
  deprecated, and the server-hosted routes (`/gateway/mlflow/v1/...`) are what current documentation
  describes. Re-run `tests/test_mlflow.py` after upgrading MLflow.
* Streaming is not covered: jevper answers in one call per question, so there is no streaming path to trace,
  and MLflow's Anthropic autolog does not record streaming calls in any case.
* The trace-attribute names (`mlflow.chat.tokenUsage`, `mlflow.llm.model`, …) are MLflow's, not jevper's; the
  suite asserts them, so a rename shows up as a failing test rather than a silent gap.
