# incident-triage

A support-incident triage service built on [jevper](https://github.com/zhulinchng/jevper), and the
harness that exercises the library through it.

The service does what a real deployment does: it holds a rubric — an intent `Choice`, an urgency
`Score` and a human-handoff `Noul` — classifies a ticket with one `system_one` call, and turns the
answer into a report its callers can act on. Everything it uses is public API: the question
models, the four methods, all three surfaces, both facades, the few-shot levels, the retry and
concurrency knobs, the debug record, and a hand-rolled client with no SDK in it at all.

```python
from openai import OpenAI

from incident_triage import Triage, TriageReport

service = Triage(
    client=OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="local"),
    options={"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}},
)
report, = service.triage_all([ticket])
print(report.to_json())
```

## What is in here

| module | what it holds |
| --- | --- |
| `rubrics.py` | the questions: a `Choice` with a JSON option description, a four-level `Score`, a `Noul` with criteria, a 30-option routing question, a 26-option label question, the 255-option maximum, and the same rubric again as raw mappings |
| `tickets.py` | sample work in every `state` form: a JSON ticket, a long one, a conversation |
| `profiles.py` | one profile per local server — base URL, model id, the request field that turns thinking off, the output budget that server's Messages route needs, and whether it enforces a JSON schema on `/v1/responses` — plus clients that count what reaches the wire |
| `duck.py` | a client with no SDK underneath, and slow variants of it for cancellation and ordering |
| `checks.py` | the invariants a consumer is entitled to hold, checked after every call |
| `service.py` | the blocking and async services, the report shape, and the debug summary |
| `sweep.py` | 38 scenarios × the surfaces each can run on: every method, every state form, the option widths, the reasoning modes, the cache key, the headers, the failures |
| `edge.py` | 49 local cases: every documented refusal (each proved to cost no request), the facade rules, cancellation, ordering, and the constants |
| `trace.py` | one call inside an MLflow trace, with what the trace is checked to hold |

## Running it

```sh
# the local suite: no server, no GPU
python -m incident_triage.cli edge --full

# one server, every scenario
python -m incident_triage.cli sweep --server vllm --full --out vllm.json

# what a caller actually gets
python -m incident_triage.cli triage --server lmstudio --api auto
python -m incident_triage.cli async --server ollama --method structured
python -m incident_triage.cli trace --server sglang --api messages   # needs mlflow
```

`sweep` exits non-zero when any scenario reports a gap, and the JSON report says which: an
invariant the answer broke, an exception the surface did not document, or a raw exception from
the library. A scenario may also declare the failures that are legitimate for the surface it
runs on — `logprobs` on the Messages API, `grammar` on a server that does not take one — so a
documented limitation is recorded rather than counted as a defect.

Install it like any other project, from the wheel:

```sh
uv venv && uv pip install -e .
```
