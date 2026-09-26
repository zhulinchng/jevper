# Jev and jevper, side by side

*Independent implementation of the documented System One wire format — not affiliated with TypeSafe.*

This page is a record of measurements, not a position. Everything below was run on **2026-09-26**,
against the live Jev service and against this branch of jevper, and every number is what came back.
Where something could not be measured, it says so rather than guessing — see
[What was not measured](#what-was-not-measured).

| Piece | Which one |
| --- | --- |
| Service | `POST https://opencode.ai/zen/v1/systemone`, the Jev endpoint opencode Zen serves |
| Model | `jev-1.13-free`; the paid `jev-1.13` answered HTTP 402 on this account |
| Reference client | `typesafe-sdk` 0.7.1 from PyPI, installed and read for the contract it encodes |
| jevper | 0.7.11 on this branch, `.venv` (Python 3.14, openai 3.19.0). The confidence and score arithmetic below was measured on 0.7.4, before the System One surface existed; every Jev wire-format measurement was taken on 2026-09-26 against 0.7.8, which this page does not re-measure. The Ollaya run in [local servers](local-servers.md#ollaya-the-decision-server) is 0.7.11 |
| Published docs | TypeSafe's API reference, confidence page, and the `jev-1.13` jaggedness page (reviewed 2026-09-17) |

## jevper can call this endpoint

```http
POST https://api.typesafe.ai/v1/systemone
Authorization: Bearer <API_KEY>
Content-Type: application/json

{"state": "…", "model": "jev-latest", "questions": {"is_urgent": {"type": "noul", "instructions": "…"}}}
```

That request is what `api="systemone"` sends. The questions go in the body as typed questions rather
than as a prompt, the model answers them together, and the answers come back as the Jev answer shapes.
It is a fourth surface beside the three prompt ones ([Architecture](architecture.md)), reached through
the same duck-typed client object — any object with `post(path, body=…, cast_to=…)`, which an `openai`
client has:

```python
from openai import OpenAI
from jevper import Choice, Noul, SystemOneClient

client = SystemOneClient(
    OpenAI(base_url="https://api.typesafe.ai/v1", api_key=key), model="jev-latest", api="systemone"
)

response = client.system_one(
    state="I was charged twice for the same order. Can someone look into this?",
    questions={
        "refund": Noul(instructions="Is the customer asking for a refund?"),
        "team": Choice(instructions="Which team should handle this?",
                       criteria={"billing": "Payments", "support": "Customer problems"}),
    },
)
```

Measured against the live endpoint on 2026-09-26, that call returned in 1.10 s: `refund` 0.72,
`team` billing at confidence 0.99, `usage.n_calls` 1, and one attempt record per question naming the
request it was read from. `client.list_models()` reads the service's other route the same way.

Three things about this surface are worth knowing before you point a call at it.

**The service's own numbers are kept.** `score`, `confidence`, `choice` and `legend` are read as they
arrived rather than recomputed by jevper, because the service computed them from a model trained for
the decision. jevper's formulas agree with it to within 0.015 (see below), which is a reason to trust
them when a general model is doing the answering — not a reason to overwrite a trained model's
arithmetic with them.

**The options this wire format has no field for are refused, not dropped.** `method` other than
`auto`, `reasoning`, `examples`, `temperature` and `prompt_cache_key` are all refused by name, in one
error, before a request is sent. The service ignores what it does not know, so accepting them would
mean reporting a method the caller did not get. A noul carrying neither instructions nor criteria is
refused for the same reason the service answers 400 for one.

**So is anything the wire format cannot carry, however reasonable it looks.** Every structured field
on this wire takes a string, an object or an array, so a number or a boolean in a `state`, an
`instructions`, a noul's criterion, a choice option's description or a score level is a 422 the
service would answer — refused here instead, with every field named at once. An empty question id is
the same: a 400 the service would have answered after the round trip.

**The surface is never chosen for you.** `api="auto"` keeps its documented order and does not post a
Jev body to a client that happens to have a `post` method — both official SDKs do.

What is still not the same as the service: the reference SDK is generated from the service's own
OpenAPI schema, and jevper is an independent implementation of the same wire format. It reads the
documented response shape and refuses a body that breaks it; it does not know about fields the
service has not published.

## One request, or one per question

| | Real Jev | jevper |
| --- | --- | --- |
| 11 questions in one call | 1 request, 1.08 s, 1094 input + 353 output tokens | 1 request on `systemone`; 11 requests on a prompt surface |
| The state | sent once, as the request's `state` field | sent once as `state` on `systemone`; in every request on a prompt surface, wrapped in `<document>` |
| The questions | become the request body | become the request body on `systemone`; a 305-character system prompt plus a per-question user turn otherwise |
| Prompt caching | not applicable | `prompt_cache_key` offered on every prompt-surface request, refused on `systemone` |
| Question isolation | one evaluation per question, by construction | the same on `systemone`, since the service evaluates each question on its own; one worker per question otherwise |

The prompt-surface column was measured against a fake provider object that records requests and
answers with a schema-conforming distribution, because the OpenRouter daily free quota was exhausted
(`X-RateLimit-Remaining: 0`, `limit_source: openrouter_free_tier_daily`) before the comparison ran;
the request count is jevper's own dispatch, not an artefact of the fake. Three questions produced
three requests, eleven produced eleven.

So the round trips now match the service on the surface that talks to it, and do not on the ones that
render a prompt. What a run costs still depends on the backend: the service billed 1094 input tokens
for eleven questions, a prompt surface bills its prompt per question, and the free-tier model behind
OpenRouter bills a different number again.

## The numbers agree

Confidence is the claim most worth checking, because jevper computes it locally
([`normalize`](api.md)) while the service computes it server-side. Given the service's own
probabilities, jevper reproduces its `confidence` to within 0.015, and the score expectation
exactly:

| Answer | Service said | jevper computes | Difference |
| --- | --- | --- | --- |
| `department`, 3 options, peak 0.82 | 0.73 | 0.7300 | 0.0000 |
| `urgent_2`, 2 options, peak 1.0 | 1.0 | 1.0000 | 0.0000 |
| `department_3`, 3 options, peak 0.91 | 0.87 | 0.8650 | −0.0050 |
| `language_4`, 4 options, peak 0.98 | 0.97 | 0.9733 | +0.0033 |
| `topic_6_vague`, 6 options, peak 0.56 | 0.47 | 0.4720 | +0.0020 |
| `tone_5_vague`, 5 options, peak 0.42 | 0.29 | 0.2750 | −0.0150 |
| `frustration`, 3 levels, all mass on level 1 | 1.0 | 1.0000 | 0.0000 |
| `score_2`, 2 levels | 0.94 | 0.9400 | 0.0000 |
| `score_3`, 3 levels | 0.62 | 0.6250 | +0.0050 |
| `score_4`, 4 levels | 0.43 | 0.4300 | 0.0000 |
| `score_6`, 6 levels | 0.64 | 0.6400 | 0.0000 |

Two things explain the residue. The service prints probabilities at two decimals and rounds
`confidence` to two decimals from its own internal distribution, so recomputing from the printed
numbers cannot always land on the printed answer: `tone_5_vague` needs a peak near 0.43 to produce
0.29, and 0.43 would have printed as `0.43`. And the service does not reproduce itself exactly —
the same noul asked twice in one call, under two ids, came back 0.42 and 0.41. A tenth of a percent
of disagreement with a live service is not evidence of a different formula.

The rest is exact. The score answers matched `Σ level × probability` to the last digit (0.97, 1.75,
2.43, 3.32, 1.0), every distribution returned summed to 1.000000000000, and jevper's `rescale` — the
step that renormalises a score distribution that arrived off 1 — moved no probability at all
(`0.00e+00`). A noul carries no confidence on either side, matching the service, which documents
that only Choice and Score answers have one.

The service publishes its own contract: `https://api.typesafe.ai/openapi.json`, an OpenAPI 3.1
document (`info.version` 0.2.0) that the server validates against, and the most authoritative source
here — more so than the prose docs, and more so than the limits below, which are the server's runtime
rules and are not in the schema at all. What it settles:

| In the schema | What it says |
| --- | --- |
| `state` | a string, an object, or an array — the three shapes the table above measured |
| `instructions` | a string, an object, or an array, on all three question types |
| `ChoiceQuestion.criteria` | required, an object; a value may be a string, an object, an array, or `null` |
| `ScoreQuestion.criteria` | required, an array, `minItems: 1` — no `maxItems` anywhere |
| `NoulQuestion.criteria` | optional, `{"true": …, "false": …}`, each side nullable |
| `SystemOneResponse` | `model`, `answers` and `usage` all required; `model` "may differ from the alias supplied in the request" |
| `ModelMetadata` | all three of `name`, `description`, `release_date` required — jevper defaults the last two, so a terser gateway is still read |
| Errors | only `200` and `422` are declared; the `400`s, `401` and `402` below are the runtime's own, and the 422 body is the pydantic `detail` list |

Two consequences for jevper. The 255-option and 10-level limits are not in the schema, so a client
built from it alone would send them and be rejected at the server — which is what `typesafe-sdk` does,
and why jevper refuses them locally. And the schema's `minItems: 1` for a score rubric is why the
service answers a single level while jevper, judging the answer degenerate, refuses to ask.

One field the schema documents and jevper does not expose: the service's own `model` in the response
names what answered, which for an alias like `jev-latest` is the concrete build. jevper reports the
model the call asked for, on every surface including this one, because that is the one it validated
and keyed its state on; the service's answer to the question "which build was that" is not on the
wire jevper hands back.

## An open model on the same wire format

[Ollaya](https://github.com/ollaya-dev/ollaya) is an independent, open implementation of this same
documented format, serving open decision models locally, and jevper's `api="systemone"` surface talks to
it unchanged — the measured run is in [local servers](local-servers.md#ollaya-the-decision-server). It
belongs on this page because it is the only accuracy figure available for this wire format that was not
published by TypeSafe.

Ollaya's own model page scores its `laya:typed-decisions` — a fine-tune of a 421M-parameter
ModernBERT-large — at **0.766 on typed-decisions, against 0.727 published for Jev 1.13**. Both numbers are
Ollaya's: jevper computed neither and has no opinion on the benchmark. What the pair is worth is that a
local, open model on this wire format is not far off the hosted one, so a rubric can be developed
entirely against Ollaya and pointed at the hosted service afterwards without changing a line.

Two things differ from the hosted service, and both are the server's rather than the format's. Ollaya's
own `/api/decide` adds a report on the request — which checkpoint a router chose, whether the state was
truncated, how long the forward pass took — which jevper reads behind `native=True`. And its `model`
field behaves as the hosted one does, naming the checkpoint that answered rather than the alias, so
`native=True` reports that as `routing.model` while `response.model` stays the name the caller asked for,
as it does on every other surface.

## Where the two disagree about input

jevper validates client-side and the service validates again on arrival, so the interesting column
is where the two sets of rules differ. `typesafe-sdk` 0.7.1 is the third column because it is the
reference client, and it turns out to enforce almost nothing:

| Input | Real service | typesafe-sdk 0.7.1 | jevper 0.7.8 |
| --- | --- | --- | --- |
| 255 options | accepted | accepted | accepted |
| 256 options | 400 `Too many choices. Must have at most 255 choices.` | accepted, fails at the server | rejected at construction, naming 255 as the Jev limit |
| 10 levels | accepted | accepted | accepted |
| 11 levels | 400 `Too many score levels. Must have at most 10 levels.` | accepted, fails at the server | rejected, `score needs 2..10 levels` |
| 1 level | **accepted** — `score` 0, `confidence` 1, `probabilities` `{"0": 1}` | accepted | rejected, `SCORE_MIN_LEVELS` is 2 |
| 1 option | accepted, `confidence` 1 | accepted | accepted |
| A choice option with `null` criteria | accepted | accepted | accepted |
| `instructions` as an object or an array | accepted | accepted | accepted |
| `state` as an object or an array | accepted | accepted | accepted |
| `state` as an empty array | accepted | — | accepted; the prompt renderer refuses it as an empty conversation, and this surface no longer asks it |
| A bare number or boolean anywhere a structured field goes — `state`, `instructions`, a noul side, a choice option's description, a score level | 422 `Input should be a valid string` naming the path | accepted, fails at the server | refused before sending, every field named in one error |
| A `null` score level | 422 | — | refused before sending |
| A `null` on a noul side beside a set one | accepted | accepted | accepted |
| An empty question id | 400 `Question key cannot be empty.` | — | refused before sending |
| An empty `questions` map | 422 with a pydantic `detail` list | `TypeSafeError` before sending | rejected before sending |
| A noul with neither instructions nor criteria | 400 `Noul question must have criteria or instructions` | accepted | refused on `systemone`, accepted on a prompt surface, where the question can be rendered from an example |
| An unknown field on a question (`temperature`) | ignored, 200 | rejected, `extra="forbid"` | rejected, `extra="forbid"` |

One rule covers the middle of that table, and it is the schema's rather than a limit the service
invents: **every structured field takes a string, an object or an array.** Measured field by field on
2026-09-26 — a number and a boolean are 422 in all five places, and an array answers 200 where a
string would, as does an object. `null` is the one value treated differently per field, and each of
those is measured too: refused as missing on `state`, 422 on a score level, and 200 on a noul side or
a choice option, where the API reference documents it as "use null when an option needs no extra
detail".

jevper's question models are wider than this on purpose, because a prompt surface can render a number
as text and no prompt surface has such a limit, so the check lives on this surface rather than in the
models: `Noul(instructions=0)` is not a malformed question until it goes on this wire.

The one row where jevper is stricter than the service it mirrors is the single-level score. The
service answers it, and the answer is degenerate — all mass on level 0, score 0, confidence 1 — so
jevper's refusal is a judgement, not a compatibility gap, and it is deliberate.

## Four error envelopes

The service signals failure four different ways, which is worth knowing before writing a client
against it:

| Status | Body |
| --- | --- |
| 400 | `{"error": {"type": "server_error", "message": "Upstream request failed: Model is unavailable."}}` |
| 401 | `{"error": {"type": "server_error", "message": "Upstream request failed: Invalid credential"}}` |
| 402 | `{"error": {"type": "server_error", "message": "Upstream request failed: Insufficient account funds"}}` |
| 400 | `{"detail": "Too many score levels. Must have at most 10 levels."}` |
| 400 | `{"detail": {"error_type": "api_usage_error", "message": "Invalid request."}}` |
| 422 | `{"detail": [{"type": "too_short", "loc": ["body", "questions"], "msg": "…"}]}` |

A client that only parses `detail` misses the first three; one that only parses `error` misses the
rest. On `api="systemone"` these reach the caller as a `ProviderError` carrying the service's own
status and message — a 400 for too many score levels arrives with the service's wording — and the
transient set (408, 409, 429 and 5xx) is retried on the same terms as every other surface. The
reference SDK raises its own exception family and parses `retry-after` and `retry-after-ms`; jevper
honours the same two headers everywhere, described in [Methods](methods.md).

## What the model does that jevper cannot promise

TypeSafe publishes a list of `jev-1.13` failure modes, and two of them reproduced on demand, on
states the docs do not use:

- **A noul and a choice on the same predicate do not agree.** On `"I'm not happy with the fit. What
  are my options here?"` the noul `"Is the customer asking for a refund?"` returned **0.22**, while
  the two-option choice on the same sentence returned `yes` 0.0, `no` 1.0, confidence 1.0. The
  published table for that exact state shows 0.22 against 0.01/0.99 — the noul matches to the digit,
  the choice is more extreme today.
- **A predicate and its negation do not sum to one.** On `"I was charged twice for the same order.
  Can someone look into this?"`, `refund` 0.71 and `not_refund` 0.46 sum to **1.17**. The published
  pair is 0.72 and 0.47.

Both are properties of the model, not of the wire format, and both are why the service's own docs
tell you not to carry a threshold from a noul to a choice. For a jevper user the same warning
applies with more force: a jevper answer is a general model's answer wearing the Jev shape.
`confidence` is a statistic of whatever distribution came back — jevper's formulas match the
service's arithmetic to 0.015, but arithmetic agreement is not calibration, and nothing in jevper
can make an arbitrary model's distribution as trustworthy as a model trained for the decision. Use
[confidence](methods.md) to route, never as a guarantee.

## What was not measured

- **A live jevper run.** OpenRouter's free daily allowance was exhausted (`X-RateLimit-Remaining: 0`)
  during this comparison, so every jevper figure here comes from a recording fake client, and no
  jevper answer was compared with a jev answer on the same state. The request counts, prompt shapes
  and validator behaviour are jevper's own and do not depend on the backend; the answer *quality*
  comparison is simply absent.
- **The paid `jev-1.13`.** Every call returned HTTP 402 on this account, so `jev-1.13-free` is the
  only model measured. The free alias may be quantised or otherwise degraded relative to the paid
  one; the arithmetic in [The numbers agree](#the-numbers-agree) is the model's, but the
  distributions are the free model's.
- **Rate-limit and retry behaviour.** Nothing in this session provoked a 429 from the service, so
  whether it sends `retry-after` or `retry-after-ms` — the headers jevper honours — is unmeasured
  here. The `typesafe-sdk` source reads both, which is evidence about the reference client, not
  about this deployment.
- **Latency on the jevper side**, for the same quota reason. The 1.08 s figure is the service's, for
  11 questions in one request.
- **Latency between the two paths is not comparable** beyond that: one is a purpose-built model on
  someone else's infrastructure, the other is N prompt-completion round trips to whatever backend
  you point jevper at.

## Reproducing this

The service side is one `curl`, and the key comes from the environment:

```sh
curl -sS -X POST "$JEV_BASE_URL" \
  -H "Authorization: Bearer $JEV_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"state": "The invoice total does not match the amount I was charged.",
       "model": "jev-1.13-free",
       "questions": {"wrong": {"type": "noul", "instructions": "Do the numbers disagree?"}}}'
```

The jevper side needs no network at all: a client object exposing `chat.completions.create` that
records its kwargs and returns a schema-conforming distribution is enough to count requests, read
the prompts and check the validators — the shape [`examples/duck_client.py`](https://github.com/zhulinchng/jevper/blob/main/examples/duck_client.py)
already demonstrates. Feeding a captured response back through
`SystemOneResponse.model_validate` checks the answer contract without a request at all.

The numbers in this page are pinned by `tests/test_jev_agreement.py`, which asserts jevper's
arithmetic against these recorded service answers and the response model's acceptance of the
recorded payloads. What it cannot pin is the service: third-party answers drift with the model, so
re-run the `curl` above when `jev-1.13-free` moves and expect the distributions to differ.
