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
| jevper | 0.7.4 on this branch, `.venv` (Python 3.14, openai 3.19.0) |
| Published docs | TypeSafe's API reference, confidence page, and the `jev-1.13` jaggedness page (reviewed 2026-09-17) |

## The difference that decides everything: jevper cannot call this endpoint

The hosted API takes one request per evaluation:

```http
POST https://api.typesafe.ai/v1/systemone
Authorization: Bearer <API_KEY>
Content-Type: application/json

{"state": "…", "model": "jev-latest", "questions": {"is_urgent": {"type": "noul", "instructions": "…"}}}
```

jevper has no transport for that. It picks a surface by probing `responses.create`, then
`chat.completions.create`, then `messages.create` ([Architecture](architecture.md)), renders the
questions into a prompt, and reads a distribution back out of the model's answer. There is no code
path in it that posts a `questions` map to `/v1/systemone`, so pointing jevper at a Jev key today
does nothing useful: the surface probe finds no route and the call fails with a capability error.

So the two are not competitors on the same endpoint. The hosted service is a purpose-built decision
model reached over its own wire format; jevper is a client that puts the Jev question and answer
shapes onto whatever OpenAI-compatible model you already have. If you hold a Jev key and want the
real model, the reference SDK is the tool. If you want the Jev shapes over a model you control,
[jevper](index.md) is.

The reverse direction is more interesting, and it works: **jevper's response model accepts the
service's real payloads verbatim.** Every response captured below was fed to
`SystemOneResponse.model_validate` and accepted — the string legend keys (`"0"`, `"1"`, `"2"`) come
back as the `int` keys `ScoreAnswer.legend` declares, and the service's two-field `usage` fills
jevper's wider `Usage` with its defaults for `n_calls`, `n_retries` and `latency`. The answer
contract is compatible. Only the transport is missing, so bridging the two is a small adapter rather
than a redesign.

There is a second, sharper limit in the way this model is served. The same free model over the
*chat* route — the shape jevper speaks — answers:

```json
{"type": "error", "error": {"type": "FreeTierError", "message": "OpenCode's free tier can only be used from within OpenCode"}}
```

HTTP 403, on `POST https://opencode.ai/zen/v1/chat/completions`. The `systemone` route serves the
same model to an ordinary HTTP client; the chat route does not. So even with a bridge, this
particular deployment is reachable only as System One.

## One request, or one per question

| | Real Jev | jevper |
| --- | --- | --- |
| 11 questions in one call | 1 request, 1.08 s, 1094 input + 353 output tokens | 11 requests, `usage.n_calls == 11` |
| The state | sent once, as the request's `state` field | sent in every request, wrapped in `<document>` |
| The questions | become the request body | become a 305-character system prompt plus a per-question user turn |
| Prompt caching | not applicable | `prompt_cache_key` offered on every request |
| Question isolation | one evaluation per question, by construction | one worker per question (`_BaseClient._run`), so also isolated |

The jevper column was measured against a fake provider object that records requests and answers with
a schema-conforming distribution, because the OpenRouter daily free quota was exhausted
(`X-RateLimit-Remaining: 0`, `limit_source: openrouter_free_tier_daily`) before the comparison ran;
the request count is jevper's own dispatch, not an artefact of the fake. Three questions produced
three requests, eleven produced eleven.

The cost shapes differ even where the totals look alike: the real service billed 1094 input tokens
for the whole batch, jevper's fake billed 100 per request for 1100. What a real run costs depends
entirely on the backend, but the round trips do not: N questions is N sequential-per-question
requests, up to `max_concurrency` in flight, and one pass is one request each.

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

## Where the two disagree about input

jevper validates client-side and the service validates again on arrival, so the interesting column
is where the two sets of rules differ. `typesafe-sdk` 0.7.1 is the third column because it is the
reference client, and it turns out to enforce almost nothing:

| Input | Real service | typesafe-sdk 0.7.1 | jevper 0.7.4 |
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
| An empty `questions` map | 422 with a pydantic `detail` list | `TypeSafeError` before sending | rejected before sending |
| A noul with neither instructions nor criteria | 400 `Noul question must have criteria or instructions` | accepted | accepted |
| An unknown field on a question (`temperature`) | ignored, 200 | rejected, `extra="forbid"` | rejected, `extra="forbid"` |

The one row where jevper is stricter than the service it mirrors is the single-level score. The
service answers it, and the answer is degenerate — all mass on level 0, score 0, confidence 1 — so
jevper's refusal is a judgement, not a compatibility gap, and it is deliberate.

## Four error envelopes, and none of them reach jevper

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
rest. jevper is in neither position, because it never calls this endpoint — and on the surfaces it
does call, it classifies by status and retries the transient set (408, 409, 429 and 5xx), which puts
400, 401 and 402 in the non-retryable branch as a `ProviderError`. That is the right outcome for all
three, though for a different reason than the service's own taxonomy. The reference SDK raises its
own exception family and parses `retry-after` and `retry-after-ms`; jevper honours the same two
headers on its own surfaces, described in [Methods](methods.md).

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
