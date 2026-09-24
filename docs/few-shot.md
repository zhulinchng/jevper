# Few-shot examples

*Independent implementation of the documented System One wire format — not affiliated with TypeSafe.*

A few-shot example is rendered as a chat turn pair: a `user` turn holding the example state plus the same
question block the real call uses, then an `assistant` turn holding the answer in the format the active method
expects. Because the demonstration goes through the same renderers as the real call, the model sees exactly the
answer shape it is being asked for — a label for `logprobs`/`grammar`, a JSON object for
`structured`/`discrete`.

```python
Example(state="Charged twice for one order", answer="billing")
Example(state="Login fails after reset", answer="technical", probabilities={"billing": 0.05, "technical": 0.9})
```

`answer` accepts a label (`"B"`, case-insensitive; two letters such as `"AB"` past 26 options), a `Choice`
criteria key, a `Score` level index, or a bool for `Noul`. An exact criteria key wins over a label spelled the
same way, so `answer="a"` with criteria `{"b": ..., "a": ...}` demonstrates the option keyed `a` rather than the
first label. `probabilities` is rendered only by `method="structured"` — no other answer shape carries a
distribution — but it is validated for every method, before any provider call: the keys must be exactly the
question's, every weight must be non-negative, and a `Noul` weight must be within `[0, 1]`.

## Where examples come from

```mermaid
flowchart TD
    A["question.examples"] -->|"non-empty"| U["use them"]
    A -->|"empty"| B["system_one(examples=...)"]
    B -->|"mapping entry for this question id, or the bare sequence"| U
    B -->|"empty"| C["SystemOneClient(examples=...)"]
    C -->|"mapping entry, or the bare sequence"| U
    C -->|"empty"| D["no examples for this question"]
```

The first level that yields anything wins; the levels do not merge. A mapping is keyed by question id, so
`{"intent": [...]}` applies to the `intent` question only; a bare sequence applies to every question in the
call. This makes it easy to keep a house style on the client and override it for one call or one question.

Examples are resolved and validated per question inside `system_one`, after the question is known, so an
answer that matches no option fails before any request:

```
InvalidQuestionError: question 'intent': example 1: answer 'nope' does not match any option of this question
```

## Rendering

For each example, in order:

| Turn | Content |
| --- | --- |
| `user` | `render_question_turn(example.state, question, labels)` — the question block, then the example state's message contents joined with `"\n\n"` |
| `assistant` | the expected answer (see below) |

The full message list for a call is therefore
`[system] + example turns + [question block] + state turns`. The state comes **last** because it is the part
that changes from call to call: a provider reuses a cached prefix only up to the first token that differs, so
putting the state second — where it used to be — made every call about a new state reprocess the whole prompt.
Measured against ollama, llama.cpp, vLLM and SGLang, moving it to the end takes the reused prefix of a
2400-token prompt from about 40 tokens to 528–1010 (see
[local-servers.md](local-servers.md#prompt-caching)). The state is never repeated.

A state that carries its own `system` or `developer` turn has that content folded into jevper's system
prompt, in the order given, and the rest of the state goes last as usual. No server here accepts a `system`
turn that is not first — llama.cpp's template raises `System message must be at the beginning.` and vLLM and
SGLang answer `400` with the same words — and jevper's own prompt always leads, so the alternative would be to
drop the caller's instruction or fail the call.

Expected answers per method:

| Method | Assistant turn |
| --- | --- |
| `logprobs`, `grammar` | the label, e.g. `B` |
| `structured` | `{"probabilities": {"billing": 1.0, "technical": 0.0, "sales": 0.0}}` — one-hot from `answer` when `Example.probabilities` is omitted, otherwise your numbers verbatim |
| `discrete` | the one-hot shape that method reads: `{"choice": "B"}`, `{"noul": true}` or `{"score": 0}` |

For `Noul` in structured mode the payload is `{"noul": 1.0}` (or `{"noul": 0.0}`); supplying
`probabilities={True: 0.9}` gives `{"noul": 0.9}`, and `{False: 0.1}` gives `{"noul": 0.9}` as well. In
`discrete` mode it is `{"noul": true}` or `{"noul": false}` — that schema has no room for a weight.

Use explicit probabilities when the demonstration should teach calibration, not just format: a one-hot example
teaches the model that answers are certain.

## Wire compatibility

`examples` is declared with `Field(exclude=True)`, so it never appears in a dump:

```python
Choice(criteria={"billing": "...", "technical": "..."}, examples=[...]).model_dump()
# {"type": "choice", "instructions": None, "criteria": {"billing": "...", "technical": "..."}}
```

That keeps `model_dump_json()` on questions — and on `SystemOneResponse` — matching the Jev wire shape exactly.

## Two-step reasoning

Examples are emitted in both passes of `mode="two_step"`: the analysis call and the answer call carry the same
example turns, so the analysis reasons with the same demonstrations in view.
