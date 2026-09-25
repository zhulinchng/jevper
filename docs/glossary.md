# Glossary

The vocabulary the other pages use, each defined the way the library implements it. The reference for
a term's fields is [API](api.md); for where it is computed, [Architecture](architecture.md).

| Term | What it means here |
| --- | --- |
| State | The untrusted input being classified: a string, a JSON value, or a chat-message list. It is rendered last in the prompt, so every call about one rubric shares the cached prefix. |
| Question | One rubric entry — `Noul`, `Choice` or `Score`, with its instructions and criteria. The key it is passed under is the question id it answers under. |
| Answer | What comes back per question: `NoulAnswer`, `ChoiceAnswer` or `ScoreAnswer`, each carrying the distribution and, where it is defined, `confidence`. |
| Probability | One option's share of the distribution: the provider's own numbers on the JSON methods, a softmax of the label logprobs on the label methods. |
| Distribution | The probabilities for every option of a question, rescaled to sum to 1 when they are off by more than 1e-6. `normalize_probabilities=False` keeps the provider's numbers. |
| Confidence | A value derived from the distribution, never asked of the model. |
| Score | The probability-weighted level index of a `Score` answer, levels numbered from zero. |
| Surface | The wire dialect a request is sent in: `chat_completions`, `responses` or `messages`. `api` picks one; `api="auto"` reads what the client can do. |
| Method | How the answer is elicited: `logprobs`, `grammar`, `structured`, `discrete`, or `auto` to let the client choose. |
| Readout | The step that turns one reply into a distribution. It raises `LabelReadoutError` or `MalformedAnswerError` rather than guessing, and the client decides whether to spend another call. |
| Label | The token a model answers with: `A`–`Z` up to 26 options, two letters past that. Only the JSON methods may use the two-letter range. |
| Native reasoning | The provider's own reasoning field, when the plan is `native`: `reasoning_effort` on Chat Completions, `reasoning` on Responses, `thinking` on Messages. |
| Two-step | The other reasoning plan: an analysis call first, its trace quoted into the answer call, for a provider that has no native reasoning field. |
| Correction turn | The retry turn after an answer that could not be read, built from the failure reason alone. `n_retry_malformed` counts how many are allowed. |
| Capability ladder | The order in which a field jevper added for capability is given up when a server refuses it: `include`, the schema carrier, the reasoning parameters, the cache key, `thinking`. |
| Server limit | A refusal remembered per `(model, surface)`, so a later call starts without the field. Reported in `debug["server_limits"]`. |
| Prompt cache key | The `prompt_cache_key` a provider uses to route requests that can share a cache. jevper derives it from the rubric prefix, never the state; your own key replaces it. |
| `extra_body` | Request fields jevper does not own, merged into the call as the provider's client sends them. A key named there is the one that reaches the wire. |
| Debug record | `response.debug`: the method and surface actually used, one entry per provider attempt including failures, the retry reasons, and anything normalization had to repair. |
| Usage | `response.usage`: provider calls and token counts for the call. `n_calls` counts successful results, and a token count any response omitted stays `None` rather than becoming `0`. |
