"""A client that is not an SDK: the whole structural contract, in one class.

jevper never imports ``openai`` or ``anthropic``. It calls whatever object it is
handed, reading attributes or mapping keys with the same helper, so a
twenty-line object can stand in for a full SDK client. This is the smallest one
that answers a question, and the "Using an existing or duck-typed client" page
embeds it.
"""

from __future__ import annotations

from types import SimpleNamespace

from jevper import Noul, SystemOneClient


class ConstantModel:
    """``client.chat.completions.create(**kwargs)``, returning what jevper reads
    off a Chat Completions answer: ``choices[0].message.content``,
    ``choices[0].finish_reason`` and ``usage.prompt_tokens`` /
    ``usage.completion_tokens``. Every one may be a mapping key instead of an
    attribute. A real client adds the rest — streaming, retries, auth — none of
    which jevper requires."""

    class _Completions:
        def create(self, **kwargs: object) -> dict[str, object]:
            assert kwargs["model"] == "constant-model", kwargs["model"]
            return {
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": '{"noul": 0.8}'},
                    }
                ],
                "usage": {"prompt_tokens": 41, "completion_tokens": 7},
            }

    def __init__(self) -> None:
        self.chat = SimpleNamespace(completions=self._Completions())


def main() -> None:
    with SystemOneClient(
        ConstantModel(),
        model="constant-model",
        method="structured",  # ask for JSON, so the answer needs no logprobs
        # api="auto" is the default and the surface follows the object: this one
        # exposes no responses.create and no messages.create, so every call goes
        # to chat_completions without probing a route that is not there.
    ) as client:
        response = client.system_one(
            state="The status page says all systems are operational.",
            questions={
                "is_outage": Noul(
                    instructions="Is the service down right now?"
                )
            },
        )

    answer = response.answers["is_outage"]
    print(answer.noul)  # 0.8
    print(response.usage.n_calls)  # 1
    print(response.debug["api"])  # chat_completions


if __name__ == "__main__":
    main()
