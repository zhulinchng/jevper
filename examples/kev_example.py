"""kev, reached the way its own documentation says to reach it.

kev (https://github.com/jaredpalmer/kev) is a family of small decision models
built on Qwen3.5/Qwen3.8. Its API matches TypeSafe's System One exactly, so
jevper's `api="systemone"` surface works against it unchanged.

Start a kev server first:
    pip install kev
    python -m kev.serve --run jaredpalmer/kev-4b --port 8009

Then run this example:
    python examples/kev_example.py
"""

from __future__ import annotations

import os

from openai import OpenAI

from jevper import Choice, Noul, Score, SystemOneClient

KEV_BASE_URL = os.environ.get("KEV_BASE_URL", "http://127.0.0.1:8009/v1")
KEV_MODEL = os.environ.get("KEV_MODEL", "kev-latest")


def main() -> None:
    client = SystemOneClient(
        OpenAI(base_url=KEV_BASE_URL, api_key="local", max_retries=0, timeout=60),
        model=KEV_MODEL,
        api="systemone",
    )

    # List the models kev serves
    models = client.list_models()
    print("Models:", [m.name for m in models])

    # Ask three typed questions about a state
    response = client.system_one(
        state="Customer: my invoice was charged twice and nobody answers the phone!",
        questions={
            "department": Choice(
                instructions="Which team should handle this?",
                criteria={
                    "billing": "Charges, invoices, refunds",
                    "technical": "Bugs and outages",
                },
            ),
            "escalate": Noul(instructions="Does this need urgent human attention?"),
            "frustration": Score(
                instructions="How frustrated is the customer?",
                criteria=["Calm", "Frustrated", "Very angry"],
            ),
        },
    )

    # Read the answers
    dept = response.answers["department"]
    print(f"Department: {dept.choice} (confidence: {dept.confidence:.2f})")
    print(f"  probabilities: {dept.probabilities}")

    escalate = response.answers["escalate"]
    print(f"Escalate: {escalate.noul:.2f}")

    frustration = response.answers["frustration"]
    print(f"Frustration: {frustration.score:.2f}")
    print(f"  legend: {frustration.legend}")
    print(f"  probabilities: {frustration.probabilities}")

    # Usage
    print(f"n_calls: {response.usage.n_calls}")
    print(f"input_tokens: {response.usage.input_tokens}")

    # A bare noul (no instructions, no criteria) — kev accepts it,
    # but jevper refuses it by default. Lift the rule:
    bare = client.system_one(
        state="Refund request.",
        questions={"ack": Noul()},
        noul_requires_question=False,
    )
    print(f"Bare noul: {bare.answers['ack'].noul:.2f}")


if __name__ == "__main__":
    main()
