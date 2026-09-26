"""A ``clm.CLMClient`` as the client object jevper calls.

**This is not the way to use jevper with CLM.** ``clm-serve`` speaks the System One
wire format directly, so the ordinary route is to point an ``openai`` client at it
and change nothing else::

    SystemOneClient(OpenAI(base_url="http://127.0.0.1:8700/v1", api_key="clm"),
                    model="clm-latest", api="systemone")

See the CLM section of ``docs/local-servers.md`` for that, and for what the server
does with a question.

This file is for the other case: a program that already holds a ``CLMClient`` —
because it also calls ``rank()``, or it reads ``CLM_BASE_URL`` itself — and does not
want to stand up a second client object to reach the same server. It is a
translation, nothing more: jevper's System One surface is reached through two
low-level methods, ``post(path, body=..., cast_to=...)`` and ``get(path,
cast_to=...)``, and a ``CLMClient`` has neither. What it does have is the two
operations those methods stand for.

Two honest caveats, both about ``CLMClient``'s own surface rather than jevper's:

* ``_post`` is **private**. It is the only way to get the raw response body out of
  a ``CLMClient`` — its public ``system_one()`` returns dataclasses, and jevper
  reads the body as sent, because the answers are the provider's numbers. So this
  recipe is tied to the ``contrastive-lm`` version you test it against; if the
  method is renamed upstream, the failure is loud and immediate.
* ``extra_headers`` are **not** plumbed through, because ``_post`` takes no
  headers. That costs nothing in practice: a ``CLMClient`` already sends
  ``Authorization`` from ``CLM_API_KEY``, which is the only header CLM reads. For
  a header CLM does not read, set it on the client's session (``clm._s.headers``)
  instead of through jevper.

There is no async twin. ``CLMClient`` is synchronous, so ``AsyncSystemOneClient``
cannot be built on this — use the ``AsyncOpenAI`` form above if the surrounding
program is async.
"""

from __future__ import annotations

from typing import Any

# jevper names the two routes relative to the client's base URL, which for an
# OpenAI-compatible client already carries the version prefix. A `CLMClient`'s
# base URL is the server root, so its version prefix is added here.
_ROUTES = {"/systemone": "/v1/systemone", "/models": "/v1/models"}


def _with_status(exc: BaseException) -> BaseException:
    """Give a ``CLMClient`` error the attribute jevper reads a status from.

    ``CLMError`` carries ``.status``; jevper's retry and surface-capability logic
    reads ``.status_code``, which is what both official SDKs set. Left as it is, a
    502 from an unreachable encoder reaches jevper as an error with no status, and
    a status is what decides whether another attempt is worth making.
    """
    if getattr(exc, "status_code", None) is not None:
        return exc
    status = getattr(exc, "status", None)
    if isinstance(status, int) and not isinstance(status, bool):
        try:
            exc.status_code = status  # type: ignore[attr-defined]
        except (AttributeError, TypeError):  # a slotted or frozen exception
            pass
    return exc


class CLMTransport:
    """A ``CLMClient`` shaped like the client object jevper calls.

    The two methods are the whole contract: ``post`` for ``POST /v1/systemone`` and
    ``get`` for ``GET /v1/models``. ``cast_to`` is accepted and not acted on — both
    operations answer with already-parsed JSON, and jevper asks for a ``dict``,
    which is what they return.
    """

    def __init__(self, client: Any) -> None:
        self.client = client

    def post(
        self,
        *,
        path: str,
        body: dict[str, Any],
        cast_to: Any = None,
        options: Any = None,
    ) -> Any:
        try:
            payload, _ = self.client._post(_ROUTES[path], body)
        except Exception as exc:  # noqa: BLE001 - re-raised with the status jevper reads
            raise _with_status(exc) from None
        return payload

    def get(self, *, path: str, cast_to: Any = None, options: Any = None) -> Any:
        try:
            # `CLMClient.models()` hands back the list inside the envelope; jevper
            # reads the envelope, because that is what the route returns.
            models = self.client.models()
        except Exception as exc:  # noqa: BLE001 - re-raised with the status jevper reads
            raise _with_status(exc) from None
        return {"models": models}


def main(clm_client: Any = None) -> None:
    from jevper import Choice, Noul, Score, SystemOneClient

    if clm_client is None:  # pragma: no cover - needs a running clm-serve
        from clm import CLMClient

        clm_client = CLMClient()  # CLM_BASE_URL, CLM_API_KEY

    with SystemOneClient(
        CLMTransport(clm_client),
        model="clm-latest",
        api="systemone",
        # CLM answers a noul with neither instructions nor criteria, reading the
        # question id in their place; the hosted Jev service answers 400 for one, so
        # jevper refuses it unless the caller opts out.
        noul_requires_question=False,
    ) as client:
        response = client.system_one(
            state="Customer: my invoice was charged twice and nobody answers the phone!",
            questions={
                "urgency": Noul(instructions="Is this urgent?"),
                "department": Choice(
                    instructions="Which team should handle this?",
                    criteria={
                        "billing": "Charges, invoices, refunds",
                        "technical": "Bugs and outages",
                    },
                ),
                "frustration": Score(
                    instructions="How frustrated is the customer?",
                    criteria=["Calm", "Frustrated", "Very angry"],
                ),
            },
        )

    print(response.answers["urgency"].noul)  # 0.41
    print(response.answers["department"].choice)  # billing
    # The rubric's own levels, on every System One server: CLM spells them as JSON
    # object keys, which are text, and jevper reads them back as the indices the
    # documented arithmetic uses.
    print(sorted(response.answers["frustration"].probabilities))  # [0, 1, 2]
    print(client.model)  # clm-latest
    # CLM reports `usage.input_tokens` and its own `billing_units` (the number of
    # questions). jevper counts the requests it made itself, so `n_calls` is jevper's
    # number and not a restatement of CLM's.
    print(response.usage.n_calls)  # 1


if __name__ == "__main__":
    main()
