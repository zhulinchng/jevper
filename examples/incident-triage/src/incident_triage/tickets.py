"""Sample work: the tickets the service triages, in every ``state`` form the library takes."""

from __future__ import annotations

__all__ = ["CONVERSATION", "LONG_TICKET", "SHORT_TICKET", "TICKETS"]

SHORT_TICKET: dict = {
    "id": "INC-1041",
    "customer": "Northwind Traders",
    "plan": "business",
    "seats": 40,
    "subject": "Charged twice for the September subscription",
    "body": (
        "Our card was charged twice for the same subscription this month. The invoice says one "
        "payment, but the bank shows two. Please refund the duplicate."
    ),
}

LONG_TICKET: dict = {
    "id": "INC-1042",
    "customer": "Contoso Health",
    "plan": "enterprise",
    "seats": 1200,
    "subject": "Export job stuck at 99% since the maintenance window",
    "body": (
        "The nightly export has been stuck at 99% since Tuesday's maintenance window. Four "
        "retries, same place. The dashboard is fine and queries are fast; only the export is "
        "affected. This blocks our morning report to the regulator, so it is time-sensitive. "
        + "Log excerpt: " + " ".join(f"step-{index}=ok" for index in range(400)) + " step-export=pending"
    ),
}

# A conversation: the customer wrote, the bot answered, the customer followed up. Handing this
# over as chat turns keeps the roles, which is what a support inbox actually holds.
CONVERSATION: list[dict] = [
    {"role": "user", "content": "Hi — our workspace admin left and nobody can reset the SSO certificate."},
    {"role": "assistant", "content": "I can help with that. Do you have workspace admin access?"},
    {"role": "user", "content": "No, that is the problem. Login for 300 people is broken before 10am daily."},
]

TICKETS: tuple[dict, ...] = (SHORT_TICKET, LONG_TICKET)
