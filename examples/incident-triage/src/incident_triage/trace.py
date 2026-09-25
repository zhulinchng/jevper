"""The service under MLflow tracing — what a deployment turns on in production.

A triage service is exactly the kind of thing an operator wants a trace for: one call per
ticket, a span per question inside it, and the request fields that explain why a server
answered the way it did. This module wires that up the way the MLflow documentation describes,
then checks the trace it produced: one trace, one child span per question, the model on the
span, and the cache key in the recorded inputs.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from . import tickets
from .checks import as_specs, check
from .profiles import ServerProfile
from .rubrics import PLAIN_RUBRIC
from .service import Triage, build_client

__all__ = ["traced_triage"]


def traced_triage(profile: ServerProfile, *, api: str = "chat_completions", method: str = "structured") -> dict[str, Any]:
    """Classify one ticket inside one MLflow trace, and report what the trace holds.

    MLflow is not a dependency of jevper and not of this project: the integration is a few
    lines a deployment adds, and this is those lines with the assertions a deployment would
    otherwise only make by eye.
    """
    import mlflow

    with tempfile.TemporaryDirectory() as workdir:
        mlflow.set_tracking_uri(f"sqlite:///{Path(workdir) / 'mlflow.db'}")
        mlflow.set_experiment("incident-triage")
        mlflow.openai.autolog()
        if api == "messages":
            mlflow.anthropic.autolog()

        body = profile.body_for(api)
        service = Triage(
            client=build_client(profile, messages=api == "messages"),
            model=profile.model,
            options={"method": method, "api": api, **({"extra_body": body} if body else {})},
            rubric=dict(PLAIN_RUBRIC),
        )

        @mlflow.trace(name="system_one")
        def classify(ticket: dict[str, Any]) -> str:
            response = service.classify(ticket)
            check(as_specs(service.rubric).keys() <= response.answers.keys(), "the rubric lost an answer")
            intent = response.answers.get("intent")
            return intent.choice if intent is not None and hasattr(intent, "choice") else ""

        answer = classify(dict(tickets.SHORT_TICKET))
        mlflow.flush_trace_async_logging()

        traces = mlflow.search_traces(return_type="list")
        check(len(traces) == 1, f"expected one trace, found {len(traces)}")
        trace = traces[0]
        spans = trace.data.spans
        check(len(spans) >= 2, f"expected a parent span and one per question, found {len(spans)} span(s)")
        names = sorted(span.name for span in spans)
        check("system_one" in names, f"the caller's own span is missing from {names}")
        sdk_spans = [span for span in spans if span.name != "system_one"]
        check(
            len(sdk_spans) == len(service.rubric),
            f"{len(sdk_spans)} provider span(s) for {len(service.rubric)} question(s)",
        )
        # MLflow's own attributes live beside the inputs: the model it read off the response, and
        # the request jevper built. The cache key travels in ``extra_body`` — a field of the API,
        # not of any SDK release — so it is in the recorded inputs and in no promoted attribute.
        attributes = getattr(sdk_spans[0], "attributes", None) or {}
        model_attribute = attributes.get("mlflow.llm.model")
        check(bool(model_attribute), "the provider span carries no mlflow.llm.model attribute")
        recorded = sdk_spans[0].inputs or {}
        check(
            recorded.get("model") == profile.model,
            f"the span records model {recorded.get('model')!r}, which is not the one sent",
        )
        cache_key = (recorded.get("extra_body") or {}).get("prompt_cache_key")
        check(bool(cache_key), "the cache key is not in the recorded request inputs")
        check(
            int(attributes.get("mlflow.spanLogLevel", 0)) == 20,
            f"a successful provider call is logged at level {attributes.get('mlflow.spanLogLevel')!r}",
        )
        return {
            "answer": answer,
            "trace_id": trace.info.request_id,
            "spans": names,
            "model": model_attribute,
            "sent_model": recorded.get("model"),
            "cache_key_chars": len(cache_key),
        }
