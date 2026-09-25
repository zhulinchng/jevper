"""The command line a deployment or a test run drives the service with.

``edge`` is local and needs no server; every other subcommand talks to one. The sweep writes
a JSON report, because a matrix of ninety attempts is not something anybody reads on a
terminal — the summary is the short version and the report is the evidence.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from . import edge as edge_suite
from . import sweep as sweep_suite
from .checks import Gap
from .profiles import ServerProfile, profile_for
from .service import AsyncTriage, Triage, TriageReport, build_client, sample_work

__all__ = ["main"]


def _profile(args: argparse.Namespace) -> ServerProfile:
    return profile_for(
        args.server,
        **{
            key: value
            for key, value in (
                ("base_url", getattr(args, "base_url", None)),
                ("model", getattr(args, "model", None)),
                ("extra_body", getattr(args, "extra_body", None)),
                ("responses_extra_body", getattr(args, "responses_extra_body", None)),
            )
            if value
        },
    )


def _emit(payload: Any, out: str | None) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True, default=str)
    if out:
        Path(out).write_text(text + "\n")
        print(f"wrote {out} ({len(text)} chars)")
    else:
        print(text)


def _service(profile: ServerProfile, args: argparse.Namespace, *, messages: bool = False) -> Triage:
    body = profile.extra_body if args.api != "responses" else profile.responses_extra_body
    return Triage(
        client=build_client(profile, messages=messages),
        model=args.model or profile.model,
        options={"method": args.method, "api": args.api, **({"extra_body": body} if body else {})},
    )


def cmd_edge(args: argparse.Namespace) -> int:
    report = edge_suite.run_edge_cases(only=tuple(args.only or ()))
    summary = edge_suite.summarize(report)
    _emit({"summary": summary, "cases": report} if args.full else summary, args.out)
    return 1 if summary["gaps"] else 0


def cmd_sweep(args: argparse.Namespace) -> int:
    profile = _profile(args)
    report = sweep_suite.run_matrix(
        profile,
        only=tuple(args.only or ()),
        methods=tuple(args.methods or ()),
    )
    summary = sweep_suite.summarize(report)
    summary["server"] = profile.name
    summary["model"] = args.model or profile.model
    _emit({"summary": summary, "attempts": report} if args.full else summary, args.out)
    return 1 if summary["gaps"] else 0


def cmd_triage(args: argparse.Namespace) -> int:
    profile = _profile(args)
    service = _service(profile, args)
    work = sample_work()
    with service.session() as client:
        for item in work:
            state = item["conversation"] if "conversation" in item else dict(item)
            response = client.system_one(state=state, questions=service.rubric)
            report = TriageReport.from_response(response, item["id"])
            print(report.to_json())
            if args.explain:
                print(json.dumps(service.explain(response), indent=2, sort_keys=True, default=str))
    return 0


def cmd_batch(args: argparse.Namespace) -> int:
    profile = _profile(args)
    service = _service(profile, args)
    for report in service.triage_all(sample_work()):
        print(report.to_json())
    return 0


def cmd_async(args: argparse.Namespace) -> int:
    profile = _profile(args)
    body = profile.extra_body if args.api != "responses" else profile.responses_extra_body
    service = AsyncTriage(
        client=build_client(profile, messages=args.api == "messages"),
        model=args.model or profile.model,
        options={"method": args.method, "api": args.api, **({"extra_body": body} if body else {})},
    )
    for report in asyncio.run(service.triage_all(sample_work())):
        print(report.to_json())
    return 0


def cmd_trace(args: argparse.Namespace) -> int:
    """One traced call, and what the trace holds — needs ``mlflow`` installed."""
    from .trace import traced_triage

    profile = _profile(args)
    try:
        _emit(traced_triage(profile, api=args.api, method=args.method), args.out)
    except Gap as exc:
        _emit({"status": "gap", "error": str(exc)}, args.out)
        return 1
    return 0


def cmd_all(args: argparse.Namespace) -> int:
    """The local suite and the live matrix, in that order, with one exit code."""
    edge_status = cmd_edge(argparse.Namespace(out=None, full=False, only=()))
    sweep_status = cmd_sweep(args)
    return edge_status or sweep_status


def _common_options() -> argparse.ArgumentParser:
    """The options every subcommand takes, so they work on either side of the subcommand."""
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--server", default="vllm", help="profile name (ollama, llamacpp, vllm, sglang, lmstudio)")
    common.add_argument("--base-url", help="override the profile's base URL")
    common.add_argument("--model", help="override the profile's model id")
    common.add_argument("--extra-body", help="JSON merged into every request (thinking off, budgets)")
    common.add_argument("--responses-extra-body", help="JSON merged into Responses requests")
    common.add_argument("--method", default="auto", help="auto, logprobs, grammar, structured, discrete")
    common.add_argument("--api", default="auto", help="auto, chat_completions, responses, messages")
    common.add_argument("--out", help="write the JSON report here")
    common.add_argument("--full", action="store_true", help="include every attempt, not only the summary")
    return common


def build_parser() -> argparse.ArgumentParser:
    common = _common_options()
    parser = argparse.ArgumentParser(prog="triage", description=__doc__, parents=[common])
    sub = parser.add_subparsers(dest="command", required=True)
    for name, handler, help_text in (
        ("edge", cmd_edge, "the local edge cases (no server needed)"),
        ("sweep", cmd_sweep, "the full live matrix against one server"),
        ("triage", cmd_triage, "triage the sample tickets"),
        ("batch", cmd_batch, "triage the sample tickets in sequence"),
        ("trace", cmd_trace, "one call inside an MLflow trace, and what it recorded"),
        ("all", cmd_all, "the local suite, then the live matrix"),
    ):
        action = sub.add_parser(name, help=help_text, parents=[common])
        action.set_defaults(func=handler)
        action.add_argument("--only", action="append", help="run only these scenarios (repeatable)")
        if name != "edge":
            action.add_argument("--methods", action="append", help="restrict the methods (repeatable)")
            action.add_argument("--apis", action="append", help="restrict the surfaces (repeatable)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover - entry point
    sys.exit(main())
