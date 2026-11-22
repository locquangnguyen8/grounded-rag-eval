"""Command line: ask, eval, serve, promote."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .evaluate import gate, load_cases, run_eval, to_markdown
from .pipeline import RAGPipeline

ROOT = Path.cwd()
EVAL_DIR = ROOT / "eval"


def cmd_ask(args) -> int:
    a = RAGPipeline(provider=args.provider).ask(args.question)
    print(a.text)
    if args.trace:
        print(json.dumps(a.trace.__dict__, indent=2))
    return 0


def cmd_eval(args) -> int:
    cases = load_cases(EVAL_DIR / "golden.jsonl", EVAL_DIR / "regression.jsonl")
    result = run_eval(RAGPipeline(provider=args.provider), cases)
    baseline_path = EVAL_DIR / f"baseline.{result['provider']}.json"

    failures = None
    if args.gate:
        if not baseline_path.exists():
            print(f"No baseline at {baseline_path}; run with --update-baseline first.", file=sys.stderr)
            return 2
        failures = gate(result["metrics"], json.loads(baseline_path.read_text(encoding="utf-8")), args.tolerance)

    out = EVAL_DIR / "results"
    out.mkdir(exist_ok=True)
    (out / f"{result['provider']}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    report = to_markdown(result, failures)
    (out / f"{result['provider']}.md").write_text(report, encoding="utf-8")
    print(report)

    if args.update_baseline:
        keep = {k: v for k, v in result["metrics"].items() if not k.startswith(("latency", "input_", "output_"))}
        baseline_path.write_text(json.dumps(keep, indent=2) + "\n")
        print(f"Baseline written to {baseline_path}")
    return 1 if failures else 0


def cmd_promote(args) -> int:
    """Turn a reported production failure into a permanent regression test."""
    issues = Path(args.feedback_log)
    records = [json.loads(line) for line in issues.read_text(encoding="utf-8").splitlines() if line.strip()]
    record = next((r for r in records if r["trace_id"] == args.trace_id), None)
    if record is None:
        print(f"trace_id {args.trace_id} not found in {issues}", file=sys.stderr)
        return 1
    case = {
        "id": f"reg-{args.trace_id}",
        "question": record["question"],
        "answerable": not args.unanswerable,
        "expected_docs": args.expected_doc or [],
        "required_facts": args.fact or [],
        "source": "regression",
    }
    with (EVAL_DIR / "regression.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(case) + "\n")
    print(f"Added {case['id']} to eval/regression.jsonl")
    return 0


def cmd_serve(args) -> int:
    import uvicorn

    from .api import create_app

    uvicorn.run(create_app(), host=args.host, port=args.port)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="grounded-rag")
    sub = p.add_subparsers(dest="command", required=True)

    a = sub.add_parser("ask", help="answer one question")
    a.add_argument("question")
    a.add_argument("--provider", default="extractive", choices=["extractive", "claude"])
    a.add_argument("--trace", action="store_true", help="print the request trace")
    a.set_defaults(func=cmd_ask)

    e = sub.add_parser("eval", help="run the evaluation set")
    e.add_argument("--provider", default="extractive", choices=["extractive", "claude"])
    e.add_argument("--gate", action="store_true", help="fail if quality drops below the baseline")
    e.add_argument("--tolerance", type=float, default=0.02)
    e.add_argument("--update-baseline", action="store_true")
    e.set_defaults(func=cmd_eval)

    pr = sub.add_parser("promote", help="add a reported failure to the regression set")
    pr.add_argument("trace_id")
    pr.add_argument("--expected-doc", action="append")
    pr.add_argument("--fact", action="append", help="a fact the correct answer must contain")
    pr.add_argument("--unanswerable", action="store_true", help="the correct behaviour is to abstain")
    pr.add_argument("--feedback-log", default="logs/feedback.jsonl")
    pr.set_defaults(func=cmd_promote)

    s = sub.add_parser("serve", help="run the HTTP API")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(func=cmd_serve)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
