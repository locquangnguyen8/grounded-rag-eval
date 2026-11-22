"""Offline evaluation and the release gate.

Metrics are computed per question, then aggregated. The gate compares the
aggregate against a committed baseline and fails if any quality metric drops
by more than the tolerance.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from pathlib import Path

from .pipeline import RAGPipeline

# Metrics where higher is better; these are gated.
QUALITY_METRICS = (
    "retrieval_recall_at_k",
    "mrr",
    "answer_accuracy",
    "fact_coverage",
    "citation_validity",
    "correct_abstention_rate",
    "answered_rate",
)


@dataclass
class Case:
    id: str
    question: str
    answerable: bool
    expected_docs: list[str]
    required_facts: list[str]
    source: str = "golden"


def load_cases(*paths: str | Path) -> list[Case]:
    cases = []
    for path in paths:
        path = Path(path)
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                data = json.loads(line)
                data.setdefault("source", path.stem)
                cases.append(Case(**data))
    return cases


def _score_case(pipeline: RAGPipeline, case: Case) -> dict:
    answer = pipeline.ask(case.question)
    docs = [h.chunk.doc_id for h in answer.hits]
    rank = next((i + 1 for i, d in enumerate(docs) if d in case.expected_docs), None)
    text = answer.text.lower()
    found = [f for f in case.required_facts if f.lower() in text]
    return {
        "id": case.id,
        "source": case.source,
        "question": case.question,
        "answerable": case.answerable,
        "answer": answer.text,
        "abstained": answer.abstained,
        "abstain_reason": answer.trace.abstain_reason,
        "retrieved_docs": docs,
        "first_relevant_rank": rank,
        "facts_found": len(found),
        "facts_required": len(case.required_facts),
        "correct": (not answer.abstained and len(found) == len(case.required_facts))
        if case.answerable
        else answer.abstained,
        "citations_valid": answer.trace.citations_valid,
        "latency_ms": answer.trace.total_ms,
        "input_tokens": answer.trace.input_tokens,
        "output_tokens": answer.trace.output_tokens,
    }


def _mean(values) -> float:
    values = list(values)
    return round(sum(values) / len(values), 4) if values else 0.0


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    return round(statistics.quantiles(values, n=100, method="inclusive")[int(q) - 1], 2)


def run_eval(pipeline: RAGPipeline, cases: list[Case]) -> dict:
    rows = [_score_case(pipeline, c) for c in cases]
    ans = [r for r in rows if r["answerable"]]
    unans = [r for r in rows if not r["answerable"]]
    latencies = [r["latency_ms"] for r in rows]
    metrics = {
        "cases": len(rows),
        "retrieval_recall_at_k": _mean(r["first_relevant_rank"] is not None for r in ans),
        "mrr": _mean(1 / r["first_relevant_rank"] if r["first_relevant_rank"] else 0 for r in ans),
        "answer_accuracy": _mean(r["correct"] for r in ans),
        "fact_coverage": _mean(r["facts_found"] / r["facts_required"] for r in ans),
        "citation_validity": _mean(r["citations_valid"] for r in rows if not r["abstained"]),
        "correct_abstention_rate": _mean(r["abstained"] for r in unans),
        "answered_rate": _mean(not r["abstained"] for r in ans),
        "latency_p50_ms": _percentile(latencies, 50),
        "latency_p95_ms": _percentile(latencies, 95),
        "input_tokens": sum(r["input_tokens"] for r in rows),
        "output_tokens": sum(r["output_tokens"] for r in rows),
    }
    return {"provider": pipeline.generator.name, "metrics": metrics, "rows": rows}


def gate(metrics: dict, baseline: dict, tolerance: float = 0.02) -> list[str]:
    """Return a list of regressions; empty means the release may ship."""
    failures = []
    for name in QUALITY_METRICS:
        if name not in baseline:
            continue
        if metrics[name] < baseline[name] - tolerance:
            failures.append(f"{name}: {metrics[name]:.3f} < baseline {baseline[name]:.3f} - {tolerance}")
    return failures


def to_markdown(result: dict, failures: list[str] | None = None) -> str:
    m = result["metrics"]
    lines = [f"# Evaluation report ({result['provider']})", "", "| Metric | Value |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in m.items()]
    wrong = [r for r in result["rows"] if not r["correct"]]
    lines += ["", f"## Failed cases ({len(wrong)})", ""]
    if not wrong:
        lines.append("None.")
    for r in wrong:
        kind = "should abstain" if not r["answerable"] else ("abstained" if r["abstained"] else "missing facts")
        lines.append(f"- **{r['id']}** ({kind}): {r['question']}  \n  Answer: {r['answer']}")
    if failures is not None:
        lines += ["", "## Release gate", "", "PASS" if not failures else "FAIL"]
        lines += [f"- {f}" for f in failures]
    return "\n".join(lines) + "\n"
