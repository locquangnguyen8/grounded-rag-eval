import json
from pathlib import Path

from grounded_rag.evaluate import Case, gate, load_cases, run_eval

ROOT = Path(__file__).resolve().parents[1]


def test_gate_flags_only_real_regressions():
    baseline = {"answer_accuracy": 0.80, "citation_validity": 1.0}
    assert gate({"answer_accuracy": 0.79, "citation_validity": 1.0}, baseline) == []
    failures = gate({"answer_accuracy": 0.70, "citation_validity": 0.95}, baseline)
    assert len(failures) == 2


def test_metrics_on_a_tiny_set(pipeline):
    cases = [
        Case("a", "How often do sondes record a reading?", True, ["sensor_network"], ["15 minutes"]),
        Case("b", "Which company manufactures the sondes?", False, [], []),
    ]
    m = run_eval(pipeline, cases)["metrics"]
    assert m["retrieval_recall_at_k"] == 1.0
    assert m["answer_accuracy"] == 1.0
    assert m["correct_abstention_rate"] == 1.0


def test_current_build_passes_the_committed_baseline(pipeline):
    cases = load_cases(ROOT / "eval" / "golden.jsonl", ROOT / "eval" / "regression.jsonl")
    baseline = json.loads((ROOT / "eval" / "baseline.extractive.json").read_text())
    assert gate(run_eval(pipeline, cases)["metrics"], baseline) == []
