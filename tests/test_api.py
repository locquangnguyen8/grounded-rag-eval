from fastapi.testclient import TestClient

from grounded_rag.api import create_app


def _client(pipeline, tmp_path):
    return TestClient(create_app(pipeline, feedback_path=tmp_path / "feedback.jsonl"))


def test_health(pipeline, tmp_path):
    r = _client(pipeline, tmp_path).get("/health")
    assert r.status_code == 200 and r.json()["provider"] == "extractive"


def test_ask_returns_answer_with_citation_text(pipeline, tmp_path):
    r = _client(pipeline, tmp_path).post("/ask", json={"question": "What is the API rate limit per key?"})
    body = r.json()
    assert r.status_code == 200 and not body["abstained"]
    assert body["citations"][0]["id"].startswith("data_platform#")
    assert "60 requests per minute" in body["citations"][0]["text"]


def test_feedback_is_stored_with_trace(pipeline, tmp_path):
    client = _client(pipeline, tmp_path)
    trace_id = client.post("/ask", json={"question": "Who is the current program manager?"}).json()["trace_id"]
    r = client.post("/feedback", json={"trace_id": trace_id, "issue": "Should have said it doesn't know"})
    assert r.status_code == 201
    assert trace_id in (tmp_path / "feedback.jsonl").read_text()


def test_feedback_for_unknown_trace_is_404(pipeline, tmp_path):
    r = _client(pipeline, tmp_path).post("/feedback", json={"trace_id": "nope", "issue": "bad"})
    assert r.status_code == 404
