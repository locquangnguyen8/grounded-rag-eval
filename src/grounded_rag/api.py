"""HTTP service: /ask, /feedback, /health."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .pipeline import RAGPipeline, pipeline_from_env

logging.basicConfig(level=logging.INFO, format="%(message)s")


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)


class Citation(BaseModel):
    id: str
    title: str
    text: str


class AskResponse(BaseModel):
    trace_id: str
    answer: str
    abstained: bool
    citations: list[Citation]
    latency_ms: float


class FeedbackRequest(BaseModel):
    trace_id: str
    issue: str = Field(min_length=3, max_length=2000)


def create_app(pipeline: RAGPipeline | None = None, feedback_path: str | Path | None = None) -> FastAPI:
    app = FastAPI(title="grounded-rag-eval", version="0.1.0")
    app.state.pipeline = pipeline or pipeline_from_env()
    issues = Path(feedback_path or os.getenv("GRAG_FEEDBACK_LOG", "logs/feedback.jsonl"))

    @app.get("/health")
    def health():
        p = app.state.pipeline
        return {"status": "ok", "provider": p.generator.name, "chunks": len(p.index.chunks)}

    @app.post("/ask", response_model=AskResponse)
    def ask(req: AskRequest):
        a = app.state.pipeline.ask(req.question)
        by_id = {h.chunk.id: h.chunk for h in a.hits}
        cites = [Citation(id=c, title=by_id[c].title, text=by_id[c].text) for c in a.citations if c in by_id]
        return AskResponse(
            trace_id=a.trace.trace_id,
            answer=a.text,
            abstained=a.abstained,
            citations=cites,
            latency_ms=a.trace.total_ms,
        )

    @app.post("/feedback", status_code=201)
    def feedback(req: FeedbackRequest):
        """A user reports a bad answer. The issue is stored with its full trace so a
        developer can triage it and promote it into the regression set."""
        a = app.state.pipeline.recent(req.trace_id)
        if a is None:
            raise HTTPException(404, "Unknown or expired trace_id")
        issues.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "reported_at": datetime.now(timezone.utc).isoformat(),
            "issue": req.issue,
            "trace_id": req.trace_id,
            "question": a.trace.question,
            "answer": a.text,
            "retrieved": a.trace.retrieved,
        }
        with issues.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        return {"status": "recorded", "trace_id": req.trace_id}

    return app
