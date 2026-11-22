"""Retrieve -> abstain or generate -> validate citations -> trace."""

from __future__ import annotations

import json
import logging
import os
import re
import time
import uuid
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .corpus import load_corpus
from .generation import ABSTAIN_TOKEN, ClaudeGenerator, ExtractiveGenerator, Generation, Generator
from .retrieval import BM25Index, Hit

log = logging.getLogger("grounded_rag")

CITATION = re.compile(r"\[([a-z0-9_]+#\d+)\]")
DEFAULT_CORPUS = Path(__file__).resolve().parents[2] / "data" / "corpus"


@dataclass
class Config:
    top_k: int = 4
    min_score: float = 2.0  # BM25 score below which the best hit is too weak to answer from
    min_coverage: float = 0.45  # share of (IDF-weighted) query terms the best hit must contain


@dataclass
class Trace:
    trace_id: str
    question: str
    provider: str
    retrieved: list[str]
    top_score: float
    coverage: float
    abstained: bool
    abstain_reason: str | None
    citations: list[str]
    citations_valid: bool
    retrieve_ms: float
    generate_ms: float
    total_ms: float
    input_tokens: int = 0
    output_tokens: int = 0
    refused: bool = False


@dataclass
class Answer:
    text: str
    abstained: bool
    citations: list[str]
    hits: list[Hit] = field(repr=False)
    trace: Trace


class RAGPipeline:
    def __init__(
        self,
        corpus_dir: str | Path = DEFAULT_CORPUS,
        provider: str = "extractive",
        config: Config | None = None,
        generator: Generator | None = None,
        log_path: str | Path | None = None,
    ):
        self.config = config or Config()
        self.index = BM25Index(load_corpus(corpus_dir))
        self.generator = generator or self._make_generator(provider)
        self.log_path = Path(log_path) if log_path else None
        self._recent: OrderedDict[str, Answer] = OrderedDict()  # for feedback lookups

    def _make_generator(self, provider: str) -> Generator:
        if provider == "extractive":
            return ExtractiveGenerator(self.index)
        if provider == "claude":
            return ClaudeGenerator()
        raise ValueError(f"Unknown provider: {provider!r} (use 'extractive' or 'claude')")

    def ask(self, question: str) -> Answer:
        start = time.perf_counter()
        hits = self.index.search(question, k=self.config.top_k)
        retrieved_at = time.perf_counter()

        top_score = hits[0].score if hits else 0.0
        coverage = self.index.evidence_coverage(question, hits[0]) if hits else 0.0
        reason = None
        if not hits or top_score < self.config.min_score:
            reason = "weak_retrieval"
        elif coverage < self.config.min_coverage:
            reason = "low_coverage"

        gen = Generation(ABSTAIN_TOKEN, abstained=True)
        if not reason:
            gen = self.generator.generate(question, hits)
            if gen.abstained:
                reason = "refused" if gen.refused else "generator_abstained"
        text, abstained = gen.text, gen.abstained
        done = time.perf_counter()

        retrieved_ids = [h.chunk.id for h in hits]
        citations = list(dict.fromkeys(CITATION.findall(text)))
        valid = abstained or (bool(citations) and all(c in retrieved_ids for c in citations))

        trace = Trace(
            trace_id=uuid.uuid4().hex[:12],
            question=question,
            provider=self.generator.name,
            retrieved=retrieved_ids,
            top_score=round(top_score, 3),
            coverage=round(coverage, 3),
            abstained=abstained,
            abstain_reason=reason,
            citations=citations,
            citations_valid=valid,
            retrieve_ms=round((retrieved_at - start) * 1000, 2),
            generate_ms=round((done - retrieved_at) * 1000, 2),
            total_ms=round((done - start) * 1000, 2),
            input_tokens=gen.input_tokens,
            output_tokens=gen.output_tokens,
            refused=gen.refused,
        )
        answer = Answer(text, abstained, citations, hits, trace)
        self._record(answer)
        return answer

    def recent(self, trace_id: str) -> Answer | None:
        return self._recent.get(trace_id)

    def _record(self, answer: Answer) -> None:
        self._recent[answer.trace.trace_id] = answer
        while len(self._recent) > 1000:
            self._recent.popitem(last=False)
        line = json.dumps(asdict(answer.trace))
        log.info(line)
        if not answer.trace.citations_valid:
            log.warning("invalid citations in trace %s: %s", answer.trace.trace_id, answer.trace.citations)
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")


def pipeline_from_env() -> RAGPipeline:
    return RAGPipeline(
        corpus_dir=os.getenv("GRAG_CORPUS_DIR", DEFAULT_CORPUS),
        provider=os.getenv("GRAG_PROVIDER", "extractive"),
        log_path=os.getenv("GRAG_TRACE_LOG", "logs/traces.jsonl"),
    )
