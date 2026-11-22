"""Answer generators. Both must cite chunk ids like [sensor_network#0]."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol

from .retrieval import BM25Index, Hit
from .text import content_tokens, split_sentences

ABSTAIN_TOKEN = "INSUFFICIENT_EVIDENCE"


@dataclass
class Generation:
    text: str
    abstained: bool
    input_tokens: int = 0
    output_tokens: int = 0
    refused: bool = False


class Generator(Protocol):
    name: str

    def generate(self, question: str, hits: list[Hit]) -> Generation: ...


class ExtractiveGenerator:
    """Offline, deterministic baseline: returns the best-matching source sentences.

    It needs no API key, so CI can run the full evaluation on every commit.
    """

    name = "extractive"

    def __init__(self, index: BM25Index, max_sentences: int = 2):
        self.index = index
        self.max_sentences = max_sentences

    def generate(self, question: str, hits: list[Hit]) -> Generation:
        terms = set(content_tokens(question))
        candidates = []
        for rank, hit in enumerate(hits):
            for sentence in split_sentences(hit.chunk.text):
                overlap = terms & set(content_tokens(sentence))
                score = sum(self.index.idf.get(t, 0.0) for t in overlap) * (1.0 - 0.1 * rank)
                if score > 0:
                    candidates.append((score, sentence, hit.chunk.id))
        if not candidates:
            return Generation(ABSTAIN_TOKEN, abstained=True)
        candidates.sort(key=lambda c: c[0], reverse=True)
        best = candidates[: self.max_sentences]
        # Drop a second sentence that is much weaker than the first.
        best = [c for c in best if c[0] >= 0.6 * best[0][0]]
        return Generation(" ".join(f"{s} [{cid}]" for _, s, cid in best), abstained=False)


SYSTEM_PROMPT = f"""You answer questions about a water-quality monitoring program using only the sources provided in the user message.

Rules:
- Use only facts stated in the sources. Do not use outside knowledge.
- After every sentence that states a fact, cite the source id in square brackets, for example [sensor_network#0]. Cite only ids that appear in the sources.
- Keep the answer to one to three sentences.
- If the sources do not contain the answer, reply with exactly {ABSTAIN_TOKEN} and nothing else."""


class ClaudeGenerator:
    """Generates cited answers with Claude. Requires the `anthropic` package and credentials."""

    name = "claude"

    def __init__(self, model: str | None = None, effort: str | None = None, client=None):
        if client is None:
            import anthropic  # optional dependency: pip install ".[claude]"

            client = anthropic.Anthropic()
        self.client = client
        self.model = model or os.getenv("GRAG_CLAUDE_MODEL", "claude-opus-5-5")
        # Short, grounded answers do not need deep reasoning; raise effort if evals show a gain.
        self.effort = effort or os.getenv("GRAG_CLAUDE_EFFORT", "low")

    def generate(self, question: str, hits: list[Hit]) -> Generation:
        sources = "\n\n".join(f'<source id="{h.chunk.id}">\n{h.chunk.text}\n</source>' for h in hits)
        response = self.client.beta.messages.create(
            model=self.model,
            max_tokens=16000,
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",  # if a safety classifier declines, the API retries on a fallback model
            output_config={"effort": self.effort},
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": f"<sources>\n{sources}\n</sources>\n\nQuestion: {question}"}],
        )
        usage = response.usage
        if response.stop_reason == "refusal":
            return Generation(ABSTAIN_TOKEN, True, usage.input_tokens, usage.output_tokens, refused=True)
        text = "".join(block.text for block in response.content if block.type == "text").strip()
        return Generation(text, text.startswith(ABSTAIN_TOKEN), usage.input_tokens, usage.output_tokens)
