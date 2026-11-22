"""BM25 retrieval with an evidence-strength signal used for abstention."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

from .corpus import Chunk
from .text import content_tokens


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float


class BM25Index:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1, self.b = k1, b
        # Index the title with the body so "flood event" finds the flood document.
        self._docs = [content_tokens(f"{c.title} {c.text}") for c in chunks]
        self._tf = [Counter(d) for d in self._docs]
        self._avgdl = sum(len(d) for d in self._docs) / len(self._docs)
        df = Counter(t for d in self._docs for t in set(d))
        n = len(self._docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def search(self, query: str, k: int = 4) -> list[Hit]:
        terms = content_tokens(query)
        scored = []
        for i, tf in enumerate(self._tf):
            dl = len(self._docs[i])
            s = 0.0
            for t in terms:
                f = tf.get(t)
                if f:
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self._avgdl))
            if s > 0:
                scored.append(Hit(self.chunks[i], s))
        scored.sort(key=lambda h: h.score, reverse=True)
        return scored[:k]

    def evidence_coverage(self, query: str, hit: Hit) -> float:
        """IDF-weighted share of the query's content terms present in the hit.

        Low coverage means the best chunk only matches generic words, which is the
        main signal that the corpus cannot answer the question.
        """
        terms = set(content_tokens(query))
        if not terms:
            return 0.0
        present = set(content_tokens(f"{hit.chunk.title} {hit.chunk.text}"))
        unseen = max(self.idf.values())  # terms absent from the corpus weigh the most

        def weight(t: str) -> float:
            return self.idf.get(t, unseen)

        total = sum(weight(t) for t in terms)
        return sum(weight(t) for t in terms if t in present) / total
