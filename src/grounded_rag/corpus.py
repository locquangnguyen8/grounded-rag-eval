"""Load markdown documents and split them into citable chunks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Chunk:
    id: str  # "<doc_id>#<n>", the string the model must cite
    doc_id: str
    title: str
    text: str


def _paragraphs(body: str) -> list[str]:
    blocks, current = [], []
    for line in body.splitlines():
        if line.strip():
            current.append(line.rstrip())
        elif current:
            blocks.append("\n".join(current))
            current = []
    if current:
        blocks.append("\n".join(current))
    return blocks


def chunk_document(doc_id: str, markdown: str, max_words: int = 90) -> list[Chunk]:
    lines = markdown.strip().splitlines()
    title = lines[0].lstrip("# ").strip() if lines and lines[0].startswith("#") else doc_id
    body = "\n".join(lines[1:]) if lines and lines[0].startswith("#") else markdown

    chunks: list[Chunk] = []
    buffer: list[str] = []
    words = 0
    for para in _paragraphs(body):
        n = len(para.split())
        if buffer and words + n > max_words:
            chunks.append(Chunk(f"{doc_id}#{len(chunks)}", doc_id, title, "\n\n".join(buffer)))
            buffer, words = [], 0
        buffer.append(para)
        words += n
    if buffer:
        chunks.append(Chunk(f"{doc_id}#{len(chunks)}", doc_id, title, "\n\n".join(buffer)))
    return chunks


def load_corpus(directory: str | Path, max_words: int = 90) -> list[Chunk]:
    paths = sorted(Path(directory).glob("*.md"))
    if not paths:
        raise FileNotFoundError(f"No .md documents found in {directory}")
    chunks: list[Chunk] = []
    for path in paths:
        chunks.extend(chunk_document(path.stem, path.read_text(encoding="utf-8"), max_words))
    return chunks
