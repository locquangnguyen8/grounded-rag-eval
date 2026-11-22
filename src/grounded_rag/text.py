"""Tokenisation shared by indexing, retrieval and extractive answering."""

from __future__ import annotations

import re

STOPWORDS = frozenset(
    """a an and are as at be been by can did do does for from has have how i in is it its
    of on or that the their this to was were what when where which who whom why will with
    after before during each every any all into than then there these those most much
    many more per about over under
    often long quickly kind type level happen mean used use need needed""".split()
)

_TOKEN = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")


def _stem(token: str) -> str:
    # Deliberately tiny: fold plurals so "sondes" matches "sonde".
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    return [_stem(t) for t in _TOKEN.findall(text.lower())]


def content_tokens(text: str) -> list[str]:
    return [t for t in tokenize(text) if t not in STOPWORDS]


_BULLET = re.compile(r"^(?:[-*]|\d+\.)\s+")
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def split_sentences(text: str) -> list[str]:
    sentences: list[str] = []
    for line in text.splitlines():
        line = _BULLET.sub("", line.strip())
        if line:
            sentences.extend(s.strip() for s in _SENTENCE.split(line) if s.strip())
    return sentences
