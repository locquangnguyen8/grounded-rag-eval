from grounded_rag.corpus import chunk_document, load_corpus
from grounded_rag.pipeline import DEFAULT_CORPUS
from grounded_rag.text import split_sentences, tokenize


def test_chunk_ids_are_unique_and_cover_every_document():
    chunks = load_corpus(DEFAULT_CORPUS)
    ids = [c.id for c in chunks]
    assert len(ids) == len(set(ids))
    assert {c.doc_id for c in chunks} == {p.stem for p in DEFAULT_CORPUS.glob("*.md")}


def test_long_documents_are_split_into_bounded_chunks():
    body = "# Title\n\n" + "\n\n".join("word " * 40 for _ in range(5))
    chunks = chunk_document("doc", body, max_words=90)
    assert len(chunks) == 3
    assert all(len(c.text.split()) <= 90 for c in chunks)
    assert chunks[0].title == "Title"


def test_tokenizer_keeps_decimals_and_folds_plurals():
    assert tokenize("Sondes hit 0.78 accuracy") == ["sonde", "hit", "0.78", "accuracy"]


def test_sentence_splitter_strips_list_markers():
    assert split_sentences("1. Range check: in range.\n- second item") == ["Range check: in range.", "second item"]


def test_relevant_document_ranks_first(pipeline):
    hits = pipeline.index.search("How are readings sent over LoRaWAN telemetry?")
    assert hits[0].chunk.doc_id == "sensor_network"
