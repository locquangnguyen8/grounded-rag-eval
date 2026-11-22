from grounded_rag.generation import ABSTAIN_TOKEN, Generation
from grounded_rag.pipeline import RAGPipeline


def test_answer_cites_retrieved_chunks(pipeline):
    a = pipeline.ask("How often are dissolved oxygen sensors recalibrated?")
    assert not a.abstained
    assert "42 days" in a.text
    assert a.citations and a.trace.citations_valid
    assert set(a.citations) <= set(a.trace.retrieved)


def test_abstains_when_corpus_lacks_the_answer(pipeline):
    a = pipeline.ask("What programming language is the forecasting model written in?")
    assert a.abstained
    assert a.text == ABSTAIN_TOKEN
    assert a.trace.abstain_reason == "low_coverage"


def test_off_topic_question_abstains_on_weak_retrieval(pipeline):
    a = pipeline.ask("Recommend a pizza restaurant nearby")
    assert a.abstained
    assert a.trace.abstain_reason == "weak_retrieval"


class _HallucinatingGenerator:
    name = "fake"

    def generate(self, question, hits):
        return Generation("Made up fact [not_a_doc#9].", abstained=False)


def test_citation_to_unretrieved_source_is_flagged():
    p = RAGPipeline(generator=_HallucinatingGenerator())
    a = p.ask("How often do sondes record a reading?")
    assert a.citations == ["not_a_doc#9"]
    assert a.trace.citations_valid is False


def test_traces_are_written_and_retrievable(tmp_path):
    log = tmp_path / "traces.jsonl"
    p = RAGPipeline(log_path=log)
    a = p.ask("What licence is the data published under?")
    assert p.recent(a.trace.trace_id) is a
    assert a.trace.trace_id in log.read_text()
