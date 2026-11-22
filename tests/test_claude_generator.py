"""ClaudeGenerator tests with a stub client, so no API key or network is needed."""

from types import SimpleNamespace

from grounded_rag.generation import ABSTAIN_TOKEN, ClaudeGenerator
from grounded_rag.pipeline import RAGPipeline


class _StubMessages:
    def __init__(self, text="", stop_reason="end_turn"):
        self.text, self.stop_reason, self.calls = text, stop_reason, []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        content = [SimpleNamespace(type="text", text=self.text)] if self.text else []
        return SimpleNamespace(
            stop_reason=self.stop_reason, content=content, usage=SimpleNamespace(input_tokens=512, output_tokens=40)
        )


def _client(messages):
    return SimpleNamespace(beta=SimpleNamespace(messages=messages))


def _hits(pipeline, q="How often do sondes record a reading?"):
    return pipeline.index.search(q)


def test_request_sends_sources_and_grounding_rules(pipeline):
    stub = _StubMessages("Every 15 minutes [sensor_network#0].")
    gen = ClaudeGenerator(client=_client(stub))
    out = gen.generate("How often do sondes record a reading?", _hits(pipeline))

    call = stub.calls[0]
    assert call["model"] == "claude-opus-5-5"
    assert call["fallbacks"] == "default"
    assert ABSTAIN_TOKEN in call["system"]
    assert '<source id="sensor_network#0">' in call["messages"][0]["content"]
    assert out.text.startswith("Every 15 minutes") and not out.abstained
    assert (out.input_tokens, out.output_tokens) == (512, 40)


def test_model_abstention_is_detected(pipeline):
    gen = ClaudeGenerator(client=_client(_StubMessages(ABSTAIN_TOKEN)))
    assert gen.generate("q", _hits(pipeline)).abstained


def test_refusal_is_treated_as_abstention(pipeline):
    gen = ClaudeGenerator(client=_client(_StubMessages(stop_reason="refusal")))
    out = gen.generate("q", _hits(pipeline))
    assert out.abstained and out.refused


def test_pipeline_records_tokens_from_claude():
    stub = _StubMessages("Every 15 minutes [sensor_network#0].")
    p = RAGPipeline(generator=ClaudeGenerator(client=_client(stub)))
    a = p.ask("How often do sondes record a reading?")
    assert a.trace.provider == "claude"
    assert a.trace.input_tokens == 512 and a.trace.citations_valid
