import pytest

from grounded_rag.pipeline import RAGPipeline


@pytest.fixture(scope="session")
def pipeline():
    return RAGPipeline()
