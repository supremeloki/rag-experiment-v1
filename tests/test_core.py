import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

from rag_experiment import (
    EmptyCorpusError,
    NoRetrievalError,
    RagPipeline,
    SimpleEmbedder,
    chunk_text,
    split_sentences,
)

VOCAB = ["database", "cache", "persian", "model", "retrieval", "vector", "query", "index"]


def make_pipeline(top_k: int = 2) -> RagPipeline:
    return RagPipeline(embedder=SimpleEmbedder(VOCAB), top_k=top_k)


DOCS = {
    "db_doc": "The database stores rows. A query hits the index. Cache warms the database.",
    "nlp_doc": "Persian text needs a model. The vector retrieval uses persian tokens.",
}


@pytest.fixture
def pipeline():
    app = make_pipeline()
    app.ingest(DOCS)
    return app


def test_chunk_text_respects_size_and_overlap():
    words = " ".join(f"w{i}" for i in range(100))
    chunks = chunk_text("d", words, chunk_size=20, overlap=5)
    assert len(chunks) == 7
    assert all(len(chunk.text.split()) <= 20 for chunk in chunks)
    assert chunks[0].text.split()[:15] == [f"w{i}" for i in range(15)]


def test_chunk_overlap_shows_shared_words():
    chunks = chunk_text("d", "a b c d e f g h", chunk_size=6, overlap=2)
    first_words = set(chunks[0].text.split())
    second_words = set(chunks[1].text.split())
    assert first_words & second_words


def test_invalid_chunk_config_rejected():
    with pytest.raises(Exception):
        chunk_text("d", "some text", chunk_size=10, overlap=10)


def test_empty_corpus_rejected():
    with pytest.raises(EmptyCorpusError):
        make_pipeline().ingest({})


def test_ask_before_ingest_raises():
    with pytest.raises(NoRetrievalError):
        make_pipeline().ask("anything")


def test_ingest_counts(pipeline):
    assert pipeline.stats.documents == 2
    assert pipeline.stats.chunks > 0
    assert pipeline.store.size() == pipeline.stats.chunks


def test_answer_includes_sources_and_scores(pipeline):
    answer = pipeline.ask("database query index")
    assert answer.sources
    assert len(answer.sources) <= 2
    assert all(score >= 0 for score in answer.retrieval_scores)


def test_relevant_doc_ranks_first(pipeline):
    answer = pipeline.ask("persian vector model")
    assert answer.sources[0].startswith("nlp_doc#")


def test_extractive_generator_returns_sentence_from_context(pipeline):
    answer = pipeline.ask("cache database")
    combined = " ".join(
        chunk.text for chunk in [pipeline.store.all_entries()[0][0]]
    )
    assert answer.text
    assert answer.text != "no relevant context found"


def test_stats_update_after_queries(pipeline):
    before = pipeline.stats.queries_answered
    pipeline.ask("vector retrieval")
    after = pipeline.stats.queries_answered
    assert after == before + 1


def test_split_sentences_persian_period():
    sentences = split_sentences("این جمله تست است. جمله دوم اینجاست")
    assert len(sentences) == 2
