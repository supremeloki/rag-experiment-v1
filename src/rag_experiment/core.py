from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Protocol, Sequence


class RagError(Exception):
    pass


class EmptyCorpusError(RagError):
    pass


class NoRetrievalError(RagError):
    pass


CHUNK_OVERLAP_DEFAULT = 50
CHUNK_SIZE_DEFAULT = 400
WORD_PATTERN: re.Pattern[str] = re.compile(r"\S+")


@dataclass(frozen=True)
class Chunk:
    doc_id: str
    chunk_index: int
    text: str

    @property
    def uid(self) -> str:
        return f"{self.doc_id}#{self.chunk_index}"


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: Chunk
    score: float


@dataclass(frozen=True)
class Answer:
    text: str
    sources: tuple[str, ...]
    retrieval_scores: tuple[float, ...]


@dataclass
class CorpusStats:
    documents: int = 0
    chunks: int = 0
    queries_answered: int = 0
    avg_chunks_per_query: float = 0.0


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?؟。])\s+", text.strip())
    return [part for part in parts if part]


def chunk_text(doc_id: str, text: str,
               chunk_size: int = CHUNK_SIZE_DEFAULT,
               overlap: int = CHUNK_OVERLAP_DEFAULT) -> list[Chunk]:
    if chunk_size <= overlap:
        raise RagError("chunk_size must exceed overlap")
    words = WORD_PATTERN.findall(text)
    if not words:
        return []
    step = chunk_size - overlap
    chunks: list[Chunk] = []
    for index, start in enumerate(range(0, len(words), step)):
        window = words[start:start + chunk_size]
        if not window:
            break
        chunks.append(Chunk(
            doc_id=doc_id,
            chunk_index=index,
            text=" ".join(window),
        ))
    return chunks


class Embedder(Protocol):
    model_name: str

    def embed(self, text: str) -> list[float]: ...


def bag_of_words_embedder(vocabulary: Sequence[str]) -> Callable[[str], list[float]]:
    vocab_index = {term: i for i, term in enumerate(vocabulary)}

    def embed(text: str) -> list[float]:
        vector = [0.0] * len(vocab_index)
        for token in WORD_PATTERN.findall(text.lower()):
            if token in vocab_index:
                vector[vocab_index[token]] += 1.0
        return vector

    return embed


class SimpleEmbedder:
    model_name = "bag-of-words"

    def __init__(self, vocabulary: Sequence[str]) -> None:
        self._embed = bag_of_words_embedder(vocabulary)

    def embed(self, text: str) -> list[float]:
        return self._embed(text)


class VectorStore:
    def __init__(self) -> None:
        self._entries: dict[str, tuple[Chunk, list[float]]] = {}

    def upsert(self, chunk: Chunk, vector: list[float]) -> None:
        self._entries[chunk.uid] = (chunk, vector)

    def size(self) -> int:
        return len(self._entries)

    def all_entries(self) -> list[tuple[Chunk, list[float]]]:
        return list(self._entries.values())


class Retriever:
    def __init__(self, store: VectorStore, embedder: Embedder, top_k: int = 3) -> None:
        self._store = store
        self._embedder = embedder
        self._top_k = max(1, top_k)

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        query_vector = self._embedder.embed(query)
        query_norm = sum(x * x for x in query_vector) ** 0.5
        scored: list[RetrievedChunk] = []
        for chunk, vector in self._store.all_entries():
            denom = sum(y * y for y in vector) ** 0.5
            if query_norm == 0.0 or denom == 0.0:
                similarity = 0.0
            else:
                similarity = sum(a * b for a, b in zip(query_vector, vector)) / (query_norm * denom)
            scored.append(RetrievedChunk(chunk=chunk, score=similarity))
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[: self._top_k]


class Generator(Protocol):
    def generate(self, query: str, contexts: Sequence[Chunk]) -> str: ...


class ExtractiveGenerator:
    def generate(self, query: str, contexts: Sequence[Chunk]) -> str:
        if not contexts:
            return "no relevant context found"
        query_terms = {token.lower() for token in WORD_PATTERN.findall(query)}
        best_sentence = ""
        best_overlap = -1
        for chunk in contexts:
            for sentence in split_sentences(chunk.text):
                terms = {token.lower() for token in WORD_PATTERN.findall(sentence)}
                overlap = len(terms & query_terms)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_sentence = sentence
        return best_sentence or contexts[0].text[:200]


@dataclass
class RagPipeline:
    embedder: Embedder
    generator: Generator = field(default_factory=ExtractiveGenerator)
    top_k: int = 3
    stats: CorpusStats = field(default_factory=CorpusStats)
    store: VectorStore | None = None
    retriever: Retriever | None = None

    def __post_init__(self) -> None:
        if self.store is None:
            self.store = VectorStore()
        if self.retriever is None:
            self.retriever = Retriever(self.store, self.embedder, self.top_k)

    def ingest(self, documents: dict[str, str]) -> int:
        if not documents:
            raise EmptyCorpusError("no documents provided")
        total_chunks = 0
        assert self.store is not None
        for doc_id, text in documents.items():
            chunks = chunk_text(doc_id, text)
            total_chunks += len(chunks)
            for chunk in chunks:
                self.store.upsert(chunk, self.embedder.embed(chunk.text))
        self.stats.documents = len(documents)
        self.stats.chunks = total_chunks
        return total_chunks

    def ask(self, query: str) -> Answer:
        assert self.retriever is not None and self.store is not None
        if self.store.size() == 0:
            raise NoRetrievalError("ingest documents before asking")
        retrieved = self.retriever.retrieve(query)
        answer_text = self.generator.generate(query, [item.chunk for item in retrieved])
        self.stats.queries_answered += 1
        self.stats.avg_chunks_per_query = (
            self.stats.queries_answered and len(retrieved) / self.stats.queries_answered
        )
        return Answer(
            text=answer_text,
            sources=tuple(item.chunk.uid for item in retrieved),
            retrieval_scores=tuple(round(item.score, 4) for item in retrieved),
        )
