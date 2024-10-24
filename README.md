# rag-experiment-v1

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A minimal, dependency-free RAG pipeline — chunking, bag-of-words embedding, cosine retrieval, and extractive answering — built to understand every moving part before scaling up.

## 🚀 Overview

RAG frameworks hide the mechanics. This experiment exposes them: documents split into overlapping word-window **chunks**, embedded with a transparent **bag-of-words** vectorizer, retrieved by plain **cosine similarity**, and answered by an **extractive generator** that returns the sentence most overlapping the query. Every stage is a swappable protocol — replace the embedder or generator with real models and the pipeline logic holds.

## ✨ Features

- **Overlapping chunking:** `chunk_text(doc_id, text, chunk_size=400, overlap=50)` with invalid-config rejection
- **Pluggable embedder:** `Embedder` protocol; `SimpleEmbedder` ships a deterministic bag-of-words
- **In-memory vector store:** dict-backed upsert + full-scan retrieval; zero infra
- **Cosine retriever:** top-k ranking with zero-norm safety (returns 0 similarity, never NaN)
- **Extractive generator:** picks the query-overlapping sentence from retrieved chunks — honest about "no relevant context"
- **Pipeline stats:** documents, chunks, queries answered, avg chunks per query
- **Typed errors:** `EmptyCorpusError`, `NoRetrievalError` under `RagError`
- **Zero dependencies**

## 🚧 Structure

```
rag-experiment-v1/
├── src/rag_experiment/
│   ├── __init__.py
│   └── core.py
├── tests/
│   └── test_core.py
├── README.md
└── pyproject.toml
```

## 📦 Installation

```bash
git clone https://github.com/supremeloki/rag-experiment-v1.git
cd rag-experiment-v1
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
```

## 📋 Requirements

- Python 3.11+
- No runtime dependencies

## 🏃 Quick Start

```python
from rag_experiment import RagPipeline, SimpleEmbedder

pipeline = RagPipeline(embedder=SimpleEmbedder(["database", "persian", "vector"]))
pipeline.ingest({
    "db_doc": "The database stores rows. A query hits the index.",
    "nlp_doc": "Persian text needs a model for vector retrieval.",
})

answer = pipeline.ask("persian vector model")
print(answer.text)
print(answer.sources)
```

## 🔧 Error Handling

```text
RagError
├── EmptyCorpusError      # ingest({}) rejected
└── NoRetrievalError      # ask() before any ingest()
```

## 🧪 Testing

```bash
pytest tests/ -v
```

## 📝 Code Quality

- Full type hints (`X | None` style), frozen chunks and answers
- Zero comments — names carry the meaning
- Protocol-based seams (`Embedder`, `Generator`) keep stages independent

## 📄 License

MIT — see [LICENSE](LICENSE).

## 👤 Author

**Kooroush Masoumi**

---

⭐ Star this repo if you find it useful!
