# Architecture

STEERING 0.1.0 is a local-first Python application. One Starlette/Uvicorn daemon serves the web UI, typed JSON
API, and Streamable HTTP MCP endpoint. The daemon owns one writable Ladybug database object for its lifetime.

## Runtime boundaries

```text
CLI / HTMX UI / MCP clients
             |
      localhost daemon
       /      |      \
ingestion  intelligence  retrieval
       \      |      /
       domain protocols
             |
     single Ladybug writer
```

Fetching, browser capture, embedding, and generation happen outside database transactions. Only validated domain
records cross the repository boundary. This keeps network latency away from the embedded database's write lock.

## Ingestion and knowledge construction

Resolvers canonicalize a source, capture immutable snapshots, and retain provenance locators. Social posts and
their same-author replies form one artifact. A directly linked paper, repository, benchmark, or documentation page
is a separate artifact with its own evidence role.

Structured LLM output passes schema, enum, evidence-span, entity-resolution, and review-state validation before it
can become graph knowledge. Original snapshots remain immutable when a reviewer resolves an entity or conflict.

## Retrieval and intelligence

Ladybug performs candidate generation through indexed `SearchTerms` for exact names and URLs, native FTS/BM25 over
chunk text, native HNSW cosine search over 768-dimensional chunk embeddings, and native relationship traversal from
the strongest seed artifacts. Filters are applied inside those queries. STEERING then loads only the bounded
candidate set for Reciprocal Rank Fusion, constraint and project-history scoring, duplicate grouping, and diversity
selection; it does not scan the full collection in Python.

Each chunk vector carries its provider, model, immutable revision, dimension, document task mode, normalization
state, and source-content hash. Ladybug stores the active index identity and refuses mixed or dimension-mismatched
vectors. Replacement FTS and HNSW indexes are built before their active metadata is switched. An API or optional
local embedding provider is required; missing configuration fails clearly instead of degrading retrieval.

Relevance, constraint fit, maturity, evidence strength, and time remain distinct signals; publication date is not a
default relevance boost.

## Extension boundaries

`SourceResolver`, `GenerationProvider`, `EmbeddingProvider`, `ArtifactRepository`, and `KnowledgeRetriever` are
typed internal protocols registered through small registries. They are designed for contributors but remain
experimental until 0.1 validates the domain contracts.

See [data-model.md](data-model.md), [privacy-security.md](privacy-security.md), and the [ADRs](adr/README.md).
