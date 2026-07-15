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

Candidate generation combines exact matches, BM25, vectors, graph traversal, and project history. Reciprocal Rank
Fusion combines lexical and vector candidates before duplicate grouping and diversity selection. Relevance,
constraint fit, maturity, evidence strength, and time remain distinct signals; publication date is not a default
relevance boost.

A configured semantic embedding model is the primary intelligence signal. The bundled hash embedder is a
deliberately lexical, keyless fallback; when it is active, STEERING weights BM25 more heavily rather than
misrepresenting feature hashing as semantic understanding.

## Extension boundaries

`SourceResolver`, `GenerationProvider`, `EmbeddingProvider`, `ArtifactRepository`, and `KnowledgeRetriever` are
typed internal protocols registered through small registries. They are designed for contributors but remain
experimental until 0.1 validates the domain contracts.

See [data-model.md](data-model.md), [privacy-security.md](privacy-security.md), and the [ADRs](adr/README.md).
