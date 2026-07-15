# STEERING evaluation corpus

This directory contains the 51-case seed corpus used to test STEERING ingestion,
knowledge extraction, source reconciliation, graph construction, and retrieval.

## Data path

1. `chat_export.json` is the private, Git-ignored Telegram export.
2. `preprocess_telegram_export.py` extracts and deduplicates saved links.
3. `enrich_candidates.py` adds publicly recoverable source metadata.
4. `curation.yaml` and `build_evaluation_dataset.py` produce `examples.yaml`.
5. The optional capture scripts collect authenticated X author threads and links.
6. `template.yaml` defines the compact extraction shape; `results/` contains the
   51 reviewed records.
7. `validate_dataset.py` and `validate_results.py` verify the corpus, while
   `result_categories.yaml` records the mutually exclusive artifact categories.

`retrieval_prompts.json` contains realistic architecture-time prompts plus seeded
relevant candidate IDs, diversity expectations, and trust checks for retrieval
and MCP evaluation.

Temporary extracted and enriched candidate files should go under `.work/`, which
is ignored by Git. Raw browser sessions are also local and ignored.

## Validation

```powershell
python evaluate/validate_dataset.py evaluate/examples.yaml
python evaluate/validate_results.py --end 51
```

The corpus contains 51 candidate records backed by 61 retained source links.
`behavioral-state-decay` and `proactive-memory-agent` intentionally preserve two
social entry points to the same canonical paper for graph-deduplication testing.
