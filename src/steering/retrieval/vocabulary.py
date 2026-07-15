from __future__ import annotations

ENGINEERING_CONCEPTS: dict[str, set[str]] = {
    "agent_memory": {
        "memory",
        "remember",
        "recall",
        "stale",
        "rollback",
        "long-running",
        "long horizon",
        "persistent state",
        "context drift",
        "across sessions",
        "rolling summaries",
        "forgets why",
    },
    "memory_governance": {"approval", "provenance", "audit", "rollback", "diff", "isolation"},
    "retrieval": {"rag", "retrieval", "index", "vector", "search", "embedding"},
    "visual_retrieval": {"visual", "screenshot", "layout", "diagram", "table", "pixel", "scanned"},
    "document_ai": {"ocr", "document", "pdf", "page", "scan", "parsing"},
    "video_retrieval": {"video", "lecture", "footage", "clip", "frame", "audio"},
    "inference_optimization": {"inference", "latency", "serving", "prefill", "decode", "throughput"},
    "kv_cache": {
        "kv",
        "cache",
        "prefix",
        "prefill",
        "repeated context",
        "same context",
        "repeatedly pays to process",
    },
    "speculative_decoding": {"speculative", "draft", "drafter", "lossless", "acceptance"},
    "model_offloading": {"moe", "expert", "nvme", "ram", "weights", "workstation", "offload"},
    "reasoning_exploration": {"exploration", "grpo", "rollout", "zero reward", "reasoning path"},
    "reasoning_efficiency": {"verbosity", "loop", "repetitive", "overthinking", "shorter reasoning"},
    "test_time_compute": {"test-time", "parallel compute", "self consistency", "voting", "deliberation"},
    "pretraining_efficiency": {"pretraining", "4-bit", "low precision", "training budget", "data selection"},
    "fine_tuning": {
        "fine-tuning",
        "finetuning",
        "generalization",
        "learned facts",
        "skill recombination",
        "recombine learned",
        "knowledge is stored but not routed",
    },
    "agent_training": {"trajectory", "reward", "tool trace", "agent training", "failed trajectory"},
    "agent_skills": {"coding agent", "skill", "project instructions", "codebase", "monorepo", "cuda"},
    "scientific_ai": {"research agent", "literature", "hypothesis", "scientific", "experiment"},
    "speech": {"speech", "transcribe", "speaker", "diarization", "voice", "tts"},
    "tabular_ml": {"tabular", "classification", "regression", "zero-shot", "spreadsheet"},
    "computer_vision": {"vision", "detection", "segmentation", "visual question", "multimodal"},
    "packaged_memory": {
        "memory framework",
        "memory library",
        "personal assistant",
        "open-source options",
        "different projects separate",
        "remember preferences",
    },
    "architecture_frontier_recall": {
        "tried everything",
        "architecture good enough",
        "ideas i may have saved",
        "diverse shortlist",
        "options exhausted",
    },
}


# Query concepts provide a deliberately small bridge from problem language to
# the strategy-family labels used by the knowledge graph. This stays generic:
# it describes engineering approaches, never evaluation prompt or artifact IDs.
CONCEPT_STRATEGY_FAMILIES: dict[str, set[str]] = {
    "agent_memory": {
        "memory_lifecycle",
        "trainable_memory",
        "versioned_memory",
        "context_compression_memory",
        "memory_compression",
        "packaged_context_engine",
        "behavioral_state_memory",
        "governed_memory_writes",
    },
    "memory_governance": {
        "governed_memory_writes",
        "versioned_memory",
        "memory_lifecycle",
    },
    "packaged_memory": {
        "memory_compression",
        "packaged_context_engine",
        "context_compression_memory",
        "versioned_memory",
    },
    "retrieval": {
        "graph_guided_retrieval",
        "embedded_vector_database",
        "compact_vector_index",
    },
    "visual_retrieval": {"visual_page_retrieval", "visual_sparse_retrieval"},
    "kv_cache": {"kv_cache_reuse"},
    "speculative_decoding": {
        "speculative_draft_training",
        "block_diffusion_speculation",
    },
    "model_offloading": {"moe_expert_offloading"},
    "reasoning_exploration": {"reasoning_exploration"},
    "reasoning_efficiency": {"reasoning_efficiency"},
    "test_time_compute": {"latent_test_time_compute", "test_time_compute"},
    "pretraining_efficiency": {"pretraining_efficiency"},
    "fine_tuning": {"activation_routing", "skill_recombination"},
    "agent_training": {"agent_training"},
    "agent_skills": {
        "code_knowledge_graph",
        "generic_skill_discovery",
        "vendor_specific_skills",
        "harness_engineering",
    },
    "scientific_ai": {
        "scientific_agent_workbench",
        "autonomous_ml_experiment_agent",
    },
    "architecture_frontier_recall": {
        "visual_page_retrieval",
        "visual_sparse_retrieval",
        "graph_guided_retrieval",
        "memory_lifecycle",
        "behavioral_state_memory",
        "kv_cache_reuse",
        "harness_engineering",
    },
}


def expand_query(query: str) -> tuple[str, set[str]]:
    lowered = query.lower()
    matched: set[str] = set()
    expansions: list[str] = []
    for concept, phrases in ENGINEERING_CONCEPTS.items():
        if any(phrase in lowered for phrase in phrases):
            matched.add(concept)
            expansions.extend(sorted(phrases))
            expansions.append(concept.replace("_", " "))
    expanded = f"{query} {' '.join(dict.fromkeys(expansions))}" if expansions else query
    return expanded, matched
