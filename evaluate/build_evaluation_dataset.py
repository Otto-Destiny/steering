"""Apply reviewed curation decisions and build evaluation YAML artifacts."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from urllib.parse import urlsplit

import yaml

HIDDEN_LINK_SIGNAL = re.compile(
    r"\b(link|repo|github|paper|code)\b.{0,30}\b(comment|reply|next post|below)\b",
    re.IGNORECASE,
)


def is_external_material(url: str) -> bool:
    host = urlsplit(url).netloc.lower()
    return not (
        host == "x.com" or host.endswith(".x.com") or host == "linkedin.com" or host.endswith(".linkedin.com")
    )


def source_record(candidate: dict[str, object]) -> dict[str, object]:
    enrichment = candidate["enrichment"]
    return {
        "candidate_id": candidate["id"],
        "url": candidate["url"],
        "canonical_url": enrichment.get("canonical_url"),
        "platform": candidate["platform"],
        "author": enrichment.get("author"),
        "title": enrichment.get("title"),
        "text": enrichment.get("post_text"),
        "discovered_urls": enrichment.get("discovered_urls", []),
        "resolution_status": enrichment.get("resolution_status"),
        "resolution_note": enrichment.get("resolution_note"),
        "telegram_message_ids": candidate["telegram"]["message_ids"],
        "saved_at": candidate["telegram"]["saved_at"],
    }


def infer_link_audit(sources: list[dict[str, object]], override: str | None) -> dict[str, object]:
    if override:
        return {"status": override, "comments_inspected": False}

    has_direct_material = any(source["platform"] not in {"x", "linkedin"} for source in sources)
    discovered = [
        url
        for source in sources
        for url in source.get("discovered_urls", [])
        if is_external_material(str(url))
    ]
    text = " ".join(str(source.get("text") or "") for source in sources)
    if has_direct_material:
        status = "direct_material_present"
    elif discovered:
        status = "public_outbound_link_found"
    elif HIDDEN_LINK_SIGNAL.search(text):
        status = "comment_or_reply_link_expected"
    else:
        status = "manual_social_link_audit_recommended"
    return {
        "status": status,
        "comments_inspected": False,
        "public_outbound_urls": list(dict.fromkeys(discovered)),
    }


def dump_yaml(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True, width=120),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("enriched", type=Path)
    parser.add_argument("curation", type=Path)
    parser.add_argument("output_directory", type=Path)
    args = parser.parse_args()

    enriched = yaml.safe_load(args.enriched.read_text(encoding="utf-8"))
    curation = yaml.safe_load(args.curation.read_text(encoding="utf-8"))
    candidates = {candidate["id"]: candidate for candidate in enriched["candidates"]}

    assigned: list[str] = []
    cases = []
    for definition in curation["cases"]:
        candidate_ids = definition["candidates"]
        assigned.extend(candidate_ids)
        sources = [source_record(candidates[candidate_id]) for candidate_id in candidate_ids]
        cases.append(
            {
                "id": definition["id"],
                "decision": "include",
                "relevance": "direct_ai_engineering_value",
                "topics": definition["topics"],
                "sources": sources,
                "link_audit": infer_link_audit(sources, definition.get("link_audit")),
                "gold_annotation": {
                    "status": "pending_human_validation",
                    "expected_artifact_types": [],
                    "expected_entities": [],
                    "expected_claims": [],
                    "expected_architecture_uses": [],
                },
            }
        )

    duplicates = sorted({candidate_id for candidate_id in assigned if assigned.count(candidate_id) > 1})
    unknown = sorted(set(assigned) - set(candidates))
    if duplicates or unknown:
        raise ValueError(f"Invalid selection: duplicates={duplicates}, unknown={unknown}")

    args.output_directory.mkdir(parents=True, exist_ok=True)
    dump_yaml(
        args.output_directory / "examples.yaml",
        {
            "schema_version": 1,
            "dataset_status": "preprocessed_provisional",
            "purpose": (
                "Seed corpus for STEERING ingestion, graph construction, retrieval, "
                "and architecture-recall evaluation."
            ),
            "important_limitations": [
                "Gold expectations still require human validation before scoring model output.",
                "Public X oEmbed and LinkedIn metadata do not include complete comment threads.",
                "Recency and repetition are not evidence of quality.",
            ],
            "summary": {
                "cases": len(cases),
                "source_candidates": sum(len(case["sources"]) for case in cases),
            },
            "cases": cases,
        },
    )


if __name__ == "__main__":
    main()
