"""Validate compact STEERING evaluation result records against dataset cases."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("evaluate/examples.yaml"))
    parser.add_argument("--results", type=Path, default=Path("evaluate/results"))
    parser.add_argument("--start", type=int, default=1, help="One-based first case")
    parser.add_argument("--end", type=int, required=True, help="Inclusive one-based last case")
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main() -> None:
    args = parse_args()
    dataset = yaml.safe_load(args.dataset.read_text(encoding="utf-8"))
    cases = dataset["cases"]
    require(1 <= args.start <= args.end <= len(cases), "Invalid case range")

    checked: list[str] = []
    for case in cases[args.start - 1 : args.end]:
        case_id = case["id"]
        result_path = args.results / f"{case_id}.yaml"
        require(result_path.exists(), f"Missing result: {result_path}")
        result = yaml.safe_load(result_path.read_text(encoding="utf-8"))

        require(result.get("schema_version") == 1, f"Invalid schema version: {case_id}")
        require(result.get("candidate_id") == case_id, f"Candidate mismatch: {case_id}")
        require(
            str(result.get("evaluation_status", "")).startswith("ready_for_human_review"),
            f"Unexpected evaluation status: {case_id}",
        )

        ingestion = result.get("ingestion", {})
        artifact = ingestion.get("selected_artifact", {})
        source_platforms = {source.get("platform") for source in case.get("sources", [])}
        social_source = ingestion.get("social_source")
        if source_platforms & {"x", "linkedin"}:
            require(str(social_source or "").startswith("https://"), f"Missing social source: {case_id}")
        else:
            require(
                social_source is None or str(social_source).startswith("https://"),
                f"Invalid optional social source: {case_id}",
            )
        require(
            str(artifact.get("canonical_url", "")).startswith("https://"),
            f"Missing selected artifact: {case_id}",
        )
        require(bool(artifact.get("type")), f"Missing artifact type: {case_id}")
        require(ingestion.get("retained_binary_artifacts") == [], f"Binary artifact retained: {case_id}")

        knowledge = result.get("knowledge", {})
        require(bool(knowledge.get("concise_idea")), f"Missing concise idea: {case_id}")
        require(bool(knowledge.get("limitations")), f"Missing limitations: {case_id}")
        require(bool(knowledge.get("engineering_relevance")), f"Missing engineering relevance: {case_id}")
        require("source_reconciliation" in result, f"Missing reconciliation: {case_id}")
        require(bool(result.get("graph_ready", {}).get("relations")), f"Missing graph relations: {case_id}")
        checked.append(case_id)

    require(len(checked) == len(set(checked)), "Duplicate candidate results")
    print(f"Validated {len(checked)} result records: {', '.join(checked)}")


if __name__ == "__main__":
    main()
