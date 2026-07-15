"""Validate the selected STEERING evaluation corpus."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

ALLOWED_LINK_AUDIT_STATES = {
    "direct_material_present",
    "public_outbound_link_found",
    "comment_or_reply_link_expected",
    "manual_social_link_audit_recommended",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args()

    data = yaml.safe_load(args.dataset.read_text(encoding="utf-8"))
    cases = data.get("cases", [])
    case_ids = [case["id"] for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("Case IDs must be unique")

    source_ids = []
    for case in cases:
        if case.get("decision") != "include":
            raise ValueError(f"Non-included decision found in {case['id']}")
        if not case.get("topics"):
            raise ValueError(f"Missing topics in {case['id']}")
        if not case.get("sources"):
            raise ValueError(f"Missing sources in {case['id']}")
        if case.get("link_audit", {}).get("status") not in ALLOWED_LINK_AUDIT_STATES:
            raise ValueError(f"Invalid link-audit state in {case['id']}")
        if case.get("gold_annotation", {}).get("status") != "pending_human_validation":
            raise ValueError(f"Unexpected gold-annotation state in {case['id']}")

        for source in case["sources"]:
            source_ids.append(source["candidate_id"])
            if source.get("resolution_status") != "resolved":
                raise ValueError(f"Unresolved source retained in {case['id']}: {source['candidate_id']}")
            if not source.get("url"):
                raise ValueError(f"Source URL missing in {case['id']}")

    if len(source_ids) != len(set(source_ids)):
        raise ValueError("A source candidate was assigned to more than one case")

    expected = data.get("summary", {})
    if expected.get("cases") != len(cases):
        raise ValueError("Summary case count does not match the dataset")
    if expected.get("source_candidates") != len(source_ids):
        raise ValueError("Summary source count does not match the dataset")

    print(f"Validated {len(cases)} cases and {len(source_ids)} resolved sources.")


if __name__ == "__main__":
    main()
