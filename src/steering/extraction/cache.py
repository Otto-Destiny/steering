from __future__ import annotations

import json
from pathlib import Path

from steering.domain.credentials import CredentialDetectedError, reject_high_confidence_credentials
from steering.extraction.schemas import KnowledgeExtraction


class ExtractionCache:
    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def get(self, key: str) -> KnowledgeExtraction | None:
        path = self.directory / f"{key}.json"
        if not path.exists():
            return None
        try:
            serialized = path.read_text(encoding="utf-8")
            reject_high_confidence_credentials(serialized)
            return KnowledgeExtraction.model_validate_json(serialized)
        except (CredentialDetectedError, OSError, ValueError):
            return None

    def put(self, key: str, value: KnowledgeExtraction) -> None:
        serialized = json.dumps(value.model_dump(mode="json"), ensure_ascii=False, indent=2)
        reject_high_confidence_credentials(serialized)
        self.directory.mkdir(parents=True, exist_ok=True)
        temporary = self.directory / f".{key}.tmp"
        target = self.directory / f"{key}.json"
        temporary.write_text(serialized, encoding="utf-8")
        temporary.replace(target)
