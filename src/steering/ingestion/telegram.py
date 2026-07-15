from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup

URL = re.compile(r"https?://[^\s<>\]\[\"']+")


def _flatten(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(_flatten(item) for item in value)
    if isinstance(value, dict):
        return _flatten(value.get("text", ""))
    return ""


def telegram_sources(path: Path) -> list[str]:
    message_sources: list[tuple[str, list[str]]] = []
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Telegram JSON export must contain an object")
        messages = payload.get("messages", [])
        for message in messages:
            if not isinstance(message, dict):
                continue
            explicit_links: list[str] = []
            for entity in message.get("text_entities", []):
                if isinstance(entity, dict) and isinstance(entity.get("href"), str):
                    explicit_links.append(entity["href"])
            message_sources.append((_flatten(message.get("text", "")), explicit_links))
    elif path.suffix.lower() in {".html", ".htm"}:
        soup = BeautifulSoup(path.read_text(encoding="utf-8"), "html.parser")
        messages = soup.select(".message")
        for message in messages:
            explicit_links = [
                str(anchor["href"])
                for anchor in message.select("a[href]")
                if str(anchor["href"]).startswith(("http://", "https://"))
            ]
            message_sources.append((message.get_text(" ", strip=True), explicit_links))
    else:
        raise ValueError("Telegram import accepts JSON or HTML exports")
    sources: list[str] = []
    for text, explicit_links in message_sources:
        sources.extend(match.rstrip(".,);]") for match in URL.findall(text))
        sources.extend(explicit_links)
    return list(dict.fromkeys(sources))
