"""Convert a Telegram Desktop HTML export into deduplicated YAML candidates."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import yaml
from bs4 import BeautifulSoup

TRACKING_QUERY_PREFIXES = ("utm_",)
TRACKING_QUERY_KEYS = {"rcm", "si", "s"}


def canonicalize_url(raw_url: str) -> str:
    """Remove known tracking parameters while preserving functional URL parts."""

    parts = urlsplit(raw_url.strip())
    host = parts.netloc.lower()
    if host == "twitter.com" or host.endswith(".twitter.com"):
        host = "x.com"
    if host == "www.x.com":
        host = "x.com"

    kept_query = []
    for key, value in parse_qsl(parts.query, keep_blank_values=True):
        lowered = key.lower()
        if lowered in TRACKING_QUERY_KEYS:
            continue
        if any(lowered.startswith(prefix) for prefix in TRACKING_QUERY_PREFIXES):
            continue
        kept_query.append((key, value))

    path = parts.path
    if path != "/":
        path = path.rstrip("/")

    return urlunsplit((parts.scheme.lower(), host, path, urlencode(kept_query), ""))


def classify_platform(url: str) -> str:
    host = urlsplit(url).netloc.lower()
    if host == "x.com" or host.endswith(".x.com"):
        return "x"
    if host == "linkedin.com" or host.endswith(".linkedin.com"):
        return "linkedin"
    if host == "github.com" or host.endswith(".github.com"):
        return "github"
    if host == "huggingface.co" or host.endswith(".huggingface.co"):
        return "huggingface"
    if host in {"youtube.com", "www.youtube.com", "youtu.be"}:
        return "youtube"
    return "web"


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def parse_export(input_path: Path) -> tuple[str, list[dict[str, object]]]:
    soup = BeautifulSoup(input_path.read_text(encoding="utf-8"), "html.parser")
    title_node = soup.select_one(".page_header .text.bold")
    chat_title = normalize_text(title_node.get_text(" ")) if title_node else "unknown"

    sender: str | None = None
    occurrences: list[dict[str, object]] = []
    for message in soup.select("div.message.default"):
        sender_node = message.select_one(".from_name")
        if sender_node:
            sender = normalize_text(sender_node.get_text(" "))

        date_node = message.select_one(".date.details[title]")
        text_node = message.select_one(".text")
        if text_node is None:
            continue

        message_id = message.get("id", "")
        message_id = message_id.removeprefix("message")
        saved_at = date_node.get("title", "") if date_node else ""
        text = normalize_text(text_node.get_text(" "))

        urls = []
        for anchor in text_node.select("a[href]"):
            href = anchor.get("href", "")
            if href.startswith(("http://", "https://")):
                urls.append(canonicalize_url(href))

        for url in dict.fromkeys(urls):
            occurrences.append(
                {
                    "message_id": message_id,
                    "saved_at": saved_at,
                    "sender": sender,
                    "telegram_text": text,
                    "url": url,
                    "platform": classify_platform(url),
                }
            )

    return chat_title, occurrences


def flatten_telegram_text(value: object) -> str:
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        return ""

    parts = []
    for item in value:
        if isinstance(item, str):
            parts.append(item)
        elif isinstance(item, dict):
            parts.append(str(item.get("text", "")))
    return "".join(parts)


def parse_json_export(input_path: Path) -> tuple[str, list[dict[str, object]]]:
    data = json.loads(input_path.read_text(encoding="utf-8"))
    chat_title = str(data.get("name", "unknown"))
    occurrences: list[dict[str, object]] = []

    for message in data.get("messages", []):
        if message.get("type") != "message":
            continue

        text = normalize_text(flatten_telegram_text(message.get("text", "")))
        urls = []
        for entity in message.get("text_entities", []):
            if entity.get("type") in {"link", "text_link"}:
                candidate = str(entity.get("href") or entity.get("text") or "")
                if candidate.startswith(("http://", "https://")):
                    urls.append(canonicalize_url(candidate))

        for url in dict.fromkeys(urls):
            occurrences.append(
                {
                    "message_id": str(message.get("id", "")),
                    "saved_at": str(message.get("date", "")),
                    "sender": message.get("from"),
                    "telegram_text": text,
                    "url": url,
                    "platform": classify_platform(url),
                }
            )

    return chat_title, occurrences


def parse_input(input_path: Path) -> tuple[str, list[dict[str, object]]]:
    if input_path.suffix.lower() == ".json":
        return parse_json_export(input_path)
    return parse_export(input_path)


def deduplicate(occurrences: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for occurrence in occurrences:
        grouped.setdefault(str(occurrence["url"]), []).append(occurrence)

    candidates = []
    for index, (url, items) in enumerate(grouped.items(), start=1):
        first = items[0]
        candidates.append(
            {
                "id": f"candidate-{index:03d}",
                "url": url,
                "platform": first["platform"],
                "telegram": {
                    "message_ids": [item["message_id"] for item in items],
                    "saved_at": [item["saved_at"] for item in items],
                    "texts": list(
                        dict.fromkeys(str(item["telegram_text"]) for item in items if item["telegram_text"])
                    ),
                    "occurrence_count": len(items),
                },
                "enrichment": {
                    "title": None,
                    "author": None,
                    "published_at": None,
                    "post_text": None,
                    "discovered_urls": [],
                    "resolution_status": "pending",
                },
                "curation": {
                    "decision": "pending",
                    "reason": None,
                    "relevance": None,
                    "topics": [],
                },
            }
        )
    return candidates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    chat_title, occurrences = parse_input(args.input)
    candidates = deduplicate(occurrences)
    platform_counts = Counter(str(candidate["platform"]) for candidate in candidates)

    payload = {
        "schema_version": 1,
        "source": {
            "kind": f"telegram_desktop_{args.input.suffix.lower().lstrip('.')}_export",
            "chat_title": chat_title,
            "export_path": args.input.as_posix(),
        },
        "summary": {
            "url_occurrences": len(occurrences),
            "unique_candidates": len(candidates),
            "platform_counts": dict(sorted(platform_counts.items())),
        },
        "candidates": candidates,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True, width=100),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
