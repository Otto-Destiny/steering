"""Publicly enrich extracted evaluation candidates without account credentials."""

from __future__ import annotations

import argparse
import html
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx
import yaml
from bs4 import BeautifulSoup

USER_AGENT = "STEERING-Evaluation-Curator/0.1 (+https://github.com/Otto-Destiny/steering)"
URL_PATTERN = re.compile(r"https?://[^\s<>\"']+")


def clean_text(value: str | None) -> str | None:
    if not value:
        return None
    normalized = re.sub(r"\s+", " ", html.unescape(value)).strip()
    return normalized or None


def unique_urls(values: list[str]) -> list[str]:
    result = []
    seen = set()
    for value in values:
        value = html.unescape(value).rstrip(".,;:)]}")
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def resolve_redirect(session: httpx.Client, url: str) -> str:
    try:
        response = session.get(url, timeout=12, follow_redirects=True)
        return str(response.url)
    except httpx.HTTPError:
        return url


def enrich_x(session: httpx.Client, url: str) -> dict[str, object]:
    response = session.get(
        "https://publish.twitter.com/oembed",
        params={"url": url, "omit_script": "true", "dnt": "true"},
        timeout=20,
    )
    response.raise_for_status()
    data = response.json()
    embed = BeautifulSoup(data.get("html", ""), "html.parser")
    paragraph = embed.select_one("blockquote p")
    post_text = clean_text(paragraph.get_text(" ")) if paragraph else None

    discovered = []
    if paragraph:
        for anchor in paragraph.select("a[href]"):
            label = clean_text(anchor.get_text(" ")) or ""
            href = str(anchor.get("href", ""))
            if not href.startswith(("http://", "https://")):
                continue
            if label.startswith("pic.twitter.com"):
                continue
            discovered.append(resolve_redirect(session, href))

    date_anchor = embed.select_one("blockquote > a:last-child")
    published_at = clean_text(date_anchor.get_text(" ")) if date_anchor else None
    return {
        "title": post_text[:160] if post_text else None,
        "author": clean_text(data.get("author_name")),
        "published_at": published_at,
        "post_text": post_text,
        "canonical_url": data.get("url") or url,
        "discovered_urls": unique_urls(discovered),
        "resolution_status": "resolved" if post_text else "partial",
        "resolution_note": "X public oEmbed; replies and comments were not inspected.",
    }


def meta_content(soup: BeautifulSoup, *selectors: str) -> str | None:
    for selector in selectors:
        node = soup.select_one(selector)
        if node and node.get("content"):
            return clean_text(str(node.get("content")))
    return None


def enrich_page(session: httpx.Client, url: str, platform: str) -> dict[str, object]:
    response = session.get(url, timeout=25, follow_redirects=True)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    title = meta_content(soup, 'meta[property="og:title"]', 'meta[name="twitter:title"]')
    if not title and soup.title:
        title = clean_text(soup.title.get_text(" "))
    description = meta_content(
        soup,
        'meta[property="og:description"]',
        'meta[name="twitter:description"]',
        'meta[name="description"]',
    )
    canonical = soup.select_one('link[rel="canonical"]')
    canonical_url = str(canonical.get("href")) if canonical and canonical.get("href") else str(response.url)

    discovered = unique_urls(URL_PATTERN.findall(description or ""))
    if platform == "linkedin" and description:
        description = re.sub(r"\s*\|\s*\d+\s+comments?\s+on\s+LinkedIn\s*$", "", description)

    note = "Public page metadata."
    if platform == "linkedin":
        note = "LinkedIn public metadata; comments were not inspected."

    return {
        "title": title,
        "author": None,
        "published_at": None,
        "post_text": description,
        "canonical_url": canonical_url,
        "discovered_urls": discovered,
        "resolution_status": "resolved" if title or description else "partial",
        "resolution_note": note,
    }


def enrich_candidate(candidate: dict[str, object]) -> tuple[str, dict[str, object]]:
    candidate_id = str(candidate["id"])
    url = str(candidate["url"])
    platform = str(candidate["platform"])
    with httpx.Client(headers={"User-Agent": USER_AGENT, "Accept-Language": "en-US,en;q=0.8"}) as session:
        try:
            result = enrich_x(session, url) if platform == "x" else enrich_page(session, url, platform)
        except (httpx.HTTPError, ValueError) as exc:
            result = {
                "title": None,
                "author": None,
                "published_at": None,
                "post_text": None,
                "canonical_url": url,
                "discovered_urls": [],
                "resolution_status": "failed",
                "resolution_note": f"{type(exc).__name__}: {exc}",
            }
    return candidate_id, result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    data = yaml.safe_load(args.input.read_text(encoding="utf-8"))
    candidates = data["candidates"]
    results: dict[str, dict[str, object]] = {}

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {executor.submit(enrich_candidate, candidate): candidate for candidate in candidates}
        for future in as_completed(futures):
            candidate_id, enrichment = future.result()
            results[candidate_id] = enrichment

    for candidate in candidates:
        candidate["enrichment"] = results[str(candidate["id"])]

    statuses = {}
    for candidate in candidates:
        status = str(candidate["enrichment"]["resolution_status"])
        statuses[status] = statuses.get(status, 0) + 1
    data["summary"]["enrichment_status_counts"] = dict(sorted(statuses.items()))

    args.output.write_text(
        yaml.safe_dump(data, sort_keys=False, allow_unicode=True, width=120),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
