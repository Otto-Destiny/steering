from __future__ import annotations

import base64
import json
import re
import warnings
from datetime import datetime
from html import unescape
from pathlib import Path
from urllib.parse import quote, urljoin, urlsplit

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from bs4.element import Tag

from steering.domain.models import ResolvedSource, SourceKind
from steering.domain.protocols import SourceResolver
from steering.ingestion.pdf import extract_bounded_pdf
from steering.ingestion.security import SafeFetcher, SourceUnavailableError

URL = re.compile(r"https?://[^\s<>\]\[\"']+")
X_HOSTS = {"x.com", "www.x.com", "twitter.com", "www.twitter.com"}
PAPER_HOSTS = {
    "arxiv.org",
    "export.arxiv.org",
    "openreview.net",
    "aclanthology.org",
    "pubmed.ncbi.nlm.nih.gov",
    "pmc.ncbi.nlm.nih.gov",
    "doi.org",
    "dx.doi.org",
}


def canonical_http_url(url: str) -> str:
    parts = urlsplit(url.strip())
    scheme = parts.scheme.lower()
    host = (parts.hostname or "").lower()
    port = f":{parts.port}" if parts.port and parts.port not in {80, 443} else ""
    path = parts.path.rstrip("/") or "/"
    return f"{scheme}://{host}{port}{path}" + (f"?{parts.query}" if parts.query else "")


def extract_links(text: str) -> list[str]:
    return list(dict.fromkeys(match.rstrip(".,);]") for match in URL.findall(text)))


def _meta(soup: BeautifulSoup, *names: str) -> str | None:
    for name in names:
        tag = soup.find("meta", attrs={"property": name}) or soup.find("meta", attrs={"name": name})
        if isinstance(tag, Tag) and tag.get("content"):
            return str(tag["content"]).strip()
    return None


def _clean_html(html: str) -> tuple[str, str, list[str], dict[str, str]]:
    soup = BeautifulSoup(html, "html.parser")
    for node in soup(["script", "style", "noscript", "svg", "nav", "footer"]):
        node.decompose()
    title = (
        _meta(soup, "og:title", "twitter:title") or (soup.title.string if soup.title else None) or "Web page"
    )
    description = _meta(soup, "og:description", "twitter:description", "description")
    main = soup.find("main") or soup.find("article") or soup.body or soup
    text = main.get_text("\n", strip=True)
    if description and description not in text:
        text = f"{description}\n\n{text}"
    links = []
    for anchor in soup.find_all("a", href=True):
        href = str(anchor["href"])
        if href.startswith(("http://", "https://")):
            links.append(href)
    metadata = {
        key: value
        for key, value in {
            "author": _meta(soup, "author", "article:author", "citation_author"),
            "published_time": _meta(soup, "article:published_time", "date", "citation_publication_date"),
        }.items()
        if value
    }
    return unescape(str(title)).strip(), text, list(dict.fromkeys(links)), metadata


def _parse_date(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _pdf_source(data: bytes, final_url: str, *, extraction_method: str) -> ResolvedSource:
    try:
        reader, pages = extract_bounded_pdf(data)
    except Exception as exc:
        raise SourceUnavailableError(f"PDF extraction failed ({type(exc).__name__})") from None
    text = "\n\n".join(f"[Page {index + 1}]\n{page}" for index, page in enumerate(pages))
    title = (
        str(reader.metadata.title)
        if reader.metadata and reader.metadata.title
        else Path(urlsplit(final_url).path).name or "Paper"
    )
    return ResolvedSource(
        canonical_url=final_url,
        source_kind=SourceKind.PDF,
        title=title,
        text=text,
        mime_type="application/pdf",
        extraction_method=extraction_method,
        partial=not any(page.strip() for page in pages),
        metadata={"page_count": len(pages)},
    )


def _is_pdf_response(data: bytes, content_type: str, final_url: str) -> bool:
    media_type = content_type.split(";", 1)[0].strip().lower()
    return (
        "pdf" in media_type
        or data.lstrip().startswith(b"%PDF-")
        or (final_url.lower().endswith(".pdf") and media_type in {"", "application/octet-stream"})
    )


def _is_scholarly_html(html: str) -> bool:
    soup = BeautifulSoup(html, "html.parser")
    scholarly_meta = (
        "citation_title",
        "citation_author",
        "citation_pdf_url",
        "citation_doi",
        "dc.identifier",
    )
    return any(_meta(soup, name) for name in scholarly_meta)


def _paper_pdf_candidate(html: str, landing_url: str) -> str | None:
    """Choose one authoritative full-text candidate without crawling every link."""

    soup = BeautifulSoup(html, "html.parser")
    declared = _meta(soup, "citation_pdf_url", "eprints.document_url")
    if declared:
        return urljoin(landing_url, declared)

    parts = urlsplit(landing_url)
    host = (parts.hostname or "").lower()
    if host in {"arxiv.org", "export.arxiv.org"} and parts.path.startswith("/abs/"):
        identifier = parts.path.removeprefix("/abs/").strip("/")
        if identifier and ".." not in identifier:
            return f"https://arxiv.org/pdf/{identifier}.pdf"
    return None


def _clean_scholarly_xml(data: bytes) -> tuple[str, str, list[str]]:
    """Extract readable JATS-like XML with no platform-specific dependency."""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", XMLParsedAsHTMLWarning)
        soup = BeautifulSoup(data.decode("utf-8", errors="replace"), "html.parser")
    title_node = soup.find(["article-title", "title"])
    title = title_node.get_text(" ", strip=True) if isinstance(title_node, Tag) else "Research paper"
    body = soup.find("article") or soup.find("body") or soup
    text = body.get_text("\n", strip=True)
    links = [
        str(node.get("href") or node.get("xlink:href"))
        for node in soup.find_all(["a", "ext-link"])
        if str(node.get("href") or node.get("xlink:href") or "").startswith(("http://", "https://"))
    ]
    return title, text, list(dict.fromkeys(links))


class TextResolver:
    name = "text"

    def can_resolve(self, source: str) -> bool:
        return source.startswith("text:") or not source.startswith(("http://", "https://"))

    async def resolve(self, source: str) -> ResolvedSource:
        text = source.removeprefix("text:").strip()
        if not text:
            raise ValueError("plain text source cannot be empty")
        digest = __import__("hashlib").sha256(text.encode("utf-8")).hexdigest()
        return ResolvedSource(
            canonical_url=f"text://{digest}",
            source_kind=SourceKind.TEXT,
            title=text.splitlines()[0][:100],
            text=text,
            extraction_method="plain_text",
            outbound_urls=extract_links(text),
        )


class XResolver:
    name = "x_oembed"

    def __init__(self, fetcher: SafeFetcher) -> None:
        self.fetcher = fetcher

    def can_resolve(self, source: str) -> bool:
        return (urlsplit(source).hostname or "").lower() in X_HOSTS and "/status/" in source

    async def resolve(self, source: str) -> ResolvedSource:
        canonical = canonical_http_url(source)
        parts = urlsplit(canonical)
        if (parts.hostname or "").lower() in {"twitter.com", "www.twitter.com"}:
            canonical = canonical.replace(f"//{parts.netloc}", "//x.com", 1)
        endpoint = "https://publish.twitter.com/oembed"
        data, _, _ = await self.fetcher.get(
            endpoint,
            params={"url": canonical, "omit_script": "1", "dnt": "1"},
        )
        try:
            payload = json.loads(data)
            html = str(payload["html"])
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise SourceUnavailableError("X oEmbed returned an invalid payload") from exc
        soup = BeautifulSoup(html, "html.parser")
        text = soup.get_text(" ", strip=True)
        links = [str(anchor["href"]) for anchor in soup.find_all("a", href=True)]
        return ResolvedSource(
            canonical_url=canonical,
            source_kind=SourceKind.X,
            title=f"X post by {payload.get('author_name') or 'unknown author'}",
            text=text,
            author=str(payload.get("author_name") or "") or None,
            mime_type="text/html",
            extraction_method="x_public_oembed",
            outbound_urls=[url for url in dict.fromkeys(links) if "status/" not in url],
            metadata={"provider_url": payload.get("provider_url")},
        )


class GitHubResolver:
    name = "github"

    def __init__(self, fetcher: SafeFetcher) -> None:
        self.fetcher = fetcher

    def can_resolve(self, source: str) -> bool:
        return (urlsplit(source).hostname or "").lower() in {"github.com", "www.github.com"}

    async def resolve(self, source: str) -> ResolvedSource:
        canonical = canonical_http_url(source)
        parts = [part for part in urlsplit(canonical).path.split("/") if part]
        if len(parts) < 2:
            return await WebResolver(self.fetcher).resolve(canonical)
        owner, repo = parts[0], parts[1].removesuffix(".git")
        endpoint = f"https://api.github.com/repos/{quote(owner, safe='')}/{quote(repo, safe='')}/readme"
        data, headers, _ = await self.fetcher.get(
            endpoint,
            headers={
                "Accept": "application/vnd.github.raw+json",
                "User-Agent": "steering/0.1",
            },
        )
        content_type = headers.get("content-type", "")
        if "json" in content_type:
            try:
                payload = json.loads(data)
                encoded = str(payload.get("content", "")).replace("\n", "")
                text = base64.b64decode(encoded, validate=True).decode("utf-8", errors="replace")
            except (json.JSONDecodeError, ValueError, TypeError) as exc:
                raise SourceUnavailableError("GitHub README API returned invalid content") from exc
        else:
            text = data.decode("utf-8", errors="replace")
        return ResolvedSource(
            canonical_url=f"https://github.com/{owner}/{repo}",
            source_kind=SourceKind.GITHUB,
            title=f"{owner}/{repo}",
            text=text,
            mime_type="text/markdown",
            extraction_method="github_readme_api",
            outbound_urls=extract_links(text),
            metadata={"owner": owner, "repository": repo},
        )


class PaperResolver:
    name = "paper"

    def __init__(self, fetcher: SafeFetcher) -> None:
        self.fetcher = fetcher

    def can_resolve(self, source: str) -> bool:
        parts = urlsplit(source)
        host = (parts.hostname or "").lower()
        return host in PAPER_HOSTS or parts.path.lower().endswith(".pdf")

    async def resolve(self, source: str) -> ResolvedSource:
        canonical = canonical_http_url(source)
        data, headers, final_url = await self.fetcher.get(canonical)
        content_type = headers.get("content-type", "").lower()
        if _is_pdf_response(data, content_type, final_url):
            return _pdf_source(data, final_url, extraction_method="pypdf_text")
        if "xml" in content_type:
            title, text, links = _clean_scholarly_xml(data)
            return ResolvedSource(
                canonical_url=final_url,
                source_kind=SourceKind.PAPER,
                title=title,
                text=text,
                mime_type=content_type.split(";", 1)[0],
                extraction_method="scholarly_xml",
                partial=len(text) < 500,
                outbound_urls=links,
            )
        html = data.decode("utf-8", errors="replace")
        pdf_candidate = _paper_pdf_candidate(html, final_url)
        if pdf_candidate and canonical_http_url(pdf_candidate) != canonical_http_url(final_url):
            try:
                pdf_data, pdf_headers, pdf_url = await self.fetcher.get(pdf_candidate)
            except SourceUnavailableError:
                pass
            else:
                pdf_content_type = pdf_headers.get("content-type", "").lower()
                if _is_pdf_response(pdf_data, pdf_content_type, pdf_url):
                    full_text = _pdf_source(
                        pdf_data,
                        pdf_url,
                        extraction_method="authoritative_full_text_pypdf",
                    )
                    return full_text.model_copy(
                        update={
                            "metadata": {
                                **full_text.metadata,
                                "landing_page_url": final_url,
                                "binary_retained": False,
                            }
                        }
                    )
        title, text, links, metadata = _clean_html(html)
        abstract = _meta(BeautifulSoup(html, "html.parser"), "citation_abstract", "dc.description")
        if abstract and abstract not in text:
            text = f"Abstract\n{abstract}\n\n{text}"
        return ResolvedSource(
            canonical_url=final_url,
            source_kind=SourceKind.PAPER,
            title=title,
            text=text,
            author=metadata.get("author"),
            published_at=_parse_date(metadata.get("published_time")),
            mime_type="text/html",
            extraction_method="scholarly_html",
            partial=len(text) < 500,
            outbound_urls=links,
            metadata=metadata,
        )


class LinkedInResolver:
    name = "linkedin_public"

    def __init__(self, fetcher: SafeFetcher) -> None:
        self.fetcher = fetcher

    def can_resolve(self, source: str) -> bool:
        return (urlsplit(source).hostname or "").lower() in {"linkedin.com", "www.linkedin.com"}

    async def resolve(self, source: str) -> ResolvedSource:
        data, _, final_url = await self.fetcher.get(canonical_http_url(source))
        html = data.decode("utf-8", errors="replace")
        lowered = html.lower()
        if any(
            marker in lowered
            for marker in (
                "linkedin authwall",
                "authwall-join-form",
                "sign in to linkedin",
                "join linkedin",
            )
        ):
            raise SourceUnavailableError(
                "LinkedIn requires authorized capture; public extraction stopped at the login boundary"
            )
        title, text, links, metadata = _clean_html(html)
        if len(text) < 40:
            raise SourceUnavailableError(
                "LinkedIn public metadata is unavailable; authorized capture is required"
            )
        return ResolvedSource(
            canonical_url=final_url,
            source_kind=SourceKind.LINKEDIN,
            title=title,
            text=text,
            author=metadata.get("author"),
            published_at=_parse_date(metadata.get("published_time")),
            mime_type="text/html",
            extraction_method="linkedin_public_page",
            partial=True,
            outbound_urls=links,
            metadata=metadata,
        )


class WebResolver:
    name = "web"

    def __init__(self, fetcher: SafeFetcher) -> None:
        self.fetcher = fetcher

    def can_resolve(self, source: str) -> bool:
        return source.startswith(("http://", "https://"))

    async def resolve(self, source: str) -> ResolvedSource:
        data, headers, final_url = await self.fetcher.get(canonical_http_url(source))
        content_type = headers.get("content-type", "").lower()
        if _is_pdf_response(data, content_type, final_url):
            return _pdf_source(data, final_url, extraction_method="platform_neutral_pypdf_text")
        if "xml" in content_type:
            title, text, links = _clean_scholarly_xml(data)
            return ResolvedSource(
                canonical_url=final_url,
                source_kind=SourceKind.PAPER,
                title=title,
                text=text,
                mime_type=content_type.split(";", 1)[0],
                extraction_method="scholarly_xml",
                partial=len(text) < 500,
                outbound_urls=links,
            )
        if "html" not in content_type and not content_type.startswith("text/"):
            raise SourceUnavailableError(f"unsupported webpage content type: {content_type or 'unknown'}")
        html = data.decode("utf-8", errors="replace")
        title, text, links, metadata = _clean_html(html)
        scholarly = _is_scholarly_html(html)
        if scholarly:
            abstract = _meta(BeautifulSoup(html, "html.parser"), "citation_abstract", "dc.description")
            if abstract and abstract not in text:
                text = f"Abstract\n{abstract}\n\n{text}"
        return ResolvedSource(
            canonical_url=final_url,
            source_kind=(
                SourceKind.PAPER
                if scholarly
                else SourceKind.DOCUMENTATION
                if "docs." in (urlsplit(final_url).hostname or "")
                else SourceKind.WEBPAGE
            ),
            title=title,
            text=text,
            author=metadata.get("author"),
            published_at=_parse_date(metadata.get("published_time")),
            mime_type=content_type.split(";", 1)[0] or "text/html",
            extraction_method="scholarly_html" if scholarly else "public_webpage",
            partial=len(text) < 200,
            outbound_urls=links,
            metadata=metadata,
        )


class ResolverRegistry:
    def __init__(self, resolvers: list[SourceResolver]) -> None:
        self.resolvers = resolvers

    def resolver_for(self, source: str) -> SourceResolver:
        for resolver in self.resolvers:
            if resolver.can_resolve(source):
                return resolver
        raise ValueError("no resolver accepted the source")

    async def resolve(self, source: str) -> ResolvedSource:
        return await self.resolver_for(source).resolve(source)


def default_registry(fetcher: SafeFetcher) -> ResolverRegistry:
    return ResolverRegistry(
        [
            XResolver(fetcher),
            LinkedInResolver(fetcher),
            GitHubResolver(fetcher),
            PaperResolver(fetcher),
            WebResolver(fetcher),
            TextResolver(),
        ]
    )
