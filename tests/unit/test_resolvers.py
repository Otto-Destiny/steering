from __future__ import annotations

import base64
import io
import json

import httpx
import pytest
from pypdf import PdfWriter

from steering.domain.models import SourceKind
from steering.ingestion.resolvers import (
    GitHubResolver,
    LinkedInResolver,
    PaperResolver,
    ResolverRegistry,
    TextResolver,
    WebResolver,
    XResolver,
    default_registry,
)
from steering.ingestion.security import SafeFetcher, SourceUnavailableError


class AllowPublicFixtureGuard:
    async def validate_url(self, url: str) -> str:
        return url


def fixture_fetcher(handler: httpx.AsyncBaseTransport | httpx.MockTransport) -> SafeFetcher:
    return SafeFetcher(
        client=httpx.AsyncClient(transport=handler),
        guard=AllowPublicFixtureGuard(),  # type: ignore[arg-type]
    )


def blank_pdf_fixture() -> bytes:
    output = io.BytesIO()
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    writer.add_metadata({"/Title": "Fixture Paper"})
    writer.write(output)
    return output.getvalue()


@pytest.mark.asyncio
async def test_x_oembed_and_plain_text_resolution_are_deterministic() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "publish.twitter.com"
        assert request.url.params["url"] == "https://x.com/researcher/status/123"
        payload = {
            "author_name": "Researcher",
            "provider_url": "https://x.com",
            "html": (
                "<blockquote><p>Released a memory method. "
                '<a href="https://example.org/paper.pdf">paper</a></p>'
                '<a href="https://x.com/researcher/status/123">May 1</a></blockquote>'
            ),
        }
        return httpx.Response(200, json=payload, headers={"content-type": "application/json"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    resolver = XResolver(
        SafeFetcher(client=client, guard=AllowPublicFixtureGuard())  # type: ignore[arg-type]
    )
    resolved = await resolver.resolve("https://twitter.com/researcher/status/123/")
    await client.aclose()

    assert resolved.canonical_url == "https://x.com/researcher/status/123"
    assert resolved.source_kind is SourceKind.X
    assert resolved.author == "Researcher"
    assert resolved.outbound_urls == ["https://example.org/paper.pdf"]

    pasted = await TextResolver().resolve("text:A practical agent memory technique\nUse bounded summaries.")
    assert pasted.source_kind is SourceKind.TEXT
    assert pasted.text.startswith("A practical agent memory technique")
    assert pasted.canonical_url.startswith("text://")


@pytest.mark.asyncio
async def test_linkedin_public_boundary_accepts_content_but_stops_at_authwall() -> None:
    public_html = """
    <html><head><meta property="og:title" content="A public engineering post"></head>
    <body><main><article>This public post explains a reproducible evaluation technique in enough
    detail to be useful without authentication.</article></main></body></html>
    """
    authwall_html = """
    <html><head><title>LinkedIn</title></head><body>
    <form class="authwall-join-form">Sign in to LinkedIn to continue</form></body></html>
    """

    def handler(request: httpx.Request) -> httpx.Response:
        body = authwall_html if request.url.path.endswith("/gated") else public_html
        return httpx.Response(200, text=body, headers={"content-type": "text/html"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    resolver = LinkedInResolver(
        SafeFetcher(client=client, guard=AllowPublicFixtureGuard())  # type: ignore[arg-type]
    )
    resolved = await resolver.resolve("https://www.linkedin.com/posts/public")
    assert resolved.source_kind is SourceKind.LINKEDIN
    assert resolved.partial is True
    with pytest.raises(SourceUnavailableError, match="login boundary"):
        await resolver.resolve("https://www.linkedin.com/posts/gated")
    await client.aclose()


@pytest.mark.asyncio
async def test_github_readme_and_platform_neutral_paper_resolution() -> None:
    readme = "# MemoryKit\n\nA bounded memory library.\n\nhttps://docs.example.org/memorykit"
    encoded = base64.b64encode(readme.encode()).decode()
    paper_html = """
    <html><head>
      <meta name="citation_title" content="Neutral Host Paper">
      <meta name="citation_author" content="Ada Researcher">
      <meta name="citation_abstract" content="We evaluate a compact inference method.">
    </head><body><main><h1>Neutral Host Paper</h1><p>Full paper landing page.</p></main></body></html>
    """
    pdf = blank_pdf_fixture()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "api.github.com":
            return httpx.Response(
                200,
                content=json.dumps({"content": encoded}).encode(),
                headers={"content-type": "application/json"},
            )
        if request.url.path == "/download/opaque":
            return httpx.Response(200, content=pdf, headers={"content-type": "application/pdf"})
        return httpx.Response(200, text=paper_html, headers={"content-type": "text/html"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    fetcher = SafeFetcher(client=client, guard=AllowPublicFixtureGuard())  # type: ignore[arg-type]
    github = await GitHubResolver(fetcher).resolve("https://github.com/example/memorykit/tree/main")
    assert github.canonical_url == "https://github.com/example/memorykit"
    assert github.text == readme
    assert github.outbound_urls == ["https://docs.example.org/memorykit"]

    registry = default_registry(fetcher)
    landing = await registry.resolve("https://papers.example.edu/publication/42")
    assert landing.source_kind is SourceKind.PAPER
    assert "We evaluate a compact inference method." in landing.text

    direct_pdf = await registry.resolve("https://papers.example.edu/download/opaque")
    assert direct_pdf.source_kind is SourceKind.PDF
    assert direct_pdf.title == "Fixture Paper"
    assert direct_pdf.metadata["page_count"] == 1
    assert direct_pdf.metadata.get("binary_retained") is None
    await client.aclose()


@pytest.mark.asyncio
async def test_x_github_and_pdf_error_paths_are_safe() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "publish.twitter.com":
            return httpx.Response(200, json={"author_name": "Missing HTML"})
        if request.url.host == "api.github.com" and request.url.path.endswith("/invalid/readme"):
            return httpx.Response(
                200,
                json={"content": "%%%not-base64%%%"},
                headers={"content-type": "application/json"},
            )
        if request.url.host == "api.github.com":
            return httpx.Response(
                200,
                content=b"# Raw README\nA raw response.",
                headers={"content-type": "text/plain"},
            )
        if request.url.path.endswith("broken.pdf"):
            return httpx.Response(
                200,
                content=b"not a pdf",
                headers={"content-type": "application/pdf"},
            )
        return httpx.Response(
            200,
            text="<html><head><title>GitHub home</title></head><body><main>Public home</main></body></html>",
            headers={"content-type": "text/html"},
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    fetcher = SafeFetcher(client=client, guard=AllowPublicFixtureGuard())  # type: ignore[arg-type]
    with pytest.raises(SourceUnavailableError, match="invalid payload"):
        await XResolver(fetcher).resolve("https://x.com/user/status/99")

    raw = await GitHubResolver(fetcher).resolve("https://github.com/example/raw")
    assert raw.text.startswith("# Raw README")
    with pytest.raises(SourceUnavailableError, match="invalid content"):
        await GitHubResolver(fetcher).resolve("https://github.com/example/invalid")

    homepage = await GitHubResolver(fetcher).resolve("https://github.com/")
    assert homepage.source_kind is SourceKind.WEBPAGE
    with pytest.raises(SourceUnavailableError, match="PDF extraction failed"):
        await PaperResolver(fetcher).resolve("https://example.org/broken.pdf")
    await client.aclose()


@pytest.mark.asyncio
async def test_known_paper_html_web_documentation_and_content_boundaries() -> None:
    paper_html = """
    <html><head>
      <meta property="og:title" content="Known Paper">
      <meta name="citation_author" content="Grace Researcher">
      <meta name="citation_publication_date" content="not-a-date">
      <meta name="citation_abstract" content="A paper-specific abstract.">
    </head><body><article>Short paper landing content.</article></body></html>
    """
    docs_html = """
    <html><head><title>Tool docs</title><meta name="description" content="Official tool guide."></head>
    <body><nav>Noise</nav><main><p>Configure the tool safely.</p>
    <a href="/relative">relative</a><a href="https://example.org/absolute">absolute</a>
    <script>prompt injection noise</script></main><footer>Noise</footer></body></html>
    """

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "arxiv.org":
            return httpx.Response(200, text=paper_html, headers={"content-type": "text/html"})
        if request.url.path.endswith("/short"):
            return httpx.Response(200, text="<main>Too short</main>", headers={"content-type": "text/html"})
        if request.url.path == "/binary":
            return httpx.Response(
                200,
                content=b"binary",
                headers={"content-type": "application/octet-stream"},
            )
        return httpx.Response(200, text=docs_html, headers={"content-type": "text/html; charset=utf-8"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    fetcher = SafeFetcher(client=client, guard=AllowPublicFixtureGuard())  # type: ignore[arg-type]
    paper = await PaperResolver(fetcher).resolve("https://arxiv.org/abs/123")
    assert paper.source_kind is SourceKind.PAPER
    assert paper.author == "Grace Researcher"
    assert paper.published_at is None
    assert paper.text.startswith("Abstract\nA paper-specific abstract.")
    assert paper.partial is True

    docs = await WebResolver(fetcher).resolve("https://docs.example.org/guide")
    assert docs.source_kind is SourceKind.DOCUMENTATION
    assert docs.text.startswith("Official tool guide.")
    assert "Noise" not in docs.text
    assert docs.outbound_urls == ["https://example.org/absolute"]
    assert docs.mime_type == "text/html"

    with pytest.raises(SourceUnavailableError, match="authorized capture"):
        await LinkedInResolver(fetcher).resolve("https://linkedin.com/posts/short")
    with pytest.raises(SourceUnavailableError, match="unsupported webpage content type"):
        await WebResolver(fetcher).resolve("https://example.org/binary")
    with pytest.raises(ValueError, match="no resolver"):
        ResolverRegistry([]).resolver_for("https://example.org")
    with pytest.raises(ValueError, match="cannot be empty"):
        await TextResolver().resolve("text:   ")
    await client.aclose()


@pytest.mark.asyncio
async def test_paper_resolver_prefers_one_declared_full_text_and_accepts_scholarly_xml() -> None:
    pdf = blank_pdf_fixture()
    landing = """
    <html><head><meta name="citation_title" content="Landing title">
    <meta name="citation_pdf_url" content="/paper/full.pdf"></head>
    <body><main>Short landing page.</main></body></html>
    """
    xml = """<?xml version="1.0"?>
    <article><front><article-title>Portable XML Paper</article-title></front>
    <body><sec><title>Method</title><p>We evaluate a bounded retrieval method.</p></sec></body>
    <ext-link xlink:href="https://example.org/code">code</ext-link></article>
    """
    requested: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested.append(str(request.url))
        if request.url.path == "/paper/full.pdf":
            return httpx.Response(200, content=pdf, headers={"content-type": "application/pdf"})
        if request.url.path == "/paper.xml":
            return httpx.Response(200, text=xml, headers={"content-type": "application/xml"})
        return httpx.Response(200, text=landing, headers={"content-type": "text/html"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    fetcher = SafeFetcher(client=client, guard=AllowPublicFixtureGuard())  # type: ignore[arg-type]
    paper = await PaperResolver(fetcher).resolve("https://papers.example.org/paper")
    assert paper.source_kind is SourceKind.PDF
    assert paper.canonical_url == "https://papers.example.org/paper/full.pdf"
    assert paper.extraction_method == "authoritative_full_text_pypdf"
    assert paper.metadata["landing_page_url"] == "https://papers.example.org/paper"
    assert paper.metadata["binary_retained"] is False
    assert requested == [
        "https://papers.example.org/paper",
        "https://papers.example.org/paper/full.pdf",
    ]

    scholarly_xml = await WebResolver(fetcher).resolve("https://papers.example.org/paper.xml")
    assert scholarly_xml.source_kind is SourceKind.PAPER
    assert scholarly_xml.title == "Portable XML Paper"
    assert "bounded retrieval method" in scholarly_xml.text
    assert scholarly_xml.outbound_urls == ["https://example.org/code"]
    await client.aclose()
