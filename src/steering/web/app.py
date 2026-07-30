from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, TypeVar

import anyio.to_thread
from pydantic import BaseModel, ValidationError
from starlette.applications import Starlette
from starlette.datastructures import FormData, UploadFile
from starlette.exceptions import HTTPException
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, RedirectResponse, Response
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles
from starlette.templating import Jinja2Templates

from steering import SCHEMA_VERSION, __version__
from steering.database.repository import CorruptRecordError
from steering.domain.models import Project, SearchQuery
from steering.domain.protocols import ArtifactRepository, ImageUnderstandingProvider
from steering.extraction.service import EvidenceValidationError
from steering.ingestion.bookmarks import (
    MAX_EXPORT_BYTES,
    UnsupportedBookmarkExportError,
    bookmark_sources,
)
from steering.ingestion.browser import (
    BrowserAuthenticationRequired,
    BrowserCaptureUnavailable,
    BrowserDependencyUnavailable,
)
from steering.ingestion.login_session import BrowserLoginSession, LoginStatus
from steering.ingestion.security import SourceUnavailableError
from steering.ingestion.service import IngestionService, ThreadPolicy
from steering.ingestion.uploads import MAX_UPLOAD_BYTES, UnsupportedUploadError, resolve_upload
from steering.ingestion.x_api import (
    DEFAULT_REDIRECT_PATH,
    PendingAuthorization,
    XApiClient,
    XApiError,
    XApiNotAuthorized,
    start_authorization,
)
from steering.intelligence.service import SteeringEngine
from steering.providers.openai_compatible import ProviderConnectionError, ProviderTimeoutError
from steering.web.capture import AuthorizedBrowserCapture, ResolvedSourceIngestion
from steering.web.providers import ProviderSettingsService
from steering.web.schemas import (
    BackupInput,
    BrowserCaptureInput,
    DecisionInput,
    DesignInput,
    IngestionInput,
    IssueResolutionInput,
    OutcomeInput,
    ProjectInput,
    ProviderInput,
    RetireInput,
    SearchInput,
)
from steering.web.security import MAX_REQUEST_BYTES, LocalMutationGuardMiddleware

LOGGER = logging.getLogger(__name__)
TInput = TypeVar("TInput", bound=BaseModel)
WEB_ROOT = Path(__file__).resolve().parent
ISSUE_ACTIONS = ("accept_correction", "keep_both", "dismiss", "reject")
#: How many artifacts the directory lists at once. Generous enough to be a real
#: index of a personal corpus, bounded so a large graph cannot stall the browser.
DIRECTORY_LIMIT = 100
DIRECTORY_SORTS = ("newest", "oldest", "title")
#: The platforms a signed-in capture can read, and where each one signs in.
#: One managed profile is shared, so each is reported by the cookies it left.
BROWSER_PLATFORMS = (
    {"name": "X", "host": "x.com", "login_url": "https://x.com/i/flow/login"},
    {"name": "LinkedIn", "host": "linkedin.com", "login_url": "https://www.linkedin.com/login"},
)


def _bookmarklet_source() -> str:
    """The one-line bookmarklet the Add page offers for dragging."""

    return (WEB_ROOT / "static" / "bookmarklet.min.txt").read_text(encoding="utf-8").strip()


def _json(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json(item) for item in value]
    if isinstance(value, dict):
        return {key: _json(item) for key, item in value.items()}
    return value


def _validation_payload(exc: ValidationError) -> dict[str, Any]:
    return {
        "error": "Request validation failed.",
        "details": [
            {"location": list(item["loc"]), "message": item["msg"], "type": item["type"]}
            for item in exc.errors(include_input=False, include_context=False, include_url=False)
        ],
    }


async def _payload(request: Request, model: type[TInput]) -> TInput:
    body = await request.body()
    if len(body) > MAX_REQUEST_BYTES:
        raise HTTPException(413, "Request body is too large.")
    try:
        return model.model_validate_json(body)
    except ValueError as exc:
        if isinstance(exc, ValidationError):
            raise
        raise HTTPException(400, "Request body must be valid JSON.") from None


async def _form(request: Request) -> FormData:
    body = await request.body()
    if len(body) > MAX_REQUEST_BYTES:
        raise HTTPException(413, "Request body is too large.")
    return await request.form(max_files=2, max_fields=50, max_part_size=MAX_UPLOAD_BYTES)


def _string(form: FormData, name: str, default: str = "") -> str:
    value = form.get(name, default)
    return value.strip() if isinstance(value, str) else default


#: What a capture failure means in the words of someone reading the page, rather
#: than the name of the exception that carried it.
_FAILURE_REASONS = {
    "SourceUnavailableError": "could not be reached, or needs you to be signed in",
    "XPostUnavailable": "the post is deleted, private, or age-restricted",
    "BrowserAuthenticationRequired": "needs a signed-in browser session",
    "BrowserCaptureUnavailable": "the signed-in browser could not read it",
    "UnsupportedUploadError": "that kind of file is not supported",
    "EvidenceValidationError": "nothing source-backed could be extracted from it",
    "ValueError": "is not a source STEERING can read",
}


def _failure_reason(code: str) -> str:
    return _FAILURE_REASONS.get(code, "could not be captured")


def _lines(value: str) -> list[str]:
    return [line.strip() for line in value.splitlines() if line.strip()]


class WebController:
    def __init__(
        self,
        *,
        engine: SteeringEngine,
        ingestion: IngestionService,
        repository: ArtifactRepository,
        provider_settings: ProviderSettingsService,
        resolved_ingestion: ResolvedSourceIngestion | None,
        browser_capture: AuthorizedBrowserCapture | None,
        login_session: BrowserLoginSession | None,
        image_provider: ImageUnderstandingProvider | None,
        templates: Jinja2Templates,
        on_record_changed: Callable[[], None] | None = None,
        x_api: XApiClient | None = None,
        x_api_client_id: str | None = None,
    ) -> None:
        self.engine = engine
        self.ingestion = ingestion
        self.repository = repository
        self.provider_settings = provider_settings
        self.resolved_ingestion = resolved_ingestion
        self.browser_capture = browser_capture
        self.login_session = login_session
        self.image_provider = image_provider
        self.templates = templates
        self.on_record_changed = on_record_changed
        self.x_api = x_api
        self.x_api_client_id = x_api_client_id
        self._pending_x_authorization: PendingAuthorization | None = None

    def context(self, request: Request, **values: Any) -> dict[str, Any]:
        return {
            "request": request,
            "current_path": request.url.path,
            "unresolved_count": len(self.repository.list_issues(unresolved_only=True)),
            **values,
        }

    def _issue_evidence(self, issues: Sequence[Any]) -> dict[str, list[dict[str, str]]]:
        result: dict[str, list[dict[str, str]]] = {}
        for issue in issues:
            record = self.repository.get_record(issue.artifact_id)
            if record is None:
                continue
            spans = {span.id: span for span in record.evidence_spans}
            snapshots = {snapshot.id: snapshot for snapshot in record.snapshots}
            rows: list[dict[str, str]] = []
            for span_id in issue.evidence_span_ids:
                span = spans.get(span_id)
                snapshot = snapshots.get(span.snapshot_id) if span is not None else None
                if span is None or snapshot is None:
                    continue
                rows.append(
                    {
                        "role": "Primary" if snapshot.source_url == issue.primary_source_url else "Social",
                        "quote": span.quote,
                        "source_url": snapshot.source_url,
                    }
                )
            result[issue.id] = rows
        return result

    def template(
        self,
        request: Request,
        name: str,
        *,
        status_code: int = 200,
        **values: Any,
    ) -> Response:
        return self.templates.TemplateResponse(
            request=request,
            name=name,
            context=self.context(request, **values),
            status_code=status_code,
        )

    @staticmethod
    def is_htmx(request: Request) -> bool:
        return request.headers.get("HX-Request", "").lower() == "true"

    async def dashboard(self, request: Request) -> Response:
        return self.template(
            request,
            "dashboard.html",
            artifacts=self.repository.list_artifacts(limit=8),
            record_count=self.repository.count_artifacts(),
            jobs=self.repository.list_jobs(limit=8),
        )

    async def favicon(self, request: Request) -> Response:
        """Answer the path a browser probes before it has read any markup."""

        return FileResponse(WEB_ROOT / "static" / "favicon.ico", media_type="image/x-icon")

    async def add_page(self, request: Request) -> Response:
        return self.template(
            request,
            "add.html",
            bookmarklet=_bookmarklet_source(),
            x_api_configured=bool(self.x_api_client_id),
            x_api_authorized=self.x_api is not None and self.x_api.authorized,
            browser_available=self.browser_capture is not None,
            browser_signed_in=self.login_session is not None and self.login_session.signed_in,
            platforms=self._platform_status(),
        )

    def _platform_status(self) -> list[dict[str, Any]]:
        session = self.login_session
        return [
            {**platform, "signed_in": session is not None and session.is_signed_in_to(platform["host"])}
            for platform in BROWSER_PLATFORMS
        ]

    def _sources_rail(self, request: Request, **values: Any) -> Response:
        return self.template(
            request,
            "partials/sources_rail.html",
            bookmarklet=_bookmarklet_source(),
            x_api_configured=bool(self.x_api_client_id),
            x_api_authorized=self.x_api is not None and self.x_api.authorized,
            browser_available=self.browser_capture is not None,
            platforms=self._platform_status(),
            **values,
        )

    async def browser_signout_ui(self, request: Request) -> Response:
        """Forget one platform's session and redraw what is now available."""

        form = await _form(request)
        host = _string(form, "host")
        if not any(host == platform["host"] for platform in BROWSER_PLATFORMS):
            raise HTTPException(422, "Choose a platform to sign out of.")
        try:
            self._login_session().sign_out(host)
        except (BrowserCaptureUnavailable, RuntimeError) as exc:
            raise HTTPException(409, str(exc)) from None
        if self.is_htmx(request):
            return self._sources_rail(request)
        return RedirectResponse("/add", status_code=303)

    async def x_authorize_ui(self, request: Request) -> Response:
        """Send the user to X's consent screen for a read-only, opt-in token."""

        if self.x_api is None or not self.x_api_client_id:
            raise HTTPException(
                501,
                "The X API is not configured. Set STEERING_X_API_CLIENT_ID first.",
            )
        redirect_uri = str(request.url.replace(path=DEFAULT_REDIRECT_PATH, query=""))
        consent_url, pending = start_authorization(self.x_api_client_id, redirect_uri)
        self._pending_x_authorization = pending
        return RedirectResponse(consent_url, status_code=303)

    async def x_callback_ui(self, request: Request) -> Response:
        pending = self._pending_x_authorization
        self._pending_x_authorization = None
        error = request.query_params.get("error")
        code = request.query_params.get("code")
        state = request.query_params.get("state")
        if error:
            raise HTTPException(400, f"X declined the authorization ({error}).")
        if pending is None or not code:
            raise HTTPException(400, "No X authorization was in progress. Start again.")
        if state != pending.state:
            # A mismatched state means this callback did not come from the
            # request we started, so the code must not be exchanged.
            raise HTTPException(400, "The X authorization response did not match this request.")
        if self.x_api is None:
            raise HTTPException(501, "The X API is not configured.")
        await self.x_api.complete_authorization(code, pending)
        LOGGER.info("X API authorization completed and stored")
        return RedirectResponse("/add", status_code=303)

    async def api_import_x_bookmarks(self, request: Request) -> Response:
        """Read bookmarks through the authorized API instead of the bookmarklet."""

        if self.x_api is None or not self.x_api.authorized:
            raise HTTPException(409, "Authorize the X API before importing bookmarks from it.")
        try:
            sources = await self.x_api.bookmarks()
        except XApiNotAuthorized as exc:
            raise HTTPException(409, str(exc)) from None
        except XApiError as exc:
            raise HTTPException(502, str(exc)) from None
        if not sources:
            raise HTTPException(404, "The X API returned no bookmarked posts.")
        # The bookmarks response already carried each post, so ingestion reuses
        # it rather than billing a second read for the same content.
        report = await self.ingestion.add_prepared_batch_report(sources)
        return JSONResponse(
            {
                "records": _json(report.records),
                "failures": [{"source": source, "error_code": code} for source, code in report.failures],
                "aborted": report.aborted,
            },
            status_code=202,
        )

    async def import_bookmarks_ui(self, request: Request) -> Response:
        form = await _form(request)
        upload = form.get("bookmarks_file")
        if not isinstance(upload, UploadFile) or not upload.filename:
            raise HTTPException(422, "Choose the bookmarks file the bookmarklet saved.")
        report = await self._import_bookmarks(await upload.read(MAX_EXPORT_BYTES + 1))
        if self.is_htmx(request):
            return self.template(request, "partials/ingestion_result.html", records=report.records)
        return RedirectResponse("/search", status_code=303)

    async def api_import_bookmarks(self, request: Request) -> Response:
        report = await self._import_bookmarks(await request.body())
        return JSONResponse(
            {
                "records": _json(report.records),
                "failures": [{"source": source, "error_code": code} for source, code in report.failures],
                "aborted": report.aborted,
            },
            status_code=202,
        )

    async def _import_bookmarks(self, payload: bytes) -> Any:
        if len(payload) > MAX_EXPORT_BYTES:
            raise HTTPException(413, "The bookmarks export exceeds the 8 MiB limit.")
        try:
            sources = bookmark_sources(payload.decode("utf-8-sig"))
        except UnicodeDecodeError:
            raise HTTPException(422, "The bookmarks export must be UTF-8 text.") from None
        except UnsupportedBookmarkExportError as exc:
            raise HTTPException(422, str(exc)) from None
        LOGGER.info("importing %d bookmarked post(s)", len(sources))
        return await self.ingestion.add_batch_report(sources, two_pass=True)

    async def add_submit(self, request: Request) -> Response:
        form = await _form(request)
        mode = _string(form, "mode", "url")
        source = _string(form, "source")
        knowledge_upload = form.get("knowledge_file")
        if isinstance(knowledge_upload, UploadFile) and knowledge_upload.filename:
            if form.get("authorize_upload") != "on":
                raise HTTPException(403, "File processing requires explicit authorization.")
            records = [await self._ingest_upload(knowledge_upload)]
            return self._ingestion_response(request, records)
        if mode == "browser":
            browser_data = BrowserCaptureInput(
                url=source,
                authorized=form.get("authorize_browser") == "on",
            )
            records = [await self._ingest_browser(browser_data)]
            return self._ingestion_response(request, records)
        sources = _lines(_string(form, "batch"))
        upload = form.get("batch_file")
        if isinstance(upload, UploadFile) and upload.filename:
            content = await upload.read(512_001)
            if len(content) > 512_000:
                raise HTTPException(413, "Batch file is too large.")
            try:
                sources.extend(_lines(content.decode("utf-8")))
            except UnicodeDecodeError:
                raise HTTPException(422, "Batch file must be UTF-8 text.") from None
        ingestion_data = IngestionInput(
            source=source or None,
            sources=sources if mode == "batch" else [],
        )
        if not ingestion_data.sources:
            return self._ingestion_response(request, await self._single_record(ingestion_data.source))
        # Reported rather than raised. One unreachable link used to end the run and
        # return an error, so every source after it was never tried and the ones
        # already captured went unmentioned.
        report = await self.ingestion.add_batch_report(ingestion_data.sources)
        return self._ingestion_response(
            request,
            report.records,
            failures=report.failures,
            aborted=report.aborted,
        )

    def _ingestion_response(
        self,
        request: Request,
        records: list[Any],
        *,
        failures: Sequence[tuple[str, str]] = (),
        aborted: str | None = None,
    ) -> Response:
        if self.is_htmx(request):
            return self.template(
                request,
                "partials/ingestion_result.html",
                records=records,
                failures=[{"source": source, "reason": _failure_reason(code)} for source, code in failures],
                aborted=aborted,
            )
        if records:
            return RedirectResponse(f"/artifacts/{records[0].artifact.id}", status_code=303)
        raise HTTPException(422, aborted or "Nothing could be captured from what was submitted.")

    async def _ingest_upload(self, upload: UploadFile) -> Any:
        if self.resolved_ingestion is None:
            raise HTTPException(501, "Knowledge upload is not enabled in this runtime.")
        content = await upload.read(MAX_UPLOAD_BYTES + 1)
        if len(content) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "Knowledge file exceeds the 20 MiB limit.")
        try:
            resolved = await resolve_upload(
                filename=upload.filename or "upload",
                content=content,
                mime_type=upload.content_type or "application/octet-stream",
                image_provider=self.image_provider,
            )
            return await self.resolved_ingestion.add_resolved(resolved)
        except UnsupportedUploadError as exc:
            raise HTTPException(422, str(exc)) from None

    async def _ingest_browser(self, data: BrowserCaptureInput) -> Any:
        if not data.authorized:
            raise HTTPException(403, "Browser capture requires explicit authorization.")
        if self.browser_capture is None or self.resolved_ingestion is None:
            raise HTTPException(501, "Authorized browser capture is not enabled in this runtime.")
        try:
            resolved = await self.browser_capture.capture(data.url, authorized=True)
            return await self.resolved_ingestion.add_resolved(resolved)
        except PermissionError:
            raise HTTPException(403, "Browser capture was not authorized.") from None
        except BrowserDependencyUnavailable:
            raise HTTPException(
                503,
                "Managed browser capture is unavailable. Install the browser extra and Chromium, "
                "then restart STEERING.",
            ) from None
        except BrowserAuthenticationRequired as exc:
            raise HTTPException(409, str(exc)) from None
        except BrowserCaptureUnavailable as exc:
            raise HTTPException(502, f"Managed browser capture did not complete. {exc}") from None
        except Exception:
            LOGGER.exception("authorized browser capture failed")
            raise HTTPException(502, "Authorized browser capture failed.") from None

    def _login_session(self) -> BrowserLoginSession:
        if self.login_session is None:
            raise HTTPException(501, "Managed browser login is not enabled in this runtime.")
        return self.login_session

    def _start_browser_login(self, data: BrowserCaptureInput, *, force: bool = False) -> LoginStatus:
        if not data.authorized:
            raise HTTPException(403, "Opening the managed browser requires explicit authorization.")
        session = self._login_session()
        try:
            return session.start(data.url, force=force)
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from None

    def _login_response(self, request: Request, status: LoginStatus) -> Response:
        return self.template(
            request,
            "partials/browser_status.html",
            status=status,
            polling=status.running,
        )

    async def browser_login_ui(self, request: Request) -> Response:
        form = await _form(request)
        data = BrowserCaptureInput(
            url=_string(form, "login_url"),
            authorized=form.get("authorize_login") == "on",
        )
        status = self._start_browser_login(data, force=form.get("force_login") == "on")
        if self.is_htmx(request):
            return self._login_response(request, status)
        return RedirectResponse("/add", status_code=303)

    async def browser_login_status_ui(self, request: Request) -> Response:
        return self._login_response(request, self._login_session().status())

    async def _single_record(
        self,
        source: str | None,
        *,
        threads: ThreadPolicy = ThreadPolicy.AUTO,
    ) -> list[Any]:
        if source is None:
            raise HTTPException(422, "A URL or text value is required.")
        return [await self.ingestion.add(source, threads=threads)]

    async def search_page(self, request: Request) -> Response:
        query = request.query_params.get("q", "").strip()
        hits = await self.engine.search(SearchQuery(query=query, limit=10)) if query else []
        # Before a search is run the page would otherwise be empty, which hides
        # the knowledge that is already there. This page doubles as the directory
        # of everything captured, newest first, until results replace it.
        sort = self._directory_sort(request.query_params.get("sort"))
        return self.template(
            request,
            "search.html",
            query=query,
            hits=hits,
            artifacts=[] if query else self._directory(sort),
            record_count=self.repository.count_artifacts(),
            sort=sort,
            sorts=DIRECTORY_SORTS,
        )

    @staticmethod
    def _directory_sort(value: str | None) -> str:
        return value if value in DIRECTORY_SORTS else DIRECTORY_SORTS[0]

    def _directory(self, sort: str) -> list[Any]:
        """Order the whole corpus before limiting it.

        Sorting only the newest page would make "oldest" mean "oldest of the most
        recent hundred", which is not what the control says.
        """

        artifacts = self.repository.list_artifacts()
        if sort == "oldest":
            artifacts.reverse()
        elif sort == "title":
            artifacts.sort(key=lambda artifact: artifact.title.casefold())
        return artifacts[:DIRECTORY_LIMIT]

    async def retire_artifact_ui(self, request: Request) -> Response:
        """Retire one record, from its own page."""

        self._retire([request.path_params["artifact_id"]])
        return RedirectResponse("/search", status_code=303)

    async def retire_selected_ui(self, request: Request) -> Response:
        """Retire everything ticked in the knowledge directory."""

        form = await _form(request)
        selected = [str(value) for value in form.getlist("artifact_ids") if value]
        if not selected:
            raise HTTPException(422, "Select at least one record to retire.")
        self._retire(selected)
        sort = self._directory_sort(_string(form, "sort") or None)
        if self.is_htmx(request):
            return self.template(
                request,
                "partials/knowledge_directory.html",
                artifacts=self._directory(sort),
                record_count=self.repository.count_artifacts(),
                sort=sort,
                sorts=DIRECTORY_SORTS,
                retired=len(selected),
            )
        return RedirectResponse(f"/search?sort={sort}", status_code=303)

    async def api_retire_artifact(self, request: Request) -> Response:
        artifact_id = request.path_params["artifact_id"]
        self._retire([artifact_id])
        return JSONResponse({"retired": [artifact_id]})

    async def api_retire_selected(self, request: Request) -> Response:
        data = await _payload(request, RetireInput)
        self._retire(list(data.artifact_ids))
        return JSONResponse({"retired": list(data.artifact_ids)})

    def _retire(self, artifact_ids: Sequence[str]) -> None:
        """Remove records and their evidence, then invalidate retrieval.

        Retiring is permanent: the artifact, its snapshots, chunks, claims, and
        quotes are deleted. Only the shared entities and concepts other artifacts
        also reference survive.
        """

        missing = [
            artifact_id for artifact_id in artifact_ids if not self.repository.delete_record(artifact_id)
        ]
        if missing:
            raise HTTPException(404, "One or more knowledge records were not found.")
        LOGGER.info("retired %d knowledge record(s): %s", len(artifact_ids), ", ".join(artifact_ids))
        if self.on_record_changed is not None:
            # Retrieval indexes still reference a removed artifact until the
            # retriever is told the graph changed.
            self.on_record_changed()

    async def search_submit(self, request: Request) -> Response:
        form = await _form(request)
        data = SearchInput(
            query=_string(form, "query"),
            limit=_string(form, "limit", "10"),
            breadth=form.get("breadth") == "on",
            project_id=_string(form, "project_id") or None,
            published_after=_string(form, "published_after") or None,
        )
        hits = await self.engine.search(
            SearchQuery(
                query=data.query,
                limit=data.limit,
                breadth=data.breadth,
                project_id=data.project_id,
                published_after=data.published_after,
            )
        )
        if self.is_htmx(request):
            return self.template(request, "partials/search_results.html", hits=hits, query=data.query)
        return self.template(
            request,
            "search.html",
            hits=hits,
            query=data.query,
            artifacts=[],
            record_count=self.repository.count_artifacts(),
            sort=DIRECTORY_SORTS[0],
            sorts=DIRECTORY_SORTS,
        )

    async def artifact_page(self, request: Request) -> Response:
        artifact_id = request.path_params["artifact_id"]
        try:
            record = self.engine.get_knowledge_record(artifact_id)
        except CorruptRecordError as exc:
            # Its stored data cannot be read, so the only useful thing this page
            # can offer is the reason and a way to remove it.
            LOGGER.error("artifact %s is unreadable: %s", artifact_id, exc.detail)
            return self.template(
                request,
                "corrupt_artifact.html",
                artifact_id=artifact_id,
                detail=exc.detail,
                status_code=500,
            )
        if record is None:
            raise HTTPException(404, "Knowledge record not found.")
        return self.template(
            request,
            "artifact.html",
            record=record,
            actions=ISSUE_ACTIONS,
            issue_evidence=self._issue_evidence(record.issues),
            linked_sources=self._linked_sources(record),
        )

    def _linked_sources(self, record: Any) -> list[dict[str, Any]]:
        """Where the capture led, which is the point of capturing a social post.

        A post's own URL says where knowledge came from. The paper or repository
        it pointed at is the thing worth opening, so both the sources that were
        followed and the ones that only got recorded are surfaced together.
        """

        metadata = record.artifact.metadata
        followed = [str(url) for url in metadata.get("supporting_sources") or []]
        # A followed source is stored as a snapshot of this record, so its title
        # is the snapshot's rather than a separate artifact's.
        titles = {snapshot.source_url: snapshot for snapshot in record.snapshots}
        rows: list[dict[str, Any]] = []
        for url in followed:
            snapshot = titles.get(url)
            rows.append(
                {
                    "url": url,
                    "title": url,
                    "characters": len(snapshot.text) if snapshot is not None else 0,
                    "followed": True,
                }
            )
        seen = set(followed)
        for value in metadata.get("outbound_urls") or []:
            url = str(value)
            if url in seen or not url.startswith(("http://", "https://")):
                continue
            seen.add(url)
            rows.append({"url": url, "title": url, "characters": 0, "followed": False})
        return rows

    async def issues_page(self, request: Request) -> Response:
        issues = self.repository.list_issues(unresolved_only=False)
        return self.template(
            request,
            "issues.html",
            issues=issues,
            actions=ISSUE_ACTIONS,
            issue_evidence=self._issue_evidence(issues),
        )

    async def issue_resolve_ui(self, request: Request) -> Response:
        form = await _form(request)
        data = IssueResolutionInput(action=_string(form, "action"))
        issue = self._resolve_issue(request.path_params["issue_id"], data.action)
        if self.is_htmx(request):
            return self.template(
                request,
                "partials/issue.html",
                issue=issue,
                actions=ISSUE_ACTIONS,
                issue_evidence=self._issue_evidence([issue]),
            )
        return RedirectResponse("/issues", status_code=303)

    async def design_page(self, request: Request) -> Response:
        return self.template(
            request,
            "design.html",
            review=None,
            projects=self.repository.list_projects(),
        )

    async def design_submit(self, request: Request) -> Response:
        form = await _form(request)
        data = DesignInput(
            architecture=_string(form, "architecture"),
            requirements=_string(form, "requirements"),
            concerns=_lines(_string(form, "concerns")),
            project_id=_string(form, "project_id") or None,
            limit=_string(form, "limit", "12"),
        )
        review = await self.engine.review_architecture(
            data.architecture,
            data.requirements,
            data.concerns,
            project_id=data.project_id,
            limit=data.limit,
        )
        if self.is_htmx(request):
            return self.template(request, "partials/design_review.html", review=review)
        return self.template(
            request,
            "design.html",
            review=review,
            projects=self.repository.list_projects(),
        )

    async def projects_page(self, request: Request) -> Response:
        return self.template(request, "projects.html", projects=self.repository.list_projects())

    async def project_create_ui(self, request: Request) -> Response:
        form = await _form(request)
        data = ProjectInput(
            name=_string(form, "name"),
            description=_string(form, "description") or None,
            constraints=_lines(_string(form, "constraints")),
        )
        project = self.repository.save_project(
            Project(name=data.name, description=data.description, constraints=data.constraints)
        )
        return RedirectResponse(f"/projects/{project.id}", status_code=303)

    def _project_history(self, project_id: str) -> dict[str, Sequence[Any]]:
        history = dict(self.repository.project_history(project_id))
        if not history.get("projects"):
            raise HTTPException(404, "Project not found.")
        return history

    async def project_page(self, request: Request) -> Response:
        return self.template(
            request,
            "project.html",
            history=self._project_history(request.path_params["project_id"]),
        )

    async def project_decision_ui(self, request: Request) -> Response:
        project_id = request.path_params["project_id"]
        self._project_history(project_id)
        form = await _form(request)
        data = DecisionInput(
            project=project_id,
            artifact_id=_string(form, "artifact_id") or None,
            decision=_string(form, "decision"),
            rationale=_string(form, "rationale"),
        )
        self.engine.record_project_decision(
            project=data.project,
            artifact_id=data.artifact_id,
            decision=data.decision,
            rationale=data.rationale,
        )
        return RedirectResponse(f"/projects/{project_id}", status_code=303)

    async def project_outcome_ui(self, request: Request) -> Response:
        project_id = request.path_params["project_id"]
        self._project_history(project_id)
        form = await _form(request)
        succeeded_value = _string(form, "succeeded")
        data = OutcomeInput(
            project_id=project_id,
            outcome=_string(form, "outcome"),
            artifact_id=_string(form, "artifact_id") or None,
            decision_id=_string(form, "decision_id") or None,
            constraints=_lines(_string(form, "constraints")),
            succeeded=(True if succeeded_value == "true" else False if succeeded_value == "false" else None),
        )
        self.engine.record_experiment_outcome(**data.model_dump())
        return RedirectResponse(f"/projects/{project_id}", status_code=303)

    async def providers_page(self, request: Request) -> Response:
        return self.template(
            request,
            "providers.html",
            providers=self.provider_settings.list_providers(),
        )

    async def provider_save_ui(self, request: Request) -> Response:
        form = await _form(request)
        data = self._provider_from_form(form)
        try:
            provider = await self._save_provider(data)
        except Exception:
            raise HTTPException(
                502,
                "Provider connection test failed; settings were not saved.",
            ) from None
        if self.is_htmx(request):
            return self.template(
                request,
                "partials/provider_status.html",
                provider=provider,
                message="Provider settings saved.",
            )
        return RedirectResponse("/settings/providers", status_code=303)

    async def provider_test_ui(self, request: Request) -> Response:
        provider_id = request.path_params["provider_id"]
        form = await _form(request)
        role = _string(form, "role") or None
        try:
            await self.provider_settings.test_provider(provider_id, role)
        except KeyError:
            raise HTTPException(404, "Provider not found.") from None
        except Exception:
            raise HTTPException(502, "Provider connection test failed.") from None
        provider = self._provider_view(provider_id)
        return self.template(
            request,
            "partials/provider_status.html",
            provider=provider,
            message="Connection succeeded.",
        )

    async def provider_delete_ui(self, request: Request) -> Response:
        provider_id = request.path_params["provider_id"]
        form = await _form(request)
        role = _string(form, "role")
        try:
            self.provider_settings.delete_key(provider_id, role)
        except KeyError:
            raise HTTPException(404, "Provider not found.") from None
        except ValueError:
            raise HTTPException(422, "A valid provider role is required.") from None
        provider = self._provider_view(provider_id)
        return self.template(
            request,
            "partials/provider_status.html",
            provider=provider,
            message="Saved key deleted.",
        )

    def _provider_view(self, provider_id: str) -> Any:
        return next(
            (item for item in self.provider_settings.list_providers() if item.provider_id == provider_id),
            None,
        )

    @staticmethod
    def _provider_from_form(form: FormData) -> ProviderInput:
        generation_api_key = _string(form, "generation_api_key")
        embedding_api_key = _string(form, "embedding_api_key")
        return ProviderInput(
            provider_id=_string(form, "provider_id"),
            role=_string(form, "role", "both"),
            base_url=_string(form, "base_url"),
            generation_model=_string(form, "generation_model") or None,
            embedding_model=_string(form, "embedding_model") or None,
            embedding_dimension=_string(form, "embedding_dimension", "768"),
            generation_api_key=generation_api_key or None,
            embedding_api_key=embedding_api_key or None,
        )

    async def _save_provider(self, data: ProviderInput) -> Any:
        return await self.provider_settings.save_provider(
            provider_id=data.provider_id,
            role=data.role,
            base_url=data.base_url,
            generation_model=data.generation_model,
            embedding_model=data.embedding_model,
            embedding_dimension=data.embedding_dimension,
            generation_api_key=(
                data.generation_api_key.get_secret_value() if data.generation_api_key else None
            ),
            embedding_api_key=(data.embedding_api_key.get_secret_value() if data.embedding_api_key else None),
        )

    def _resolve_issue(self, issue_id: str, action: str) -> Any:
        try:
            return self.repository.resolve_issue(issue_id, action)
        except KeyError:
            raise HTTPException(404, "Review issue not found.") from None
        except ValueError:
            raise HTTPException(422, "Unsupported review action.") from None

    async def api_health(self, request: Request) -> Response:
        return JSONResponse(
            {
                "status": "ok",
                "version": __version__,
                "schema_version": SCHEMA_VERSION,
                "unresolved_issues": len(self.repository.list_issues(unresolved_only=True)),
            }
        )

    async def api_search(self, request: Request) -> Response:
        data = await _payload(request, SearchInput)
        hits = await self.engine.search(
            SearchQuery(
                query=data.query,
                limit=data.limit,
                breadth=data.breadth,
                project_id=data.project_id,
            )
        )
        return JSONResponse({"results": _json(hits)})

    async def api_records(self, request: Request) -> Response:
        try:
            limit = min(max(int(request.query_params.get("limit", "50")), 1), 100)
        except ValueError:
            raise HTTPException(422, "limit must be an integer.") from None
        records = self.repository.list_records()[:limit]
        return JSONResponse({"records": _json(records)})

    async def api_record(self, request: Request) -> Response:
        record = self.engine.get_knowledge_record(request.path_params["artifact_id"])
        if record is None:
            raise HTTPException(404, "Knowledge record not found.")
        return JSONResponse(_json(record))

    async def api_design(self, request: Request) -> Response:
        data = await _payload(request, DesignInput)
        review = await self.engine.review_architecture(
            data.architecture,
            data.requirements,
            data.concerns,
            project_id=data.project_id,
            limit=data.limit,
        )
        return JSONResponse(_json(review))

    async def api_issues(self, request: Request) -> Response:
        unresolved = request.query_params.get("unresolved", "true").lower() != "false"
        return JSONResponse(
            {
                "issues": _json(self.repository.list_issues(unresolved_only=unresolved)),
                "unresolved_count": len(self.repository.list_issues(unresolved_only=True)),
            }
        )

    async def api_issue_resolve(self, request: Request) -> Response:
        data = await _payload(request, IssueResolutionInput)
        return JSONResponse(_json(self._resolve_issue(request.path_params["issue_id"], data.action)))

    async def api_ingestion(self, request: Request) -> Response:
        data = await _payload(request, IngestionInput)
        if not data.sources:
            records = await self._single_record(data.source, threads=data.threads)
            return JSONResponse({"records": _json(records), "failures": []}, status_code=202)
        # A batch is reported in full: one dead link must not discard the rest of
        # an unattended run, and a stopped run must say why.
        report = await self.ingestion.add_batch_report(
            data.sources,
            threads=data.threads,
            two_pass=data.two_pass,
        )
        return JSONResponse(
            {
                "records": _json(report.records),
                "failures": [{"source": source, "error_code": code} for source, code in report.failures],
                "aborted": report.aborted,
            },
            status_code=202,
        )

    async def api_ingestion_upload(self, request: Request) -> Response:
        form = await _form(request)
        if _string(form, "authorized").lower() not in {"true", "1", "yes"}:
            raise HTTPException(403, "File processing requires explicit authorization.")
        upload = form.get("file")
        if not isinstance(upload, UploadFile) or not upload.filename:
            raise HTTPException(422, "A knowledge file is required.")
        record = await self._ingest_upload(upload)
        return JSONResponse({"record": _json(record)}, status_code=202)

    async def api_ingestion_browser(self, request: Request) -> Response:
        data = await _payload(request, BrowserCaptureInput)
        record = await self._ingest_browser(data)
        return JSONResponse({"record": _json(record)}, status_code=202)

    async def api_browser_login(self, request: Request) -> Response:
        data = await _payload(request, BrowserCaptureInput)
        status = self._start_browser_login(data)
        return JSONResponse({"state": status.state, "message": status.message}, status_code=202)

    async def api_browser_login_status(self, request: Request) -> Response:
        status = self._login_session().status()
        return JSONResponse({"state": status.state, "message": status.message})

    async def api_jobs(self, request: Request) -> Response:
        return JSONResponse({"jobs": _json(self.repository.list_jobs(limit=100))})

    async def api_providers(self, request: Request) -> Response:
        if request.method == "GET":
            return JSONResponse({"providers": _json(self.provider_settings.list_providers())})
        data = await _payload(request, ProviderInput)
        try:
            provider = await self._save_provider(data)
        except Exception:
            raise HTTPException(
                502,
                "Provider connection test failed; settings were not saved.",
            ) from None
        return JSONResponse(_json(provider), status_code=201)

    async def api_provider_test(self, request: Request) -> Response:
        role = request.query_params.get("role")
        try:
            await self.provider_settings.test_provider(request.path_params["provider_id"], role)
        except KeyError:
            raise HTTPException(404, "Provider not found.") from None
        except Exception:
            raise HTTPException(502, "Provider connection test failed.") from None
        return JSONResponse({"status": "ok"})

    async def api_provider_key(self, request: Request) -> Response:
        role = request.query_params.get("role", "")
        try:
            deleted = self.provider_settings.delete_key(request.path_params["provider_id"], role)
        except KeyError:
            raise HTTPException(404, "Provider not found.") from None
        except ValueError:
            raise HTTPException(422, "A valid provider role is required.") from None
        return JSONResponse({"deleted": deleted})

    async def api_projects(self, request: Request) -> Response:
        if request.method == "GET":
            return JSONResponse({"projects": _json(self.repository.list_projects())})
        data = await _payload(request, ProjectInput)
        project = self.repository.save_project(
            Project(name=data.name, description=data.description, constraints=data.constraints)
        )
        return JSONResponse(_json(project), status_code=201)

    async def api_project(self, request: Request) -> Response:
        history = self.repository.project_history(request.path_params["project_id"])
        return JSONResponse(_json(dict(history)))

    async def api_decisions(self, request: Request) -> Response:
        data = await _payload(request, DecisionInput)
        decision = self.engine.record_project_decision(
            project=data.project,
            artifact_id=data.artifact_id,
            decision=data.decision,
            rationale=data.rationale,
        )
        return JSONResponse(_json(decision), status_code=201)

    async def api_outcomes(self, request: Request) -> Response:
        data = await _payload(request, OutcomeInput)
        outcome = self.engine.record_experiment_outcome(
            project_id=data.project_id,
            outcome=data.outcome,
            artifact_id=data.artifact_id,
            decision_id=data.decision_id,
            constraints=data.constraints,
            succeeded=data.succeeded,
        )
        return JSONResponse(_json(outcome), status_code=201)

    async def api_maintenance_backup(self, request: Request) -> Response:
        data = await _payload(request, BackupInput)
        destination = self.repository.backup(data.destination)
        return JSONResponse({"backup": str(destination)})

    async def api_maintenance_reindex(self, request: Request) -> Response:
        count, backup = await self.engine.retriever.reembed_all()
        return JSONResponse({"reindexed": True, "reembedded_chunks": count, "verified_backup": backup})

    async def api_maintenance_audit(self, request: Request) -> Response:
        damaged = await anyio.to_thread.run_sync(self.repository.unreadable_records)
        return JSONResponse(
            {
                "checked": self.repository.count_artifacts(),
                "unreadable": damaged,
                "unreadable_count": len(damaged),
            }
        )


def create_web_app(
    *,
    engine: SteeringEngine,
    ingestion: IngestionService,
    repository: ArtifactRepository,
    provider_settings: ProviderSettingsService,
    resolved_ingestion: ResolvedSourceIngestion | None = None,
    browser_capture: AuthorizedBrowserCapture | None = None,
    login_session: BrowserLoginSession | None = None,
    image_provider: ImageUnderstandingProvider | None = None,
    on_record_changed: Callable[[], None] | None = None,
    x_api: XApiClient | None = None,
    x_api_client_id: str | None = None,
    templates_directory: Path | None = None,
    static_directory: Path | None = None,
) -> Starlette:
    """Create an injected, mountable web application without opening a database."""

    templates = Jinja2Templates(directory=templates_directory or WEB_ROOT / "templates")
    controller = WebController(
        engine=engine,
        ingestion=ingestion,
        repository=repository,
        provider_settings=provider_settings,
        resolved_ingestion=resolved_ingestion,
        browser_capture=browser_capture,
        login_session=login_session,
        image_provider=image_provider,
        templates=templates,
        on_record_changed=on_record_changed,
        x_api=x_api,
        x_api_client_id=x_api_client_id,
    )
    routes = [
        Route("/", controller.dashboard, methods=["GET"]),
        Route("/add", controller.add_page, methods=["GET"]),
        Route("/add", controller.add_submit, methods=["POST"]),
        Route("/bookmarks/import", controller.import_bookmarks_ui, methods=["POST"]),
        Route("/oauth/x/authorize", controller.x_authorize_ui, methods=["POST"]),
        Route(DEFAULT_REDIRECT_PATH, controller.x_callback_ui, methods=["GET"]),
        Route("/browser/login", controller.browser_login_ui, methods=["POST"]),
        Route("/browser/signout", controller.browser_signout_ui, methods=["POST"]),
        Route("/browser/login/status", controller.browser_login_status_ui, methods=["GET"]),
        Route("/search", controller.search_page, methods=["GET"]),
        Route("/search", controller.search_submit, methods=["POST"]),
        Route("/artifacts/{artifact_id:str}", controller.artifact_page, methods=["GET"]),
        Route("/artifacts/retire", controller.retire_selected_ui, methods=["POST"]),
        Route(
            "/artifacts/{artifact_id:str}/retire",
            controller.retire_artifact_ui,
            methods=["POST"],
        ),
        Route("/issues", controller.issues_page, methods=["GET"]),
        Route("/issues/{issue_id:str}/resolve", controller.issue_resolve_ui, methods=["POST"]),
        Route("/design", controller.design_page, methods=["GET"]),
        Route("/design", controller.design_submit, methods=["POST"]),
        Route("/projects", controller.projects_page, methods=["GET"]),
        Route("/projects", controller.project_create_ui, methods=["POST"]),
        Route("/projects/{project_id:str}", controller.project_page, methods=["GET"]),
        Route(
            "/projects/{project_id:str}/decisions",
            controller.project_decision_ui,
            methods=["POST"],
        ),
        Route(
            "/projects/{project_id:str}/outcomes",
            controller.project_outcome_ui,
            methods=["POST"],
        ),
        Route("/settings/providers", controller.providers_page, methods=["GET"]),
        Route("/settings/providers", controller.provider_save_ui, methods=["POST"]),
        Route(
            "/settings/providers/{provider_id:str}/test",
            controller.provider_test_ui,
            methods=["POST"],
        ),
        Route(
            "/settings/providers/{provider_id:str}/delete-key",
            controller.provider_delete_ui,
            methods=["POST"],
        ),
        Route("/favicon.ico", controller.favicon, methods=["GET"]),
        Route("/api/health", controller.api_health, methods=["GET"]),
        Route("/api/search", controller.api_search, methods=["POST"]),
        Route("/api/records", controller.api_records, methods=["GET"]),
        Route("/api/records/{artifact_id:str}", controller.api_record, methods=["GET"]),
        Route("/api/records/retire", controller.api_retire_selected, methods=["POST"]),
        Route(
            "/api/records/{artifact_id:str}",
            controller.api_retire_artifact,
            methods=["DELETE"],
        ),
        Route("/api/design", controller.api_design, methods=["POST"]),
        Route("/api/issues", controller.api_issues, methods=["GET"]),
        Route("/api/issues/{issue_id:str}/resolve", controller.api_issue_resolve, methods=["POST"]),
        Route("/api/ingestion", controller.api_ingestion, methods=["POST"]),
        Route("/api/ingestion/upload", controller.api_ingestion_upload, methods=["POST"]),
        Route("/api/ingestion/browser", controller.api_ingestion_browser, methods=["POST"]),
        Route("/api/ingestion/bookmarks", controller.api_import_bookmarks, methods=["POST"]),
        Route("/api/ingestion/x-bookmarks", controller.api_import_x_bookmarks, methods=["POST"]),
        Route("/api/browser/login", controller.api_browser_login, methods=["POST"]),
        Route("/api/browser/login/status", controller.api_browser_login_status, methods=["GET"]),
        Route("/api/jobs", controller.api_jobs, methods=["GET"]),
        Route("/api/providers", controller.api_providers, methods=["GET", "POST"]),
        Route("/api/providers/{provider_id:str}/test", controller.api_provider_test, methods=["POST"]),
        Route("/api/providers/{provider_id:str}/key", controller.api_provider_key, methods=["DELETE"]),
        Route("/api/projects", controller.api_projects, methods=["GET", "POST"]),
        Route("/api/projects/{project_id:str}", controller.api_project, methods=["GET"]),
        Route("/api/decisions", controller.api_decisions, methods=["POST"]),
        Route("/api/outcomes", controller.api_outcomes, methods=["POST"]),
        Route(
            "/api/maintenance/backup",
            controller.api_maintenance_backup,
            methods=["POST"],
        ),
        Route(
            "/api/maintenance/reindex",
            controller.api_maintenance_reindex,
            methods=["POST"],
        ),
        Route(
            "/api/maintenance/audit",
            controller.api_maintenance_audit,
            methods=["GET"],
        ),
        Mount(
            "/static",
            app=StaticFiles(directory=static_directory or WEB_ROOT / "static"),
            name="static",
        ),
    ]

    async def http_error(request: Request, exc: Exception) -> Response:
        assert isinstance(exc, HTTPException)
        message = str(exc.detail) if isinstance(exc.detail, str) else "Request failed."
        if request.url.path.startswith("/api/"):
            return JSONResponse({"error": message}, status_code=exc.status_code)
        if controller.is_htmx(request) and request.url.path == "/add":
            return templates.TemplateResponse(
                request=request,
                name="partials/ingestion_error.html",
                context=controller.context(
                    request,
                    message=message,
                    retry_url="/add",
                    retry_label="Retry ingestion",
                ),
            )
        if controller.is_htmx(request) and request.url.path == "/browser/login":
            return templates.TemplateResponse(
                request=request,
                name="partials/browser_error.html",
                context=controller.context(request, message=message),
            )
        return templates.TemplateResponse(
            request=request,
            name="error.html",
            context=controller.context(request, message=message, status_code=exc.status_code),
            status_code=exc.status_code,
        )

    async def validation_error(request: Request, exc: Exception) -> Response:
        assert isinstance(exc, ValidationError)
        if request.url.path.startswith("/api/"):
            return JSONResponse(_validation_payload(exc), status_code=422)
        return templates.TemplateResponse(
            request=request,
            name="error.html",
            context=controller.context(
                request, message="Please correct the submitted fields.", status_code=422
            ),
            status_code=422,
        )

    async def unexpected_error(request: Request, exc: Exception) -> Response:
        del exc
        if request.url.path.startswith("/api/"):
            return JSONResponse({"error": "Request could not be completed."}, status_code=500)
        return templates.TemplateResponse(
            request=request,
            name="error.html",
            context=controller.context(
                request,
                message="Request could not be completed.",
                status_code=500,
            ),
            status_code=500,
        )

    async def provider_error(request: Request, exc: Exception) -> Response:
        timed_out = isinstance(exc, ProviderTimeoutError)
        ingestion_request = request.url.path == "/add" or request.url.path.startswith("/api/ingestion")
        if ingestion_request:
            message = (
                "Ingestion failed because the generation provider timed out. Retry."
                if timed_out
                else "Ingestion failed because the generation provider was unavailable. Retry."
            )
            retry_url = "/add"
            retry_label = "Retry ingestion"
        else:
            message = (
                "The provider request timed out. Retry."
                if timed_out
                else "The provider request could not be completed. Retry."
            )
            retry_url = request.url.path
            retry_label = "Retry"
        status_code = 503 if timed_out else 502
        if request.url.path.startswith("/api/"):
            return JSONResponse({"error": message}, status_code=status_code)
        if controller.is_htmx(request):
            return templates.TemplateResponse(
                request=request,
                name="partials/ingestion_error.html",
                context=controller.context(
                    request,
                    message=message,
                    retry_url=retry_url,
                    retry_label=retry_label,
                ),
            )
        return templates.TemplateResponse(
            request=request,
            name="error.html",
            context=controller.context(
                request,
                message=message,
                status_code=status_code,
                retry_url=retry_url,
                retry_label=retry_label,
            ),
            status_code=status_code,
        )

    async def source_unavailable_error(request: Request, exc: Exception) -> Response:
        del exc
        message = "Ingestion failed because the source could not be reached. Retry."
        if request.url.path.startswith("/api/"):
            return JSONResponse({"error": message}, status_code=502)
        if controller.is_htmx(request):
            return templates.TemplateResponse(
                request=request,
                name="partials/ingestion_error.html",
                context=controller.context(
                    request,
                    message=message,
                    retry_url="/add",
                    retry_label="Retry ingestion",
                ),
            )
        return templates.TemplateResponse(
            request=request,
            name="error.html",
            context=controller.context(
                request,
                message=message,
                status_code=502,
                retry_url="/add",
                retry_label="Retry ingestion",
            ),
            status_code=502,
        )

    async def evidence_error(request: Request, exc: Exception) -> Response:
        del exc
        message = (
            "Ingestion failed because the extracted claims could not be verified against the "
            "source text. Nothing was stored."
        )
        if request.url.path.startswith("/api/"):
            return JSONResponse({"error": message}, status_code=422)
        if controller.is_htmx(request):
            return templates.TemplateResponse(
                request=request,
                name="partials/ingestion_error.html",
                context=controller.context(
                    request,
                    message=message,
                    retry_url="/add",
                    retry_label="Retry ingestion",
                ),
            )
        return templates.TemplateResponse(
            request=request,
            name="error.html",
            context=controller.context(request, message=message, status_code=422),
            status_code=422,
        )

    app = Starlette(
        debug=False,
        routes=routes,
        exception_handlers={
            HTTPException: http_error,
            ValidationError: validation_error,
            ProviderTimeoutError: provider_error,
            ProviderConnectionError: provider_error,
            SourceUnavailableError: source_unavailable_error,
            EvidenceValidationError: evidence_error,
            Exception: unexpected_error,
        },
    )
    app.add_middleware(LocalMutationGuardMiddleware)
    return app
