from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError
from starlette.applications import Starlette
from starlette.datastructures import FormData, UploadFile
from starlette.exceptions import HTTPException
from starlette.requests import Request
from starlette.responses import JSONResponse, RedirectResponse, Response
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles
from starlette.templating import Jinja2Templates

from steering import SCHEMA_VERSION, __version__
from steering.domain.models import Project, SearchQuery
from steering.domain.protocols import ArtifactRepository, ImageUnderstandingProvider
from steering.ingestion.service import IngestionService
from steering.ingestion.uploads import MAX_UPLOAD_BYTES, UnsupportedUploadError, resolve_upload
from steering.intelligence.service import SteeringEngine
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
    SearchInput,
)
from steering.web.security import MAX_REQUEST_BYTES, LocalMutationGuardMiddleware

TInput = TypeVar("TInput", bound=BaseModel)
WEB_ROOT = Path(__file__).resolve().parent
ISSUE_ACTIONS = ("accept_correction", "keep_both", "dismiss", "reject")


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
        image_provider: ImageUnderstandingProvider | None,
        templates: Jinja2Templates,
    ) -> None:
        self.engine = engine
        self.ingestion = ingestion
        self.repository = repository
        self.provider_settings = provider_settings
        self.resolved_ingestion = resolved_ingestion
        self.browser_capture = browser_capture
        self.image_provider = image_provider
        self.templates = templates

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

    def template(self, request: Request, name: str, **values: Any) -> Response:
        return self.templates.TemplateResponse(
            request=request,
            name=name,
            context=self.context(request, **values),
        )

    @staticmethod
    def is_htmx(request: Request) -> bool:
        return request.headers.get("HX-Request", "").lower() == "true"

    async def dashboard(self, request: Request) -> Response:
        records = sorted(
            self.repository.list_records(),
            key=lambda item: item.artifact.captured_at,
            reverse=True,
        )
        jobs = self.repository.list_jobs(limit=8)
        return self.template(
            request,
            "dashboard.html",
            records=records[:8],
            record_count=len(records),
            jobs=jobs,
        )

    async def add_page(self, request: Request) -> Response:
        return self.template(request, "add.html")

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
        records = await (
            self.ingestion.add_batch(ingestion_data.sources)
            if ingestion_data.sources
            else self._single_record(ingestion_data.source)
        )
        return self._ingestion_response(request, records)

    def _ingestion_response(self, request: Request, records: list[Any]) -> Response:
        if self.is_htmx(request):
            return self.template(request, "partials/ingestion_result.html", records=records)
        return RedirectResponse(f"/artifacts/{records[0].artifact.id}", status_code=303)

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
        except Exception:
            raise HTTPException(502, "Authorized browser capture failed.") from None

    async def _open_browser_login(self, data: BrowserCaptureInput) -> None:
        if not data.authorized:
            raise HTTPException(403, "Opening the managed browser requires explicit authorization.")
        if self.browser_capture is None:
            raise HTTPException(501, "Managed browser login is not enabled in this runtime.")
        try:
            await self.browser_capture.open_login(data.url, authorized=True)
        except PermissionError:
            raise HTTPException(403, "Opening the managed browser was not authorized.") from None
        except Exception:
            raise HTTPException(502, "Managed browser login failed.") from None

    async def browser_login_ui(self, request: Request) -> Response:
        form = await _form(request)
        data = BrowserCaptureInput(
            url=_string(form, "login_url"),
            authorized=form.get("authorize_login") == "on",
        )
        await self._open_browser_login(data)
        if self.is_htmx(request):
            return self.template(
                request,
                "partials/browser_status.html",
                message="Managed browser session saved. You can now capture an authorized URL.",
            )
        return RedirectResponse("/add", status_code=303)

    async def _single_record(self, source: str | None) -> list[Any]:
        if source is None:
            raise HTTPException(422, "A URL or text value is required.")
        return [await self.ingestion.add(source)]

    async def search_page(self, request: Request) -> Response:
        query = request.query_params.get("q", "").strip()
        hits = await self.engine.search(SearchQuery(query=query, limit=10)) if query else []
        return self.template(request, "search.html", query=query, hits=hits)

    async def search_submit(self, request: Request) -> Response:
        form = await _form(request)
        data = SearchInput(
            query=_string(form, "query"),
            limit=_string(form, "limit", "10"),
            breadth=form.get("breadth") == "on",
            project_id=_string(form, "project_id") or None,
        )
        hits = await self.engine.search(
            SearchQuery(
                query=data.query,
                limit=data.limit,
                breadth=data.breadth,
                project_id=data.project_id,
            )
        )
        if self.is_htmx(request):
            return self.template(request, "partials/search_results.html", hits=hits, query=data.query)
        return self.template(request, "search.html", hits=hits, query=data.query)

    async def artifact_page(self, request: Request) -> Response:
        record = self.engine.get_knowledge_record(request.path_params["artifact_id"])
        if record is None:
            raise HTTPException(404, "Knowledge record not found.")
        return self.template(
            request,
            "artifact.html",
            record=record,
            actions=ISSUE_ACTIONS,
            issue_evidence=self._issue_evidence(record.issues),
        )

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
        records = (
            await self.ingestion.add_batch(data.sources)
            if data.sources
            else await self._single_record(data.source)
        )
        return JSONResponse({"records": _json(records)}, status_code=202)

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
        await self._open_browser_login(data)
        return JSONResponse({"status": "ready"})

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


def create_web_app(
    *,
    engine: SteeringEngine,
    ingestion: IngestionService,
    repository: ArtifactRepository,
    provider_settings: ProviderSettingsService,
    resolved_ingestion: ResolvedSourceIngestion | None = None,
    browser_capture: AuthorizedBrowserCapture | None = None,
    image_provider: ImageUnderstandingProvider | None = None,
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
        image_provider=image_provider,
        templates=templates,
    )
    routes = [
        Route("/", controller.dashboard, methods=["GET"]),
        Route("/add", controller.add_page, methods=["GET"]),
        Route("/add", controller.add_submit, methods=["POST"]),
        Route("/browser/login", controller.browser_login_ui, methods=["POST"]),
        Route("/search", controller.search_page, methods=["GET"]),
        Route("/search", controller.search_submit, methods=["POST"]),
        Route("/artifacts/{artifact_id:str}", controller.artifact_page, methods=["GET"]),
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
        Route("/api/health", controller.api_health, methods=["GET"]),
        Route("/api/search", controller.api_search, methods=["POST"]),
        Route("/api/records", controller.api_records, methods=["GET"]),
        Route("/api/records/{artifact_id:str}", controller.api_record, methods=["GET"]),
        Route("/api/design", controller.api_design, methods=["POST"]),
        Route("/api/issues", controller.api_issues, methods=["GET"]),
        Route("/api/issues/{issue_id:str}/resolve", controller.api_issue_resolve, methods=["POST"]),
        Route("/api/ingestion", controller.api_ingestion, methods=["POST"]),
        Route("/api/ingestion/upload", controller.api_ingestion_upload, methods=["POST"]),
        Route("/api/ingestion/browser", controller.api_ingestion_browser, methods=["POST"]),
        Route("/api/browser/login", controller.api_browser_login, methods=["POST"]),
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

    app = Starlette(
        debug=False,
        routes=routes,
        exception_handlers={
            HTTPException: http_error,
            ValidationError: validation_error,
            Exception: unexpected_error,
        },
    )
    app.add_middleware(LocalMutationGuardMiddleware)
    return app
