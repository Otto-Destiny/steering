from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol
from urllib.parse import urlsplit

import anyio.to_thread

from steering.domain.credentials import sanitized_persistence_source
from steering.domain.models import (
    ArtifactRecord,
    IngestionJob,
    JobStatus,
    ResolvedSource,
    SourceKind,
)
from steering.domain.protocols import ArtifactRepository, ImageUnderstandingProvider
from steering.extraction.service import ExtractionService
from steering.ingestion.browser import BrowserAuthenticationRequired, BrowserCaptureUnavailable
from steering.ingestion.media_fallback import append_social_image_fallback
from steering.ingestion.resolvers import PAPER_HOSTS, ResolverRegistry
from steering.ingestion.security import SafeFetcher, SourceUnavailableError
from steering.ingestion.x import AUTHOR_THREAD, PUBLIC_METHODS, ROOT_POST_ONLY, parse_x_post_url

LOGGER = logging.getLogger(__name__)


FOLLOW_PRIORITY_FLOOR = 80
#: A thread usually cites a paper and its repository. Following a few keeps both
#: without letting one link-heavy thread become an unbounded crawl.
MAX_PRIMARY_SOURCES = 3
_PRIORITY_BY_HOST: tuple[tuple[frozenset[str], int], ...] = (
    (PAPER_HOSTS, 100),
    (frozenset({"github.com"}), 95),
    (frozenset({"huggingface.co", "modelscope.ai", "modelscope.cn"}), 90),
    (frozenset({"readthedocs.io", "readthedocs.org"}), 80),
)


def _host_matches(host: str, candidates: frozenset[str]) -> bool:
    """Match a host exactly or as a subdomain, never as a substring."""

    return any(host == candidate or host.endswith(f".{candidate}") for candidate in candidates)


def _source_priority(url: str) -> int:
    """Rank a linked source so the strongest primary material is followed first.

    Ranking is host-based rather than substring-based so a URL such as
    ``https://attacker.example/?ref=arxiv.org`` cannot borrow another host's rank.
    """

    parts = urlsplit(url)
    host = (parts.hostname or "").lower()
    if not host:
        return 0
    if parts.path.lower().endswith(".pdf"):
        return 98
    for hosts, priority in _PRIORITY_BY_HOST:
        if _host_matches(host, hosts):
            return priority
    if host.startswith("docs."):
        return 80
    return 20


@dataclass(slots=True)
class BatchReport:
    """The outcome of one unattended run, including what it could not capture."""

    records: list[ArtifactRecord] = field(default_factory=list)
    failures: list[tuple[str, str]] = field(default_factory=list)
    aborted: str | None = None

    @property
    def captured(self) -> int:
        return len(self.records)


class ThreadReading(StrEnum):
    """When a post's replies are worth opening a browser for.

    Reading a thread costs about twelve seconds, or a billed API call. Most
    posts do not need it, so the default spends that only where the root post
    cannot stand on its own.
    """

    #: Only when the root post leaves something out: it carries no source worth
    #: following, or it says the link is in the replies.
    WHEN_NEEDED = "when-needed"
    #: Whenever the post has any replies at all.
    ALWAYS = "always"
    #: Never; keep every capture to the root post.
    NEVER = "never"


#: Phrases authors use when the link is deliberately held back from the root
#: post. Their presence is the clearest signal that the replies matter.
_REPLY_LINK_HINTS = (
    "link in the reply",
    "link in reply",
    "link in the replies",
    "link in replies",
    "link in the comment",
    "link in comment",
    "link below",
    "links below",
    "link in thread",
    "see the reply",
    "see reply",
    "in the reply below",
)


class ThreadPolicy(StrEnum):
    """How hard to try to read an X post's self-reply thread."""

    NEVER = "never"
    #: Capture publicly first, then read the thread only when the post actually
    #: has replies and a signed-in session already exists. Costs nothing when
    #: either is untrue, which is why it is safe as the default.
    AUTO = "auto"
    #: Always read the thread, and stop rather than quietly settle for less.
    ALWAYS = "always"


class ThreadCapture(Protocol):
    """The authorized capture that can read a whole author thread."""

    async def capture(self, url: str, *, authorized: bool = False) -> ResolvedSource: ...

    def has_stored_session(self) -> bool: ...


class ThreadReader(Protocol):
    """One way of reading an X post's self-reply thread.

    Several exist because none is universally available: the signed-in browser
    is free but needs a session, and the API is unattended but billed. They are
    tried in order so an unavailable reader costs nothing.
    """

    name: str

    def available(self) -> bool: ...

    async def read_thread(self, url: str) -> ResolvedSource: ...


class BrowserThreadReader:
    """Reads a thread through the authorized signed-in browser."""

    name = "signed_in_browser"

    def __init__(self, capture: ThreadCapture) -> None:
        self._capture = capture

    def available(self) -> bool:
        """Whether X specifically is signed in, not merely some platform.

        One browser profile is shared, so asking whether *a* session exists made a
        LinkedIn sign-in look like an X one. The reader then launched a browser
        that could not authenticate, spending real time to arrive back at the
        public capture it already had.
        """

        per_platform = getattr(self._capture, "is_signed_in_to", None)
        if callable(per_platform):
            return bool(per_platform("x.com"))
        probe = getattr(self._capture, "has_stored_session", None)
        return bool(probe()) if callable(probe) else True

    async def read_thread(self, url: str) -> ResolvedSource:
        return await self._capture.capture(url, authorized=True)


class IngestionService:
    def __init__(
        self,
        *,
        registry: ResolverRegistry,
        extraction: ExtractionService,
        repository: ArtifactRepository,
        media_fetcher: SafeFetcher | None = None,
        image_provider: ImageUnderstandingProvider | None = None,
        thread_capture: ThreadCapture | None = None,
        thread_readers: Sequence[ThreadReader] | None = None,
        thread_reading: ThreadReading = ThreadReading.WHEN_NEEDED,
        on_record_changed: Callable[[], None] | None = None,
    ) -> None:
        self.registry = registry
        self.extraction = extraction
        self.repository = repository
        self.media_fetcher = media_fetcher
        self.image_provider = image_provider
        self._thread_capture = thread_capture
        self._extra_readers: list[ThreadReader] = list(thread_readers or ())
        self.thread_reading = thread_reading
        self.on_record_changed = on_record_changed

    @property
    def thread_capture(self) -> ThreadCapture | None:
        return self._thread_capture

    @thread_capture.setter
    def thread_capture(self, capture: ThreadCapture | None) -> None:
        self._thread_capture = capture

    def add_thread_reader(self, reader: ThreadReader) -> None:
        """Register a reader tried after the ones already present."""

        self._extra_readers.append(reader)

    @property
    def thread_readers(self) -> list[ThreadReader]:
        """Readers in priority order: the free signed-in browser before the API."""

        readers: list[ThreadReader] = []
        if self._thread_capture is not None:
            readers.append(BrowserThreadReader(self._thread_capture))
        readers.extend(self._extra_readers)
        return readers

    async def _store_record(self, record: ArtifactRecord) -> ArtifactRecord:
        """Persist a record without stalling the event loop.

        The repository is synchronous and serializes on one embedded-database
        connection. Writing a record with its snapshots, chunks, and evidence is
        the longest blocking call in ingestion, so it runs on a worker thread and
        the daemon keeps serving other requests meanwhile.
        """

        stored = await anyio.to_thread.run_sync(self.repository.upsert_record, record)
        if self.on_record_changed is not None:
            self.on_record_changed()
        return stored

    async def add(
        self,
        source: str,
        *,
        threads: ThreadPolicy = ThreadPolicy.AUTO,
        prepared: ResolvedSource | None = None,
    ) -> ArtifactRecord:
        """Ingest one source, reading an X post's self-reply thread when it helps.

        Authors routinely keep the paper or repository link out of the root post
        to drive engagement into the replies, and no public X representation
        exposes replies at all. Under the default policy a post that actually has
        replies is escalated to the signed-in browser automatically, so pasting
        the root link is enough to reach a link buried three replies down.
        """

        job = IngestionJob(source=sanitized_persistence_source(source), status=JobStatus.RUNNING)
        self.repository.save_job(job)
        try:
            resolved = await self._resolve(source, threads=threads, prepared=prepared)
            return await self._ingest_resolved(resolved, job=job, follow_primary=True)
        except Exception as exc:
            self._fail_job(job, exc)
            raise

    async def _resolve(
        self,
        source: str,
        *,
        threads: ThreadPolicy,
        prepared: ResolvedSource | None = None,
    ) -> ResolvedSource:
        """Resolve a source, reusing a capture the caller already paid for.

        ``prepared`` exists because the X API returns a bookmark's full content
        alongside its id. Fetching it again would bill a second read for text
        already in hand.
        """

        if threads is ThreadPolicy.NEVER or not self.thread_readers:
            return prepared or await self.registry.resolve(source)
        if parse_x_post_url(source) is None:
            return prepared or await self.registry.resolve(source)
        if threads is ThreadPolicy.ALWAYS:
            return await self._capture_thread(source, fallback=None)

        public = prepared or await self.registry.resolve(source)
        if not self._worth_escalating(public):
            return public
        LOGGER.info(
            "%s has %s repl(ies); escalating to signed-in thread capture",
            public.canonical_url,
            public.metadata.get("reply_count"),
        )
        return await self._capture_thread(source, fallback=public)

    def _worth_escalating(self, public: ResolvedSource) -> bool:
        """Escalate only when there is something to gain and a way to gain it."""

        if self.thread_reading is ThreadReading.NEVER:
            return False
        if public.source_kind is not SourceKind.X:
            return False
        if not isinstance(public.metadata.get("reply_count"), int):
            return False
        if public.metadata["reply_count"] < 1:
            return False
        if self.thread_reading is ThreadReading.WHEN_NEEDED and not self._root_post_falls_short(public):
            LOGGER.debug(
                "%s already carries a followable source; not reading its thread",
                public.canonical_url,
            )
            return False
        return any(reader.available() for reader in self.thread_readers)

    @staticmethod
    def _root_post_falls_short(public: ResolvedSource) -> bool:
        """Report whether the root post leaves the reader somewhere to go.

        A post that already links a paper or repository has nothing to gain from
        its replies. One that says the link is in the replies plainly does.
        """

        if any(hint in public.text.lower() for hint in _REPLY_LINK_HINTS):
            return True
        return not any(_source_priority(url) >= FOLLOW_PRIORITY_FLOOR for url in public.outbound_urls)

    async def _capture_thread(
        self,
        source: str,
        *,
        fallback: ResolvedSource | None,
    ) -> ResolvedSource:
        last_error: Exception | None = None
        for reader in self.thread_readers:
            if not reader.available():
                LOGGER.debug("thread reader %s is not available; trying the next", reader.name)
                continue
            try:
                captured = await reader.read_thread(source)
            except (BrowserCaptureUnavailable, PermissionError, SourceUnavailableError) as exc:
                # One reader failing is not the end: another may still succeed.
                LOGGER.info(
                    "thread reader %s could not read %s (%s: %s)",
                    reader.name,
                    source,
                    type(exc).__name__,
                    exc,
                )
                last_error = exc
                continue
            if reader.name != "signed_in_browser":
                LOGGER.info("read the thread for %s through %s", source, reader.name)
            return captured

        if last_error is None:
            last_error = BrowserAuthenticationRequired(
                "no thread reader is available; sign in to X or configure the X API"
            )
        reason = type(last_error).__name__
        if fallback is None:
            if isinstance(last_error, BrowserAuthenticationRequired):
                # Under an explicit policy, every remaining item would fail the
                # same way; quietly storing root-post-only captures hides that.
                raise last_error
            LOGGER.warning(
                "signed-in capture of %s failed (%s: %s); using public capture",
                source,
                reason,
                last_error,
            )
            return await self.registry.resolve(source)
        LOGGER.warning(
            "thread escalation for %s failed (%s: %s); keeping the public root-post capture",
            source,
            reason,
            last_error,
        )
        # The degradation is recorded on the artifact, not just in the log.
        return fallback.model_copy(
            update={
                "metadata": {
                    **fallback.metadata,
                    "thread_escalation": f"failed_{reason}",
                }
            }
        )

    async def add_resolved(self, resolved: ResolvedSource) -> ArtifactRecord:
        """Ingest an already authorized capture without performing any network resolution."""

        job = IngestionJob(
            source=sanitized_persistence_source(resolved.canonical_url),
            status=JobStatus.RUNNING,
        )
        self.repository.save_job(job)
        try:
            follow_primary = resolved.source_kind in {SourceKind.X, SourceKind.LINKEDIN}
            return await self._ingest_resolved(resolved, job=job, follow_primary=follow_primary)
        except Exception as exc:
            self._fail_job(job, exc)
            raise

    async def add_batch(
        self,
        sources: Sequence[str],
        *,
        threads: ThreadPolicy = ThreadPolicy.AUTO,
    ) -> list[ArtifactRecord]:
        results: list[ArtifactRecord] = []
        for source in dict.fromkeys(item.strip() for item in sources if item.strip()):
            results.append(await self.add(source, threads=threads))
        return results

    async def _two_pass_batch(self, sources: Sequence[str]) -> BatchReport:
        first = await self.add_batch_report(sources, threads=ThreadPolicy.NEVER)
        if first.aborted:
            return first
        pending = [
            record.artifact.canonical_url
            for record in first.records
            if record.artifact.canonical_url and self._needs_a_second_pass(record)
        ]
        LOGGER.info(
            "first pass captured %d source(s); %d need their thread read",
            first.captured,
            len(pending),
        )
        if not pending:
            return first
        second = await self.add_batch_report(pending, threads=ThreadPolicy.AUTO)
        upgraded = {record.artifact.id for record in second.records}
        # The second pass replaces the root-post-only record it upgraded, so the
        # report lists each source once, at its best captured state.
        merged = [record for record in first.records if record.artifact.id not in upgraded]
        return BatchReport(
            records=[*merged, *second.records],
            failures=[*first.failures, *second.failures],
            aborted=second.aborted,
        )

    def _needs_a_second_pass(self, record: ArtifactRecord) -> bool:
        """Report whether a stored capture still lacks what its thread would add."""

        metadata = record.artifact.metadata
        if metadata.get("capture_scope") != ROOT_POST_ONLY:
            return False
        if self.thread_reading is ThreadReading.NEVER:
            return False
        if self.thread_reading is ThreadReading.ALWAYS:
            return True
        followable = any(
            _source_priority(str(url)) >= FOLLOW_PRIORITY_FLOOR for url in metadata.get("outbound_urls") or ()
        )
        if followable:
            return False
        text = " ".join(snapshot.text for snapshot in record.snapshots).lower()
        return bool(metadata.get("reply_count")) or any(hint in text for hint in _REPLY_LINK_HINTS)

    async def add_prepared_batch_report(
        self,
        sources: Sequence[ResolvedSource],
        *,
        threads: ThreadPolicy = ThreadPolicy.AUTO,
    ) -> BatchReport:
        """Ingest captures the caller already holds, without re-fetching them."""

        report = BatchReport()
        for source in sources:
            try:
                report.records.append(await self.add(source.canonical_url, threads=threads, prepared=source))
            except BrowserAuthenticationRequired:
                report.aborted = (
                    "The managed browser is not signed in to X. Run the sign-in step, "
                    "then restart this batch."
                )
                break
            except Exception as exc:
                LOGGER.warning("batch item %s failed (%s)", source.canonical_url, type(exc).__name__)
                report.failures.append((source.canonical_url, type(exc).__name__))
        return report

    async def add_batch_report(
        self,
        sources: Sequence[str],
        *,
        threads: ThreadPolicy = ThreadPolicy.AUTO,
        two_pass: bool = False,
    ) -> BatchReport:
        """Ingest many sources, surviving individual failures.

        An unattended overnight run must not lose four hundred good captures to
        one dead link, so per-source failures are recorded and the run continues.
        A missing sign-in session is the exception: it stops the run, because it
        would otherwise degrade every remaining item in the same invisible way.

        With ``two_pass`` the run captures everything publicly first, which is
        free and takes seconds per source, then reads threads only for the posts
        that came back without one. Browser time is then spent solely on posts
        that demonstrably need it, rather than on the whole list.
        """

        if two_pass:
            return await self._two_pass_batch(sources)
        report = BatchReport()
        for source in dict.fromkeys(item.strip() for item in sources if item.strip()):
            try:
                report.records.append(await self.add(source, threads=threads))
            except BrowserAuthenticationRequired:
                report.aborted = (
                    "The managed browser is not signed in to X. Run the sign-in step, "
                    "then restart this batch."
                )
                LOGGER.error("batch aborted at %s: no signed-in X session", source)
                break
            except Exception as exc:
                LOGGER.warning("batch item %s failed (%s)", source, type(exc).__name__)
                report.failures.append((source, type(exc).__name__))
        LOGGER.info(
            "batch finished: %d captured, %d failed%s",
            len(report.records),
            len(report.failures),
            " (aborted)" if report.aborted else "",
        )
        return report

    async def _ingest_resolved(
        self,
        resolved: ResolvedSource,
        *,
        job: IngestionJob,
        follow_primary: bool,
    ) -> ArtifactRecord:
        existing = self.repository.get_by_url(resolved.canonical_url)
        if existing is not None and not self._upgrades_public_social_capture(existing, resolved):
            job.status = JobStatus.COMPLETED
            job.artifact_id = existing.artifact.id
            self.repository.save_job(job)
            return existing
        supporting = await self._follow_primary_sources(resolved) if follow_primary else []
        if self.media_fetcher is not None:
            resolved = await append_social_image_fallback(
                resolved,
                fetcher=self.media_fetcher,
                image_provider=self.image_provider,
                stronger_source_available=bool(supporting),
            )
        # A followed source becomes a snapshot of this record, so its full text is
        # already stored, chunked, and searchable here. Emitting a second artifact
        # for it as well produced a duplicate of every captured link, titled and
        # summarized from raw README markup because no model ever described it.
        record = await self.extraction.extract(resolved, supporting)
        stored = await self._store_record(record)
        job.status = JobStatus.NEEDS_REVIEW if stored.issues else JobStatus.COMPLETED
        job.artifact_id = stored.artifact.id
        self.repository.save_job(job)
        return stored

    @staticmethod
    def _reaches_a_new_primary_source(
        existing: ArtifactRecord,
        incoming: ResolvedSource,
    ) -> bool:
        """Whether the new capture found a followable link the record never held.

        Records captured before long-form link recovery hold no destinations and
        can never gain one on their own, so a re-capture that does find one is
        worth taking. Comparing against what is already stored keeps a repeated
        submission from rewriting a record that would not change.
        """

        metadata = existing.artifact.metadata
        stored = {str(url) for url in metadata.get("outbound_urls") or ()}
        stored |= {str(url) for url in metadata.get("supporting_sources") or ()}
        return any(
            _source_priority(url) >= FOLLOW_PRIORITY_FLOOR and url not in stored
            for url in incoming.outbound_urls
        )

    @classmethod
    def _upgrades_public_social_capture(
        cls,
        existing: ArtifactRecord,
        incoming: ResolvedSource,
    ) -> bool:
        """Allow a stronger capture to replace an earlier, weaker record.

        Keyed on what the capture reached rather than which reader produced it,
        so the API upgrades a public record exactly as the browser does. Only a
        record that is itself root-post-only or publicly captured can be replaced,
        which is what stops a public re-capture from overwriting a signed-in
        thread.
        """

        if incoming.source_kind not in {SourceKind.X, SourceKind.LINKEDIN}:
            return False
        existing_metadata = existing.artifact.metadata
        replaces_weaker_record = (
            existing_metadata.get("capture_scope") == ROOT_POST_ONLY
            or existing_metadata.get("resolver") in PUBLIC_METHODS
        )
        if not replaces_weaker_record:
            return False
        if incoming.metadata.get("capture_scope") == AUTHOR_THREAD:
            return True
        return cls._reaches_a_new_primary_source(existing, incoming)

    def _fail_job(self, job: IngestionJob, exc: Exception) -> None:
        # The stored message stays safe for display; the full cause goes to logs
        # so a user can actually diagnose a repeated failure.
        LOGGER.warning("ingestion failed for job %s: %s: %s", job.id, type(exc).__name__, exc, exc_info=True)
        job.status = JobStatus.FAILED
        job.error_code = type(exc).__name__
        job.safe_error = "Ingestion failed; inspect the source, provider, and network settings."
        self.repository.save_job(job)

    async def _follow_primary_sources(self, source: ResolvedSource) -> list[ResolvedSource]:
        """Resolve the primary sources a social post points at, strongest first.

        A thread commonly cites a paper *and* its repository, and keeping only the
        paper discards half of what the author was pointing at. Following is
        bounded by :data:`MAX_PRIMARY_SOURCES` so one link-heavy thread cannot
        turn into an unbounded crawl, and all of them still share a single
        extraction call.
        """

        if source.source_kind not in {SourceKind.X, SourceKind.LINKEDIN}:
            return []
        candidates = sorted(dict.fromkeys(source.outbound_urls), key=_source_priority, reverse=True)
        if not candidates:
            LOGGER.debug("social capture %s carried no outbound links", source.canonical_url)
            return []

        followed: list[ResolvedSource] = []
        seen: set[str] = {source.canonical_url}
        for url in candidates:
            if len(followed) >= MAX_PRIMARY_SOURCES:
                break
            priority = _source_priority(url)
            if priority < FOLLOW_PRIORITY_FLOOR:
                break
            try:
                primary = await self.registry.resolve(url)
            except (RuntimeError, ValueError) as exc:
                LOGGER.info(
                    "linked primary source %s could not be resolved (%s: %s)",
                    url,
                    type(exc).__name__,
                    exc,
                )
                continue
            if primary.canonical_url in seen:
                continue
            seen.add(primary.canonical_url)
            followed.append(primary)
            LOGGER.info(
                "followed primary source %s (priority %d) from %s",
                primary.canonical_url,
                priority,
                source.canonical_url,
            )
        return followed
