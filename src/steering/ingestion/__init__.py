"""Source resolution and ingestion orchestration."""

from steering.ingestion.browser import (
    BrowserCaptureUnavailable,
    BrowserDependencyUnavailable,
    ManagedBrowserCapture,
)
from steering.ingestion.media_fallback import append_social_image_fallback
from steering.ingestion.resolvers import ResolverRegistry, default_registry
from steering.ingestion.security import NetworkGuard, SafeFetcher, SourceUnavailableError, UnsafeSourceError
from steering.ingestion.service import IngestionService
from steering.ingestion.telegram import telegram_sources
from steering.ingestion.uploads import UnsupportedUploadError, resolve_upload

__all__ = [
    "BrowserCaptureUnavailable",
    "BrowserDependencyUnavailable",
    "IngestionService",
    "ManagedBrowserCapture",
    "NetworkGuard",
    "ResolverRegistry",
    "SafeFetcher",
    "SourceUnavailableError",
    "UnsafeSourceError",
    "UnsupportedUploadError",
    "append_social_image_fallback",
    "default_registry",
    "resolve_upload",
    "telegram_sources",
]
