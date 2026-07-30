"""Source resolution and ingestion orchestration."""

from steering.ingestion.browser import (
    BrowserAuthenticationRequired,
    BrowserCaptureUnavailable,
    BrowserDependencyUnavailable,
    LoginOutcome,
    ManagedBrowserCapture,
)
from steering.ingestion.login_session import BrowserLoginSession, LoginState, LoginStatus
from steering.ingestion.media_fallback import append_social_image_fallback
from steering.ingestion.resolvers import ResolverRegistry, default_registry
from steering.ingestion.security import NetworkGuard, SafeFetcher, SourceUnavailableError, UnsafeSourceError
from steering.ingestion.service import IngestionService
from steering.ingestion.telegram import telegram_sources
from steering.ingestion.uploads import UnsupportedUploadError, resolve_upload
from steering.ingestion.x import XPostRef, XResolver, parse_x_post_url

__all__ = [
    "BrowserAuthenticationRequired",
    "BrowserCaptureUnavailable",
    "BrowserDependencyUnavailable",
    "BrowserLoginSession",
    "IngestionService",
    "LoginOutcome",
    "LoginState",
    "LoginStatus",
    "ManagedBrowserCapture",
    "NetworkGuard",
    "ResolverRegistry",
    "SafeFetcher",
    "SourceUnavailableError",
    "UnsafeSourceError",
    "UnsupportedUploadError",
    "XPostRef",
    "XResolver",
    "append_social_image_fallback",
    "default_registry",
    "parse_x_post_url",
    "resolve_upload",
    "telegram_sources",
]
