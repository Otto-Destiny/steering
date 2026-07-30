"""Drive the knowledge directory's selection behaviour in real Chromium.

Selection is CSS and event behaviour, so a rendered-HTML assertion cannot show
whether a checkbox is actually visible or whether cancelling clears a hidden
selection. Both matter: a checkbox on every card at rest is visual noise, and a
selection surviving a cancel would retire records the user thought they had
deselected.
"""

from __future__ import annotations

import contextlib
import socket
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import uvicorn
from tests.web.test_app import (
    FakeEngine,
    FakeIngestion,
    FakeProviders,
    FakeRepository,
    make_record,
)

from steering.web import create_web_app

pytestmark = pytest.mark.e2e


@contextlib.contextmanager
def _serving() -> Iterator[str]:
    repository = FakeRepository()
    for index in range(3):
        record = make_record(f"art_directory_{index}")
        record.artifact.title = f"Record {index}"
        record.artifact.canonical_url = f"https://example.test/{index}"
        repository.records.append(record)
    app = create_web_app(
        engine=FakeEngine(repository),  # type: ignore[arg-type]
        ingestion=FakeIngestion(repository),  # type: ignore[arg-type]
        repository=repository,  # type: ignore[arg-type]
        provider_settings=FakeProviders(),
    )
    with contextlib.closing(socket.socket()) as probe:
        probe.bind(("127.0.0.1", 0))
        port = int(probe.getsockname()[1])
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        for _ in range(60):
            try:
                with contextlib.closing(socket.create_connection(("127.0.0.1", port), 0.2)):
                    break
            except OSError:
                time.sleep(0.25)
        else:
            raise RuntimeError("the directory fixture server did not start")
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=10)


@contextlib.contextmanager
def _directory(base_url: str) -> Iterator[Any]:
    playwright_api = pytest.importorskip("playwright.sync_api")
    with playwright_api.sync_playwright() as runtime:
        if not Path(runtime.chromium.executable_path).is_file():
            pytest.skip("the managed Chromium runtime is not installed")
        browser = runtime.chromium.launch(headless=True)
        try:
            page = browser.new_page(viewport={"width": 1280, "height": 900})
            page.goto(f"{base_url}/search", wait_until="load")
            yield page
        finally:
            browser.close()


def test_the_directory_is_free_of_selection_chrome_until_it_is_wanted() -> None:
    with _serving() as base_url, _directory(base_url) as page:
        assert page.locator('input[name="artifact_ids"]').first.is_visible() is False
        assert page.locator(".selection-actions").is_visible() is False
        assert page.locator(".selection-enter").is_visible() is True


def test_hovering_a_card_reveals_its_checkbox() -> None:
    with _serving() as base_url, _directory(base_url) as page:
        card = page.locator("#directory-list .card").first
        card.hover()
        assert page.locator('input[name="artifact_ids"]').first.is_visible() is True


def test_selecting_one_card_enters_selection_mode() -> None:
    with _serving() as base_url, _directory(base_url) as page:
        card = page.locator("#directory-list .card").first
        card.hover()
        page.locator('input[name="artifact_ids"]').first.check()

        assert page.locator("#selection-mode").is_checked() is True
        assert page.locator(".selection-actions").is_visible() is True
        assert page.locator(".selection-enter").is_visible() is False
        assert page.locator(".selection-count").inner_text() == "1 selected"
        assert "is-selected" in (card.get_attribute("class") or "")
        # Every card's checkbox is now reachable without hovering each one.
        assert page.locator('input[name="artifact_ids"]').last.is_visible() is True


def test_select_all_counts_every_record() -> None:
    with _serving() as base_url, _directory(base_url) as page:
        page.locator(".selection-enter").click()
        page.locator("#directory-select-all").check()

        total = page.locator('input[name="artifact_ids"]').count()
        assert page.locator(".selection-count").inner_text() == f"{total} selected"


def test_cancelling_clears_the_selection_rather_than_hiding_it() -> None:
    """A hidden but still-ticked box would retire records the user deselected."""

    with _serving() as base_url, _directory(base_url) as page:
        page.locator(".selection-enter").click()
        page.locator("#directory-select-all").check()
        page.get_by_text("Cancel", exact=True).click()

        assert page.locator(".selection-actions").is_visible() is False
        assert page.locator('input[name="artifact_ids"]:checked').count() == 0
        assert page.locator('input[name="artifact_ids"]').first.is_visible() is False


def test_retiring_repeatedly_keeps_working_without_a_page_reload() -> None:
    """An earlier per-card delete stopped working after the first removal.

    The directory is re-rendered whole after each retirement, so the controls in
    the replacement have to be live again. If the swap left stale markup behind,
    the second attempt would silently do nothing.
    """

    with _serving() as base_url, _directory(base_url) as page:
        errors: list[str] = []
        page.on("dialog", lambda dialog: dialog.accept())
        page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)

        counts = [page.locator("#directory-list .card").count()]
        for _ in range(3):
            page.locator(".selection-enter").click()
            page.locator('input[name="artifact_ids"]').first.check()
            page.get_by_role("button", name="Retire").click()
            page.wait_for_timeout(700)
            counts.append(page.locator("#directory-list .card").count())

        # One record leaves on every attempt, not just the first.
        assert counts == [counts[0], counts[0] - 1, counts[0] - 2, counts[0] - 3]
        assert page.locator(".selection-enter").count() == 1
        assert errors == []
