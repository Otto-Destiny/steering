from __future__ import annotations

import pytest

from steering.providers.registry import Registry


def test_registry_normalizes_names_creates_values_and_sorts() -> None:
    registry: Registry[str] = Registry()
    registry.register(" Zebra ", lambda *, suffix: f"z-{suffix}")
    registry.register("alpha", lambda *, suffix: f"a-{suffix}")
    assert registry.names() == ("alpha", "zebra")
    assert registry.create(" ZEBRA ", suffix="one") == "z-one"


def test_registry_rejects_empty_duplicate_and_unknown_names() -> None:
    registry: Registry[str] = Registry()
    with pytest.raises(ValueError, match="cannot be empty"):
        registry.register("  ", str)
    registry.register("entry", str)
    with pytest.raises(ValueError, match="already exists"):
        registry.register(" ENTRY ", str)
    with pytest.raises(KeyError, match="unknown registry entry"):
        registry.create("missing")
