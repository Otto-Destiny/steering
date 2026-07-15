from __future__ import annotations

from collections.abc import Callable
from typing import Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """Small explicit registry for experimental extension points."""

    def __init__(self) -> None:
        self._factories: dict[str, Callable[..., T]] = {}

    def register(self, name: str, factory: Callable[..., T]) -> None:
        normalized = name.strip().lower()
        if not normalized:
            raise ValueError("registry name cannot be empty")
        if normalized in self._factories:
            raise ValueError(f"registry entry already exists: {normalized}")
        self._factories[normalized] = factory

    def create(self, name: str, **kwargs: object) -> T:
        try:
            factory = self._factories[name.strip().lower()]
        except KeyError as exc:
            raise KeyError(f"unknown registry entry: {name}") from exc
        return factory(**kwargs)

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._factories))
