from __future__ import annotations

from pathlib import Path
from types import TracebackType

import ladybug as lb

from steering.database.migrations import apply_migrations
from steering.database.repository import LadybugArtifactRepository


class DatabaseRuntime:
    """Own the one writable Ladybug Database for the daemon lifetime."""

    def __init__(self, database_path: str | Path) -> None:
        self.path = Path(database_path).expanduser().resolve(strict=False)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.database = lb.Database(str(self.path))
        connection: lb.Connection | None = None
        try:
            connection = lb.Connection(self.database)
            self.connection = connection
            apply_migrations(self.connection)
            self.repository = LadybugArtifactRepository(self.connection, self.path)
        except BaseException:
            if connection is not None:
                connection.close()
            self.database.close()
            raise
        self._closed = False

    def close(self) -> None:
        if self._closed:
            return
        self.repository.close()
        self.database.close()
        self._closed = True

    def __enter__(self) -> DatabaseRuntime:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
