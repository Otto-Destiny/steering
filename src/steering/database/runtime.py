from __future__ import annotations

from pathlib import Path
from types import TracebackType

import ladybug as lb

from steering.database.backup import create_backup, verify_backup
from steering.database.migrations import apply_migrations, current_revision
from steering.database.repository import LadybugArtifactRepository


def _prepare_revision_two_backup(connection: lb.Connection, database_path: Path) -> None:
    result = connection.execute("CALL SHOW_TABLES() RETURN name")
    if isinstance(result, list):
        raise RuntimeError("table query unexpectedly returned multiple result sets")
    tables = {str(row.get("name") if isinstance(row, dict) else row[0]) for row in result.get_all()}
    if "SchemaMigrations" not in tables or current_revision(connection) != 1:
        return
    backup_path = Path(f"{database_path}.pre-v2-backup")
    if not backup_path.exists():
        create_backup(connection, backup_path)
    verify_backup(backup_path)


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
            _prepare_revision_two_backup(self.connection, self.path)
            apply_migrations(self.connection)
            self.repository = LadybugArtifactRepository(self.connection, self.path)
            self.repository.load_active_search_extensions()
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
