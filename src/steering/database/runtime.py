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


class DatabaseCorruptedError(RuntimeError):
    """Raised when the database cannot be opened and needs manual recovery."""


def _open_database(path: Path) -> lb.Database:
    """Open the database, turning storage damage into an actionable instruction.

    A write-ahead log left behind by a killed process is the common way this
    fails, and it is recoverable: the log holds only writes since the last
    checkpoint, so setting it aside restores the database minus those. The raw
    storage error says none of that, which leaves a user with a working database
    and no idea it is one command away.
    """

    try:
        return lb.Database(str(path))
    except RuntimeError as exc:
        detail = str(exc)
        if "wal" not in detail.lower():
            raise
        wal = Path(f"{path}.wal")
        raise DatabaseCorruptedError(
            f"the database write-ahead log at '{wal}' is corrupt, which usually means a "
            "previous run was killed. Move that file elsewhere (keep it, do not delete it) "
            f"and start again; '{path}' reopens at its last checkpoint. "
            f"Original error: {detail}"
        ) from exc


class DatabaseRuntime:
    """Own the one writable Ladybug Database for the daemon lifetime."""

    def __init__(self, database_path: str | Path) -> None:
        self.path = Path(database_path).expanduser().resolve(strict=False)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.database = _open_database(self.path)
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
