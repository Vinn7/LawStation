import shutil
import sqlite3
from pathlib import Path

from alembic.config import Config

from alembic import command
from backend.app.core.config import Settings, get_settings

ROOT = Path(__file__).resolve().parents[3]
TARGET_REVISION = "20260826_05"


def _sqlite_path(database_url: str) -> Path | None:
    prefix = "sqlite:///"
    if not database_url.startswith(prefix) or database_url == "sqlite:///:memory:":
        return None
    raw = database_url[len(prefix):]
    path = Path(raw)
    return path if path.is_absolute() else ROOT / path


def _current_revision(path: Path) -> str | None:
    try:
        with sqlite3.connect(path) as connection:
            row = connection.execute(
                "SELECT version_num FROM alembic_version LIMIT 1"
            ).fetchone()
            return str(row[0]) if row else None
    except sqlite3.Error:
        return None


def upgrade_database(settings: Settings | None = None) -> Path | None:
    settings = settings or get_settings()
    database_path = _sqlite_path(settings.database_url)
    backup_path = None
    if database_path and database_path.is_file() and _current_revision(database_path) != TARGET_REVISION:
        backup_path = database_path.with_name(database_path.name + ".pre-agent-runs.bak")
        if not backup_path.exists():
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(database_path, backup_path)
    config = Config(str(ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(ROOT / "alembic"))
    config.set_main_option("sqlalchemy.url", settings.database_url)
    config.attributes["database_url"] = settings.database_url
    command.upgrade(config, "head")
    return backup_path
