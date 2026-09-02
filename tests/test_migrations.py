import sqlite3

from backend.app.core.config import Settings
from backend.app.db.migrations import TARGET_REVISION, upgrade_database


def test_existing_sqlite_memory_schema_is_backed_up_and_upgraded(tmp_path):
    database = tmp_path / "legacy.db"
    with sqlite3.connect(database) as connection:
        connection.executescript("""
            CREATE TABLE conversation_summaries (
                id TEXT PRIMARY KEY, tenant_id TEXT, user_id TEXT, conversation_id TEXT,
                content TEXT, covered_until_message_id TEXT, version INTEGER, updated_at DATETIME
            );
            CREATE TABLE user_memories (
                id TEXT PRIMARY KEY, tenant_id TEXT, user_id TEXT, conversation_id TEXT,
                memory_type TEXT, content TEXT, source_message_id TEXT, active BOOLEAN,
                created_at DATETIME, updated_at DATETIME
            );
            CREATE TABLE memory_jobs (
                id TEXT PRIMARY KEY, tenant_id TEXT, user_id TEXT, conversation_id TEXT,
                source_message_id TEXT, status TEXT, attempts INTEGER,
                candidate_count INTEGER, summary_updated BOOLEAN, last_error TEXT,
                created_at DATETIME, updated_at DATETIME
            );
            INSERT INTO user_memories VALUES (
                'm1','t','u','c','case_fact','旧记忆','source-1',1,CURRENT_TIMESTAMP,CURRENT_TIMESTAMP
            );
            INSERT INTO memory_jobs VALUES (
                'j1','t','u','c','source-1','failed',3,0,0,
                'Thinking mode does not support this tool_choice',CURRENT_TIMESTAMP,CURRENT_TIMESTAMP
            );
            INSERT INTO memory_jobs VALUES (
                'j2','t','u','c','source-2','failed',3,0,0,
                '记忆模型调用方式与模型不兼容',CURRENT_TIMESTAMP,CURRENT_TIMESTAMP
            );
            INSERT INTO memory_jobs VALUES (
                'j3','t','u','c','source-3','failed',3,0,0,
                'unrelated transport error',CURRENT_TIMESTAMP,CURRENT_TIMESTAMP
            );
        """)
    settings = Settings(_env_file=None, database_url=f"sqlite:///{database}")

    backup = upgrade_database(settings)

    assert backup and backup.is_file()
    with sqlite3.connect(database) as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(user_memories)")}
        status, active = connection.execute(
            "SELECT status, active FROM user_memories WHERE id='m1'"
        ).fetchone()
        revision = connection.execute("SELECT version_num FROM alembic_version").fetchone()[0]
        repaired_job = connection.execute(
            "SELECT status, attempts, last_error FROM memory_jobs WHERE id='j1'"
        ).fetchone()
        sanitized_job = connection.execute(
            "SELECT status, attempts, last_error FROM memory_jobs WHERE id='j2'"
        ).fetchone()
        unrelated_job = connection.execute(
            "SELECT status, attempts, last_error FROM memory_jobs WHERE id='j3'"
        ).fetchone()
    assert {"scope", "status", "canonical_key", "confidence", "version"} <= columns
    assert (status, active) == ("pending", 0)
    assert revision == TARGET_REVISION
    assert repaired_job == ("pending", 0, "")
    assert sanitized_job == ("pending", 0, "")
    assert unrelated_job == ("failed", 3, "unrelated transport error")
