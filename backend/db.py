import json
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import aiosqlite

from backend.config import DB_PATH, DEFAULT_SETTINGS, LEGACY_CHATS_FILE


class Database:
    """Async SQLite persistence store (WAL mode) for chat sessions and settings."""

    def __init__(self, db_path: Path = DB_PATH, legacy_json_path: Path | None = LEGACY_CHATS_FILE):
        self.db_path = Path(db_path)
        self.legacy_json_path = Path(legacy_json_path) if legacy_json_path is not None else None

    @asynccontextmanager
    async def _connect(self):
        conn = await aiosqlite.connect(self.db_path)
        try:
            conn.row_factory = aiosqlite.Row
            await conn.execute("PRAGMA busy_timeout=5000")
            await conn.execute("PRAGMA synchronous=NORMAL")
            yield conn
        finally:
            await conn.close()

    async def init(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        async with self._connect() as conn:
            await conn.execute("PRAGMA journal_mode=WAL")
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chats (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    provider TEXT NOT NULL DEFAULT 'ollama',
                    model TEXT NOT NULL DEFAULT '',
                    messages_json TEXT NOT NULL DEFAULT '[]',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """
            )
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS db_connectors (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    db_type TEXT NOT NULL DEFAULT 'postgresql',
                    host TEXT NOT NULL DEFAULT '127.0.0.1',
                    port INTEGER NOT NULL DEFAULT 5432,
                    database_name TEXT NOT NULL DEFAULT '',
                    username TEXT NOT NULL DEFAULT '',
                    password TEXT NOT NULL DEFAULT '',
                    ssl_mode TEXT NOT NULL DEFAULT '',
                    extra_params TEXT NOT NULL DEFAULT '',
                    is_active INTEGER NOT NULL DEFAULT 0,
                    schema_cache TEXT NOT NULL DEFAULT '',
                    last_tested_at TEXT NOT NULL DEFAULT '',
                    last_status TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            await conn.commit()
            await self._migrate_legacy_json(conn)

    async def _migrate_legacy_json(self, conn: aiosqlite.Connection) -> None:
        if not self.legacy_json_path or not self.legacy_json_path.is_file():
            return
        async with conn.execute("SELECT COUNT(*) AS cnt FROM chats") as cur:
            row = await cur.fetchone()
            if row and row["cnt"] > 0:
                return

        try:
            raw = json.loads(self.legacy_json_path.read_text(encoding="utf-8"))
        except Exception:
            return
        if not isinstance(raw, dict) or not raw:
            return

        now = time.strftime("%Y-%m-%d %H:%M:%S")
        rows_to_insert = []
        for cid, cdata in raw.items():
            if not isinstance(cdata, dict):
                continue
            messages = cdata.get("messages")
            if not isinstance(messages, list):
                messages = []
            created_at = str(cdata.get("created_at") or now)
            updated_at = str(cdata.get("updated_at") or created_at)
            rows_to_insert.append(
                (
                    str(cdata.get("id") or cid),
                    str(cdata.get("title") or "New Chat"),
                    str(cdata.get("provider") or "ollama"),
                    str(cdata.get("model") or ""),
                    json.dumps(messages, ensure_ascii=False),
                    created_at,
                    updated_at,
                )
            )
        if rows_to_insert:
            await conn.executemany(
                """
                INSERT OR IGNORE INTO chats (id, title, provider, model, messages_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                rows_to_insert,
            )
            await conn.commit()

    async def get_settings(self) -> dict:
        merged = dict(DEFAULT_SETTINGS)
        async with self._connect() as conn:
            async with conn.execute("SELECT key, value FROM settings") as cur:
                rows = await cur.fetchall()
        for row in rows:
            key = row["key"]
            if key not in DEFAULT_SETTINGS:
                continue
            try:
                merged[key] = json.loads(row["value"])
            except Exception:
                merged[key] = row["value"]
        return merged

    async def update_settings(self, patch: dict) -> dict:
        current = await self.get_settings()
        updated = dict(current)

        for key, raw_val in patch.items():
            if key not in DEFAULT_SETTINGS or raw_val is None:
                continue
            if key == "default_provider":
                val = str(raw_val).strip().lower()
                if val not in {"ollama", "llamacpp"}:
                    raise ValueError(f"Unsupported default_provider: {raw_val}")
                updated[key] = val
            elif key in {"ollama_url", "llamacpp_url"}:
                val = str(raw_val).strip().rstrip("/")
                if not val.startswith(("http://", "https://")):
                    raise ValueError(f"{key} must start with http:// or https://")
                updated[key] = val
            elif key == "wsl_auto_detect":
                updated[key] = bool(raw_val)
            elif key in {"wsl_distro", "wsl_gguf_dir", "api_key"}:
                updated[key] = str(raw_val).strip()
            elif key == "system_prompt":
                updated[key] = str(raw_val)
            elif key == "db_context_prompt":
                updated[key] = str(raw_val)
            elif key == "db_non_sql_fallback":
                updated[key] = bool(raw_val)
            elif key == "temperature":
                temp = float(raw_val)
                updated[key] = max(0.0, min(2.0, round(temp, 3)))
            elif key == "max_tokens":
                updated[key] = max(0, int(raw_val))

        async with self._connect() as conn:
            await conn.executemany(
                """
                INSERT INTO settings (key, value) VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                [(k, json.dumps(updated[k], ensure_ascii=False)) for k in DEFAULT_SETTINGS],
            )
            await conn.commit()
        return updated

    async def list_chats(self) -> list[dict]:
        async with self._connect() as conn:
            async with conn.execute(
                """
                SELECT id, title, provider, model, messages_json, updated_at
                FROM chats
                ORDER BY updated_at DESC, rowid DESC
                """
            ) as cur:
                rows = await cur.fetchall()

        summary: list[dict] = []
        for row in rows:
            try:
                msgs = json.loads(row["messages_json"])
                msg_count = len(msgs) if isinstance(msgs, list) else 0
            except Exception:
                msg_count = 0
            summary.append(
                {
                    "id": row["id"],
                    "title": row["title"] or "New Chat",
                    "provider": row["provider"] or "ollama",
                    "model": row["model"] or "",
                    "updated_at": row["updated_at"],
                    "message_count": msg_count,
                }
            )
        return summary

    async def get_chat(self, chat_id: str) -> dict | None:
        async with self._connect() as conn:
            async with conn.execute(
                """
                SELECT id, title, provider, model, messages_json, created_at, updated_at
                FROM chats
                WHERE id = ?
                """,
                (chat_id,),
            ) as cur:
                row = await cur.fetchone()
        if not row:
            return None
        try:
            messages = json.loads(row["messages_json"])
            if not isinstance(messages, list):
                messages = []
        except Exception:
            messages = []
        return {
            "id": row["id"],
            "title": row["title"] or "New Chat",
            "provider": row["provider"] or "ollama",
            "model": row["model"] or "",
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "messages": messages,
        }

    async def upsert_chat(self, chat_id: str, data: dict) -> dict:
        existing = await self.get_chat(chat_id)
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        settings = await self.get_settings()

        title = data.get("title") if "title" in data and data["title"] is not None else (existing["title"] if existing else "New Chat")
        provider = (
            data.get("provider")
            if "provider" in data and data["provider"] in {"ollama", "llamacpp"}
            else (existing["provider"] if existing else settings["default_provider"])
        )
        model = data.get("model") if "model" in data and data["model"] is not None else (existing["model"] if existing else "")
        messages = data.get("messages") if "messages" in data and isinstance(data["messages"], list) else (existing["messages"] if existing else [])
        created_at = existing["created_at"] if existing else now
        updated_at = now

        async with self._connect() as conn:
            await conn.execute(
                """
                INSERT INTO chats (id, title, provider, model, messages_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    title = excluded.title,
                    provider = excluded.provider,
                    model = excluded.model,
                    messages_json = excluded.messages_json,
                    updated_at = excluded.updated_at
                """,
                (
                    chat_id,
                    str(title),
                    str(provider),
                    str(model),
                    json.dumps(messages, ensure_ascii=False),
                    created_at,
                    updated_at,
                ),
            )
            await conn.commit()

        return {
            "id": chat_id,
            "title": str(title),
            "provider": str(provider),
            "model": str(model),
            "created_at": created_at,
            "updated_at": updated_at,
            "messages": messages,
        }

    async def delete_chat(self, chat_id: str) -> bool:
        async with self._connect() as conn:
            cur = await conn.execute("DELETE FROM chats WHERE id = ?", (chat_id,))
            await conn.commit()
            return cur.rowcount > 0

    async def export_all_chats(self) -> list[dict]:
        async with self._connect() as conn:
            async with conn.execute(
                """
                SELECT id, title, provider, model, messages_json, created_at, updated_at
                FROM chats
                ORDER BY updated_at DESC, rowid DESC
                """
            ) as cur:
                rows = await cur.fetchall()

        chats: list[dict] = []
        for row in rows:
            try:
                messages = json.loads(row["messages_json"])
                if not isinstance(messages, list):
                    messages = []
            except Exception:
                messages = []
            chats.append(
                {
                    "id": row["id"],
                    "title": row["title"] or "New Chat",
                    "provider": row["provider"] or "ollama",
                    "model": row["model"] or "",
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                    "messages": messages,
                }
            )
        return chats

    async def delete_all_chats(self) -> int:
        async with self._connect() as conn:
            cur = await conn.execute("DELETE FROM chats")
            await conn.commit()
            return cur.rowcount if cur.rowcount and cur.rowcount > 0 else 0

    @staticmethod
    def _connector_from_row(row: aiosqlite.Row, include_secret: bool = False) -> dict:
        raw_pw = row["password"] or ""
        item = {
            "id": row["id"],
            "name": row["name"] or "Database",
            "db_type": row["db_type"] or "postgresql",
            "host": row["host"] or "127.0.0.1",
            "port": int(row["port"] or 0),
            "database_name": row["database_name"] or "",
            "username": row["username"] or "",
            "has_password": bool(raw_pw),
            "ssl_mode": row["ssl_mode"] or "",
            "extra_params": row["extra_params"] or "",
            "is_active": bool(row["is_active"]),
            "schema_cache": row["schema_cache"] or "",
            "last_tested_at": row["last_tested_at"] or "",
            "last_status": row["last_status"] or "",
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }
        if include_secret:
            item["password"] = raw_pw
        return item

    async def list_connectors(self, include_secret: bool = False) -> list[dict]:
        async with self._connect() as conn:
            async with conn.execute(
                """
                SELECT * FROM db_connectors
                ORDER BY is_active DESC, updated_at DESC, name ASC
                """
            ) as cur:
                rows = await cur.fetchall()
        return [self._connector_from_row(r, include_secret=include_secret) for r in rows]

    async def get_connector(self, connector_id: str, include_secret: bool = False) -> dict | None:
        if not connector_id:
            return None
        async with self._connect() as conn:
            async with conn.execute(
                "SELECT * FROM db_connectors WHERE id = ?",
                (connector_id,),
            ) as cur:
                row = await cur.fetchone()
        if not row:
            return None
        return self._connector_from_row(row, include_secret=include_secret)

    async def get_active_connector(self, include_secret: bool = False) -> dict | None:
        async with self._connect() as conn:
            async with conn.execute(
                "SELECT * FROM db_connectors WHERE is_active = 1 ORDER BY updated_at DESC LIMIT 1"
            ) as cur:
                row = await cur.fetchone()
        if not row:
            return None
        return self._connector_from_row(row, include_secret=include_secret)

    async def upsert_connector(self, data: dict) -> dict:
        default_ports = {"postgresql": 5432, "mysql": 3306, "mariadb": 3306, "mssql": 1433, "sqlite": 0}
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        cid = str(data.get("id") or "").strip() or str(uuid.uuid4())

        async with self._connect() as conn:
            async with conn.execute("SELECT * FROM db_connectors WHERE id = ?", (cid,)) as cur:
                existing = await cur.fetchone()

            db_type = str(data.get("db_type") or (existing["db_type"] if existing else "postgresql")).strip().lower()
            if db_type in ("postgres", "pg"):
                db_type = "postgresql"
            elif db_type in ("sqlserver", "sql_server"):
                db_type = "mssql"

            raw_port = data.get("port")
            if db_type == "sqlite":
                port = 0
            elif raw_port is not None and str(raw_port).strip() != "" and int(raw_port) > 0:
                port = int(raw_port)
            elif existing and existing["db_type"] == db_type and int(existing["port"] or 0) > 0:
                port = int(existing["port"])
            else:
                port = default_ports.get(db_type, 5432)

            if data.get("clear_password"):
                password = ""
            elif data.get("password") not in (None, ""):
                password = str(data["password"])
            else:
                password = existing["password"] if existing else ""

            if "is_active" in data and data["is_active"] is not None:
                is_active = 1 if data["is_active"] else 0
            else:
                is_active = int(existing["is_active"]) if existing else 0

            name = str(data.get("name") if data.get("name") is not None else (existing["name"] if existing else "Database")).strip() or "Database"
            host = str(data.get("host") if data.get("host") is not None else (existing["host"] if existing else "127.0.0.1")).strip()
            if db_type != "sqlite" and not host:
                host = "127.0.0.1"
            database_name = str(
                data.get("database_name") if data.get("database_name") is not None else (existing["database_name"] if existing else "")
            ).strip()
            username = str(data.get("username") if data.get("username") is not None else (existing["username"] if existing else "")).strip()
            ssl_mode = str(data.get("ssl_mode") if data.get("ssl_mode") is not None else (existing["ssl_mode"] if existing else "")).strip()
            extra_params = str(
                data.get("extra_params") if data.get("extra_params") is not None else (existing["extra_params"] if existing else "")
            ).strip()
            schema_cache = str(
                data.get("schema_cache") if data.get("schema_cache") is not None else (existing["schema_cache"] if existing else "")
            )
            last_tested_at = str(
                data.get("last_tested_at") if data.get("last_tested_at") is not None else (existing["last_tested_at"] if existing else "")
            )
            last_status = str(
                data.get("last_status") if data.get("last_status") is not None else (existing["last_status"] if existing else "")
            )
            created_at = str(existing["created_at"]) if existing else now

            if is_active == 1:
                await conn.execute("UPDATE db_connectors SET is_active = 0 WHERE id != ?", (cid,))

            await conn.execute(
                """
                INSERT INTO db_connectors (
                    id, name, db_type, host, port, database_name, username, password,
                    ssl_mode, extra_params, is_active, schema_cache, last_tested_at, last_status,
                    created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    name = excluded.name,
                    db_type = excluded.db_type,
                    host = excluded.host,
                    port = excluded.port,
                    database_name = excluded.database_name,
                    username = excluded.username,
                    password = excluded.password,
                    ssl_mode = excluded.ssl_mode,
                    extra_params = excluded.extra_params,
                    is_active = excluded.is_active,
                    schema_cache = excluded.schema_cache,
                    last_tested_at = excluded.last_tested_at,
                    last_status = excluded.last_status,
                    updated_at = excluded.updated_at
                """,
                (
                    cid,
                    name,
                    db_type,
                    host,
                    port,
                    database_name,
                    username,
                    password,
                    ssl_mode,
                    extra_params,
                    is_active,
                    schema_cache,
                    last_tested_at,
                    last_status,
                    created_at,
                    now,
                ),
            )
            await conn.commit()

        saved = await self.get_connector(cid, include_secret=False)
        assert saved is not None
        return saved

    async def set_active_connector(self, connector_id: str | None) -> dict | None:
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        async with self._connect() as conn:
            if not connector_id:
                await conn.execute("UPDATE db_connectors SET is_active = 0, updated_at = ? WHERE is_active != 0", (now,))
                await conn.commit()
                return None

            async with conn.execute("SELECT id FROM db_connectors WHERE id = ?", (connector_id,)) as cur:
                row = await cur.fetchone()
            if not row:
                return None

            await conn.execute("UPDATE db_connectors SET is_active = 0 WHERE id != ?", (connector_id,))
            await conn.execute(
                "UPDATE db_connectors SET is_active = 1, updated_at = ? WHERE id = ?",
                (now, connector_id),
            )
            await conn.commit()
        return await self.get_connector(connector_id, include_secret=False)

    async def update_connector_schema_cache(self, connector_id: str, schema_text: str, status: str = "ok") -> dict | None:
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        async with self._connect() as conn:
            cur = await conn.execute(
                """
                UPDATE db_connectors
                SET schema_cache = ?, last_status = ?, last_tested_at = ?, updated_at = ?
                WHERE id = ?
                """,
                (schema_text, status, now, now, connector_id),
            )
            await conn.commit()
            if not cur.rowcount:
                return None
        return await self.get_connector(connector_id, include_secret=False)

    async def update_connector_test_status(self, connector_id: str, status: str) -> dict | None:
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        async with self._connect() as conn:
            cur = await conn.execute(
                """
                UPDATE db_connectors
                SET last_status = ?, last_tested_at = ?, updated_at = ?
                WHERE id = ?
                """,
                (status, now, now, connector_id),
            )
            await conn.commit()
            if not cur.rowcount:
                return None
        return await self.get_connector(connector_id, include_secret=False)

    async def delete_connector(self, connector_id: str) -> bool:
        async with self._connect() as conn:
            cur = await conn.execute("DELETE FROM db_connectors WHERE id = ?", (connector_id,))
            await conn.commit()
            return cur.rowcount > 0
