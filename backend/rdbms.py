import asyncio
import datetime
import decimal
from pathlib import Path
import re
import time
import uuid
from typing import Any, Callable
from urllib.parse import parse_qsl

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine, URL
from sqlalchemy.exc import OperationalError, SQLAlchemyError
from sqlalchemy.pool import NullPool

from backend.wsl import get_wsl_ip

DEFAULT_PORTS: dict[str, int] = {
    "postgresql": 5432,
    "mysql": 3306,
    "mariadb": 3306,
    "mssql": 1433,
    "sqlite": 0,
}

SUPPORTED_DB_TYPES = set(DEFAULT_PORTS.keys())

_ALLOWED_START_KEYWORDS = {
    "SELECT",
    "WITH",
    "SHOW",
    "DESCRIBE",
    "DESC",
    "EXPLAIN",
    "PRAGMA",
}

_FORBIDDEN_KEYWORDS_RE = re.compile(
    r"\b("
    r"INSERT|UPDATE|DELETE|DROP|ALTER|TRUNCATE|CREATE|REPLACE|MERGE|"
    r"GRANT|REVOKE|CALL|EXEC|EXECUTE|COPY|ATTACH|DETACH|VACUUM|LOCK"
    r")\b",
    re.IGNORECASE,
)

_SELECT_INTO_RE = re.compile(
    r"\bSELECT\b[\s\S]*\bINTO\s+(OUTFILE|DUMPFILE|[A-Za-z_\"`\[])",
    re.IGNORECASE,
)


def normalize_db_type(db_type: str | None) -> str:
    val = str(db_type or "postgresql").strip().lower()
    if val in ("postgres", "pg"):
        return "postgresql"
    if val in ("sqlserver", "sql_server"):
        return "mssql"
    return val


def is_loopback_host(host: str | None) -> bool:
    h = str(host or "").strip().lower()
    return h in ("", "127.0.0.1", "localhost", "::1", "[::1]")


def build_sqlalchemy_url(cfg: dict, host_override: str | None = None) -> URL:
    db_type = normalize_db_type(cfg.get("db_type"))
    if db_type not in SUPPORTED_DB_TYPES:
        raise ValueError(
            f"Unsupported database type '{db_type}'. Supported: postgresql, mysql, mariadb, sqlite, mssql."
        )

    database_name = str(cfg.get("database_name") or "").strip()
    extra_raw = str(cfg.get("extra_params") or "").strip().lstrip("?")
    query: dict[str, str] = {}
    if extra_raw:
        for k, v in parse_qsl(extra_raw, keep_blank_values=False):
            query[k] = v

    if db_type == "sqlite":
        if not database_name:
            raise ValueError("SQLite requires a database file path in Database Name / Path.")
        return URL.create(drivername="sqlite", database=database_name)

    host = (host_override if host_override is not None else str(cfg.get("host") or "")).strip() or "127.0.0.1"
    raw_port = cfg.get("port")
    port = int(raw_port) if raw_port not in (None, "", 0) else DEFAULT_PORTS[db_type]
    username = str(cfg.get("username") or "").strip() or None
    password = str(cfg.get("password") or "") or None
    ssl_mode = str(cfg.get("ssl_mode") or "").strip()

    if db_type == "postgresql":
        if ssl_mode:
            query.setdefault("sslmode", ssl_mode)
        return URL.create(
            drivername="postgresql+psycopg",
            username=username,
            password=password,
            host=host,
            port=port,
            database=database_name or "postgres",
            query=query,
        )

    if db_type in ("mysql", "mariadb"):
        query.setdefault("charset", "utf8mb4")
        return URL.create(
            drivername="mysql+pymysql",
            username=username,
            password=password,
            host=host,
            port=port,
            database=database_name or None,
            query=query,
        )

    # mssql
    query.setdefault("driver", "ODBC Driver 18 for SQL Server")
    return URL.create(
        drivername="mssql+pyodbc",
        username=username,
        password=password,
        host=host,
        port=port,
        database=database_name or None,
        query=query,
    )


def _create_engine_sync(cfg: dict, host_override: str | None = None, timeout_sec: int = 8) -> Engine:
    db_type = normalize_db_type(cfg.get("db_type"))
    database_name = str(cfg.get("database_name") or "").strip()
    if db_type == "sqlite" and database_name != ":memory:":
        if not database_name:
            raise ValueError("SQLite requires a database file path in Database Name / Path.")
        if not Path(database_name).is_file():
            raise FileNotFoundError(f"SQLite database file not found: {database_name}")

    url = build_sqlalchemy_url(cfg, host_override=host_override)
    connect_args: dict[str, Any] = {}
    if db_type == "postgresql":
        connect_args["connect_timeout"] = timeout_sec
    elif db_type in ("mysql", "mariadb"):
        connect_args["connect_timeout"] = timeout_sec
        connect_args["read_timeout"] = 15
        ssl_mode = str(cfg.get("ssl_mode") or "").strip().lower()
        if ssl_mode and ssl_mode not in ("disable", "off", "false"):
            connect_args["ssl"] = {}
    elif db_type == "sqlite":
        connect_args["timeout"] = timeout_sec

    try:
        return create_engine(url, poolclass=NullPool, pool_pre_ping=False, connect_args=connect_args)
    except ModuleNotFoundError as exc:
        if db_type == "mssql":
            raise RuntimeError(
                "SQL Server driver 'pyodbc' is not installed in the Python environment."
            ) from exc
        raise RuntimeError(f"Missing Python database driver for {db_type}: {exc}") from exc


def _strip_comments_and_scan_sql(sql: str) -> tuple[str, str, int, list[int]]:
    """
    Scan SQL tracking string literals ('...'), quoted identifiers ("..." / `...`),
    and SQL comments (-- ... and /* ... */).

    Returns:
      comment_free: SQL preserving literal contents (comments replaced by spaces)
      unquoted_upper_skeleton: SQL with comments removed and quoted literals replaced
          by '' / "" so keyword checks ignore text inside string literals
      semicolon_count: number of semicolons outside quotes/comments
      non_ws_after_semicolons: positions of any non-whitespace token after a semicolon
    """
    n = len(sql)
    i = 0
    comment_free_chars: list[str] = []
    skeleton_chars: list[str] = []
    semicolon_positions: list[int] = []
    non_ws_after_semi: list[int] = []
    seen_semi = False

    while i < n:
        ch = sql[i]
        nxt = sql[i + 1] if i + 1 < n else ""

        # Single-line comment
        if ch == "-" and nxt == "-":
            while i < n and sql[i] != "\n":
                i += 1
            comment_free_chars.append(" ")
            skeleton_chars.append(" ")
            continue

        # Block comment
        if ch == "/" and nxt == "*":
            i += 2
            while i < n - 1 and not (sql[i] == "*" and sql[i + 1] == "/"):
                i += 1
            i = min(n, i + 2)
            comment_free_chars.append(" ")
            skeleton_chars.append(" ")
            continue

        # Quoted string or identifier ('...', "...", `...`)
        if ch in ("'", '"', "`"):
            quote = ch
            start = i
            i += 1
            while i < n:
                if sql[i] == "\\" and quote != '"':
                    i += 2
                    continue
                if sql[i] == quote:
                    if i + 1 < n and sql[i + 1] == quote:
                        i += 2
                        continue
                    i += 1
                    break
                i += 1
            literal_text = sql[start:i]
            comment_free_chars.append(literal_text)
            skeleton_chars.append(f"{quote}{quote}")
            if seen_semi:
                non_ws_after_semi.append(start)
            continue

        if ch == ";":
            semicolon_positions.append(i)
            seen_semi = True
            comment_free_chars.append(ch)
            skeleton_chars.append(ch)
            i += 1
            continue

        if seen_semi and not ch.isspace():
            non_ws_after_semi.append(i)

        comment_free_chars.append(ch)
        skeleton_chars.append(ch)
        i += 1

    return (
        "".join(comment_free_chars).strip(),
        "".join(skeleton_chars).strip(),
        len(semicolon_positions),
        non_ws_after_semi,
    )


def validate_readonly_sql(sql: str) -> tuple[bool, str, str]:
    """
    Validate that SQL is a single read-only query.
    Returns (ok, cleaned_sql, error_message).
    """
    raw = str(sql or "").strip()
    if not raw:
        return False, "", "Empty SQL query."

    comment_free, skeleton, semi_count, non_ws_after_semi = _strip_comments_and_scan_sql(raw)
    if not comment_free:
        return False, "", "SQL query contains only comments or whitespace."

    if non_ws_after_semi or semi_count > 1:
        return False, "", "Multiple SQL statements or trailing tokens after ';' are blocked for safety."

    # Strip a single trailing semicolon outside quotes/comments
    cleaned = comment_free[:-1].strip() if comment_free.endswith(";") else comment_free
    skeleton_clean = skeleton[:-1].strip() if skeleton.endswith(";") else skeleton

    first_match = re.match(r"^\s*\(*\s*([A-Za-z_]+)", skeleton_clean)
    if not first_match:
        return False, "", "Could not determine leading SQL keyword."

    lead_kw = first_match.group(1).upper()
    if lead_kw not in _ALLOWED_START_KEYWORDS:
        return (
            False,
            "",
            f"Only read-only queries (SELECT, WITH, SHOW, DESCRIBE, EXPLAIN, PRAGMA) are allowed; got '{lead_kw}'.",
        )

    if lead_kw == "PRAGMA" and "=" in skeleton_clean:
        return False, "", "Modifying PRAGMA statements with '=' are not allowed."

    forbidden_match = _FORBIDDEN_KEYWORDS_RE.search(skeleton_clean)
    if forbidden_match:
        return (
            False,
            "",
            f"Write or DDL keyword '{forbidden_match.group(1).upper()}' is blocked in read-only mode.",
        )

    if _SELECT_INTO_RE.search(skeleton_clean):
        return False, "", "SELECT ... INTO statements are blocked in read-only mode."

    return True, cleaned, ""


def extract_sql_from_llm_output(text_out: str) -> str:
    """Extract raw SQL statement from LLM output, stripping <think> blocks and markdown fences."""
    cleaned = re.sub(r"<think>[\s\S]*?</think>", "", str(text_out or ""), flags=re.IGNORECASE).strip()
    if not cleaned:
        return ""

    sql_fence = re.search(r"```sql\s*([\s\S]*?)```", cleaned, flags=re.IGNORECASE)
    if sql_fence:
        return sql_fence.group(1).strip()

    any_fence = re.search(r"```\s*([\s\S]*?)```", cleaned)
    if any_fence:
        candidate = any_fence.group(1).strip()
        if candidate:
            return candidate

    sql_start_re = re.compile(
        r"(?:^|\n|:\s*)\s*("
        r"SELECT\b[\s\S]*"
        r"|WITH\s+[A-Za-z_\"`\[][A-Za-z0-9_\"`\]]*\s+AS\s*\([\s\S]*"
        r"|(?:SHOW|DESCRIBE|EXPLAIN|PRAGMA)\b[\s\S]*"
        r"|(?:INSERT\s+INTO|UPDATE\s+[A-Za-z_\"`\[]|DELETE\s+FROM|DROP\s+|ALTER\s+|CREATE\s+|TRUNCATE\s+)[\s\S]*"
        r")",
        re.IGNORECASE,
    )
    m = sql_start_re.search(cleaned)
    if m:
        tail = m.group(1).strip()
        semi_split = re.split(r";\s*\n+", tail, maxsplit=1)
        return semi_split[0].strip().rstrip(";")

    return ""


def _json_safe_value(val: Any) -> Any:
    if val is None or isinstance(val, (bool, int, float, str)):
        return val
    if isinstance(val, decimal.Decimal):
        return float(val) if val.as_tuple().exponent < 0 else int(val)
    if isinstance(val, (datetime.datetime, datetime.date, datetime.time)):
        return val.isoformat()
    if isinstance(val, uuid.UUID):
        return str(val)
    if isinstance(val, (bytes, bytearray, memoryview)):
        raw = bytes(val)
        return f"<binary {len(raw)} bytes>"
    return str(val)


def test_connection_sync(cfg: dict, *, host_override: str | None = None) -> dict:
    db_type = normalize_db_type(cfg.get("db_type"))
    engine = _create_engine_sync(cfg, host_override=host_override, timeout_sec=6)
    t0 = time.perf_counter()
    try:
        probe_sql = "SELECT 1"
        version_sql = {
            "postgresql": "SELECT version()",
            "mysql": "SELECT VERSION()",
            "mariadb": "SELECT VERSION()",
            "sqlite": "SELECT sqlite_version()",
            "mssql": "SELECT @@VERSION",
        }.get(db_type, "SELECT 1")
        with engine.connect() as conn:
            conn.execute(text(probe_sql)).fetchone()
            server_version = ""
            try:
                vrow = conn.execute(text(version_sql)).fetchone()
                if vrow and vrow[0] is not None:
                    server_version = str(vrow[0]).splitlines()[0][:160]
            except Exception:
                server_version = ""
            inspector = inspect(conn)
            table_count = len(inspector.get_table_names())
        latency_ms = round((time.perf_counter() - t0) * 1000, 1)
        return {
            "ok": True,
            "db_type": db_type,
            "host_used": host_override or cfg.get("host") or ("sqlite" if db_type == "sqlite" else "127.0.0.1"),
            "latency_ms": latency_ms,
            "server_version": server_version,
            "table_count": table_count,
            "message": f"Connected ({db_type}, {table_count} tables, {latency_ms} ms)",
        }
    finally:
        engine.dispose()


def introspect_schema_sync(cfg: dict, max_tables: int = 60, *, host_override: str | None = None) -> dict:
    db_type = normalize_db_type(cfg.get("db_type"))
    engine = _create_engine_sync(cfg, host_override=host_override, timeout_sec=8)
    try:
        with engine.connect() as conn:
            inspector = inspect(conn)
            schema_targets: list[str | None] = [None]
            if db_type == "postgresql":
                try:
                    all_schemas = inspector.get_schema_names()
                    user_schemas = [
                        s
                        for s in all_schemas
                        if s not in ("information_schema", "pg_catalog", "pg_toast")
                        and not s.startswith("pg_temp_")
                        and not s.startswith("pg_toast_temp_")
                    ]
                    if "public" in user_schemas:
                        user_schemas = ["public"] + [s for s in user_schemas if s != "public"]
                    if user_schemas:
                        schema_targets = user_schemas[:5]
                except Exception:
                    schema_targets = ["public"]

            tables_info: list[dict] = []
            text_lines: list[str] = []

            for schema_name in schema_targets:
                table_names = sorted(inspector.get_table_names(schema=schema_name))
                for tname in table_names:
                    if len(tables_info) >= max_tables:
                        break
                    full_name = f"{schema_name}.{tname}" if schema_name and schema_name != "public" else tname
                    try:
                        cols = inspector.get_columns(tname, schema=schema_name)
                    except Exception:
                        cols = []
                    try:
                        pk_cols = set((inspector.get_pk_constraint(tname, schema=schema_name) or {}).get("constrained_columns") or [])
                    except Exception:
                        pk_cols = set()
                    fk_map: dict[str, str] = {}
                    try:
                        for fk in inspector.get_foreign_keys(tname, schema=schema_name) or []:
                            ref_table = fk.get("referred_table") or ""
                            ref_cols = fk.get("referred_columns") or []
                            for idx, col_name in enumerate(fk.get("constrained_columns") or []):
                                ref_col = ref_cols[idx] if idx < len(ref_cols) else (ref_cols[0] if ref_cols else "")
                                if ref_table:
                                    fk_map[col_name] = f"{ref_table}.{ref_col}" if ref_col else ref_table
                    except Exception:
                        fk_map = {}

                    col_entries = []
                    col_summaries = []
                    for col in cols:
                        cname = str(col.get("name") or "")
                        ctype = str(col.get("type") or "TEXT")
                        is_pk = cname in pk_cols
                        fk_ref = fk_map.get(cname, "")
                        col_entries.append(
                            {
                                "name": cname,
                                "type": ctype,
                                "nullable": bool(col.get("nullable", True)),
                                "primary_key": is_pk,
                                "foreign_key": fk_ref,
                            }
                        )
                        parts = [f"{cname} {ctype}"]
                        if is_pk:
                            parts.append("[PK]")
                        if fk_ref:
                            parts.append(f"-> {fk_ref}")
                        col_summaries.append(" ".join(parts))

                    tables_info.append(
                        {
                            "schema": schema_name or "",
                            "name": tname,
                            "full_name": full_name,
                            "columns": col_entries,
                        }
                    )
                    text_lines.append(f"TABLE {full_name} ({', '.join(col_summaries)})")

                if len(tables_info) >= max_tables:
                    break

            schema_text = "\n".join(text_lines) if text_lines else "(No tables found in database.)"
            return {
                "db_type": db_type,
                "table_count": len(tables_info),
                "tables": tables_info,
                "schema_text": schema_text,
            }
    finally:
        engine.dispose()


def execute_query_sync(
    cfg: dict,
    sql: str,
    max_rows: int = 200,
    *,
    host_override: str | None = None,
) -> dict:
    ok, cleaned_sql, err_msg = validate_readonly_sql(sql)
    if not ok:
        raise ValueError(err_msg)

    row_limit = max(1, min(int(max_rows or 200), 1000))
    db_type = normalize_db_type(cfg.get("db_type"))
    engine = _create_engine_sync(cfg, host_override=host_override, timeout_sec=10)
    t0 = time.perf_counter()

    try:
        with engine.connect() as conn:
            if db_type == "postgresql":
                try:
                    conn.execute(text("SET TRANSACTION READ ONLY"))
                    conn.execute(text("SET LOCAL statement_timeout = 10000"))
                except Exception:
                    pass
            result = conn.execute(text(cleaned_sql))
            columns = [str(c) for c in result.keys()] if result.returns_rows else []
            fetched = result.fetchmany(row_limit + 1) if result.returns_rows else []
            truncated = len(fetched) > row_limit
            rows_slice = fetched[:row_limit]
            rows = [[_json_safe_value(cell) for cell in row] for row in rows_slice]
            try:
                conn.rollback()
            except Exception:
                pass

        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
        return {
            "sql": cleaned_sql,
            "columns": columns,
            "rows": rows,
            "row_count": len(rows),
            "truncated": truncated,
            "elapsed_ms": elapsed_ms,
        }
    except SQLAlchemyError as exc:
        orig = getattr(exc, "orig", exc)
        raise RuntimeError(str(orig)) from exc
    finally:
        engine.dispose()


_NON_NETWORK_ERROR_MARKERS = (
    "password authentication failed",
    "authentication failed",
    "access denied",
    "1045",
    "does not exist",
    "unknown database",
    "1049",
    "3d000",
    "fe_sendauth",
    "pg_hba.conf",
    "login failed",
    "syntax error",
)


def _is_connection_error(exc: Exception) -> bool:
    if isinstance(exc, FileNotFoundError):
        return False
    orig = getattr(exc, "orig", exc)
    msg = f"{orig} {exc}".lower()
    if any(marker in msg for marker in _NON_NETWORK_ERROR_MARKERS):
        return False
    if isinstance(exc, (ConnectionError, TimeoutError)):
        return True
    if isinstance(exc, OSError) and not isinstance(exc, FileNotFoundError):
        return True
    return any(
        needle in msg
        for needle in (
            "connection refused",
            "could not connect",
            "can't connect",
            "timed out",
            "timeout",
            "network is unreachable",
            "actively refused",
            "winerror 10061",
            "winerror 10060",
            "(2003,",
            "getaddrinfo failed",
            "could not translate host",
            "name or service not known",
        )
    )


def format_db_error(
    exc: Exception,
    cfg: dict,
    tried_hosts: list[str] | None = None,
) -> dict:
    db_type = normalize_db_type(cfg.get("db_type"))
    host = str(cfg.get("host") or ("127.0.0.1" if db_type != "sqlite" else "")).strip()
    port = cfg.get("port") or DEFAULT_PORTS.get(db_type, "")
    db_name = str(cfg.get("database_name") or "").strip()
    username = str(cfg.get("username") or "").strip()
    target = db_name if db_type == "sqlite" else f"{host}:{port}/{db_name or '(default)'}"

    orig = getattr(exc, "orig", exc)
    raw_str = str(orig if orig else exc)
    cleaned = re.sub(
        r"\(Background on this error at:\s*https?://sqlalche\.me/[^\)]+\)",
        "",
        raw_str,
        flags=re.IGNORECASE,
    )
    cleaned = " ".join(cleaned.split()).strip() or type(exc).__name__
    msg_lower = cleaned.lower()
    tried = list(dict.fromkeys([h for h in (tried_hosts or []) if h]))

    if isinstance(exc, FileNotFoundError) or "file not found" in msg_lower or "unable to open database file" in msg_lower:
        error = f"SQLite database file not found: {db_name or '(empty)'}"
        hint = (
            "Check the full .db/.sqlite file path (e.g. E:\\data\\sample.db). "
            "If the file is inside WSL, use \\\\wsl.localhost\\<Distro>\\path or copy it to Windows."
        )
    elif "cryptography" in msg_lower:
        error = "MySQL 8+ authentication requires the 'cryptography' Python package."
        hint = "Install dependencies (`pip install cryptography`) in .venv and restart the server."
    elif "pyodbc" in msg_lower or isinstance(exc, ModuleNotFoundError):
        error = f"Database driver unavailable for {db_type}: {cleaned}"
        hint = "Ensure the required driver (e.g. pyodbc and ODBC Driver 18 for SQL Server) is installed."
    elif any(
        tok in msg_lower
        for tok in (
            "password authentication failed",
            "authentication failed",
            "access denied",
            "1045",
            "fe_sendauth",
            "pg_hba.conf",
            "login failed",
            "28p01",
            "28000",
        )
    ):
        error = f"Authentication failed for {db_type} user '{username or '(empty)'}' at {target}."
        hint = (
            "Verify username and password. For PostgreSQL in WSL, ensure pg_hba.conf allows "
            "scram-sha-256 or md5 connections from your Windows host (e.g. 172.16.0.0/12)."
        )
    elif (
        ("database" in msg_lower and "does not exist" in msg_lower)
        or "unknown database" in msg_lower
        or "1049" in msg_lower
        or "3d000" in msg_lower
        or "cannot open database" in msg_lower
    ):
        error = f"Database '{db_name}' does not exist on {host}:{port}."
        hint = "Check the spelling and case of Database Name, and verify that the database exists."
    elif any(tok in msg_lower for tok in ("ssl", "tls", "certificate", "handshake")):
        error = f"SSL/TLS negotiation failed connecting to {target}."
        hint = "Open Advanced Options and try setting SSL Mode to 'disable' or 'prefer' for local or WSL databases."
    elif _is_connection_error(exc) or any(
        tok in msg_lower
        for tok in (
            "connection refused",
            "10061",
            "10060",
            "(2003,",
            "timeout",
            "could not connect",
            "can't connect",
            "getaddrinfo failed",
            "name or service not known",
            "could not translate host",
            "network is unreachable",
        )
    ):
        hosts_str = ", ".join(tried) if tried else (host or "127.0.0.1")
        error = f"Cannot reach {db_type} server at {target} (tried: {hosts_str})."
        hint = (
            "Verify the database service is running and listening on that port. "
            "If running inside WSL, configure listen_addresses = '*' (PostgreSQL) or "
            "bind-address = 0.0.0.0 (MySQL/MariaDB) and allow the host subnet."
        )
    else:
        error = f"{db_type} error: {cleaned}"
        hint = "Verify your host, port, credentials, and database permissions."

    return {
        "error": error,
        "detail": cleaned,
        "hint": hint,
        "tried_hosts": tried,
    }


def _attach_tried_hosts(exc: Exception, tried_hosts: list[str]) -> None:
    try:
        setattr(exc, "_tried_hosts", list(dict.fromkeys(tried_hosts)))
    except Exception:
        pass


async def run_with_wsl_fallback(
    cfg: dict,
    wsl_auto_detect: bool,
    wsl_distro: str,
    fn: Callable[..., Any],
    *args: Any,
) -> Any:
    """
    Run synchronous SQLAlchemy work in a thread.
    For non-SQLite loopback hosts on connection error, falls back from
    localhost -> 127.0.0.1 -> detected WSL IP and records all tried hosts.
    """
    db_type = normalize_db_type(cfg.get("db_type"))
    initial_host = (
        "sqlite"
        if db_type == "sqlite"
        else (str(cfg.get("host") or "").strip() or "127.0.0.1")
    )
    tried_hosts: list[str] = [initial_host]

    try:
        return await asyncio.to_thread(fn, cfg, *args, host_override=None)
    except Exception as first_exc:
        last_exc: Exception = first_exc
        if db_type != "sqlite" and _is_connection_error(first_exc):
            if initial_host.lower() == "localhost":
                tried_hosts.append("127.0.0.1")
                try:
                    return await asyncio.to_thread(fn, cfg, *args, host_override="127.0.0.1")
                except Exception as exc2:
                    last_exc = exc2

            if (
                wsl_auto_detect
                and is_loopback_host(initial_host)
                and _is_connection_error(last_exc)
            ):
                wsl_ip = await get_wsl_ip(distro=wsl_distro)
                if wsl_ip and wsl_ip not in tried_hosts:
                    tried_hosts.append(wsl_ip)
                    try:
                        return await asyncio.to_thread(fn, cfg, *args, host_override=wsl_ip)
                    except Exception as exc3:
                        last_exc = exc3

        _attach_tried_hosts(last_exc, tried_hosts)
        raise last_exc
