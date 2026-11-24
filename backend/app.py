import asyncio
import base64
import binascii
import json
import time
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import httpx
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from backend.config import DB_PATH, LEGACY_CHATS_FILE, WEB_DIR
from backend.db import Database
from backend.pdf_utils import extract_uploaded_text
from backend.providers import PROVIDERS
from backend.providers.base import ProviderName
from backend.rdbms import (
    DEFAULT_PORTS,
    execute_query_sync,
    extract_sql_from_llm_output,
    format_db_error,
    introspect_schema_sync,
    normalize_db_type,
    run_with_wsl_fallback,
    test_connection_sync,
    validate_readonly_sql,
)
from backend.wsl import get_wsl_ip, list_wsl_distros, list_wsl_gguf_files

_UNSET = object()


class ChatRequest(BaseModel):
    provider: ProviderName | None = None
    model: str = ""
    messages: list[dict] = Field(default_factory=list)
    system: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    reasoning_effort: str | None = None
    sampling: dict | None = None
    custom_json: dict | None = None
    disable_reasoning_parsing: bool = False
    connector_id: str | None = None
    stream: bool = True


class ConnectorUpsertRequest(BaseModel):
    id: str | None = None
    name: str = "Database"
    db_type: str = "postgresql"
    host: str = "127.0.0.1"
    port: int | None = None
    database_name: str = ""
    username: str = ""
    password: str | None = None
    clear_password: bool = False
    ssl_mode: str = ""
    extra_params: str = ""
    is_active: bool | None = None


class ConnectorTestRequest(BaseModel):
    id: str | None = None
    name: str = ""
    db_type: str = "postgresql"
    host: str = "127.0.0.1"
    port: int | None = None
    database_name: str = ""
    username: str = ""
    password: str | None = None
    ssl_mode: str = ""
    extra_params: str = ""


class ConnectorQueryRequest(BaseModel):
    sql: str
    max_rows: int = 200


class ModelActionRequest(BaseModel):
    model: str


class SettingsPatch(BaseModel):
    default_provider: Literal["ollama", "llamacpp"] | None = None
    ollama_url: str | None = None
    llamacpp_url: str | None = None
    api_key: str | None = None
    wsl_auto_detect: bool | None = None
    wsl_distro: str | None = None
    wsl_gguf_dir: str | None = None
    system_prompt: str | None = None
    db_context_prompt: str | None = None
    db_non_sql_fallback: bool | None = None
    temperature: float | None = None
    max_tokens: int | None = None


class UploadRequest(BaseModel):
    filename: str = "document.txt"
    data_b64: str = ""


def _provider_url_from_settings(settings: dict, provider: ProviderName) -> str:
    return settings["ollama_url"] if provider == "ollama" else settings["llamacpp_url"]


def create_app(
    db_path: Path | None = None,
    legacy_json_path: Path | None | object = _UNSET,
    http_client_factory: Callable[[], httpx.AsyncClient] | None = None,
    cli_overrides: dict | None = None,
) -> FastAPI:
    effective_db_path = db_path or DB_PATH
    resolved_legacy = LEGACY_CHATS_FILE if legacy_json_path is _UNSET else legacy_json_path

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        db = Database(db_path=effective_db_path, legacy_json_path=resolved_legacy)
        await db.init()

        if cli_overrides:
            clean_overrides = {k: v for k, v in cli_overrides.items() if v is not None}
            if clean_overrides:
                await db.update_settings(clean_overrides)

        client = (
            http_client_factory()
            if http_client_factory is not None
            else httpx.AsyncClient(limits=httpx.Limits(max_connections=50, max_keepalive_connections=20))
        )
        app.state.db = db
        app.state.http = client
        app.state.providers = PROVIDERS
        try:
            yield
        finally:
            await client.aclose()

    app = FastAPI(title="ChatUI Local", version="2.0.0", lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    async def api_health(request: Request):
        settings = await request.app.state.db.get_settings()
        return {
            "status": "ok",
            "app": "ChatUI Local",
            "backend": "fastapi",
            "default_provider": settings["default_provider"],
            "ollama_url": settings["ollama_url"],
            "llamacpp_url": settings["llamacpp_url"],
        }

    @app.get("/api/providers")
    async def api_providers(request: Request):
        settings = await request.app.state.db.get_settings()
        client = request.app.state.http
        wsl_auto = settings["wsl_auto_detect"]
        wsl_distro = settings["wsl_distro"]
        api_key = settings["api_key"]

        ollama_status, llamacpp_status = await asyncio.gather(
            PROVIDERS["ollama"].check_status(client, settings["ollama_url"], wsl_auto, wsl_distro, api_key=api_key),
            PROVIDERS["llamacpp"].check_status(client, settings["llamacpp_url"], wsl_auto, wsl_distro, api_key=api_key),
        )
        return {
            "default_provider": settings["default_provider"],
            "providers": {
                "ollama": ollama_status.model_dump(),
                "llamacpp": llamacpp_status.model_dump(),
            },
        }

    @app.get("/api/models")
    async def api_models(request: Request, provider: ProviderName | None = Query(default=None)):
        settings = await request.app.state.db.get_settings()
        selected: ProviderName = provider or settings["default_provider"]
        adapter = PROVIDERS[selected]
        base_url = _provider_url_from_settings(settings, selected)
        res = await adapter.list_models(
            request.app.state.http,
            base_url,
            settings["wsl_auto_detect"],
            settings["wsl_distro"],
            api_key=settings["api_key"],
        )
        return res.model_dump()

    @app.get("/api/tags")
    async def api_tags(request: Request):
        settings = await request.app.state.db.get_settings()
        res = await PROVIDERS["ollama"].list_models(
            request.app.state.http,
            settings["ollama_url"],
            settings["wsl_auto_detect"],
            settings["wsl_distro"],
            api_key=settings["api_key"],
        )
        return {
            "provider": "ollama",
            "online": res.online,
            "url": res.resolved_url,
            "via_wsl_ip": res.via_wsl_ip,
            "error": res.error,
            "hint": res.hint,
            "models": [m.model_dump() for m in res.models],
        }

    @app.post("/api/providers/llamacpp/load")
    async def api_llamacpp_load(request: Request, body: ModelActionRequest):
        settings = await request.app.state.db.get_settings()
        try:
            res = await PROVIDERS["llamacpp"].load_model(
                request.app.state.http,
                settings["llamacpp_url"],
                body.model,
                settings["wsl_auto_detect"],
                settings["wsl_distro"],
                api_key=settings["api_key"],
            )
        except Exception as exc:
            return JSONResponse({"ok": False, "error": str(exc)}, status_code=502)
        if not res.get("ok"):
            return JSONResponse(res, status_code=res.get("status_code", 502))
        return res

    @app.post("/api/providers/llamacpp/unload")
    async def api_llamacpp_unload(request: Request, body: ModelActionRequest):
        settings = await request.app.state.db.get_settings()
        try:
            res = await PROVIDERS["llamacpp"].unload_model(
                request.app.state.http,
                settings["llamacpp_url"],
                body.model,
                settings["wsl_auto_detect"],
                settings["wsl_distro"],
                api_key=settings["api_key"],
            )
        except Exception as exc:
            return JSONResponse({"ok": False, "error": str(exc)}, status_code=502)
        if not res.get("ok"):
            return JSONResponse(res, status_code=res.get("status_code", 502))
        return res

    @app.post("/api/chat")
    async def api_chat(request: Request, body: ChatRequest):
        db = request.app.state.db
        settings = await db.get_settings()
        selected: ProviderName = body.provider or settings["default_provider"]
        adapter = PROVIDERS[selected]
        base_url = _provider_url_from_settings(settings, selected)
        system_prompt = body.system if body.system is not None else settings["system_prompt"]
        temperature = body.temperature if body.temperature is not None else settings["temperature"]
        max_tokens = body.max_tokens if body.max_tokens is not None else settings["max_tokens"]

        def _sse(payload: dict) -> str:
            return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

        async def _collect_completion(
            messages: list[dict],
            sys_prompt: str,
            temp: float = 0.0,
            tok_limit: int = 1024,
        ) -> tuple[str, dict | None]:
            stream_iter = adapter.stream_chat(
                client=request.app.state.http,
                base_url=base_url,
                model=body.model,
                messages=messages,
                system_prompt=sys_prompt,
                temperature=temp,
                max_tokens=tok_limit,
                reasoning_effort=None,
                sampling=None,
                custom_json=None,
                disable_reasoning_parsing=True,
                wsl_auto_detect=settings["wsl_auto_detect"],
                wsl_distro=settings["wsl_distro"],
                api_key=settings["api_key"],
            )
            parts: list[str] = []
            try:
                async for ev in stream_iter:
                    if await request.is_disconnected():
                        return "", {"type": "error", "error": "Client disconnected."}
                    etype = ev.get("type")
                    if etype == "delta":
                        if ev.get("content"):
                            parts.append(str(ev["content"]))
                    elif etype == "error":
                        return "", ev
            finally:
                aclose = getattr(stream_iter, "aclose", None)
                if callable(aclose):
                    await aclose()
            return "".join(parts).strip(), None

        async def _stream_connector_chat():
            cid = str(body.connector_id or "").strip()
            connector = await db.get_connector(cid, include_secret=True)
            if not connector:
                yield _sse({"type": "error", "error": f"Database connector '{cid}' not found."})
                return

            cname = connector["name"]
            db_type = normalize_db_type(connector["db_type"])
            yield _sse({
                "type": "db_status",
                "stage": "schema",
                "message": f"Inspecting schema on {cname} ({db_type})...",
                "connector_id": cid,
                "connector_name": cname,
            })

            schema_text = str(connector.get("schema_cache") or "").strip()
            if not schema_text:
                try:
                    schema_info = await run_with_wsl_fallback(
                        connector,
                        settings["wsl_auto_detect"],
                        settings["wsl_distro"],
                        introspect_schema_sync,
                    )
                    schema_text = schema_info["schema_text"]
                    await db.update_connector_schema_cache(cid, schema_text, "ok")
                except Exception as exc:
                    err_info = format_db_error(exc, connector, getattr(exc, "_tried_hosts", None))
                    yield _sse({
                        "type": "error",
                        "error": f"Failed to inspect schema for '{cname}': {err_info['error']}",
                        "hint": err_info.get("hint") or "",
                    })
                    return

            db_context_prompt = str(settings.get("db_context_prompt") or "").strip()
            allow_non_sql_fallback = bool(settings.get("db_non_sql_fallback", True))
            ctx_section = f"\n\nADDITIONAL DATABASE CONTEXT & RULES:\n{db_context_prompt}" if db_context_prompt else ""

            sql_system_prompt = (
                f"You are a strict read-only SQL generator for a {db_type} database named \"{cname}\".\n"
                "Given the database schema below, write ONE single read-only SQL query to answer the user's latest question.\n"
                "Rules:\n"
                "- Use ONLY tables and columns declared in SCHEMA.\n"
                f"- Write valid {db_type} SQL syntax.\n"
                "- Output ONLY the SQL query (inside a ```sql ... ``` code block or raw SQL). Do NOT add explanations, greetings, or phrases like 'Here is...'.\n"
                "- Only read-only queries (SELECT, WITH, SHOW, DESCRIBE, EXPLAIN, PRAGMA). Never write INSERT, UPDATE, DELETE, DROP, ALTER, or CREATE.\n"
                "- Unless computing an aggregate/count, include a reasonable LIMIT (e.g. LIMIT 50)."
                f"{ctx_section}\n\n"
                f"SCHEMA:\n{schema_text}"
            )
            recent_msgs = body.messages[-6:] if body.messages else []

            yield _sse({
                "type": "db_status",
                "stage": "generating_sql",
                "message": f"Generating SQL for {cname}...",
                "connector_id": cid,
                "connector_name": cname,
            })

            sql_raw_text, sql_err_ev = await _collect_completion(
                messages=recent_msgs,
                sys_prompt=sql_system_prompt,
                temp=0.0,
                tok_limit=min(int(max_tokens or 1024), 1024),
            )
            if sql_err_ev:
                yield _sse(sql_err_ev)
                return

            extracted_sql = extract_sql_from_llm_output(sql_raw_text)
            if not extracted_sql:
                if allow_non_sql_fallback:
                    yield _sse({
                        "type": "db_status",
                        "stage": "answering",
                        "message": f"Answering using {cname} schema context...",
                        "connector_id": cid,
                        "connector_name": cname,
                    })
                    fallback_sys = (
                        f"{system_prompt}{ctx_section}\n\n"
                        f"ACTIVE DATABASE SCHEMA ({cname}, {db_type}):\n{schema_text}\n\n"
                        "No SQL query was executed for this message. Answer clearly using the schema and context above when relevant, and do not invent query results."
                    )
                    fallback_stream = adapter.stream_chat(
                        client=request.app.state.http,
                        base_url=base_url,
                        model=body.model,
                        messages=body.messages,
                        system_prompt=fallback_sys,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        reasoning_effort=body.reasoning_effort,
                        sampling=body.sampling,
                        custom_json=body.custom_json,
                        disable_reasoning_parsing=body.disable_reasoning_parsing,
                        wsl_auto_detect=settings["wsl_auto_detect"],
                        wsl_distro=settings["wsl_distro"],
                        api_key=settings["api_key"],
                    )
                    try:
                        async for event in fallback_stream:
                            if await request.is_disconnected():
                                break
                            yield _sse(event)
                    finally:
                        aclose = getattr(fallback_stream, "aclose", None)
                        if callable(aclose):
                            await aclose()
                    return

                yield _sse({
                    "type": "delta",
                    "content": "No SQL query was generated for this message. You can enable **Fallback to schema-aware chat** or add **Database Context & SQL Instructions** under **Settings → Connector**.",
                    "thinking": "",
                })
                yield _sse({"type": "done", "stats": {}})
                return

            ok_sql, clean_sql, invalid_reason = validate_readonly_sql(extracted_sql)
            if not ok_sql:
                yield _sse({
                    "type": "db_sql",
                    "sql": extracted_sql,
                    "connector_id": cid,
                    "connector_name": cname,
                    "attempt": 1,
                    "blocked": True,
                })
                yield _sse({
                    "type": "db_result",
                    "sql": extracted_sql,
                    "columns": [],
                    "rows": [],
                    "row_count": 0,
                    "truncated": False,
                    "elapsed_ms": 0.0,
                    "error": invalid_reason,
                    "connector_id": cid,
                    "connector_name": cname,
                })
                yield _sse({
                    "type": "delta",
                    "content": f"I could not run a database query safely: {invalid_reason}",
                    "thinking": "",
                })
                yield _sse({"type": "done", "stats": {}})
                return

            yield _sse({
                "type": "db_sql",
                "sql": clean_sql,
                "connector_id": cid,
                "connector_name": cname,
                "attempt": 1,
                "retried": False,
            })
            yield _sse({
                "type": "db_status",
                "stage": "executing",
                "message": f"Executing read-only SQL on {cname}...",
                "connector_id": cid,
                "connector_name": cname,
            })

            exec_res: dict | None = None
            retried = False
            try:
                exec_res = await run_with_wsl_fallback(
                    connector,
                    settings["wsl_auto_detect"],
                    settings["wsl_distro"],
                    execute_query_sync,
                    clean_sql,
                    200,
                )
            except Exception as first_err:
                retried = True
                yield _sse({
                    "type": "db_status",
                    "stage": "retry",
                    "message": f"Query failed ({first_err}); retrying SQL generation...",
                    "connector_id": cid,
                    "connector_name": cname,
                })
                retry_msgs = [
                    *recent_msgs,
                    {"role": "assistant", "content": f"```sql\n{clean_sql}\n```"},
                    {
                        "role": "user",
                        "content": (
                            f"That SQL query failed with error:\n{first_err}\n\n"
                            "Fix the SQL query using ONLY valid table and column names from SCHEMA. "
                            "Return ONLY the corrected single read-only SQL query."
                        ),
                    },
                ]
                retry_text, retry_err_ev = await _collect_completion(
                    messages=retry_msgs,
                    sys_prompt=sql_system_prompt,
                    temp=0.0,
                    tok_limit=min(int(max_tokens or 1024), 1024),
                )
                if retry_err_ev:
                    yield _sse(retry_err_ev)
                    return

                retry_extracted = extract_sql_from_llm_output(retry_text)
                ok_retry, retry_clean, retry_invalid = validate_readonly_sql(retry_extracted)
                if not ok_retry:
                    yield _sse({
                        "type": "db_sql",
                        "sql": retry_extracted,
                        "connector_id": cid,
                        "connector_name": cname,
                        "attempt": 2,
                        "retried": True,
                        "blocked": True,
                    })
                    yield _sse({
                        "type": "db_result",
                        "sql": retry_extracted,
                        "columns": [],
                        "rows": [],
                        "row_count": 0,
                        "truncated": False,
                        "elapsed_ms": 0.0,
                        "error": retry_invalid,
                        "connector_id": cid,
                        "connector_name": cname,
                        "retried": True,
                    })
                    yield _sse({
                        "type": "delta",
                        "content": f"The retry SQL query could not be validated safely: {retry_invalid}",
                        "thinking": "",
                    })
                    yield _sse({"type": "done", "stats": {}})
                    return

                yield _sse({
                    "type": "db_sql",
                    "sql": retry_clean,
                    "connector_id": cid,
                    "connector_name": cname,
                    "attempt": 2,
                    "retried": True,
                })
                try:
                    exec_res = await run_with_wsl_fallback(
                        connector,
                        settings["wsl_auto_detect"],
                        settings["wsl_distro"],
                        execute_query_sync,
                        retry_clean,
                        200,
                    )
                except Exception as second_err:
                    yield _sse({
                        "type": "db_result",
                        "sql": retry_clean,
                        "columns": [],
                        "rows": [],
                        "row_count": 0,
                        "truncated": False,
                        "elapsed_ms": 0.0,
                        "error": str(second_err),
                        "connector_id": cid,
                        "connector_name": cname,
                        "retried": True,
                    })
                    yield _sse({
                        "type": "delta",
                        "content": f"Database query failed after retry: `{second_err}`",
                        "thinking": "",
                    })
                    yield _sse({"type": "done", "stats": {}})
                    return

            assert exec_res is not None
            yield _sse({
                "type": "db_result",
                "sql": exec_res["sql"],
                "columns": exec_res["columns"],
                "rows": exec_res["rows"],
                "row_count": exec_res["row_count"],
                "truncated": exec_res["truncated"],
                "elapsed_ms": exec_res["elapsed_ms"],
                "connector_id": cid,
                "connector_name": cname,
                "retried": retried,
            })
            yield _sse({
                "type": "db_status",
                "stage": "answering",
                "message": f"Synthesizing answer from {exec_res['row_count']} row(s)...",
                "connector_id": cid,
                "connector_name": cname,
            })

            preview_rows = exec_res["rows"][:50]
            rows_json = json.dumps(preview_rows, ensure_ascii=False)
            if len(rows_json) > 6000:
                rows_json = rows_json[:6000] + "...(truncated)"

            answer_sys_prompt = (
                f"{system_prompt}\n\n"
                f"You executed a read-only SQL query against the \"{cname}\" ({db_type}) database to answer the user's question.\n"
                "Answer clearly and concisely based ONLY on the query results below. Do not invent rows or numbers.\n\n"
                f"Executed SQL:\n```sql\n{exec_res['sql']}\n```\n\n"
                f"Returned {exec_res['row_count']} row(s) in {exec_res['elapsed_ms']} ms"
                + (" (output truncated)" if exec_res["truncated"] else "")
                + f".\nColumns: {json.dumps(exec_res['columns'], ensure_ascii=False)}\n"
                f"Rows JSON: {rows_json}"
            )

            answer_stream = adapter.stream_chat(
                client=request.app.state.http,
                base_url=base_url,
                model=body.model,
                messages=body.messages,
                system_prompt=answer_sys_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                reasoning_effort=body.reasoning_effort,
                sampling=body.sampling,
                custom_json=body.custom_json,
                disable_reasoning_parsing=body.disable_reasoning_parsing,
                wsl_auto_detect=settings["wsl_auto_detect"],
                wsl_distro=settings["wsl_distro"],
                api_key=settings["api_key"],
            )
            try:
                async for event in answer_stream:
                    if await request.is_disconnected():
                        break
                    yield _sse(event)
            finally:
                aclose = getattr(answer_stream, "aclose", None)
                if callable(aclose):
                    await aclose()

        async def event_stream():
            if not body.model.strip():
                err_event = {
                    "type": "error",
                    "error": f"No model selected for {adapter.label}. Choose an installed model first.",
                    "hint": adapter.offline_hint,
                }
                yield _sse(err_event)
                return

            if body.connector_id and str(body.connector_id).strip():
                async for chunk in _stream_connector_chat():
                    yield chunk
                return

            stream_iter = adapter.stream_chat(
                client=request.app.state.http,
                base_url=base_url,
                model=body.model,
                messages=body.messages,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                reasoning_effort=body.reasoning_effort,
                sampling=body.sampling,
                custom_json=body.custom_json,
                disable_reasoning_parsing=body.disable_reasoning_parsing,
                wsl_auto_detect=settings["wsl_auto_detect"],
                wsl_distro=settings["wsl_distro"],
                api_key=settings["api_key"],
            )
            try:
                async for event in stream_iter:
                    if await request.is_disconnected():
                        break
                    yield _sse(event)
            finally:
                aclose = getattr(stream_iter, "aclose", None)
                if callable(aclose):
                    await aclose()

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream; charset=utf-8",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @app.get("/api/chats")
    async def api_list_chats(request: Request):
        return await request.app.state.db.list_chats()

    @app.get("/api/chats/export")
    async def api_export_all_chats(request: Request):
        return {"chats": await request.app.state.db.export_all_chats()}

    @app.delete("/api/chats")
    async def api_delete_all_chats(request: Request):
        deleted_count = await request.app.state.db.delete_all_chats()
        return {"ok": True, "deleted_count": deleted_count}

    @app.get("/api/chats/{chat_id}")
    async def api_get_chat(request: Request, chat_id: str):
        chat = await request.app.state.db.get_chat(chat_id)
        if chat is not None:
            return chat
        settings = await request.app.state.db.get_settings()
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        return {
            "id": chat_id,
            "title": "New Chat",
            "provider": settings["default_provider"],
            "model": "",
            "created_at": now,
            "updated_at": now,
            "messages": [],
        }

    @app.post("/api/chats/{chat_id}")
    async def api_save_chat(request: Request, chat_id: str):
        try:
            body = await request.json()
            if not isinstance(body, dict):
                body = {}
        except Exception:
            body = {}
        chat = await request.app.state.db.upsert_chat(chat_id, body)
        return {"ok": True, "chat": chat}

    @app.delete("/api/chats/{chat_id}")
    async def api_delete_chat(request: Request, chat_id: str):
        deleted = await request.app.state.db.delete_chat(chat_id)
        if not deleted:
            return JSONResponse({"ok": False, "error": "Chat not found"}, status_code=404)
        return {"ok": True, "deleted": chat_id}

    @app.get("/api/settings")
    async def api_get_settings(request: Request):
        return await request.app.state.db.get_settings()

    async def _update_settings(request: Request, patch: SettingsPatch):
        try:
            return await request.app.state.db.update_settings(patch.model_dump(exclude_none=True))
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.put("/api/settings")
    async def api_put_settings(request: Request, patch: SettingsPatch):
        return await _update_settings(request, patch)

    @app.post("/api/settings")
    async def api_post_settings(request: Request, patch: SettingsPatch):
        return await _update_settings(request, patch)

    @app.get("/api/wsl/detect")
    async def api_wsl_detect(request: Request):
        settings = await request.app.state.db.get_settings()
        distro = settings["wsl_distro"]
        gguf_dir = settings["wsl_gguf_dir"]
        client = request.app.state.http
        wsl_auto = settings["wsl_auto_detect"]

        distros, wsl_ip, ollama_status, llamacpp_status, gguf_files = await asyncio.gather(
            list_wsl_distros(),
            get_wsl_ip(distro, force_refresh=True),
            PROVIDERS["ollama"].check_status(client, settings["ollama_url"], wsl_auto, distro, api_key=settings["api_key"]),
            PROVIDERS["llamacpp"].check_status(client, settings["llamacpp_url"], wsl_auto, distro, api_key=settings["api_key"]),
            list_wsl_gguf_files(gguf_dir, distro),
        )

        gguf_arg = gguf_dir or "~/models"
        return {
            "distros": distros,
            "selected_distro": distro,
            "wsl_ip": wsl_ip,
            "wsl_auto_detect": wsl_auto,
            "providers": {
                "ollama": ollama_status.model_dump(),
                "llamacpp": llamacpp_status.model_dump(),
            },
            "gguf_dir": gguf_dir,
            "gguf_files": gguf_files,
            "commands": {
                "ollama": "OLLAMA_HOST=0.0.0.0:11434 ollama serve",
                "llamacpp_router": f"llama-server --models-dir {gguf_arg} --host 0.0.0.0 --port 8080",
                "llamacpp_single": "llama-server -m ~/models/your-model.gguf --host 0.0.0.0 --port 8080",
            },
        }

    @app.post("/api/upload")
    async def api_upload(body: UploadRequest):
        try:
            file_bytes = base64.b64decode(body.data_b64 or "", validate=False)
        except (binascii.Error, ValueError) as exc:
            return JSONResponse({"ok": False, "error": f"Invalid base64 payload: {exc}"}, status_code=400)
        return extract_uploaded_text(body.filename or "document.txt", file_bytes)

    @app.get("/api/connectors")
    async def api_list_connectors(request: Request):
        items = await request.app.state.db.list_connectors(include_secret=False)
        active = next((c["id"] for c in items if c.get("is_active")), None)
        return {
            "connectors": items,
            "active_connector_id": active,
            "default_ports": DEFAULT_PORTS,
        }

    @app.post("/api/connectors/test")
    async def api_test_connector(request: Request, body: ConnectorTestRequest):
        db = request.app.state.db
        settings = await db.get_settings()
        cfg = body.model_dump()
        if body.id and not body.password:
            saved = await db.get_connector(body.id, include_secret=True)
            if saved:
                cfg["password"] = saved.get("password", "")
        try:
            res = await run_with_wsl_fallback(
                cfg,
                settings["wsl_auto_detect"],
                settings["wsl_distro"],
                test_connection_sync,
            )
            if body.id:
                await db.update_connector_test_status(body.id, "ok")
            return res
        except Exception as exc:
            err_info = format_db_error(exc, cfg, getattr(exc, "_tried_hosts", None))
            if body.id:
                await db.update_connector_test_status(body.id, f"error: {err_info['error'][:180]}")
            return JSONResponse({"ok": False, **err_info}, status_code=400)

    @app.post("/api/connectors/deactivate")
    async def api_deactivate_connectors(request: Request):
        await request.app.state.db.set_active_connector(None)
        return {"ok": True, "active_connector_id": None}

    @app.post("/api/connectors")
    async def api_create_connector(request: Request, body: ConnectorUpsertRequest):
        payload = body.model_dump()
        try:
            payload["db_type"] = normalize_db_type(payload.get("db_type"))
            if payload["db_type"] not in DEFAULT_PORTS:
                raise ValueError(f"Unsupported database type: {payload['db_type']}")
            saved = await request.app.state.db.upsert_connector(payload)
        except Exception as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {"ok": True, "connector": saved}

    @app.put("/api/connectors/{connector_id}")
    async def api_update_connector(request: Request, connector_id: str, body: ConnectorUpsertRequest):
        db = request.app.state.db
        existing = await db.get_connector(connector_id, include_secret=False)
        if not existing:
            raise HTTPException(status_code=404, detail="Connector not found")
        payload = body.model_dump()
        payload["id"] = connector_id
        try:
            payload["db_type"] = normalize_db_type(payload.get("db_type"))
            if payload["db_type"] not in DEFAULT_PORTS:
                raise ValueError(f"Unsupported database type: {payload['db_type']}")
            saved = await db.upsert_connector(payload)
        except Exception as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {"ok": True, "connector": saved}

    @app.delete("/api/connectors/{connector_id}")
    async def api_delete_connector(request: Request, connector_id: str):
        deleted = await request.app.state.db.delete_connector(connector_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Connector not found")
        return {"ok": True, "deleted": True}

    @app.post("/api/connectors/{connector_id}/activate")
    async def api_activate_connector(request: Request, connector_id: str):
        if connector_id in ("none", "off", "clear"):
            await request.app.state.db.set_active_connector(None)
            return {"ok": True, "active_connector_id": None}
        saved = await request.app.state.db.set_active_connector(connector_id)
        if not saved:
            raise HTTPException(status_code=404, detail="Connector not found")
        return {"ok": True, "active_connector_id": connector_id, "connector": saved}

    @app.get("/api/connectors/{connector_id}/schema")
    async def api_get_connector_schema(
        request: Request,
        connector_id: str,
        refresh: bool = Query(default=False),
    ):
        db = request.app.state.db
        conn = await db.get_connector(connector_id, include_secret=True)
        if not conn:
            raise HTTPException(status_code=404, detail="Connector not found")

        if not refresh and conn.get("schema_cache"):
            return {
                "ok": True,
                "connector_id": connector_id,
                "cached": True,
                "schema_text": conn["schema_cache"],
            }

        settings = await db.get_settings()
        try:
            info = await run_with_wsl_fallback(
                conn,
                settings["wsl_auto_detect"],
                settings["wsl_distro"],
                introspect_schema_sync,
            )
            updated = await db.update_connector_schema_cache(connector_id, info["schema_text"], "ok")
            return {
                "ok": True,
                "connector_id": connector_id,
                "cached": False,
                "connector": updated,
                **info,
            }
        except Exception as exc:
            err_info = format_db_error(exc, conn, getattr(exc, "_tried_hosts", None))
            await db.update_connector_test_status(connector_id, f"error: {err_info['error'][:180]}")
            return JSONResponse({"ok": False, **err_info}, status_code=400)

    @app.post("/api/connectors/{connector_id}/schema")
    async def api_refresh_connector_schema(request: Request, connector_id: str):
        return await api_get_connector_schema(request, connector_id, refresh=True)

    @app.post("/api/connectors/{connector_id}/query")
    async def api_query_connector(request: Request, connector_id: str, body: ConnectorQueryRequest):
        db = request.app.state.db
        conn = await db.get_connector(connector_id, include_secret=True)
        if not conn:
            raise HTTPException(status_code=404, detail="Connector not found")
        settings = await db.get_settings()
        try:
            res = await run_with_wsl_fallback(
                conn,
                settings["wsl_auto_detect"],
                settings["wsl_distro"],
                execute_query_sync,
                body.sql,
                body.max_rows,
            )
            return {"ok": True, "connector_id": connector_id, **res}
        except Exception as exc:
            err_info = format_db_error(exc, conn, getattr(exc, "_tried_hosts", None))
            return JSONResponse({"ok": False, **err_info}, status_code=400)

    @app.get("/", include_in_schema=False)
    @app.get("/{full_path:path}", include_in_schema=False)
    async def serve_spa(full_path: str = ""):
        clean_path = full_path.lstrip("/")
        if clean_path == "api" or clean_path.startswith("api/"):
            return JSONResponse({"error": "Endpoint not found"}, status_code=404)

        web_root = WEB_DIR.resolve()
        if clean_path:
            candidate = (web_root / clean_path).resolve()
            if candidate.is_file() and candidate.is_relative_to(web_root):
                return FileResponse(candidate)

        index_file = web_root / "index.html"
        if index_file.is_file():
            return FileResponse(index_file, media_type="text/html; charset=utf-8")
        return JSONResponse({"error": "Frontend index.html not found"}, status_code=404)

    return app


app = create_app()
