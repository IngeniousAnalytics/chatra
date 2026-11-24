import base64
import json
from pathlib import Path

import httpx
from fastapi.testclient import TestClient

from backend.app import create_app


def _parse_sse_events(body_text: str) -> list[dict]:
    events = []
    for line in body_text.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload:
            events.append(json.loads(payload))
    return events


def _make_client(tmp_path: Path, handler, legacy_json_path: Path | None = None):
    db_path = tmp_path / "test_chatui.db"
    app = create_app(
        db_path=db_path,
        legacy_json_path=legacy_json_path,
        http_client_factory=lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    return TestClient(app)


def test_health_spa_and_api_404(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("offline", request=request)

    with _make_client(tmp_path, handler) as client:
        r_health = client.get("/api/health")
        assert r_health.status_code == 200
        h = r_health.json()
        assert h["status"] == "ok"
        assert h["backend"] == "fastapi"
        assert h["default_provider"] == "ollama"

        r_root = client.get("/")
        assert r_root.status_code == 200
        assert "Advisor Chat" in r_root.text
        assert "Chatra" in r_root.text
        assert "IngeniousAnalytics.com" in r_root.text

        for asset_path in ("/favicon.svg", "/chatra-logo.svg", "/ingenious-logo.svg"):
            r_asset = client.get(asset_path)
            assert r_asset.status_code == 200
            assert "<svg" in r_asset.text

        r_png = client.get("/ingenious-logo.png")
        assert r_png.status_code == 200
        assert r_png.headers["content-type"].startswith("image/png")
        assert len(r_png.content) > 1000

        r_spa = client.get("/chat/961ff56d-2f0b-4c65-b130-e4d2135f8c75")
        assert r_spa.status_code == 200
        assert "Advisor Chat" in r_spa.text

        r_404 = client.get("/api/unknown")
        assert r_404.status_code == 404
        assert r_404.headers["content-type"].startswith("application/json")


def test_settings_persistence(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("offline", request=request)

    db_path = tmp_path / "settings.db"
    app1 = create_app(
        db_path=db_path,
        legacy_json_path=None,
        http_client_factory=lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    with TestClient(app1) as client:
        r0 = client.get("/api/settings")
        assert r0.status_code == 200
        assert r0.json()["default_provider"] == "ollama"

        r_put = client.put(
            "/api/settings",
            json={
                "default_provider": "llamacpp",
                "ollama_url": "http://127.0.0.1:11435/",
                "llamacpp_url": "http://127.0.0.1:8081/",
                "wsl_auto_detect": False,
                "wsl_distro": "Ubuntu-24.04",
                "wsl_gguf_dir": "~/models",
                "temperature": 0.7,
                "max_tokens": 512,
                "system_prompt": "Custom Persona",
            },
        )
        assert r_put.status_code == 200
        saved = r_put.json()
        assert saved["default_provider"] == "llamacpp"
        assert saved["ollama_url"] == "http://127.0.0.1:11435"
        assert saved["llamacpp_url"] == "http://127.0.0.1:8081"
        assert saved["wsl_auto_detect"] is False
        assert saved["wsl_distro"] == "Ubuntu-24.04"
        assert saved["wsl_gguf_dir"] == "~/models"
        assert saved["temperature"] == 0.7
        assert saved["max_tokens"] == 512
        assert saved["system_prompt"] == "Custom Persona"

    # Re-instantiate app on the same SQLite DB to verify persistence across restarts
    app2 = create_app(
        db_path=db_path,
        legacy_json_path=None,
        http_client_factory=lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    with TestClient(app2) as client2:
        r_get = client2.get("/api/settings").json()
        assert r_get["default_provider"] == "llamacpp"
        assert r_get["llamacpp_url"] == "http://127.0.0.1:8081"
        assert r_get["system_prompt"] == "Custom Persona"


def test_ollama_models_loaded_detection_and_streaming(tmp_path):
    captured_payload = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(
                200,
                json={
                    "models": [
                        {
                            "name": "qwen2.5:3b",
                            "size": 1929912320,
                            "details": {
                                "parameter_size": "3B",
                                "quantization_level": "Q4_K_M",
                                "family": "qwen2",
                            },
                        },
                        {
                            "name": "deepseek-r1:7b",
                            "size": 4683072000,
                            "details": {
                                "parameter_size": "7B",
                                "quantization_level": "Q4_K_M",
                                "family": "qwen2",
                            },
                        },
                    ]
                },
            )
        if request.url.path == "/api/ps":
            return httpx.Response(200, json={"models": [{"name": "qwen2.5:3b"}]})
        if request.url.path == "/api/chat":
            captured_payload.update(json.loads(request.content.decode("utf-8")))
            ndjson = (
                json.dumps({"message": {"role": "assistant", "thinking": "Let me think..."}, "done": False})
                + "\n"
                + json.dumps({"message": {"role": "assistant", "content": "Hello "}, "done": False})
                + "\n"
                + json.dumps({"message": {"role": "assistant", "content": "world!"}, "done": False})
                + "\n"
                + json.dumps(
                    {
                        "message": {"role": "assistant", "content": ""},
                        "done": True,
                        "eval_count": 20,
                        "prompt_eval_count": 10,
                        "eval_duration": 500_000_000,
                        "prompt_eval_duration": 2_000_000_000,
                    }
                )
                + "\n"
            ).encode("utf-8")
            return httpx.Response(200, content=ndjson, headers={"Content-Type": "application/x-ndjson"})
        return httpx.Response(404)

    with _make_client(tmp_path, handler) as client:
        r_models = client.get("/api/models?provider=ollama")
        assert r_models.status_code == 200
        mdata = r_models.json()
        assert mdata["online"] is True
        assert mdata["provider"] == "ollama"
        assert len(mdata["models"]) == 2
        by_id = {m["id"]: m for m in mdata["models"]}
        assert by_id["qwen2.5:3b"]["loaded"] is True
        assert by_id["deepseek-r1:7b"]["loaded"] is False
        assert by_id["qwen2.5:3b"]["parameter_size"] == "3B"

        with client.stream(
            "POST",
            "/api/chat",
            json={
                "provider": "ollama",
                "model": "deepseek-r1:7b",
                "reasoning_effort": "high",
                "messages": [{"role": "user", "content": "Hi"}],
            },
        ) as stream_resp:
            assert stream_resp.status_code == 200
            body = stream_resp.read().decode("utf-8")

        assert captured_payload["think"] is True
        events = _parse_sse_events(body)
        assert any(e["type"] == "delta" and e.get("thinking") == "Let me think..." for e in events)
        content = "".join(e.get("content", "") for e in events if e["type"] == "delta")
        assert content == "Hello world!"
        done_ev = [e for e in events if e["type"] == "done"][-1]
        assert done_ev["stats"]["provider"] == "ollama"
        assert done_ev["stats"]["tokens_per_sec"] == 40.0
        assert done_ev["stats"]["completion_tokens"] == 20
        assert done_ev["stats"]["prompt_seconds"] == 2.0
        assert done_ev["stats"]["prompt_tokens_per_sec"] == 5.0
        assert done_ev["stats"]["completion_seconds"] == 0.5


def test_llamacpp_router_single_html_guard_load_and_streaming(tmp_path):
    mode = {"case": "router"}
    captured_payload = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            if mode["case"] == "router":
                return httpx.Response(
                    200,
                    json={
                        "data": [
                            {"id": "qwen2.5-coder-7b.gguf", "meta": {"n_params": 7_600_000_000}},
                            {"id": "llama-3.2-3b.gguf"},
                        ],
                        "models": [
                            {"name": "qwen2.5-coder-7b.gguf", "status": {"value": "loaded"}},
                            {"name": "llama-3.2-3b.gguf", "status": {"value": "unloaded"}},
                        ],
                    },
                )
            if mode["case"] == "single":
                return httpx.Response(
                    200,
                    json={"data": [{"id": "/home/user/models/mistral-7b-q4.gguf", "owned_by": "llamacpp"}]},
                )
            if mode["case"] == "html":
                return httpx.Response(200, text="<html><body>Not llama.cpp</body></html>", headers={"Content-Type": "text/html"})

        if request.url.path == "/props":
            return httpx.Response(200, json={"default_generation_settings": {"n_ctx": 32768}})

        if request.url.path == "/models/load":
            body = json.loads(request.content.decode("utf-8"))
            return httpx.Response(200, json={"success": True, "model": body["model"]})

        if request.url.path == "/v1/chat/completions":
            captured_payload.update(json.loads(request.content.decode("utf-8")))
            sse = (
                'data: {"choices":[{"delta":{"reasoning_content":"step 1"}}]}\n\n'
                'data: {"choices":[{"delta":{"content":"Answer: "}}]}\n\n'
                'data: {"choices":[{"delta":{"content":"42"}}]}\n\n'
                'data: {"choices":[],"usage":{"prompt_tokens":15,"completion_tokens":12},"timings":{"predicted_per_second":64.25,"predicted_n":12,"prompt_n":15,"prompt_ms":1500.0,"prompt_per_second":10.0,"predicted_ms":300.0}}\n\n'
                "data: [DONE]\n\n"
            ).encode("utf-8")
            return httpx.Response(200, content=sse, headers={"Content-Type": "text/event-stream"})
        return httpx.Response(404)

    with _make_client(tmp_path, handler) as client:
        # 1. Router mode multi-model
        r_router = client.get("/api/models?provider=llamacpp").json()
        assert r_router["online"] is True
        assert r_router["mode"] == "router"
        assert len(r_router["models"]) == 2
        by_id = {m["id"]: m for m in r_router["models"]}
        assert by_id["qwen2.5-coder-7b.gguf"]["loaded"] is True
        assert by_id["qwen2.5-coder-7b.gguf"]["parameter_size"] == "7.6B"
        assert by_id["qwen2.5-coder-7b.gguf"]["context_length"] == 32768
        assert by_id["llama-3.2-3b.gguf"]["loaded"] is False
        assert by_id["llama-3.2-3b.gguf"]["status"] == "unloaded"

        # 2. Load endpoint
        r_load = client.post("/api/providers/llamacpp/load", json={"model": "llama-3.2-3b.gguf"})
        assert r_load.status_code == 200
        assert r_load.json()["ok"] is True

        # 3. Streaming chat with reasoning_content, empty choices chunk, timings & [DONE]
        with client.stream(
            "POST",
            "/api/chat",
            json={
                "provider": "llamacpp",
                "model": "qwen2.5-coder-7b.gguf",
                "reasoning_effort": "medium",
                "messages": [{"role": "user", "content": "What is the answer?"}],
            },
        ) as resp:
            assert resp.status_code == 200
            events = _parse_sse_events(resp.read().decode("utf-8"))

        assert captured_payload["reasoning_effort"] == "medium"
        assert any(e["type"] == "delta" and e.get("thinking") == "step 1" for e in events)
        text = "".join(e.get("content", "") for e in events if e["type"] == "delta")
        assert text == "Answer: 42"
        done_ev = [e for e in events if e["type"] == "done"][-1]
        assert done_ev["stats"]["provider"] == "llamacpp"
        assert done_ev["stats"]["tokens_per_sec"] == 64.2 or done_ev["stats"]["tokens_per_sec"] == 64.3
        assert done_ev["stats"]["completion_tokens"] == 12
        assert done_ev["stats"]["prompt_tokens"] == 15
        assert done_ev["stats"]["prompt_seconds"] == 1.5
        assert done_ev["stats"]["prompt_tokens_per_sec"] == 10.0
        assert done_ev["stats"]["completion_seconds"] == 0.3

        # 4. Single-model .gguf path cleaning
        mode["case"] = "single"
        r_single = client.get("/api/models?provider=llamacpp").json()
        assert r_single["online"] is True
        assert r_single["mode"] == "single-model"
        assert r_single["models"][0]["id"] == "/home/user/models/mistral-7b-q4.gguf"
        assert r_single["models"][0]["name"] == "mistral-7b-q4.gguf"

        # 5. HTML guard on port clash
        mode["case"] = "html"
        r_html = client.get("/api/models?provider=llamacpp").json()
        assert r_html["online"] is False
        assert r_html["models"] == []
        assert "port conflict" in (r_html["error"] or "").lower() or "expected json" in (r_html["error"] or "").lower()


def test_offline_handling_no_fake_models_and_error_sse(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Connection refused", request=request)

    with _make_client(tmp_path, handler) as client:
        for prov in ("ollama", "llamacpp"):
            m_res = client.get(f"/api/models?provider={prov}").json()
            assert m_res["online"] is False
            assert m_res["models"] == []
            assert m_res["hint"]

            with client.stream(
                "POST",
                "/api/chat",
                json={
                    "provider": prov,
                    "model": "some-model",
                    "messages": [{"role": "user", "content": "Hello"}],
                },
            ) as resp:
                assert resp.status_code == 200
                events = _parse_sse_events(resp.read().decode("utf-8"))
                assert len(events) == 1
                assert events[0]["type"] == "error"
                assert events[0]["hint"]


def test_wsl_ip_fallback(tmp_path, monkeypatch):
    async def fake_wsl_ip(distro: str = "", force_refresh: bool = False):
        return "172.24.16.5"

    monkeypatch.setattr("backend.providers.base.get_wsl_ip", fake_wsl_ip)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "127.0.0.1":
            raise httpx.ConnectError("localhost forwarding disabled", request=request)
        if request.url.host == "172.24.16.5" and request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "qwen2.5:7b"}]})
        if request.url.host == "172.24.16.5" and request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "phi-4.gguf"}]})
        return httpx.Response(404)

    with _make_client(tmp_path, handler) as client:
        r_ollama = client.get("/api/models?provider=ollama").json()
        assert r_ollama["online"] is True
        assert r_ollama["via_wsl_ip"] is True
        assert r_ollama["resolved_url"] == "http://172.24.16.5:11434"
        assert r_ollama["models"][0]["id"] == "qwen2.5:7b"

        r_llama = client.get("/api/models?provider=llamacpp").json()
        assert r_llama["online"] is True
        assert r_llama["via_wsl_ip"] is True
        assert r_llama["resolved_url"] == "http://172.24.16.5:8080"
        assert r_llama["models"][0]["id"] == "phi-4.gguf"


def test_concurrent_providers_endpoint(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "llama3.2:3b"}]})
        if request.url.path == "/api/ps":
            return httpx.Response(200, json={"models": []})
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "gemma-2.gguf"}]})
        return httpx.Response(404)

    with _make_client(tmp_path, handler) as client:
        r = client.get("/api/providers")
        assert r.status_code == 200
        data = r.json()
        assert data["providers"]["ollama"]["online"] is True
        assert data["providers"]["ollama"]["model_count"] == 1
        assert data["providers"]["llamacpp"]["online"] is True
        assert data["providers"]["llamacpp"]["model_count"] == 1


def test_chat_crud_and_legacy_migration(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("offline", request=request)

    legacy_file = tmp_path / "chats.json"
    legacy_file.write_text(
        json.dumps(
            {
                "legacy-1": {
                    "id": "legacy-1",
                    "title": "Legacy Chat",
                    "model": "qwen2.5:3b",
                    "created_at": "2026-01-01 10:00:00",
                    "updated_at": "2026-01-01 10:05:00",
                    "messages": [{"role": "user", "content": "Old message"}],
                }
            }
        ),
        encoding="utf-8",
    )

    with _make_client(tmp_path, handler, legacy_json_path=legacy_file) as client:
        # Legacy migrated chat exists
        r_list = client.get("/api/chats").json()
        assert len(r_list) == 1
        assert r_list[0]["id"] == "legacy-1"
        assert r_list[0]["provider"] == "ollama"
        assert r_list[0]["message_count"] == 1

        # Create/update a llama.cpp chat
        r_save = client.post(
            "/api/chats/new-chat-1",
            json={
                "title": "Llama Test",
                "provider": "llamacpp",
                "model": "mistral-7b.gguf",
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there"},
                ],
            },
        )
        assert r_save.status_code == 200
        assert r_save.json()["chat"]["provider"] == "llamacpp"

        r_get = client.get("/api/chats/new-chat-1").json()
        assert r_get["title"] == "Llama Test"
        assert r_get["model"] == "mistral-7b.gguf"
        assert len(r_get["messages"]) == 2

        # Delete chat
        r_del = client.delete("/api/chats/new-chat-1")
        assert r_del.status_code == 200
        assert r_del.json()["deleted"] == "new-chat-1"

        r_del_missing = client.delete("/api/chats/new-chat-1")
        assert r_del_missing.status_code == 404


def test_document_upload(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("offline", request=request)

    with _make_client(tmp_path, handler) as client:
        payload_txt = base64.b64encode(b"Hello FastAPI upload test!").decode("ascii")
        r = client.post("/api/upload", json={"filename": "notes.txt", "data_b64": payload_txt})
        assert r.status_code == 200
        res = r.json()
        assert res["ok"] is True
        assert res["filename"] == "notes.txt"
        assert res["text"] == "Hello FastAPI upload test!"


def test_api_key_sampling_custom_json_and_reasoning_format(tmp_path):
    captured = {"ollama_auth": None, "ollama_body": None, "llama_auth": None, "llama_body": None}

    def handler(request: httpx.Request) -> httpx.Response:
        auth = request.headers.get("authorization")
        if request.url.path == "/api/tags":
            captured["ollama_auth"] = auth
            return httpx.Response(200, json={"models": [{"name": "qwen2.5:3b"}]})
        if request.url.path == "/api/ps":
            return httpx.Response(200, json={"models": []})
        if request.url.path == "/api/chat":
            captured["ollama_auth"] = auth
            captured["ollama_body"] = json.loads(request.content.decode("utf-8"))
            ndjson = (json.dumps({"message": {"role": "assistant", "content": "ok"}, "done": True}) + "\n").encode()
            return httpx.Response(200, content=ndjson, headers={"Content-Type": "application/x-ndjson"})
        if request.url.path == "/v1/models":
            captured["llama_auth"] = auth
            return httpx.Response(200, json={"data": [{"id": "qwen.gguf"}]})
        if request.url.path == "/v1/chat/completions":
            captured["llama_auth"] = auth
            captured["llama_body"] = json.loads(request.content.decode("utf-8"))
            sse = b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\ndata: [DONE]\n\n'
            return httpx.Response(200, content=sse, headers={"Content-Type": "text/event-stream"})
        return httpx.Response(404)

    with _make_client(tmp_path, handler) as client:
        r_set = client.put("/api/settings", json={"api_key": "secret-key-123"})
        assert r_set.status_code == 200
        assert r_set.json()["api_key"] == "secret-key-123"

        sampling = {
            "temperature": 0.6,
            "top_k": 25,
            "top_p": 0.9,
            "min_p": 0.08,
            "repeat_penalty": 1.15,
            "max_tokens": 256,
            "dry_multiplier": 0.8,
            "xtc_probability": 0.2,
            "samplers": "top_k,top_p,temperature",
        }
        custom_json = {"seed": 42}

        with client.stream(
            "POST",
            "/api/chat",
            json={
                "provider": "ollama",
                "model": "qwen2.5:3b",
                "messages": [{"role": "user", "content": "hi"}],
                "sampling": sampling,
                "custom_json": {"seed": 42, "options": {"num_ctx": 8192}},
            },
        ) as r_ol:
            assert r_ol.status_code == 200
            r_ol.read()

        assert captured["ollama_auth"] == "Bearer secret-key-123"
        assert captured["ollama_body"]["seed"] == 42
        assert captured["ollama_body"]["options"]["temperature"] == 0.6
        assert captured["ollama_body"]["options"]["top_k"] == 25
        assert captured["ollama_body"]["options"]["num_predict"] == 256
        assert captured["ollama_body"]["options"]["num_ctx"] == 8192

        with client.stream(
            "POST",
            "/api/chat",
            json={
                "provider": "llamacpp",
                "model": "qwen.gguf",
                "messages": [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "hello", "reasoning_content": "chain"},
                    {"role": "user", "content": "next"},
                ],
                "sampling": sampling,
                "custom_json": custom_json,
                "disable_reasoning_parsing": True,
            },
        ) as r_ll:
            assert r_ll.status_code == 200
            r_ll.read()

        assert captured["llama_auth"] == "Bearer secret-key-123"
        assert captured["llama_body"]["temperature"] == 0.6
        assert captured["llama_body"]["top_k"] == 25
        assert captured["llama_body"]["dry_multiplier"] == 0.8
        assert captured["llama_body"]["xtc_probability"] == 0.2
        assert captured["llama_body"]["samplers"] == ["top_k", "top_p", "temperature"]
        assert captured["llama_body"]["reasoning_format"] == "none"
        assert captured["llama_body"]["seed"] == 42
        assert captured["llama_body"]["messages"][2]["reasoning_content"] == "chain"


def test_bulk_chat_export_and_delete_all(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("offline", request=request)

    with _make_client(tmp_path, handler) as client:
        client.post("/api/chats/c1", json={"title": "First", "messages": [{"role": "user", "content": "1"}]})
        client.post("/api/chats/c2", json={"title": "Second", "messages": [{"role": "user", "content": "2"}]})

        r_exp = client.get("/api/chats/export")
        assert r_exp.status_code == 200
        exported = r_exp.json()["chats"]
        assert len(exported) == 2
        assert {c["id"] for c in exported} == {"c1", "c2"}
        assert all("messages" in c and len(c["messages"]) == 1 for c in exported)

        r_del_all = client.delete("/api/chats")
        assert r_del_all.status_code == 200
        assert r_del_all.json()["deleted_count"] == 2
        assert client.get("/api/chats").json() == []


def test_rdbms_connector_crud_masking_and_sqlite_query(tmp_path):
    import sqlite3

    sample_db = tmp_path / "sample_store.db"
    conn = sqlite3.connect(sample_db)
    conn.execute("CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT NOT NULL, country TEXT)")
    conn.execute(
        "CREATE TABLE orders (id INTEGER PRIMARY KEY, customer_id INTEGER REFERENCES customers(id), amount REAL, note TEXT)"
    )
    conn.execute("INSERT INTO customers VALUES (1, 'Alice', 'US'), (2, 'Bob', 'IN')")
    conn.execute("INSERT INTO orders VALUES (10, 1, 99.5, 'normal'), (11, 2, 150.0, 'DELETE')")
    conn.commit()
    conn.close()

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("offline", request=request)

    with _make_client(tmp_path, handler) as client:
        # 1. Save PostgreSQL connector with password and verify password masking
        r_pg = client.post(
            "/api/connectors",
            json={
                "name": "Prod Postgres",
                "db_type": "postgresql",
                "host": "127.0.0.1",
                "port": 5432,
                "database_name": "dvdrental",
                "username": "postgres",
                "password": "super-secret-password",
            },
        )
        assert r_pg.status_code == 200
        pg_conn = r_pg.json()["connector"]
        pg_id = pg_conn["id"]
        assert pg_conn["has_password"] is True
        assert "password" not in pg_conn

        # Updating without resending password preserves saved password
        r_pg_upd = client.put(
            f"/api/connectors/{pg_id}",
            json={
                "name": "Renamed Postgres",
                "db_type": "postgresql",
                "host": "127.0.0.1",
                "port": 5432,
                "database_name": "dvdrental",
                "username": "postgres",
            },
        )
        assert r_pg_upd.status_code == 200
        assert r_pg_upd.json()["connector"]["has_password"] is True
        assert "password" not in r_pg_upd.json()["connector"]

        # 2. Save SQLite connector, test, inspect schema, activate, query
        r_sq = client.post(
            "/api/connectors",
            json={
                "name": "Local Sample SQLite",
                "db_type": "sqlite",
                "database_name": str(sample_db),
            },
        )
        assert r_sq.status_code == 200
        sq_id = r_sq.json()["connector"]["id"]

        r_test = client.post(
            "/api/connectors/test",
            json={"id": sq_id, "db_type": "sqlite", "database_name": str(sample_db)},
        )
        assert r_test.status_code == 200
        tjson = r_test.json()
        assert tjson["ok"] is True
        assert tjson["table_count"] == 2

        r_act = client.post(f"/api/connectors/{sq_id}/activate")
        assert r_act.status_code == 200
        assert r_act.json()["active_connector_id"] == sq_id

        r_list = client.get("/api/connectors")
        assert r_list.status_code == 200
        assert r_list.json()["active_connector_id"] == sq_id
        assert len(r_list.json()["connectors"]) == 2

        r_schema = client.get(f"/api/connectors/{sq_id}/schema")
        assert r_schema.status_code == 200
        sjson = r_schema.json()
        assert sjson["ok"] is True
        assert sjson["table_count"] == 2
        assert "customers" in sjson["schema_text"]
        assert "orders" in sjson["schema_text"]

        # Read-only query containing string literal 'DELETE' MUST succeed
        r_ok_lit = client.post(
            f"/api/connectors/{sq_id}/query",
            json={"sql": "SELECT id, amount, note FROM orders WHERE note = 'DELETE'"},
        )
        assert r_ok_lit.status_code == 200
        qjson = r_ok_lit.json()
        assert qjson["ok"] is True
        assert qjson["columns"] == ["id", "amount", "note"]
        assert qjson["rows"] == [[11, 150.0, "DELETE"]]

        # Mutating / multi-statement queries MUST be blocked
        for bad_sql in (
            "DROP TABLE orders",
            "DELETE FROM orders WHERE id = 10",
            "INSERT INTO customers VALUES (99, 'Mallory', 'XX')",
            "SELECT 1; DROP TABLE orders",
            "PRAGMA writable_schema = ON",
        ):
            r_bad = client.post(f"/api/connectors/{sq_id}/query", json={"sql": bad_sql})
            assert r_bad.status_code == 400, bad_sql
            assert r_bad.json()["ok"] is False

        # Deactivate and delete
        r_deact = client.post("/api/connectors/deactivate")
        assert r_deact.status_code == 200
        assert r_deact.json()["active_connector_id"] is None

        r_del = client.delete(f"/api/connectors/{pg_id}")
        assert r_del.status_code == 200
        assert client.delete(f"/api/connectors/{pg_id}").status_code == 404


def test_rdbms_chat_qa_end_to_end_retry_and_write_block(tmp_path):
    import sqlite3

    dvd_db = tmp_path / "dvd.db"
    conn = sqlite3.connect(dvd_db)
    conn.execute("CREATE TABLE films (id INTEGER PRIMARY KEY, title TEXT, length INTEGER)")
    conn.executemany(
        "INSERT INTO films VALUES (?, ?, ?)",
        [(1, "Short Film", 65), (2, "Medium Epic", 142), (3, "Longest Saga", 210)],
    )
    conn.commit()
    conn.close()

    call_log: list[dict] = []
    mode = {"scenario": "retry_then_answer"}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/chat":
            body = json.loads(request.content.decode("utf-8"))
            call_log.append(body)
            idx = len(call_log)
            if mode["scenario"] == "retry_then_answer":
                if idx == 1:
                    reply = "```sql\nSELECT title, nonexistent_col FROM films\n```"
                elif idx == 2:
                    reply = "```sql\nSELECT title, length FROM films ORDER BY length DESC LIMIT 2\n```"
                else:
                    reply = "The longest films are Longest Saga (210) and Medium Epic (142)."
            else:
                reply = "```sql\nDELETE FROM films WHERE id = 1\n```"
            ndjson = (json.dumps({"message": {"role": "assistant", "content": reply}, "done": True}) + "\n").encode()
            return httpx.Response(200, content=ndjson, headers={"Content-Type": "application/x-ndjson"})
        return httpx.Response(404)

    with _make_client(tmp_path, handler) as client:
        r_conn = client.post(
            "/api/connectors",
            json={"name": "DVD DB", "db_type": "sqlite", "database_name": str(dvd_db), "is_active": True},
        )
        cid = r_conn.json()["connector"]["id"]

        # Scenario 1: Attempt 1 fails -> 1-shot self-correction retry succeeds -> grounded answer streams
        with client.stream(
            "POST",
            "/api/chat",
            json={
                "provider": "ollama",
                "model": "qwen2.5:7b",
                "connector_id": cid,
                "messages": [{"role": "user", "content": "What are the two longest films?"}],
            },
        ) as resp:
            assert resp.status_code == 200
            events = _parse_sse_events(resp.read().decode("utf-8"))

        assert len(call_log) == 3
        retry_prompt = call_log[1]["messages"][-1]["content"]
        assert "nonexistent_col" in retry_prompt
        answer_sys = call_log[2]["messages"][0]["content"]
        assert "Longest Saga" in answer_sys

        sql_events = [e for e in events if e.get("type") == "db_sql"]
        assert len(sql_events) == 2
        assert sql_events[0]["attempt"] == 1
        assert sql_events[1]["attempt"] == 2 and sql_events[1]["retried"] is True

        res_events = [e for e in events if e.get("type") == "db_result"]
        assert len(res_events) == 1
        assert res_events[0]["columns"] == ["title", "length"]
        assert res_events[0]["rows"] == [["Longest Saga", 210], ["Medium Epic", 142]]
        assert res_events[0]["retried"] is True

        delta_text = "".join(e.get("content", "") for e in events if e.get("type") == "delta")
        assert "Longest Saga" in delta_text
        assert any(e.get("type") == "done" for e in events)

        # Scenario 2: Unsafe DELETE SQL generated by LLM is blocked & never mutates DB
        call_log.clear()
        mode["scenario"] = "unsafe_delete"
        with client.stream(
            "POST",
            "/api/chat",
            json={
                "provider": "ollama",
                "model": "qwen2.5:7b",
                "connector_id": cid,
                "messages": [{"role": "user", "content": "Delete film 1"}],
            },
        ) as resp2:
            assert resp2.status_code == 200
            events2 = _parse_sse_events(resp2.read().decode("utf-8"))

        res2 = [e for e in events2 if e.get("type") == "db_result"]
        assert len(res2) == 1
        assert "blocked" in res2[0]["error"].lower() or "read-only" in res2[0]["error"].lower()

        check_conn = sqlite3.connect(dvd_db)
        remaining = check_conn.execute("SELECT COUNT(*) FROM films").fetchone()[0]
        check_conn.close()
        assert remaining == 3


def test_rdbms_connector_error_diagnostics_and_sqlite_missing_file(tmp_path):
    from backend.rdbms import format_db_error

    missing_sqlite = tmp_path / "does_not_exist.sqlite"

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("offline", request=request)

    with _make_client(tmp_path, handler) as client:
        r_test = client.post(
            "/api/connectors/test",
            json={"db_type": "sqlite", "database_name": str(missing_sqlite)},
        )
        assert r_test.status_code == 400
        err_json = r_test.json()
        assert err_json["ok"] is False
        assert "not found" in err_json["error"].lower()
        assert err_json["hint"]
        assert not missing_sqlite.exists()

    pg_cfg = {"db_type": "postgresql", "host": "127.0.0.1", "port": 5432, "database_name": "dvdrental", "username": "postgres"}
    auth_info = format_db_error(RuntimeError('password authentication failed for user "postgres"'), pg_cfg)
    assert "authentication failed" in auth_info["error"].lower()
    assert "pg_hba.conf" in auth_info["hint"].lower()

    db_missing_info = format_db_error(RuntimeError('FATAL: database "dvdrental" does not exist'), pg_cfg)
    assert "does not exist" in db_missing_info["error"].lower()
    assert "database name" in db_missing_info["hint"].lower()

    conn_info = format_db_error(
        RuntimeError("(psycopg.OperationalError) connection refused\n(Background on this error at: https://sqlalche.me/e/21/e3q8)"),
        {"db_type": "postgresql", "host": "localhost", "port": 5432, "database_name": "dvdrental"},
        tried_hosts=["localhost", "127.0.0.1", "172.25.10.2"],
    )
    assert "sqlalche.me" not in conn_info["detail"]
    assert "172.25.10.2" in conn_info["error"]
    assert conn_info["tried_hosts"] == ["localhost", "127.0.0.1", "172.25.10.2"]

    ssl_info = format_db_error(RuntimeError("SSL SYSCALL error: EOF detected"), pg_cfg)
    assert "ssl" in ssl_info["error"].lower()
    assert "disable" in ssl_info["hint"].lower()

    mysql_cfg = {"db_type": "mysql", "host": "127.0.0.1", "port": 3306, "database_name": "appdb", "username": "root"}
    crypto_info = format_db_error(RuntimeError("'cryptography' package is required for sha256_password"), mysql_cfg)
    assert "cryptography" in crypto_info["error"].lower()

    pyodbc_info = format_db_error(RuntimeError("SQL Server driver 'pyodbc' is not installed"), {"db_type": "mssql", "host": "127.0.0.1", "port": 1433, "database_name": "master"})
    assert "pyodbc" in pyodbc_info["hint"].lower() or "pyodbc" in pyodbc_info["error"].lower()


def test_rdbms_prose_sql_extraction_and_non_sql_fallback_context(tmp_path):
    import sqlite3
    from backend.rdbms import extract_sql_from_llm_output

    assert extract_sql_from_llm_output("Here is a query with joins:\nSELECT id FROM films LIMIT 5;") == "SELECT id FROM films LIMIT 5"
    assert extract_sql_from_llm_output("Here are some ways I can help you today!") == ""
    assert extract_sql_from_llm_output("Here:\nDELETE FROM films WHERE id = 1") == "DELETE FROM films WHERE id = 1"

    dvd_db = tmp_path / "ctx.db"
    conn = sqlite3.connect(dvd_db)
    conn.execute("CREATE TABLE films (id INTEGER PRIMARY KEY, title TEXT)")
    conn.execute("INSERT INTO films VALUES (1, 'Alpha')")
    conn.commit()
    conn.close()

    seen_prompts: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/chat":
            body = json.loads(request.content.decode("utf-8"))
            seen_prompts.append(body["messages"][0]["content"])
            if len(seen_prompts) == 1:
                reply = "Here is how I can help with your database!"
            else:
                reply = "Your database contains the films table (id, title)."
            ndjson = (json.dumps({"message": {"role": "assistant", "content": reply}, "done": True}) + "\n").encode()
            return httpx.Response(200, content=ndjson, headers={"Content-Type": "application/x-ndjson"})
        return httpx.Response(404)

    with _make_client(tmp_path, handler) as client:
        r_set = client.put(
            "/api/settings",
            json={"db_context_prompt": "Always prefer table films.", "db_non_sql_fallback": True},
        )
        assert r_set.status_code == 200
        assert r_set.json()["db_context_prompt"] == "Always prefer table films."
        assert r_set.json()["db_non_sql_fallback"] is True

        r_conn = client.post(
            "/api/connectors",
            json={"name": "Ctx DB", "db_type": "sqlite", "database_name": str(dvd_db), "is_active": True},
        )
        cid = r_conn.json()["connector"]["id"]

        with client.stream(
            "POST",
            "/api/chat",
            json={
                "provider": "ollama",
                "model": "qwen2.5:7b",
                "connector_id": cid,
                "messages": [{"role": "user", "content": "hi"}],
            },
        ) as resp:
            assert resp.status_code == 200
            events = _parse_sse_events(resp.read().decode("utf-8"))

        assert len(seen_prompts) == 2
        assert "Always prefer table films." in seen_prompts[0]
        assert "Always prefer table films." in seen_prompts[1]
        delta_text = "".join(e.get("content", "") for e in events if e.get("type") == "delta")
        assert "films table" in delta_text
        assert not any(e.get("type") == "db_result" for e in events)
