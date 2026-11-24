#!/usr/bin/env python3
"""
Chatra — AI Chat & Analytics Studio (Powered by IngeniousAnalytics.com)
=======================================================================
Async FastAPI + Uvicorn + httpx + SQLite (WAL) server supporting
both Ollama and llama.cpp (llama-server) running in WSL plus RDBMS Q&A.
"""

import argparse
import os
import socket
import sys

import uvicorn

from backend.app import create_app
from backend.config import DEFAULT_LLAMACPP_URL, DEFAULT_OLLAMA_URL, DEFAULT_PORT


def _pick_free_port(host: str, preferred_port: int) -> int:
    """Return the first bindable port from preferred_port and safe fallbacks."""
    bind_host = "127.0.0.1" if host in ("", "localhost") else host
    candidates = []
    for port in (preferred_port, 8888, 8780, 8000, 8788):
        if port not in candidates:
            candidates.append(port)

    for port in candidates:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind((bind_host, port))
            return port
        except OSError:
            continue
        finally:
            sock.close()

    sys.stderr.write(f"Error: Could not bind to any port in {candidates}.\n")
    sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="ChatUI Local FastAPI Server")
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("PORT", DEFAULT_PORT)),
        help="Port to listen on",
    )
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address to bind")
    parser.add_argument("--ollama-url", type=str, default=None, help="Override Ollama API base URL")
    parser.add_argument("--llamacpp-url", type=str, default=None, help="Override llama.cpp API base URL")
    parser.add_argument("--reload", action="store_true", help="Enable Uvicorn auto-reload for development")
    args = parser.parse_args()

    bind_port = _pick_free_port(args.host, args.port)
    display_host = "localhost" if args.host in ("127.0.0.1", "0.0.0.0", "") else args.host
    url = f"http://{display_host}:{bind_port}"

    cli_overrides: dict[str, str] = {}
    if args.ollama_url:
        cli_overrides["ollama_url"] = args.ollama_url.rstrip("/")
        os.environ["OLLAMA_URL"] = cli_overrides["ollama_url"]
    if args.llamacpp_url:
        cli_overrides["llamacpp_url"] = args.llamacpp_url.rstrip("/")
        os.environ["LLAMACPP_URL"] = cli_overrides["llamacpp_url"]

    display_ollama = cli_overrides.get("ollama_url", DEFAULT_OLLAMA_URL)
    display_llamacpp = cli_overrides.get("llamacpp_url", DEFAULT_LLAMACPP_URL)

    print("=" * 68)
    print("  Chatra — AI & Database Analytics Studio")
    print("  Powered by      : IngeniousAnalytics.com")
    print(f"  Web Interface   : {url}")
    print(f"  Direct Chat     : {url}/#/chat/961ff56d-2f0b-4c65-b130-e4d2135f8c75")
    print(f"  Ollama Engine   : {display_ollama}")
    print(f"  llama.cpp Engine: {display_llamacpp}")
    print("=" * 68)
    print("  Server is active. Press Ctrl+C to stop.\n", flush=True)

    target = "backend.app:app" if args.reload else create_app(cli_overrides=cli_overrides or None)
    uvicorn.run(target, host=args.host, port=bind_port, reload=args.reload, log_level="info")


if __name__ == "__main__":
    main()
