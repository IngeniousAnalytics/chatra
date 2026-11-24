# Changelog

All notable changes to **Chatra** (Powered by [IngeniousAnalytics.com](https://ingeniousanalytics.com)) will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2026-11-24

### Added
- Public community release of **Chatra — AI Chat & Database Analytics Studio** under [LICENSE](LICENSE), [NOTICE](NOTICE), and [BRAND_POLICY.md](BRAND_POLICY.md).
- Redesigned light-surface **Chatra** product logo (`web/chatra-logo.svg`) and matching favicon (`web/favicon.svg`).
- Full SEO optimization (`web/index.html` Open Graph, Twitter Cards, JSON-LD `SoftwareApplication`, `web/robots.txt`, `web/sitemap.xml`, `web/manifest.webmanifest`).
- Comprehensive community documentation (`README.md` with Ollama & `llama.cpp` installation guides and screenshots, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`, GitHub Actions CI & issue/PR templates).
- Async FastAPI + Uvicorn + SQLite (WAL) backend with zero-buffer SSE streaming and instant stop/abort.
- Dual local LLM provider adapters: **Ollama** (`/api/tags`, `/api/ps`, `/api/chat`) and **`llama.cpp` (`llama-server`)** (single-model & multi-model router modes) with automatic WSL2 smart networking fallback.
- **RDBMS Database Analytics Studio** supporting PostgreSQL, MySQL, MariaDB, SQLite, and SQL Server with schema introspection and read-only SQL execution.
- Voice input/output, PDF/TXT/MD/JSON document attachment Q&A, Paper/Matte light themes and Dark mode.
