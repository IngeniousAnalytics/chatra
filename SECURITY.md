# Security Policy

## Supported Versions

| Version | Supported |
| ------- | --------- |
| 2.0.x   | Yes       |

## Local-First Security Architecture

Chatra is designed to run locally (`127.0.0.1` by default) with:
- Read-only SQL validation for RDBMS analytics queries (`SELECT` / `WITH` / `EXPLAIN` / `SHOW` / `DESCRIBE`)
- Encrypted database connector password storage in local SQLite (`data/chatui.db`)
- Zero telemetry or external data exfiltration when used with local Ollama or `llama.cpp` models

## Reporting a Vulnerability

If you discover a security vulnerability in Chatra, **please do not open a public GitHub issue**.

Instead, please report it responsibly via:
- GitHub Security Advisories on `https://github.com/IngeniousAnalytics/chatra/security/advisories/new`, or
- Contacting **[IngeniousAnalytics.com](https://ingeniousanalytics.com)** directly.

Please include steps to reproduce, affected version, and potential impact. We aim to acknowledge reports within 48 hours.
