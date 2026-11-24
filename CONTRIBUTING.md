# Contributing to Chatra

Thank you for your interest in contributing to **Chatra** (Powered by [IngeniousAnalytics.com](https://ingeniousanalytics.com))!

## Ground Rules

1. **Preserve Brand & Copyright**: Per [LICENSE](LICENSE), [NOTICE](NOTICE), and [BRAND_POLICY.md](BRAND_POLICY.md), pull requests must not alter, obscure, or remove the copyright message, **IngeniousAnalytics.com** company name/attribution, or official logo assets (`web/chatra-logo.svg`, `web/favicon.svg`, `web/ingenious-logo.png`, `web/ingenious-logo.svg`).
2. **Code Quality**: Keep changes focused, readable, and well-tested.
3. **Privacy**: Never commit personal `.db` files, `chats.json`, API keys, or database credentials.

## Local Development Setup

```bash
git clone https://github.com/IngeniousAnalytics/chatra.git
cd chatra
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# Linux / macOS / WSL2
source .venv/bin/activate

pip install -r requirements.txt
pip install pytest
```

## Running the Test Suite

```bash
python -m pytest tests/test_api.py -v
```

## Submitting a Pull Request

1. Fork the repository and create a feature branch from `main` (e.g. `feat/postgres-ssl-options` or `fix/stream-abort`).
2. Make your changes and add/update tests in `tests/test_api.py`.
3. Ensure `python -m pytest tests/test_api.py -v` passes completely.
4. Open a Pull Request describing the motivation, changes, and verification steps.
