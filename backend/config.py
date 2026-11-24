import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
WEB_DIR = ROOT_DIR / "web"
DATA_DIR = ROOT_DIR / "data"
DB_PATH = DATA_DIR / "chatui.db"
LEGACY_CHATS_FILE = DATA_DIR / "chats.json"

DEFAULT_PORT = 8085
DEFAULT_OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
DEFAULT_LLAMACPP_URL = os.environ.get("LLAMACPP_URL", "http://127.0.0.1:8080").rstrip("/")
DEFAULT_SYSTEM_PROMPT = (
    "You are Advisor, a sharp, hyper-competent AI assistant and operating partner. "
    "Deliver clear, direct, and well-structured responses. Use markdown formatting where appropriate."
)

DEFAULT_SETTINGS: dict = {
    "default_provider": "ollama",
    "ollama_url": DEFAULT_OLLAMA_URL,
    "llamacpp_url": DEFAULT_LLAMACPP_URL,
    "api_key": "",
    "wsl_auto_detect": True,
    "wsl_distro": "",
    "wsl_gguf_dir": "",
    "system_prompt": DEFAULT_SYSTEM_PROMPT,
    "db_context_prompt": "",
    "db_non_sql_fallback": True,
    "temperature": 0.3,
    "max_tokens": 0,
}
