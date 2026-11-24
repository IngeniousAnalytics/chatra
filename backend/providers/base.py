from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Literal, TypeVar

import httpx
from pydantic import BaseModel, Field

from backend.wsl import get_wsl_ip, is_loopback_url, rewrite_url_with_host

ProviderName = Literal["ollama", "llamacpp"]
T = TypeVar("T")


class ModelInfo(BaseModel):
    id: str
    name: str
    provider: ProviderName
    size_bytes: int | None = None
    parameter_size: str | None = None
    quantization: str | None = None
    family: str | None = None
    context_length: int | None = None
    loaded: bool | None = None
    status: str | None = None


class ProviderStatus(BaseModel):
    provider: ProviderName
    label: str
    online: bool
    url: str
    resolved_url: str
    via_wsl_ip: bool = False
    mode: str | None = None
    latency_ms: float | None = None
    model_count: int = 0
    error: str | None = None
    hint: str | None = None


class ModelsResponse(BaseModel):
    provider: ProviderName
    online: bool
    url: str
    resolved_url: str
    via_wsl_ip: bool = False
    mode: str | None = None
    error: str | None = None
    hint: str | None = None
    models: list[ModelInfo] = Field(default_factory=list)


def auth_headers(api_key: str) -> dict[str, str]:
    clean = (api_key or "").strip()
    return {"Authorization": f"Bearer {clean}"} if clean else {}


def prepare_messages(messages: list[dict], system_prompt: str | None) -> list[dict]:
    """Normalize conversation messages and prepend system prompt when needed."""
    formatted: list[dict] = []
    clean_system = (system_prompt or "").strip()
    if clean_system and (not messages or messages[0].get("role") != "system"):
        formatted.append({"role": "system", "content": clean_system})
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role in {"system", "user", "assistant"}:
            item: dict = {"role": role, "content": str(msg.get("content", ""))}
            if role == "assistant" and msg.get("reasoning_content"):
                item["reasoning_content"] = str(msg["reasoning_content"])
            formatted.append(item)
    return formatted


async def resolve_reachable_url(
    configured_url: str,
    wsl_auto_detect: bool,
    wsl_distro: str,
    probe_fn: Callable[[str], Awaitable[T]],
) -> tuple[str, bool, T]:
    """Try configured_url first, then fall back to WSL2 VM IP on loopback connect failure."""
    clean_url = configured_url.rstrip("/")
    try:
        result = await probe_fn(clean_url)
        return clean_url, False, result
    except (httpx.ConnectError, httpx.ConnectTimeout):
        if not (wsl_auto_detect and is_loopback_url(clean_url)):
            raise
        wsl_ip = await get_wsl_ip(wsl_distro)
        if not wsl_ip:
            raise
        fallback_url = rewrite_url_with_host(clean_url, wsl_ip)
        result = await probe_fn(fallback_url)
        return fallback_url, True, result


@asynccontextmanager
async def stream_post_with_fallback(
    client: httpx.AsyncClient,
    configured_url: str,
    path: str,
    payload: dict,
    timeout: httpx.Timeout,
    wsl_auto_detect: bool,
    wsl_distro: str,
    headers: dict[str, str] | None = None,
):
    """Open an upstream POST stream with automatic WSL2 IP fallback if loopback fails."""
    clean_url = configured_url.rstrip("/")
    req_headers = headers or {}
    entered = False
    try:
        async with client.stream("POST", f"{clean_url}{path}", json=payload, headers=req_headers, timeout=timeout) as resp:
            entered = True
            yield clean_url, False, resp
            return
    except (httpx.ConnectError, httpx.ConnectTimeout):
        if entered or not (wsl_auto_detect and is_loopback_url(clean_url)):
            raise
        wsl_ip = await get_wsl_ip(wsl_distro)
        if not wsl_ip:
            raise
        fallback_url = rewrite_url_with_host(clean_url, wsl_ip)
        async with client.stream("POST", f"{fallback_url}{path}", json=payload, headers=req_headers, timeout=timeout) as resp:
            yield fallback_url, True, resp


class BaseProvider(ABC):
    provider: ProviderName
    label: str
    offline_hint: str

    @abstractmethod
    async def list_models(
        self,
        client: httpx.AsyncClient,
        base_url: str,
        wsl_auto_detect: bool = True,
        wsl_distro: str = "",
        api_key: str = "",
    ) -> ModelsResponse:
        ...

    @abstractmethod
    async def check_status(
        self,
        client: httpx.AsyncClient,
        base_url: str,
        wsl_auto_detect: bool = True,
        wsl_distro: str = "",
        api_key: str = "",
    ) -> ProviderStatus:
        ...

    @abstractmethod
    def stream_chat(
        self,
        client: httpx.AsyncClient,
        base_url: str,
        model: str,
        messages: list[dict],
        system_prompt: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 0,
        reasoning_effort: str | None = None,
        sampling: dict | None = None,
        custom_json: dict | None = None,
        disable_reasoning_parsing: bool = False,
        wsl_auto_detect: bool = True,
        wsl_distro: str = "",
        api_key: str = "",
    ) -> AsyncIterator[dict]:
        ...
