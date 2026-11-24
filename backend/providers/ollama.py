import json
import time
from collections.abc import AsyncIterator

import httpx

from backend.providers.base import (
    BaseProvider,
    ModelInfo,
    ModelsResponse,
    ProviderStatus,
    auth_headers,
    prepare_messages,
    resolve_reachable_url,
    stream_post_with_fallback,
)


class OllamaProvider(BaseProvider):
    provider = "ollama"
    label = "Ollama"
    offline_hint = "Start Ollama in WSL with: OLLAMA_HOST=0.0.0.0:11434 ollama serve"

    async def list_models(
        self,
        client: httpx.AsyncClient,
        base_url: str,
        wsl_auto_detect: bool = True,
        wsl_distro: str = "",
        api_key: str = "",
    ) -> ModelsResponse:
        clean_url = base_url.rstrip("/")
        headers = auth_headers(api_key)

        async def _probe(target_url: str) -> dict:
            resp = await client.get(f"{target_url}/api/tags", headers=headers, timeout=3.0)
            resp.raise_for_status()
            ctype = resp.headers.get("content-type", "").lower()
            if ctype and "application/json" not in ctype:
                raise ValueError(f"Expected JSON from Ollama /api/tags, got {ctype}")
            data = resp.json()
            if not isinstance(data, dict) or "models" not in data:
                raise ValueError("Unexpected response format from Ollama /api/tags")
            return data

        try:
            resolved_url, via_wsl, tags_data = await resolve_reachable_url(
                clean_url, wsl_auto_detect, wsl_distro, _probe
            )
        except Exception as exc:
            return ModelsResponse(
                provider="ollama",
                online=False,
                url=clean_url,
                resolved_url=clean_url,
                via_wsl_ip=False,
                error=str(exc),
                hint=self.offline_hint,
                models=[],
            )

        loaded_names: set[str] = set()
        try:
            ps_resp = await client.get(f"{resolved_url}/api/ps", headers=headers, timeout=1.5)
            if ps_resp.status_code == 200:
                ps_data = ps_resp.json()
                for item in ps_data.get("models", []) if isinstance(ps_data, dict) else []:
                    if isinstance(item, dict):
                        m_name = item.get("name") or item.get("model")
                        if m_name:
                            loaded_names.add(str(m_name))
        except Exception:
            pass

        models: list[ModelInfo] = []
        for item in tags_data.get("models") or []:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or item.get("model") or "").strip()
            if not name:
                continue
            details = item.get("details") if isinstance(item.get("details"), dict) else {}
            is_loaded = name in loaded_names
            models.append(
                ModelInfo(
                    id=name,
                    name=name,
                    provider="ollama",
                    size_bytes=item.get("size"),
                    parameter_size=details.get("parameter_size"),
                    quantization=details.get("quantization_level"),
                    family=details.get("family"),
                    loaded=is_loaded,
                    status="loaded" if is_loaded else "available",
                )
            )

        return ModelsResponse(
            provider="ollama",
            online=True,
            url=clean_url,
            resolved_url=resolved_url,
            via_wsl_ip=via_wsl,
            mode="daemon",
            models=models,
        )

    async def check_status(
        self,
        client: httpx.AsyncClient,
        base_url: str,
        wsl_auto_detect: bool = True,
        wsl_distro: str = "",
        api_key: str = "",
    ) -> ProviderStatus:
        t0 = time.perf_counter()
        res = await self.list_models(client, base_url, wsl_auto_detect, wsl_distro, api_key=api_key)
        latency_ms = round((time.perf_counter() - t0) * 1000, 1) if res.online else None
        return ProviderStatus(
            provider="ollama",
            label=self.label,
            online=res.online,
            url=res.url,
            resolved_url=res.resolved_url,
            via_wsl_ip=res.via_wsl_ip,
            mode=res.mode,
            latency_ms=latency_ms,
            model_count=len(res.models),
            error=res.error,
            hint=res.hint,
        )

    async def stream_chat(
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
        clean_url = base_url.rstrip("/")
        options: dict = {"temperature": temperature}
        if max_tokens and max_tokens > 0:
            options["num_predict"] = max_tokens

        if isinstance(sampling, dict):
            float_keys = (
                "temperature",
                "top_p",
                "min_p",
                "typical_p",
                "repeat_penalty",
                "presence_penalty",
                "frequency_penalty",
            )
            int_keys = ("top_k", "repeat_last_n")
            for k in float_keys:
                if sampling.get(k) is not None:
                    options[k] = float(sampling[k])
            for k in int_keys:
                if sampling.get(k) is not None:
                    options[k] = int(sampling[k])
            if sampling.get("max_tokens") is not None:
                mt = int(sampling["max_tokens"])
                if mt > 0:
                    options["num_predict"] = mt
                else:
                    options.pop("num_predict", None)

        payload: dict = {
            "model": model,
            "messages": prepare_messages(messages, system_prompt),
            "stream": True,
            "options": options,
        }
        if reasoning_effort:
            eff = reasoning_effort.strip().lower()
            if eff in {"off", "none"}:
                payload["think"] = False
            elif eff in {"low", "medium", "high"}:
                payload["think"] = True

        if isinstance(custom_json, dict) and custom_json:
            for k, v in custom_json.items():
                if k == "options" and isinstance(v, dict):
                    options.update(v)
                else:
                    payload[k] = v

        headers = auth_headers(api_key)
        timeout = httpx.Timeout(connect=10.0, read=None, write=30.0, pool=10.0)
        t0 = time.perf_counter()
        done_emitted = False

        try:
            async with stream_post_with_fallback(
                client, clean_url, "/api/chat", payload, timeout, wsl_auto_detect, wsl_distro, headers=headers
            ) as (resolved_url, _via_wsl, resp):
                if resp.status_code >= 400:
                    raw_err = await resp.aread()
                    err_text = raw_err.decode("utf-8", errors="replace")
                    try:
                        err_json = json.loads(err_text)
                        err_text = str(err_json.get("error") or err_text)
                    except Exception:
                        pass
                    yield {
                        "type": "error",
                        "error": f"Ollama HTTP {resp.status_code} ({resolved_url}): {err_text}",
                        "hint": self.offline_hint,
                    }
                    return

                async for line in resp.aiter_lines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    if isinstance(obj, dict) and obj.get("error"):
                        yield {"type": "error", "error": str(obj["error"]), "hint": self.offline_hint}
                        return

                    msg = obj.get("message") if isinstance(obj, dict) else None
                    if isinstance(msg, dict):
                        content = msg.get("content") or ""
                        thinking = msg.get("thinking") or ""
                        if content or thinking:
                            yield {"type": "delta", "content": content, "thinking": thinking}

                    if isinstance(obj, dict) and obj.get("done"):
                        eval_count = obj.get("eval_count") or 0
                        prompt_eval_count = obj.get("prompt_eval_count") or 0
                        eval_duration = obj.get("eval_duration") or 0
                        prompt_eval_duration = obj.get("prompt_eval_duration") or 0
                        tps = (
                            round(eval_count / (eval_duration / 1e9), 1)
                            if eval_count and eval_duration and eval_duration > 0
                            else None
                        )
                        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
                        prompt_seconds = (
                            round(prompt_eval_duration / 1e9, 2)
                            if prompt_eval_duration and prompt_eval_duration > 0
                            else None
                        )
                        prompt_tps = (
                            round(prompt_eval_count / (prompt_eval_duration / 1e9), 2)
                            if prompt_eval_count and prompt_eval_duration and prompt_eval_duration > 0
                            else None
                        )
                        completion_seconds = (
                            round(eval_duration / 1e9, 2)
                            if eval_duration and eval_duration > 0
                            else round(elapsed_ms / 1000, 2)
                        )
                        yield {
                            "type": "done",
                            "stats": {
                                "provider": "ollama",
                                "model": model,
                                "tokens_per_sec": tps,
                                "completion_tokens": eval_count or None,
                                "prompt_tokens": prompt_eval_count or None,
                                "prompt_seconds": prompt_seconds,
                                "prompt_tokens_per_sec": prompt_tps,
                                "completion_seconds": completion_seconds,
                                "elapsed_ms": elapsed_ms,
                            },
                        }
                        done_emitted = True
                        return

            if not done_emitted:
                elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
                yield {
                    "type": "done",
                    "stats": {
                        "provider": "ollama",
                        "model": model,
                        "tokens_per_sec": None,
                        "completion_tokens": None,
                        "prompt_tokens": None,
                        "prompt_seconds": None,
                        "prompt_tokens_per_sec": None,
                        "completion_seconds": round(elapsed_ms / 1000, 2),
                        "elapsed_ms": elapsed_ms,
                    },
                }
        except Exception as exc:
            yield {
                "type": "error",
                "error": f"Ollama connection error ({clean_url}): {exc}",
                "hint": self.offline_hint,
            }
