import json
import re
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


def _clean_model_display_name(model_id: str) -> str:
    """Turn long .gguf file paths into clean basenames for display while preserving exact model ID."""
    if model_id.lower().endswith(".gguf") and ("/" in model_id or "\\" in model_id):
        return model_id.replace("\\", "/").rsplit("/", 1)[-1]
    return model_id


def _extract_status_value(item: dict) -> str | None:
    status = item.get("status")
    if isinstance(status, dict):
        val = status.get("value")
        return str(val).lower() if val else None
    if isinstance(status, str):
        return status.lower()
    return None


def _format_param_count(n_params: int | float | None) -> str | None:
    if not isinstance(n_params, (int, float)) or n_params <= 0:
        return None
    if n_params >= 1e9:
        return f"{n_params / 1e9:.1f}B"
    if n_params >= 1e6:
        return f"{n_params / 1e6:.0f}M"
    return str(int(n_params))


class LlamaCppProvider(BaseProvider):
    provider = "llamacpp"
    label = "llama.cpp"
    offline_hint = (
        "Start llama-server in WSL (multi-model router mode): "
        "llama-server --models-dir ~/models --host 0.0.0.0 --port 8080"
    )

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
            resp = await client.get(f"{target_url}/v1/models", headers=headers, timeout=3.0)
            ctype = resp.headers.get("content-type", "").lower()
            if "application/json" not in ctype:
                raise ValueError(
                    f"Expected JSON from llama.cpp /v1/models, got {ctype or 'unknown'} (possible port conflict)"
                )
            resp.raise_for_status()
            data = resp.json()
            if not isinstance(data, dict) or ("data" not in data and "models" not in data):
                raise ValueError("Unexpected response format from llama.cpp /v1/models")
            return data

        try:
            resolved_url, via_wsl, payload = await resolve_reachable_url(
                clean_url, wsl_auto_detect, wsl_distro, _probe
            )
        except Exception as exc:
            return ModelsResponse(
                provider="llamacpp",
                online=False,
                url=clean_url,
                resolved_url=clean_url,
                via_wsl_ip=False,
                error=str(exc),
                hint=self.offline_hint,
                models=[],
            )

        merged_items: dict[str, dict] = {}
        for source_key in ("data", "models"):
            raw_list = payload.get(source_key) or []
            if not isinstance(raw_list, list):
                continue
            for item in raw_list:
                if not isinstance(item, dict):
                    continue
                mid = str(item.get("id") or item.get("name") or item.get("model") or "").strip()
                if not mid:
                    continue
                if mid not in merged_items:
                    merged_items[mid] = dict(item)
                else:
                    merged_items[mid].update(item)

        has_router_status = any(_extract_status_value(item) is not None for item in merged_items.values())
        mode = "router" if (len(merged_items) > 1 or has_router_status) else "single-model"

        server_n_ctx: int | None = None
        if merged_items:
            try:
                props_resp = await client.get(f"{resolved_url}/props", headers=headers, timeout=1.2)
                if props_resp.status_code == 200 and "json" in props_resp.headers.get("content-type", "").lower():
                    props_data = props_resp.json()
                    if isinstance(props_data, dict):
                        dgs = props_data.get("default_generation_settings")
                        if isinstance(dgs, dict) and isinstance(dgs.get("n_ctx"), int):
                            server_n_ctx = dgs["n_ctx"]
            except Exception:
                pass

        models: list[ModelInfo] = []
        for mid, item in merged_items.items():
            meta = item.get("meta") if isinstance(item.get("meta"), dict) else {}
            details = item.get("details") if isinstance(item.get("details"), dict) else {}
            status_val = _extract_status_value(item)
            if status_val == "loaded":
                is_loaded = True
            elif status_val in {"unloaded", "loading", "failed"}:
                is_loaded = False
            else:
                is_loaded = mode == "single-model"

            size_bytes = item.get("size") or item.get("size_bytes") or meta.get("size")
            param_size = (
                meta.get("parameter_size")
                or details.get("parameter_size")
                or _format_param_count(meta.get("n_params"))
            )
            quant = meta.get("quantization") or details.get("quantization_level") or item.get("quantization")
            family = meta.get("arch") or details.get("family") or item.get("owned_by")
            ctx_raw = server_n_ctx or meta.get("n_ctx_train") or item.get("context_length")
            context_length = int(ctx_raw) if isinstance(ctx_raw, (int, float)) and ctx_raw > 0 else None

            models.append(
                ModelInfo(
                    id=mid,
                    name=_clean_model_display_name(mid),
                    provider="llamacpp",
                    size_bytes=size_bytes,
                    parameter_size=param_size,
                    quantization=quant,
                    family=family,
                    context_length=context_length,
                    loaded=is_loaded,
                    status=status_val or ("loaded" if is_loaded else "available"),
                )
            )

        hint = None
        if not models:
            try:
                health_resp = await client.get(f"{resolved_url}/health", headers=headers, timeout=1.5)
                h_data = health_resp.json() if "json" in health_resp.headers.get("content-type", "").lower() else {}
                h_status = str(h_data.get("status") or "").lower()
                if health_resp.status_code == 503 or h_status == "loading":
                    mode = "loading"
                    hint = "llama-server is currently loading a model..."
            except Exception:
                pass
            if not hint:
                hint = (
                    "llama-server is online but reported 0 models. "
                    "Verify --models-dir contains .gguf files or start with -m <model.gguf>."
                )

        return ModelsResponse(
            provider="llamacpp",
            online=True,
            url=clean_url,
            resolved_url=resolved_url,
            via_wsl_ip=via_wsl,
            mode=mode,
            hint=hint,
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
            provider="llamacpp",
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

    async def _model_action(
        self,
        client: httpx.AsyncClient,
        base_url: str,
        action_path: str,
        model_id: str,
        wsl_auto_detect: bool = True,
        wsl_distro: str = "",
        api_key: str = "",
    ) -> dict:
        clean_url = base_url.rstrip("/")
        headers = auth_headers(api_key)

        async def _post(target_url: str) -> httpx.Response:
            return await client.post(f"{target_url}{action_path}", json={"model": model_id}, headers=headers, timeout=60.0)

        resolved_url, via_wsl, resp = await resolve_reachable_url(clean_url, wsl_auto_detect, wsl_distro, _post)
        raw = resp.text
        try:
            body = resp.json() if raw else {}
        except Exception:
            body = {"raw": raw}

        if resp.status_code >= 400:
            err = body.get("error", body) if isinstance(body, dict) else raw
            if isinstance(err, dict):
                err = err.get("message") or str(err)
            return {
                "ok": False,
                "status_code": resp.status_code,
                "model": model_id,
                "url": resolved_url,
                "via_wsl_ip": via_wsl,
                "error": str(err or f"HTTP {resp.status_code}"),
            }

        return {
            "ok": True,
            "model": model_id,
            "url": resolved_url,
            "via_wsl_ip": via_wsl,
            "data": body,
        }

    async def load_model(
        self,
        client: httpx.AsyncClient,
        base_url: str,
        model_id: str,
        wsl_auto_detect: bool = True,
        wsl_distro: str = "",
        api_key: str = "",
    ) -> dict:
        return await self._model_action(client, base_url, "/models/load", model_id, wsl_auto_detect, wsl_distro, api_key=api_key)

    async def unload_model(
        self,
        client: httpx.AsyncClient,
        base_url: str,
        model_id: str,
        wsl_auto_detect: bool = True,
        wsl_distro: str = "",
        api_key: str = "",
    ) -> dict:
        return await self._model_action(client, base_url, "/models/unload", model_id, wsl_auto_detect, wsl_distro, api_key=api_key)

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
        payload: dict = {
            "model": model,
            "messages": prepare_messages(messages, system_prompt),
            "stream": True,
            "stream_options": {"include_usage": True},
            "temperature": temperature,
        }
        if max_tokens and max_tokens > 0:
            payload["max_tokens"] = max_tokens

        if isinstance(sampling, dict):
            passthrough_keys = (
                "temperature",
                "dynatemp_range",
                "dynatemp_exponent",
                "top_k",
                "top_p",
                "min_p",
                "xtc_probability",
                "xtc_threshold",
                "typical_p",
                "backend_sampling",
                "repeat_last_n",
                "repeat_penalty",
                "presence_penalty",
                "frequency_penalty",
                "dry_multiplier",
                "dry_base",
                "dry_allowed_length",
                "dry_penalty_last_n",
            )
            for k in passthrough_keys:
                if sampling.get(k) is not None:
                    payload[k] = sampling[k]
            if sampling.get("max_tokens") is not None:
                mt = int(sampling["max_tokens"])
                if mt > 0:
                    payload["max_tokens"] = mt
                else:
                    payload.pop("max_tokens", None)
            if sampling.get("samplers") is not None:
                raw_samplers = sampling["samplers"]
                if isinstance(raw_samplers, str):
                    items = [s.strip() for s in re.split(r"[,;]", raw_samplers) if s.strip()]
                    if items:
                        payload["samplers"] = items
                elif isinstance(raw_samplers, list):
                    payload["samplers"] = raw_samplers

        if disable_reasoning_parsing:
            payload["reasoning_format"] = "none"

        if reasoning_effort:
            eff = reasoning_effort.strip().lower()
            if eff in {"off", "none", "low", "medium", "high"}:
                payload["reasoning_effort"] = "none" if eff == "off" else eff

        if isinstance(custom_json, dict) and custom_json:
            payload.update(custom_json)

        headers = auth_headers(api_key)
        timeout = httpx.Timeout(connect=10.0, read=None, write=30.0, pool=10.0)
        t0 = time.perf_counter()
        predicted_tps: float | None = None
        completion_tokens: int | None = None
        prompt_tokens: int | None = None
        prompt_ms: float | None = None
        prompt_tps: float | None = None
        predicted_ms: float | None = None

        try:
            async with stream_post_with_fallback(
                client, clean_url, "/v1/chat/completions", payload, timeout, wsl_auto_detect, wsl_distro, headers=headers
            ) as (resolved_url, _via_wsl, resp):
                if resp.status_code >= 400:
                    raw_err = await resp.aread()
                    err_text = raw_err.decode("utf-8", errors="replace")
                    try:
                        err_json = json.loads(err_text)
                        err_obj = err_json.get("error")
                        if isinstance(err_obj, dict):
                            err_text = str(err_obj.get("message") or err_obj)
                        elif err_obj:
                            err_text = str(err_obj)
                    except Exception:
                        pass
                    yield {
                        "type": "error",
                        "error": f"llama.cpp HTTP {resp.status_code} ({resolved_url}): {err_text}",
                        "hint": self.offline_hint,
                    }
                    return

                ctype = resp.headers.get("content-type", "").lower()
                if "text/html" in ctype:
                    yield {
                        "type": "error",
                        "error": f"Received HTML instead of SSE from {resolved_url}/v1/chat/completions (check llama.cpp port)",
                        "hint": self.offline_hint,
                    }
                    return

                async for raw_line in resp.aiter_lines():
                    line = raw_line.strip()
                    if not line or line.startswith(":") or line.startswith("event:"):
                        continue

                    if line.startswith("data:"):
                        data_str = line[5:].strip()
                    elif line.startswith("{"):
                        data_str = line
                    else:
                        continue

                    if data_str == "[DONE]":
                        break

                    try:
                        obj = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    if not isinstance(obj, dict):
                        continue

                    if obj.get("error"):
                        err_obj = obj["error"]
                        err_msg = err_obj.get("message") if isinstance(err_obj, dict) else str(err_obj)
                        yield {"type": "error", "error": str(err_msg), "hint": self.offline_hint}
                        return

                    timings = obj.get("timings")
                    if isinstance(timings, dict):
                        if timings.get("predicted_per_second") is not None:
                            try:
                                predicted_tps = float(timings["predicted_per_second"])
                            except (TypeError, ValueError):
                                pass
                        if timings.get("predicted_n") is not None:
                            try:
                                completion_tokens = int(timings["predicted_n"])
                            except (TypeError, ValueError):
                                pass
                        if timings.get("prompt_n") is not None:
                            try:
                                prompt_tokens = int(timings["prompt_n"])
                            except (TypeError, ValueError):
                                pass
                        if timings.get("prompt_ms") is not None:
                            try:
                                prompt_ms = float(timings["prompt_ms"])
                            except (TypeError, ValueError):
                                pass
                        if timings.get("prompt_per_second") is not None:
                            try:
                                prompt_tps = float(timings["prompt_per_second"])
                            except (TypeError, ValueError):
                                pass
                        if timings.get("predicted_ms") is not None:
                            try:
                                predicted_ms = float(timings["predicted_ms"])
                            except (TypeError, ValueError):
                                pass

                    usage = obj.get("usage")
                    if isinstance(usage, dict):
                        if usage.get("completion_tokens") is not None:
                            completion_tokens = int(usage["completion_tokens"])
                        if usage.get("prompt_tokens") is not None:
                            prompt_tokens = int(usage["prompt_tokens"])

                    choices = obj.get("choices") or []
                    if choices and isinstance(choices[0], dict):
                        delta = choices[0].get("delta") or {}
                        if isinstance(delta, dict):
                            content = delta.get("content") or ""
                            thinking = delta.get("reasoning_content") or delta.get("thinking") or ""
                            if content or thinking:
                                yield {"type": "delta", "content": content, "thinking": thinking}

            elapsed_s = max(0.001, time.perf_counter() - t0)
            tps = (
                round(predicted_tps, 1)
                if predicted_tps is not None
                else (round(completion_tokens / elapsed_s, 1) if completion_tokens else None)
            )
            prompt_seconds = round(prompt_ms / 1000.0, 2) if prompt_ms is not None and prompt_ms >= 0 else None
            prompt_tokens_per_sec = round(prompt_tps, 2) if prompt_tps is not None else None
            completion_seconds = (
                round(predicted_ms / 1000.0, 2)
                if predicted_ms is not None and predicted_ms > 0
                else round(elapsed_s, 2)
            )
            yield {
                "type": "done",
                "stats": {
                    "provider": "llamacpp",
                    "model": model,
                    "tokens_per_sec": tps,
                    "completion_tokens": completion_tokens,
                    "prompt_tokens": prompt_tokens,
                    "prompt_seconds": prompt_seconds,
                    "prompt_tokens_per_sec": prompt_tokens_per_sec,
                    "completion_seconds": completion_seconds,
                    "elapsed_ms": round(elapsed_s * 1000, 1),
                },
            }
        except Exception as exc:
            yield {
                "type": "error",
                "error": f"llama.cpp connection error ({clean_url}): {exc}",
                "hint": self.offline_hint,
            }
