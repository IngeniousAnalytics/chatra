from backend.providers.llamacpp import LlamaCppProvider
from backend.providers.ollama import OllamaProvider

PROVIDERS = {
    "ollama": OllamaProvider(),
    "llamacpp": LlamaCppProvider(),
}

__all__ = ["OllamaProvider", "LlamaCppProvider", "PROVIDERS"]
