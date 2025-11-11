from typing import Any

from src.llm.base import BaseLLMProvider
from src.llm.providers.openai import OpenAIProvider
from src.llm.providers.groq import GroqProvider
from src.llm.providers.azure import AzureOpenAIProvider
from src.llm.providers.ollama import OllamaProvider


def create_provider(
    provider: str,
    api_key: str,
    base_url: str,
    model: str,
    *,
    azure_api_version: str = "",
    **kwargs: Any,
) -> BaseLLMProvider:
    key = (provider or "").strip().lower()
    if key == "openai":
        return OpenAIProvider(api_key, base_url, model, **kwargs)
    if key == "groq":
        return GroqProvider(api_key, base_url, model, **kwargs)
    if key == "azure":
        return AzureOpenAIProvider(api_key, base_url, model, azure_api_version=azure_api_version, **kwargs)
    if key == "ollama":
        return OllamaProvider(api_key, base_url, model, **kwargs)
    raise ValueError(f"Unsupported provider: {provider}")
