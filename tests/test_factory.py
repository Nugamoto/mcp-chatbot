import os

from src.llm.factory import create_provider


def test_create_openai_provider():
    provider = create_provider(
        provider="openai",
        api_key="sk-xxx",
        base_url="https://api.openai.com/v1",
        model="gpt-4o-mini",
    )
    assert provider.__class__.__name__ == "OpenAIProvider"


def test_create_groq_provider():
    provider = create_provider(
        provider="groq",
        api_key="gk-xxx",
        base_url="https://api.groq.com/openai/v1",
        model="llama-3.1-70b-versatile",
    )
    assert provider.__class__.__name__ == "GroqProvider"


def test_create_azure_provider():
    provider = create_provider(
        provider="azure",
        api_key="azure-xxx",
        base_url="https://my-azure.openai.azure.com",
        model="my-deployment",
        azure_api_version="2024-08-01-preview",
    )
    assert provider.__class__.__name__ == "AzureOpenAIProvider"


def test_create_ollama_provider():
    provider = create_provider(
        provider="ollama",
        api_key="",
        base_url="http://localhost:11434/v1",
        model="llama3.1",
    )
    assert provider.__class__.__name__ == "OllamaProvider"


def test_create_provider_unsupported():
    try:
        create_provider(provider="unknown", api_key="", base_url="", model="m")
    except ValueError as e:
        assert "Unsupported provider" in str(e)
    else:
        assert False, "Expected ValueError for unsupported provider"
