import logging
from typing import Any, Dict, List

from src.llm.factory import create_provider
from src.utils.exceptions import ProviderError


class LLMClient:
    """Manages communication with the LLM provider via a provider factory."""

    def __init__(self, api_key: str, provider: str, base_url: str, model: str, azure_api_version: str = "") -> None:
        self.provider_name = (provider or "").strip().lower()
        # Create concrete provider implementation
        self.provider_impl = create_provider(
            provider=self.provider_name,
            api_key=(api_key or "").strip(),
            base_url=(base_url or "").rstrip("/"),
            model=model,
            azure_api_version=azure_api_version or "2024-08-01-preview",
        )

    def get_response(self, messages: List[Dict[str, str]]) -> str:
        """Get a response from the LLM through the selected provider."""
        try:
            return self.provider_impl.chat(messages)
        except ProviderError as e:
            error_message = f"LLM provider error: {str(e)}"
            logging.error(error_message)
            return f"I encountered an error: {error_message}. Please try again or rephrase your request."
        except Exception as e:
            logging.exception("Unexpected error while getting LLM response")
            return f"I encountered an unexpected error: {str(e)}"