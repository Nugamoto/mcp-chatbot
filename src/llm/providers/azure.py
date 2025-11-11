import logging
from typing import Any, Dict, List

import requests

from src.llm.base import BaseLLMProvider
from src.utils.exceptions import LLMRequestError, LLMResponseParseError
from src.utils.retry import retry_call


class AzureOpenAIProvider(BaseLLMProvider):
    """Provider for Azure OpenAI chat completions endpoint."""

    def chat(self, messages: List[Dict[str, str]]) -> str:
        api_version = self.extra.get("azure_api_version") or self.extra.get("api_version") or "2024-08-01-preview"
        url = f"{self.base_url}/openai/deployments/{self.model}/chat/completions?api-version={api_version}"
        headers = {
            "Content-Type": "application/json",
            "api-key": self.api_key,
        }

        payload: Dict[str, Any] = {
            "messages": messages,
            "top_p": 1,
            "stream": False,
        }
        if self._is_reasoning_model():
            payload["temperature"] = 1
            payload["max_completion_tokens"] = 1024
        else:
            payload["temperature"] = 0.7
            payload["max_tokens"] = 1024

        try:
            response = retry_call(
                lambda: requests.post(url, headers=headers, json=payload, timeout=60),
                tries=self.extra.get("tries", 3),
                backoff=self.extra.get("backoff", 0.5),
                jitter=self.extra.get("jitter", 0.25),
                exceptions=(requests.exceptions.RequestException,),
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except requests.exceptions.RequestException as e:
            logging.error("Azure OpenAI request failed: %s", e)
            if getattr(e, "response", None) is not None:
                logging.error("Status code: %s", e.response.status_code)
                logging.error("Response details: %s", e.response.text)
            raise LLMRequestError(str(e)) from e
        except (KeyError, IndexError, ValueError) as e:
            logging.error("Failed to parse Azure OpenAI response: %s", e)
            raise LLMResponseParseError(str(e)) from e
