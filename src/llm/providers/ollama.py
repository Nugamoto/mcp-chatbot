import logging
from typing import Any, Dict, List

import requests

from src.llm.base import BaseLLMProvider
from src.utils.exceptions import LLMRequestError, LLMResponseParseError
from src.utils.retry import retry_call


class OllamaProvider(BaseLLMProvider):
    """Provider for Ollama's OpenAI-compatible chat completions endpoint."""

    def chat(self, messages: List[Dict[str, str]]) -> str:
        url = f"{self.base_url}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload: Dict[str, Any] = {
            "messages": messages,
            "model": self.model,
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
            logging.error("Ollama request failed: %s", e)
            if getattr(e, "response", None) is not None:
                logging.error("Status code: %s", e.response.status_code)
                logging.error("Response details: %s", e.response.text)
            raise LLMRequestError(str(e)) from e
        except (KeyError, IndexError, ValueError) as e:
            logging.error("Failed to parse Ollama response: %s", e)
            raise LLMResponseParseError(str(e)) from e
