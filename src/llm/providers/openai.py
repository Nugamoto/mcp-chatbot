import logging
from typing import Any, Dict, List

import requests

from src.llm.base import BaseLLMProvider
from src.utils.exceptions import LLMRequestError, LLMResponseParseError
from src.utils.retry import retry_call


class OpenAIProvider(BaseLLMProvider):
    """Provider for OpenAI-compatible chat completions endpoint."""

    def chat(self, messages: List[Dict[str, str]]) -> str:
        url = f"{self.base_url}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        def build_payload(use_max_completion: bool, include_temperature: bool = True) -> Dict[str, Any]:
            max_tokens = int(self.extra.get("max_tokens", 1024))
            temp = float(self.extra.get("temperature", 0.7))
            payload: Dict[str, Any] = {
                "messages": messages,
                "model": self.model,
                "top_p": 1,
                "stream": False,
            }
            # Some models (gpt-5*, o1/o3) disallow custom temperature; omit it
            if include_temperature and not self._disallows_custom_temperature():
                payload["temperature"] = temp
            if use_max_completion:
                payload["max_completion_tokens"] = max_tokens
            else:
                payload["max_tokens"] = max_tokens
            return payload

        # Heuristic: choose correct token parameter
        use_max_completion = self._requires_max_completion_tokens()
        payload = build_payload(use_max_completion, include_temperature=True)

        try:
            response = retry_call(
                lambda: requests.post(url, headers=headers, json=payload, timeout=60),
                tries=self.extra.get("tries", 3),
                backoff=self.extra.get("backoff", 0.5),
                jitter=self.extra.get("jitter", 0.25),
                exceptions=(requests.exceptions.RequestException,),
            )
            # If request failed with 400 unsupported parameter/value, retry once adjusting payload
            if response.status_code == 400:
                try:
                    err = response.json()
                    error_obj = (err.get("error", {}) or {})
                    msg = error_obj.get("message", "")
                    code = error_obj.get("code", "")
                    param = error_obj.get("param", "")
                except Exception:
                    msg = ""; code = ""; param = ""
                # Toggle token field if unsupported_parameter
                if code == "unsupported_parameter" or "Unsupported parameter" in msg:
                    logging.warning("Retrying OpenAI call toggling token parameter due to unsupported parameter: %s", msg)
                    payload = build_payload(not use_max_completion, include_temperature=True)
                    response = requests.post(url, headers=headers, json=payload, timeout=60)
                # Remove temperature if unsupported_value on temperature
                elif code == "unsupported_value" and (param == "temperature" or "temperature" in msg.lower()):
                    logging.warning("Retrying OpenAI call removing temperature due to unsupported value: %s", msg)
                    payload = build_payload(use_max_completion, include_temperature=False)
                    response = requests.post(url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except requests.exceptions.RequestException as e:
            logging.error("OpenAI request failed: %s", e)
            if getattr(e, "response", None) is not None:
                logging.error("Status code: %s", e.response.status_code)
                logging.error("Response details: %s", e.response.text)
            raise LLMRequestError(str(e)) from e
        except (KeyError, IndexError, ValueError) as e:
            logging.error("Failed to parse OpenAI response: %s", e)
            raise LLMResponseParseError(str(e)) from e
