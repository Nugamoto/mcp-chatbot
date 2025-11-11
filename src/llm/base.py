from abc import ABC, abstractmethod
from typing import Any, Dict, List


class BaseLLMProvider(ABC):
    """Abstract base class for chat-completion providers.

    Providers implement a synchronous `chat` method that receives an array of
    OpenAI-compatible messages and returns the assistant's text response.
    """

    def __init__(self, api_key: str, base_url: str, model: str, **kwargs: Any) -> None:
        self.api_key = (api_key or "").strip()
        self.base_url = (base_url or "").rstrip("/")
        self.model = model
        self.extra: Dict[str, Any] = kwargs

    @abstractmethod
    def chat(self, messages: List[Dict[str, str]]) -> str:
        """Perform a chat completion and return the assistant message text."""
        raise NotImplementedError

    def _is_reasoning_model(self) -> bool:
        """Check if the current model is a reasoning model (o1/o3 series)."""
        reasoning_models = ["o1", "o1-mini", "o1-preview", "o3", "o3-mini", "o3-preview"]
        return any(r in (self.model or "").lower() for r in reasoning_models)

    def _requires_max_completion_tokens(self) -> bool:
        """Heuristic: models in the gpt-4o* and gpt-5* families and reasoning models
        expect `max_completion_tokens` instead of `max_tokens`.
        """
        m = (self.model or "").lower().strip()
        if self._is_reasoning_model():
            return True
        return m.startswith("gpt-4o") or m.startswith("gpt-5")

    def _disallows_custom_temperature(self) -> bool:
        """Heuristic: some models reject any non-default temperature.
        - gpt-5* family currently only accepts the default temperature (1)
        - reasoning models (o1/o3) also effectively ignore/forbid custom temps
        Policy: when True, omit the `temperature` field from payloads.
        """
        m = (self.model or "").lower().strip()
        return self._is_reasoning_model() or m.startswith("gpt-5")
