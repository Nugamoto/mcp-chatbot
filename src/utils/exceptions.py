class ConfigurationError(Exception):
    """Raised when configuration or environment variables are invalid or missing."""


class ProviderError(Exception):
    """Base class for provider-related errors."""


class LLMRequestError(ProviderError):
    """Raised when an HTTP request to the LLM provider fails."""


class LLMResponseParseError(ProviderError):
    """Raised when the LLM provider returns an unexpected response format."""


class ServerError(Exception):
    """Raised for MCP server interaction errors."""


class ToolExecutionError(ServerError):
    """Raised when executing a tool fails after retries."""