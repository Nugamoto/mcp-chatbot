import json
import os
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv


class Configuration:
    """Manages configuration and environment variables for the MCP client."""

    def __init__(self) -> None:
        """Initialize configuration with environment variables."""
        self.load_env()
        # Provider selection: openai | groq | azure | ollama
        raw_provider = os.getenv("LLM_PROVIDER", "openai").strip().lower()
        # Clean provider value - take only the first word if multiple are present
        self.provider = raw_provider.split()[0].split(',')[0].split('#')[0]
        
        # Validate provider
        valid_providers = ["openai", "groq", "azure", "ollama"]
        if self.provider not in valid_providers:
            self.provider = "openai"
        
        # API key per provider
        self.api_key = (
            os.getenv("OPENAI_API_KEY") if self.provider == "openai"
            else os.getenv("GROQ_API_KEY") if self.provider == "groq"
            else os.getenv("AZURE_OPENAI_API_KEY") if self.provider == "azure"
            else os.getenv("OLLAMA_API_KEY")  # optional
        )
        # Base URLs and models with sensible defaults
        self.base_url = (
            os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1") if self.provider == "openai"
            else os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1") if self.provider == "groq"
            else os.getenv("AZURE_OPENAI_BASE_URL", "") if self.provider == "azure"
            else os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        )
        self.model = (
            os.getenv("OPENAI_MODEL", "gpt-4o-mini") if self.provider == "openai"
            else os.getenv("GROQ_MODEL", "llama-3.1-70b-versatile") if self.provider == "groq"
            else os.getenv("AZURE_OPENAI_DEPLOYMENT") if self.provider == "azure"
            else os.getenv("OLLAMA_MODEL", "llama3.1")
        )
        # Azure API version
        self.azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
        # Optional max tokens setting (applies to providers that support it)
        try:
            self.max_tokens = int(os.getenv("LLM_MAX_TOKENS", "1024"))
        except ValueError:
            self.max_tokens = 1024

    @staticmethod
    def _find_project_root() -> Path:
        """Best-effort find the project root containing .env and servers_config.json."""
        here = Path(__file__).resolve()
        for parent in [here.parent, *here.parents]:
            if (parent / ".env").exists() or (parent / "servers_config.json").exists():
                return parent
        # Fallback to repository root three levels up
        return here.parents[3] if len(here.parents) >= 4 else here.parent

    @classmethod
    def load_env(cls) -> None:
        """Load environment variables from a .env file near project root."""
        root = cls._find_project_root()
        env_path = root / ".env"
        if env_path.exists():
            load_dotenv(dotenv_path=env_path, override=True)
        else:
            # Also try current working directory
            alt = Path.cwd() / ".env"
            if alt.exists():
                load_dotenv(dotenv_path=alt, override=True)

    @staticmethod
    def load_config(file_path: str) -> Dict[str, Any]:
        """Load server configuration from a JSON file.

        Args:
            file_path: Path to the JSON configuration file.

        Returns:
            Dict containing server configuration.

        Raises:
            FileNotFoundError: If the configuration file doesn't exist.
            JSONDecodeError: If the configuration file is invalid JSON.
        """
        with open(file_path) as f:
            return json.load(f)

    @property
    def llm_api_key(self) -> str:
        """Get the LLM API key."""
        # Ollama typically doesn't require an API key
        if self.provider == "ollama":
            return self.api_key or ""
        
        if not self.api_key or not self.api_key.strip():
            env_var_name = (
                "OPENAI_API_KEY" if self.provider == "openai"
                else "GROQ_API_KEY" if self.provider == "groq"
                else "AZURE_OPENAI_API_KEY" if self.provider == "azure"
                else "OLLAMA_API_KEY"
            )
            raise ValueError(
                f"API key not found. Please set {env_var_name} in your environment variables or .env file.\n"
                f"Current provider: {self.provider}"
            )
        return self.api_key.strip()

    @property
    def llm_base_url(self) -> str:
        return self.base_url

    @property
    def llm_model(self) -> str:
        return self.model

    @property
    def llm_provider(self) -> str:
        return self.provider

    @property
    def llm_azure_api_version(self) -> str:
        return self.azure_api_version
