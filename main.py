import asyncio
from pathlib import Path

from src.utils.logger import configure_logging
from src.config.configuration import Configuration
from src.core.server import Server
from src.llm.client import LLMClient
from src.chat.session import ChatSession


async def main() -> None:
    """Initialize and run the chat session using the modular src/ package."""
    # Configure logging once, centrally
    configure_logging()

    # Load configuration and servers config
    config = Configuration()
    cfg_path = Path(__file__).with_name("servers_config.json")
    server_config = config.load_config(str(cfg_path))

    # Initialize servers from config
    servers = [Server(name, srv_config) for name, srv_config in server_config['mcpServers'].items()]

    # Initialize LLM client
    llm_client = LLMClient(
        api_key=config.llm_api_key,
        provider=config.llm_provider,
        base_url=config.llm_base_url,
        model=config.llm_model,
        azure_api_version=config.llm_azure_api_version,
        max_tokens=getattr(config, "max_tokens", 1024),
        temperature=getattr(config, "temperature", 0.7),
    )

    # Start chat session
    chat_session = ChatSession(servers, llm_client)
    await chat_session.start()


if __name__ == "__main__":
    asyncio.run(main())