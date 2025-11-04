import asyncio
import json
import logging
import os
import shutil
from typing import Dict, List, Optional, Any

import requests
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class Configuration:
    """Manages configuration and environment variables for the MCP client."""

    def __init__(self) -> None:
        """Initialize configuration with environment variables."""
        self.load_env()
        # Provider selection: openai | groq | azure | ollama
        self.provider = os.getenv("LLM_PROVIDER", "openai").strip().lower()
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

    @staticmethod
    def load_env() -> None:
        """Load environment variables from .env file."""
        from pathlib import Path
        env_path = Path(__file__).with_name(".env")
        load_dotenv(dotenv_path=env_path, override=True)

    @staticmethod
    def load_config(file_path: str) -> Dict[str, Any]:
        """Load server configuration from JSON file.

        Args:
            file_path: Path to the JSON configuration file.

        Returns:
            Dict containing server configuration.

        Raises:
            FileNotFoundError: If configuration file doesn't exist.
            JSONDecodeError: If configuration file is invalid JSON.
        """
        with open(file_path, 'r') as f:
            return json.load(f)

    @property
    def llm_api_key(self) -> str:
        """Get the LLM API key.

        Returns:
            The API key as a string.

        Raises:
            ValueError: If the API key is not found in environment variables.
        """
        if not self.api_key or not self.api_key.strip():
            raise ValueError(f"{self.provider.upper()}_API_KEY not found in environment variables")
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


class Server:
    """Manages MCP server connections and tool execution."""

    def __init__(self, name: str, config: Dict[str, Any]) -> None:
        self.name: str = name
        self.config: Dict[str, Any] = config
        self.stdio_context: Optional[Any] = None
        self.session: Optional[ClientSession] = None
        self._cleanup_lock: asyncio.Lock = asyncio.Lock()
        self.capabilities: Optional[Dict[str, Any]] = None

    async def initialize(self) -> None:
        """Initialize the server connection."""
        # Resolve command and ensure npx exists if configured
        cmd = self.config['command']
        if cmd == "npx":
            npx_path = shutil.which("npx")
            if not npx_path:
                raise RuntimeError("npx not found on PATH")
            cmd = npx_path
        server_params = StdioServerParameters(
            command=cmd,
            args=self.config['args'],
            env={**os.environ, **self.config['env']} if self.config.get('env') else None
        )
        try:
            self.stdio_context = stdio_client(server_params)
            read, write = await self.stdio_context.__aenter__()
            self.session = ClientSession(read, write)
            await self.session.__aenter__()
            init = await self.session.initialize()
            self.capabilities = getattr(init, "capabilities", None) or (init.get("capabilities", {}) if isinstance(init, dict) else {})
        except Exception as e:
            logging.error(f"Error initializing server {self.name}: {e}")
            await self.cleanup()
            raise

    async def list_tools(self) -> List[Any]:
        if not self.session:
            raise RuntimeError(f"Server {self.name} not initialized")

        tools_resp = await self.session.list_tools()
        tools_list = getattr(tools_resp, "tools", None) or (tools_resp.get("tools", []) if isinstance(tools_resp, dict) else [])
        tools: List[Tool] = []
        for t in tools_list:
            tools.append(Tool(t.name, getattr(t, "description", "") or t.description, t.inputSchema))
        return tools

    async def execute_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        retries: int = 2,
        delay: float = 1.0
    ) -> Any:
        """Execute a tool with retry mechanism.

        Args:
            tool_name: Name of the tool to execute.
            arguments: Tool arguments.
            retries: Number of retry attempts.
            delay: Delay between retries in seconds.

        Returns:
            Tool execution result.

        Raises:
            RuntimeError: If server is not initialized.
            Exception: If tool execution fails after all retries.
        """
        if not self.session:
            raise RuntimeError(f"Server {self.name} not initialized")

        attempt = 0
        while attempt < retries:
            try:
                supports_progress = (
                    self.capabilities
                    and 'progress' in self.capabilities
                )

                if supports_progress:
                    logging.info(f"Executing {tool_name} with progress tracking...")
                    result = await self.session.call_tool(
                        tool_name,
                        arguments,
                        progress_token=f"{tool_name}_execution"
                    )
                else:
                    logging.info(f"Executing {tool_name}...")
                    result = await self.session.call_tool(tool_name, arguments)

                return result

            except Exception as e:
                attempt += 1
                logging.warning(f"Error executing tool: {e}. Attempt {attempt} of {retries}.")
                if attempt < retries:
                    logging.info(f"Retrying in {delay} seconds...")
                    await asyncio.sleep(delay)
                else:
                    logging.error("Max retries reached. Failing.")
                    raise

    async def cleanup(self) -> None:
        """Clean up server resources."""
        async with self._cleanup_lock:
            try:
                if self.session:
                    try:
                        await self.session.__aexit__(None, None, None)
                    except Exception as e:
                        logging.warning(f"Warning during session cleanup for {self.name}: {e}")
                    finally:
                        self.session = None

                if self.stdio_context:
                    try:
                        await self.stdio_context.__aexit__(None, None, None)
                    except (RuntimeError, asyncio.CancelledError) as e:
                        logging.info(f"Note: Normal shutdown message for {self.name}: {e}")
                    except Exception as e:
                        logging.warning(f"Warning during stdio cleanup for {self.name}: {e}")
                    finally:
                        self.stdio_context = None
            except Exception as e:
                logging.error(f"Error during cleanup of server {self.name}: {e}")


class Tool:
    """Represents a tool with its properties and formatting."""

    def __init__(self, name: str, description: str, input_schema: Dict[str, Any]) -> None:
        self.name: str = name
        self.description: str = description
        self.input_schema: Dict[str, Any] = input_schema

    def format_for_llm(self) -> str:
        """Format tool information for LLM.

        Returns:
            A formatted string describing the tool.
        """
        args_desc = []
        if 'properties' in self.input_schema:
            for param_name, param_info in self.input_schema['properties'].items():
                arg_desc = f"- {param_name}: {param_info.get('description', 'No description')}"
                if param_name in self.input_schema.get('required', []):
                    arg_desc += " (required)"
                args_desc.append(arg_desc)

        return f"""
Tool: {self.name}
Description: {self.description}
Arguments:
{chr(10).join(args_desc)}
"""


class LLMClient:
    """Manages communication with the LLM provider."""

    def __init__(self, api_key: str, provider: str, base_url: str, model: str, azure_api_version: str = "2024-08-01-preview") -> None:
        self.api_key = (api_key or "").strip()
        self.provider = provider.strip().lower()
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.azure_api_version = azure_api_version

    def get_response(self, messages: List[Dict[str, str]]) -> str:
        """Get a response from the LLM."""
        provider = self.provider
        if provider in ("openai", "groq", "ollama"):
            url = f"{self.base_url}/chat/completions"
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            payload = {
                "messages": messages,
                "model": self.model,
                "temperature": 0.7,
                "max_tokens": 1024,
                "top_p": 1,
                "stream": False,
            }
        elif provider == "azure":
            url = f"{self.base_url}/openai/deployments/{self.model}/chat/completions?api-version={self.azure_api_version}"
            headers = {
                "Content-Type": "application/json",
                "api-key": self.api_key,
            }
            payload = {
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 1024,
                "top_p": 1,
                "stream": False,
            }
        else:
            raise ValueError(f"Unsupported provider: {provider}")

        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            return data['choices'][0]['message']['content']
        except requests.exceptions.RequestException as e:
            error_message = f"Error getting LLM response: {str(e)}"
            logging.error(error_message)
            if getattr(e, "response", None) is not None:
                status_code = e.response.status_code
                logging.error(f"Status code: {status_code}")
                logging.error(f"Response details: {e.response.text}")
            return f"I encountered an error: {error_message}. Please try again or rephrase your request."


class ChatSession:
    """Orchestrates the interaction between user, LLM, and tools."""

    def __init__(self, servers: List[Server], llm_client: LLMClient) -> None:
        self.servers: List[Server] = servers
        self.llm_client: LLMClient = llm_client

    async def cleanup_servers(self) -> None:
        """Clean up all servers properly."""
        cleanup_tasks = []
        for server in self.servers:
            cleanup_tasks.append(asyncio.create_task(server.cleanup()))

        if cleanup_tasks:
            try:
                await asyncio.gather(*cleanup_tasks, return_exceptions=True)
            except Exception as e:
                logging.warning(f"Warning during final cleanup: {e}")

    async def process_llm_response(self, llm_response: str) -> str:
        """Process the LLM response and execute tools if needed.

        Args:
            llm_response: The response from the LLM.

        Returns:
            The result of tool execution or the original response.
        """
        import json
        try:
            tool_call = json.loads(llm_response)
            if "tool" in tool_call and "arguments" in tool_call:
                logging.info(f"Executing tool: {tool_call['tool']}")
                logging.info(f"With arguments: {tool_call['arguments']}")

                for server in self.servers:
                    tools = await server.list_tools()
                    if any(tool.name == tool_call["tool"] for tool in tools):
                        try:
                            result = await server.execute_tool(tool_call["tool"], tool_call["arguments"])

                            if isinstance(result, dict) and 'progress' in result:
                                progress = result['progress']
                                total = result['total']
                                logging.info(f"Progress: {progress}/{total} ({(progress/total)*100:.1f}%)")

                            return f"Tool execution result: {result}"
                        except Exception as e:
                            error_msg = f"Error executing tool: {str(e)}"
                            logging.error(error_msg)
                            return error_msg

                return f"No server found with tool: {tool_call['tool']}"
            return llm_response
        except json.JSONDecodeError:
            return llm_response

    async def start(self) -> None:
        """Main chat session handler."""
        try:
            for server in self.servers:
                try:
                    await server.initialize()
                except Exception as e:
                    logging.error(f"Failed to initialize server: {e}")
                    await self.cleanup_servers()
                    return

            all_tools = []
            for server in self.servers:
                tools = await server.list_tools()
                all_tools.extend(tools)

            tools_description = "\n".join([tool.format_for_llm() for tool in all_tools])

            system_message = f"""You are a helpful assistant with access to these tools: 

{tools_description}
Choose the appropriate tool based on the user's question. If no tool is needed, reply directly.

IMPORTANT: When you need to use a tool, you must ONLY respond with the exact JSON object format below, nothing else:
{{
    "tool": "tool-name",
    "arguments": {{
        "argument-name": "value"
    }}
}}

After receiving a tool's response:
1. Transform the raw data into a natural, conversational response
2. Keep responses concise but informative
3. Focus on the most relevant information
4. Use appropriate context from the user's question
5. Avoid simply repeating the raw data

Please use only the tools that are explicitly defined above."""

            messages = [
                {
                    "role": "system",
                    "content": system_message
                }
            ]

            while True:
                try:
                    user_input = input("You: ").strip()
                    if user_input in ['quit', 'exit']:
                        logging.info("\nExiting...")
                        break

                    messages.append({"role": "user", "content": user_input})

                    llm_response = self.llm_client.get_response(messages)
                    logging.info("\nAssistant: %s", llm_response)

                    result = await self.process_llm_response(llm_response)

                    if result != llm_response:
                        messages.append({"role": "assistant", "content": llm_response})
                        messages.append({"role": "system", "content": result})

                        final_response = self.llm_client.get_response(messages)
                        logging.info("\nFinal response: %s", final_response)
                        messages.append({"role": "assistant", "content": final_response})
                    else:
                        messages.append({"role": "assistant", "content": llm_response})

                except KeyboardInterrupt:
                    logging.info("\nExiting...")
                    break

        finally:
            await self.cleanup_servers()


async def main() -> None:
    """Initialize and run the chat session."""
    config = Configuration()
    from pathlib import Path
    cfg_path = Path(__file__).with_name("servers_config.json")
    server_config = config.load_config(str(cfg_path))
    servers = [Server(name, srv_config) for name, srv_config in server_config['mcpServers'].items()]
    llm_client = LLMClient(
        api_key=config.llm_api_key,
        provider=config.llm_provider,
        base_url=config.llm_base_url,
        model=config.llm_model,
        azure_api_version=config.llm_azure_api_version,
    )
    chat_session = ChatSession(servers, llm_client)
    await chat_session.start()

if __name__ == "__main__":
    asyncio.run(main())