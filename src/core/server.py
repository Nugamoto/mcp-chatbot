import asyncio
import logging
import os
import shutil
from typing import Any, Dict, List, Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from src.core.tool import Tool
from src.utils.retry import async_retry


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
        """Execute a tool with a retry mechanism (async backoff)."""
        if not self.session:
            raise RuntimeError(f"Server {self.name} not initialized")

        @async_retry(tries=retries, backoff=delay, exceptions=(Exception,))
        async def do_call() -> Any:
            supports_progress = (
                self.capabilities
                and 'progress' in self.capabilities
            )

            if supports_progress:
                logging.info(f"Executing {tool_name} with progress tracking...")
                return await self.session.call_tool(
                    tool_name,
                    arguments,
                    progress_token=f"{tool_name}_execution"
                )
            else:
                logging.info(f"Executing {tool_name}...")
                return await self.session.call_tool(tool_name, arguments)

        try:
            return await do_call()
        except Exception:
            logging.error("Max retries reached for tool %s on server %s.", tool_name, self.name)
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
