import asyncio
import logging
from typing import Any, Dict, List

from src.core.server import Server
from src.llm.client import LLMClient


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

            messages: List[Dict[str, str]] = [
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
