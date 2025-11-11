from src.core.tool import Tool


def test_tool_format_for_llm():
    tool = Tool(
        name="search",
        description="Searches the web",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "top_k": {"type": "integer", "description": "Number of results"},
            },
            "required": ["query"],
        },
    )

    out = tool.format_for_llm()
    assert "Tool: search" in out
    assert "Description: Searches the web" in out
    assert "- query:" in out and "(required)" in out
    assert "- top_k:" in out
