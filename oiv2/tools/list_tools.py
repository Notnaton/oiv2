from oiv2.tools.tools import function_tool, ToolRegistry
from ..conversation import Message

@function_tool
def list_tools() -> Message:
    """List all available tools"""
    return Message(
        role="tool",
        message=f"Available tools: \n{ToolRegistry.get_dspy_tools()}"
    )