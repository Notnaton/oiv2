"""
Tool definition + registry that works out-of-the-box with DSPy.

• Decorate any function with @function_tool
• The original OpenAI-style JSON schema is kept (ToolRegistry._tools)
• A matching dspy.Tool wrapper is created automatically (ToolRegistry._dspy_tools)
"""

from __future__ import annotations
from typing import Dict, Any, Callable, Type, List, get_type_hints
import inspect
from pydantic import BaseModel
import dspy

# -------------------------------------------------------------------- #
#  Internal representation of a tool                                   #
# -------------------------------------------------------------------- #

class Tool(BaseModel):
    type: str = "function"
    function: Dict[str, Any]      # OpenAI / LiteLLM JSON schema spec
    func: Callable                # actual Python callable

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)


# -------------------------------------------------------------------- #
#  Registry                                                            #
# -------------------------------------------------------------------- #

class ToolRegistry:
    _tools: Dict[str, Tool] = {}          # our own JSON-schema tools
    _dspy_tools: List[dspy.Tool] = []     # thin wrappers for DSPy

    _TYPE_MAP: Dict[Type, str] = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
    }

    # ---------------- decorator ------------------------------------- #
    @classmethod
    def register(cls, func: Callable) -> Tool:
        """Decorator: turn *func* into a registry entry **and** a dspy.Tool."""
        sig = inspect.signature(func)
        hints = get_type_hints(func)

        # Build the OpenAI-style parameter schema
        properties = {}
        required = []
        for name, param in sig.parameters.items():
            if name in ("self", "cls"):
                continue
            if param.default is inspect.Parameter.empty:
                required.append(name)
            properties[name] = {
                "type": cls._TYPE_MAP.get(hints.get(name, str), "string"),
                "description": f"{name} parameter",
            }

        schema = {
            "name": func.__name__,
            "description": func.__doc__ or f"Function {func.__name__}",
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }

        # Our canonical tool object
        tool_obj = Tool(function=schema, func=func)

        # ------------ DSPy wrapper (single line does all the magic) ----
        dsp_tool = dspy.Tool(
            func,                                   # callable
            name=schema["name"],
            desc=schema["description"],
            args=schema["parameters"]["properties"],  # arg -> json schema
        )

        # Register
        cls._tools[schema["name"]] = tool_obj
        cls._dspy_tools.append(dsp_tool)
        return tool_obj

    # ---------------- helpers --------------------------------------- #
    @classmethod
    def get(cls, name: str) -> Tool | None:
        return cls._tools.get(name)

    @classmethod
    def list_schema(cls) -> Dict[str, Any]:
        """Return all tool JSON schemas (for OpenAI-style function calling)."""
        return {n: t.function for n, t in cls._tools.items()}

    @classmethod
    def get_dspy_tools(cls) -> List[dspy.Tool]:
        """Return tools ready to hand to DSPy ReAct."""
        return cls._dspy_tools


# Convenience alias so users can simply write `@function_tool`
function_tool = ToolRegistry.register
