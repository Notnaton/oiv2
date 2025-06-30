from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, HttpUrl

class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    # plain-text OR multimodal list of parts
    content: Optional[Union[str, List[Dict[Literal["type"], Union[str, HttpUrl]]]]] = None
    # assistant -> tool(s)
    tool_calls: Optional[List[Dict[str, Any]]] = None
    # tool -> assistant
    tool_call_id: Optional[str] = None

class Conversation(BaseModel):
    messages: List[Message] = []
    def save(self, path: Union[str, Path]) -> None:
        Path(path).write_text(self.json(indent=2, ensure_ascii=False))
    @classmethod
    def load(cls, path: Union[str, Path]) -> "Conversation":
        return cls.parse_raw(Path(path).read_text())

import importlib
import inspect
import dspy
from mem0 import Memory

class Interpreter:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.memory = Memory()  # Initialize Mem0 memory (default in-memory store)
        self.tools = []
        self.llm = None
        # Apply configuration if provided (for custom LLM or memory settings)
        if config:
            self.from_config(config)
        else:
            # Default LLM setup: use OpenAI GPT-4 (requires OPENAI_API_KEY in env)
            self.llm = dspy.LM("openai/gpt-4")
            dspy.configure(lm=self.llm)
        # Dynamically load all tool modules from the "tools" folder
        tools_path = Path(__file__).parent / "tools"
        if tools_path.is_dir():
            for tool_file in tools_path.glob("*.py"):
                if tool_file.name.startswith("__"):
                    continue
                module_name = tool_file.stem
                module = importlib.import_module(f"tools.{module_name}")
                # Register each function defined in the module as a tool
                for name, func in inspect.getmembers(module, inspect.isfunction):
                    if inspect.getmodule(func) == module:
                        self.tools.append(func)
        # Initialize a DSPy ReAct agent with all collected tools (tool-augmented reasoning)
        self.react = dspy.ReAct(signature="question->answer", tools=self.tools)  # DSPy ReAct:contentReference[oaicite:3]{index=3}
        # Ensure the ReAct agent is using the configured language model
        if self.llm:
            self.react.set_lm(self.llm)

    def from_config(self, config: Dict[str, Any]) -> None:
        """Configure the interpreter using a config dictionary (for memory/LLM settings)."""
        # Custom memory configuration (e.g., vector or graph store)
        mem_conf = config.get("memory") or config.get("mem0")
        if mem_conf:
            try:
                self.memory = Memory.from_config(config_dict=mem_conf)
            except Exception:
                # Fallback to default memory if config loading fails
                self.memory = Memory()
        # Custom LLM configuration
        model_name = None
        api_key = None
        if "llm" in config:
            llm_conf = config["llm"]
            model_name = llm_conf.get("model") or llm_conf.get("name")
            api_key = llm_conf.get("api_key")
            provider = llm_conf.get("provider")
            if provider and model_name and not model_name.startswith(provider):
                model_name = f"{provider}/{model_name}"
        else:
            # Top-level config keys for model
            model_name = config.get("model")
            api_key = config.get("api_key")
        if model_name:
            self.llm = dspy.LM(model_name, api_key=api_key)
            dspy.configure(lm=self.llm)
        # (No explicit return; modifies self in-place)

    def respond(self, user_message: str) -> str:
        """Generate a response to the user_message using the ReAct agent and Mem0 memory."""
        # Retrieve relevant conversation history from Mem0 for context:contentReference[oaicite:4]{index=4}
        context = ""
        if self.memory:
            results = self.memory.search(query=user_message, user_id="user")
            if results:
                # Mem0 v1.1 returns {'results': [...]} vs v1.0 returns list directly
                results_list = results.get("results", results) if isinstance(results, dict) else results
                # Compile up to a few top memory snippets as additional context
                snippets = []
                for item in results_list[:3]:
                    memory_text = item.get("memory")
                    if memory_text:
                        snippets.append(memory_text)
                if snippets:
                    context = " ".join(snippets)
        # Prepend retrieved context to the user message if any context is found
        prompt = user_message if not context else f"Context: {context}\nUser: {user_message}"
        # Use the DSPy ReAct agent to get a response (it will decide on tool use as needed)
        result = self.react(question=prompt)
        # Extract the assistant's final answer from the ReAct result
        answer = getattr(result, "answer", None)
        if answer is None:  # If result is a Prediction object or dict
            try:
                answer = result["answer"]
            except Exception:
                answer = str(result)
        # Persist the new conversation turn into memory for future queries:contentReference[oaicite:5]{index=5}
        messages = [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": str(answer)}
        ]
        self.memory.add(messages, user_id="user")
        return str(answer)
