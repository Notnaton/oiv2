from __future__ import annotations
import dspy
import asyncio
from dspy.adapters import ChatAdapter
from dspy.streaming import StreamListener, StreamResponse
from dspy import streamify

from oiv2.tools.tools import ToolRegistry   # our registry with DSPy wrappers


class Interpreter:
    """
    Minimal CLI interpreter:
    • Streams tokens live to stdout.
    • Every @function_tool is automatically exposed to the model.
    """

    def __init__(self):
        # 1️⃣  Collect tools for the agent
        self.tools = ToolRegistry.get_dspy_tools()

        # 2️⃣  Language model (leave `stream` unset; streamify handles it)
        self.llm = dspy.LM(
            model="lm_studio/local",
            base_url="http://localhost:1234/v1",
            api_key="lm-studio",
            model_type="chat",
            temperature=0.0,
        )
        dspy.configure(lm=self.llm, )

        # 3️⃣  ReAct agent with access to tools
        self.react = dspy.ReAct("question->answer", tools=self.tools)

        # 4️⃣  Enable streaming
        listener = StreamListener(signature_field_name="answer")
        self.streamed_react = streamify(
            self.react,
            stream_listeners=[listener],
            async_streaming=False,   # sync generator for simple REPL
        )

    # ---------------------------------------------------------------- #
    def respond(self, user_input: str) -> str:
        """Send *user_input*, stream tokens, return the final answer."""
        gen = self.streamed_react(question=user_input)
        answer = ""

        for chunk in gen:
            if isinstance(chunk, StreamResponse):
                print(chunk.chunk, end="", flush=True)
            elif isinstance(chunk, dspy.Prediction):
                answer = chunk.answer
        return answer
