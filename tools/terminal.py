import os
import subprocess
import platform
from typing import Optional
from pydantic import BaseModel
from cli_utils import Text
from tools.tools import function_tool
from conversation import Message

@function_tool
def shell(command: str) -> Message:
    __doc__ = "Runs shell commands. Make sure you only run commands suitable for the user's native shell environment."
    import subprocess
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    return Message(
        role="user", 
        message=result.stdout or result.stderr or "Command executed successfully with no output", 
        summary=f'Assistant executed the shell function with the command `{command}`'
    )

@function_tool
def user_input(prompt: Optional[str]) -> Message:
    if prompt:
        print(prompt)
    text = input(Text(text="You: ", color="blue"))
    return Message(role="user", message=text, summary="") 