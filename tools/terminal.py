import os
from typing import Optional
from cli_utils import Text
from tools.tools import function_tool
from conversation import Message

@function_tool
def shell(command: str) -> Message:
    __doc__ = "Runs shell commands. Make sure you only run commands suitable for the user's native shell environment."
    import subprocess
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    output = result.stdout or result.stderr or "Command executed successfully with no output"
    return Message(
        role="tool", 
        message=f"{os.getcwd()}: {command}\n{output}",
        summary=f'Assistant executed the shell function with the command `{command}`'
    )