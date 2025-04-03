from tools.tools import function_tool
from conversation import Message
from typing import Optional
from cli_utils import Text

@function_tool
def user_input(prompt: Optional[str]) -> Message:
    if prompt:
        print(prompt)
    text = input(Text(text="You: ", color="blue"))
    return Message(role="user", message=text, summary="") 