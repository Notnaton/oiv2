import asyncio
import locale
from typing import Dict, List, Optional
from pydantic import BaseModel
from litellm import acompletion

class Message(BaseModel):
    role: str
    message: str
    summary: str = ""
    tool_call_id: Optional[str] = None
    name: Optional[str] = None

    async def generate_summary(self) -> str:
        if self.message == "" or self.role.lower() == "system": return None
        result = await acompletion(
            model="openai/local",
            base_url="http://localhost:1234/v1",
            api_key="dummy",
            messages=[
                {"role": "system", "content": f"You are a helpful assistant that summarizes messages for an AI. Summary must be a single sentence and not longer than 10 words or the original message. Do not answer questions, only summarize them. The native locale is {locale.getlocale()[0]}"},
                {"role": self.role, "content": self.message},
            ],
            max_tokens=100,
        )
        return result.choices[0].message.content

    def model_post_init(self, __context) -> None: asyncio.create_task(self._update_summary())
    async def _update_summary(self): self.summary = await self.generate_summary()

class Conversation(BaseModel):
    messages: List[Message]
    max_recent: int

    def get_messages(self) -> List[Dict]:
        return [{
            "role": msg.role, 
            "content": msg.message if i < self.max_recent or msg.role.lower() == "system" else msg.summary,
            "summary": msg.summary if msg.summary else None,
            "tool_call_id": msg.tool_call_id if msg.tool_call_id else None,
            "name": msg.name if msg.name else None,}
            for i, msg in enumerate(self.messages[::-1])][::-1]

    def save(self, filename: str):
        import json
        with open(filename, "w") as f: json.dump([msg.model_dump() for msg in self.messages], f)

    def load(self, filename: str):
        import json
        with open(filename, "r") as f: self.messages = [Message(**msg) for msg in json.load(f)] 