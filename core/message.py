from dataclasses import dataclass, field
from typing import Literal, Optional, Dict, Any
from datetime import datetime

Role = Literal["system", "user", "assistant", "tool"]

@dataclass
class Message:
    role: Role
    content: str = ""
    tool_call_id: Optional[str] = None  # for tool result messages
    timestamp: datetime = field(default_factory=datetime.utcnow)
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_chat(self) -> Dict[str, Any]:
        d = {"role": self.role, "content": self.content}
        if self.role == "tool" and self.tool_call_id:
            d["tool_call_id"] = self.tool_call_id
        return d
