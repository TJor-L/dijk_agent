import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TypeVar

from pydantic import BaseModel, Field
from pydantic import ConfigDict

from ..core.message import Message
from ..core.memory import ShortTermMemory
from ..llms.base import LLM

T = TypeVar("T", bound=BaseModel)


@dataclass
class PromptSpec:
    role: str = "You are dijk-agent. You solve tasks step-by-step."
    style: str = ""
    output_contract: str = (
        "Return a structured object with fields:\n"
        " - thought (optional string)\n"
        " - final_answer (string)\n"
    )
    def render(self, context: str) -> str:
        parts = [self.role, self.style, self.output_contract]
        if context:
            parts.append(f"Context:\n{context}")
        return "\n\n".join([p for p in parts if p])


class DirectTurn(BaseModel):
    thought: Optional[str] = None
    final_answer: Optional[str] = None
    model_config = ConfigDict(extra="forbid")


class BaseAgent:
    TURN_MODEL = DirectTurn
    def __init__(
        self,
        llm: LLM,
        memory: Optional[ShortTermMemory] = None,
    ):
        self.llm = llm
        self.memory = memory or ShortTermMemory(k=20)
        self.prompt: PromptSpec = self.default_prompt()
    def default_prompt(self) -> PromptSpec:
        return PromptSpec()
    def _messages(self, context: str = "") -> List[Dict[str, Any]]:
        sys_text = self.prompt.render(context)
        msgs: List[Dict[str, Any]] = []
        if sys_text:
            msgs.append({"role": "system", "content": sys_text})
        for m in self.memory.window():
            if m.role != "tool":
                msgs.append(m.to_chat())
        return msgs
    def ask(self, user_text: str, context: str = "") -> str:
        self.memory.append(Message(role="user", content=user_text))
        turn: DirectTurn = self.llm.parse(self._messages(context), self.TURN_MODEL)  # type: ignore
        ans = (turn.final_answer or turn.thought or "").strip()
        self.memory.append(Message(role="assistant", content=ans))
        return ans
