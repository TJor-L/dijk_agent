import json
from typing import List, Dict, Any, Optional, Type, TypeVar
from dataclasses import dataclass
from pydantic import BaseModel, Field
from ..core.message import Message
from ..core.memory import ShortTermMemory
from ..tools.base import Tool
from ..llms.base import LLM

T = TypeVar("T", bound=BaseModel)

@dataclass
class PromptSpec:
    role: str = (
        "You are dijk-agent's default agent. You should always remind the user that you are the default agent for debug."
        "You solve tasks step-by-step and can optionally call tools."
    )
    style: str = ""
    output_contract: str = (
        "Always return a structured object with fields:\n"
        " - thought (optional string)\n"
        " - tool_calls (optional list of {name, arguments})\n"
        " - final_answer (string when ready)\n"
        "If tools are needed, first emit tool_calls; after tool_results are provided "
        "back via system message, produce final_answer."
    )

    def render(self, tools_manifest: str, context: str) -> str:
        parts = [self.role, self.style, self.output_contract, tools_manifest]
        if context:
            parts.append(f"Context:\n{context}")
        return "\n\n".join([p for p in parts if p])

class StructuredToolCall(BaseModel):
    name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)

class AgentTurn(BaseModel):
    thought: Optional[str] = None
    tool_calls: List[StructuredToolCall] = Field(default_factory=list)
    final_answer: Optional[str] = None

class BaseAgent:

    def __init__(
        self,
        llm: LLM,
        memory: Optional[ShortTermMemory] = None,
        max_tool_hops: int = 3,
    ):
        self.llm = llm
        self.memory = memory or ShortTermMemory(k=20)
        self.max_tool_hops = max_tool_hops

        self.prompt: PromptSpec = self.default_prompt()
        tool_list: List[Tool] = self.build_tools()
        self.tools: Dict[str, Tool] = {t.name: t for t in tool_list}

    # ---- Implement in Subclass ----
    def default_prompt(self) -> PromptSpec:
        return PromptSpec() # For debug, return a basic prompt

    def build_tools(self) -> List[Tool]:
        return []

    def _tools_manifest(self) -> str:
        if not self.tools:
            return "No external tools are available."
        lines = [
            "You can request tool calls by filling `tool_calls` in the structured output.",
            "Available tools (name, description, JSON input schema):",
        ]
        for name, t in self.tools.items():
            sch = json.dumps(t.input_json_schema(), ensure_ascii=False)
            lines.append(f"- {name}: {t.description}\n  schema: {sch}")
        lines.append("When tool results are provided, use them to continue and produce `final_answer`.")
        return "\n".join(lines)

    def _messages(self, context: str = "") -> List[Dict[str, Any]]:
        sys_text = self.prompt.render(self._tools_manifest(), context)
        msgs: List[Dict[str, Any]] = []
        if sys_text:
            msgs.append({"role": "system", "content": sys_text})
        for m in self.memory.window():
            if m.role != "tool":
                msgs.append(m.to_chat())
        return msgs

    def _append_tool_results_note(self, results: List[Dict[str, Any]]):
        note = {"tool_results": results}
        self.memory.append(Message(role="system", content=json.dumps(note, ensure_ascii=False)))

    def _execute_tool_calls(self, calls: List[StructuredToolCall]) -> List[Dict[str, Any]]:
        results = []
        for c in calls:
            tool = self.tools.get(c.name)
            if not tool:
                results.append({"name": c.name, "error": f"Unknown tool '{c.name}'"})
                continue
            try:
                out = tool.run(**(c.arguments or {}))
                results.append({"name": c.name, "arguments": c.arguments, "output": out})
            except Exception as e:
                results.append({"name": c.name, "arguments": c.arguments, "error": str(e)})
        return results

    def ask(self, user_text: str, context: str = "") -> str:
        self.memory.append(Message(role="user", content=user_text))
        hops = 0
        final_answer: Optional[str] = None

        while hops <= self.max_tool_hops:
            turn: AgentTurn = self.llm.parse(self._messages(context), AgentTurn) # Provide both prompt, context and respond format

            if turn.final_answer and not turn.tool_calls: # If we have a final answer and no tool calls, we're done
                final_answer = turn.final_answer
                self.memory.append(Message(role="assistant", content=final_answer))
                break

            if turn.tool_calls: # If we have tool calls, we need to execute them
                results = self._execute_tool_calls(turn.tool_calls)
                self._append_tool_results_note(results)
                hops += 1
                if turn.final_answer: # If we have both, we prefer the final answer
                    final_answer = turn.final_answer
                continue

            final_answer = turn.final_answer or (turn.thought or "")
            self.memory.append(Message(role="assistant", content=final_answer))
            break

        return final_answer or ""
