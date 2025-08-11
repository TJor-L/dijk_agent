import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TypeVar

from pydantic import BaseModel, Field
from pydantic import ConfigDict

from ..core.message import Message
from ..core.memory import ShortTermMemory
from ..llms.base import LLM
from ..tools.base import Tool
from ..tools import registry as toolreg

from .base import BaseAgent

T = TypeVar("T", bound=BaseModel)


@dataclass
class PromptSpec:
    role: str = (
        "You are dijk-agent. "
        "You solve tasks step-by-step and can optionally call tools."
    )
    style: str = ""
    output_contract: str = (
        "Always return a structured object with fields:\n"
        " - thought (optional string)\n"
        " - tool_calls (optional list of {name, arguments_json})\n"
        " - final_answer (string when ready)\n"
        "If tools are needed, first emit tool_calls; after tool_results are provided "
        "back via system message, produce final_answer.\n"
        "For each tool call, `arguments_json` MUST be a compact JSON string "
        "that validates against the tool's input schema."
    )

    def render(self, tools_manifest: str, context: str) -> str:
        parts = [self.role, self.style, self.output_contract, tools_manifest]
        if context:
            parts.append(f"Context:\n{context}")
        return "\n\n".join([p for p in parts if p])


class StructuredToolCall(BaseModel):
    name: str
    arguments_json: str = Field(
        description="Compact JSON string for tool arguments (must pass the tool input schema)."
    )
    model_config = ConfigDict(extra="forbid")


class AgentTurn(BaseModel):
    thought: Optional[str] = None
    tool_calls: List[StructuredToolCall] = Field(default_factory=list)
    final_answer: Optional[str] = None
    model_config = ConfigDict(extra="forbid")


class ToolAgent(BaseAgent):
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

    def default_prompt(self) -> PromptSpec:
        return PromptSpec()

    def build_tools(self) -> List[Tool]:
        return [toolreg.create(name) for name in toolreg.available()]

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
        lines.append(
            "When tool results are provided, use them to continue and produce `final_answer`."
        )
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
        self.memory.append(
            Message(role="system", content=json.dumps(note, ensure_ascii=False))
        )

    def _execute_tool_calls(
        self, calls: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        results = []
        for c in calls:
            tool = self.tools.get(c.name) if hasattr(c, "name") else self.tools.get(c.get("name"))
            if not tool:
                name = c.name if hasattr(c, "name") else c.get("name")
                results.append({"name": name, "error": f"Unknown tool '{name}'"})
                continue
            try:
                arguments_json = c.arguments_json if hasattr(c, "arguments_json") else c.get("arguments_json", "")
                args = json.loads(arguments_json) if arguments_json else {}
                if not isinstance(args, dict):
                    raise ValueError("arguments_json must decode to an object")
            except Exception as e:
                name = c.name if hasattr(c, "name") else c.get("name")
                results.append({"name": name, "error": f"Invalid arguments_json: {e}"})
                continue
            try:
                out = tool.run(**args)
                results.append({"name": tool.name, "arguments": args, "output": out})
            except Exception as e:
                results.append({"name": tool.name, "arguments": args, "error": str(e)})
        return results

    def ask(self, user_text: str, context: str = "") -> str:
        self.memory.append(Message(role="user", content=user_text))
        hops = 0
        final_answer: Optional[str] = None
        while hops <= self.max_tool_hops:
            turn: AgentTurn = self.llm.parse(self._messages(context), AgentTurn)
            if turn.final_answer and not turn.tool_calls:
                final_answer = turn.final_answer
                self.memory.append(Message(role="assistant", content=final_answer))
                break
            if turn.tool_calls:
                results = self._execute_tool_calls(turn.tool_calls)
                self._append_tool_results_note(results)
                hops += 1
                if turn.final_answer:
                    final_answer = turn.final_answer
                continue
            final_answer = turn.final_answer or (turn.thought or "")
            self.memory.append(Message(role="assistant", content=final_answer))
            break
        return final_answer or ""
