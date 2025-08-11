from typing import Dict, Any
from ..llms import registry as llms
from ..core.memory import ShortTermMemory
from ..agents.simple_agent import SimpleAgent

class SimpleQAPipeline:
    def __init__(
        self,
        model: str = "gpt-4o-2024-08-06",
        temperature: float = 0.0,
        memory_k: int = 10,
        max_tool_hops: int = 3,
    ):
        llm = llms.create("openai", model=model, temperature=temperature)
        self.agent = SimpleAgent(
            llm=llm,
            memory=ShortTermMemory(k=memory_k),
            max_tool_hops=max_tool_hops,
        )

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        q = state.get("query", "")
        ctx = state.get("context", "")
        ans = self.agent.ask(q, context=ctx)
        state["answer"] = ans
        return state
