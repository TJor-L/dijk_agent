from typing import Dict, Any, Optional
from ..llms import registry as llms
from ..core.memory import ShortTermMemory
from ..agents.direct_agent import DirectAgent

class DirectQAPipeline:
    def __init__(
        self,
        model: str = "gpt-4o-2024-08-06",
        temperature: float = 0.0,
        memory_k: int = 10,
        memory: Optional[ShortTermMemory] = None,
    ):
        llm = llms.create("openai", model=model, temperature=temperature)
        self.agent = DirectAgent(
            llm=llm,
            memory= memory if memory else ShortTermMemory(k=memory_k),
        )

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        q = state.get("query", "")
        ctx = state.get("context", "")
        ans = self.agent.ask(q, context=ctx)
        state["answer"] = ans
        return state
