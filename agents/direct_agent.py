from .base import BaseAgent, PromptSpec

class DirectAgent(BaseAgent):
    def default_prompt(self) -> PromptSpec:
        base = super().default_prompt()
        return PromptSpec(
            role="You are dijk-agent. Provide concise, precise answers.",
            style="Reply in Chinese if the user speaks Chinese.",
            output_contract=base.output_contract,
        )
