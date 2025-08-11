from .base import BaseAgent, PromptSpec
from ..tools import registry as toolreg

class SimpleAgent(BaseAgent):

    def default_prompt(self) -> PromptSpec:
        base = super().default_prompt()
        return PromptSpec(
            role="You are dijk-agent in an academic lab setting.",
            style="Reply in Chinese if user speaks Chinese. Be concise and precise.",
            output_contract=base.output_contract,
        )

    def build_tools(self):
        return [toolreg.create("calc"), toolreg.create("now")]
