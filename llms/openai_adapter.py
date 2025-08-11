from typing import List, Dict, Any, Optional, Type, TypeVar
from pydantic import BaseModel
from .base import LLM, LLMResponse
from openai import OpenAI

T = TypeVar("T", bound=BaseModel)

class OpenAIAdapter(LLM):

    def __init__(self, model: str = "gpt-4o-2024-08-06", api_key: Optional[str] = None, **kwargs):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.extra = kwargs  # temperature/top_p/max_tokens/…

    def parse(self, messages: List[Dict[str, Any]], schema: Type[T]) -> T:
        resp = self.client.responses.parse(
            model=self.model,
            input=messages,
            text_format=schema,
            **self.extra
        )
        return resp.output_parsed

    def chat(self, messages, tools=None, tool_choice="auto") -> LLMResponse:
        out = self.client.responses.create(model=self.model, input=messages, **self.extra)
        return LLMResponse(content=out.output_text or "", tool_calls=[])
