from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Type, TypeVar
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

@dataclass
class LLMResponse:
    content: str
    tool_calls: List[Dict[str, Any]]

class LLM:
    def parse(
        self,
        messages: List[Dict[str, Any]],
        schema: Type[T],
    ) -> T:
        
        raise NotImplementedError

    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: str = "auto",
    ) -> LLMResponse:
        raise NotImplementedError
