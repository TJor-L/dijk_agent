from typing import Dict, Type
from .base import LLM
from .openai_adapter import OpenAIAdapter

_REGISTRY: Dict[str, Type[LLM]] = {
    "openai": OpenAIAdapter,
}

def create(name: str, **kwargs) -> LLM:
    try:
        cls = _REGISTRY[name]
    except KeyError:
        raise KeyError(f"Unknown LLM '{name}'. Available: {list(_REGISTRY)}")
    return cls(**kwargs)

def register(name: str, cls: Type[LLM]):
    if name in _REGISTRY:
        raise KeyError(f"Duplicate LLM key: {name}")
    _REGISTRY[name] = cls

def available() -> list[str]:
    return list(_REGISTRY)
