from typing import Dict, Type
from .base import BaseAgent
from .simple_agent import SimpleAgent

_REGISTRY: Dict[str, Type[BaseAgent]] = {
    "simple": SimpleAgent,
}

def create(name: str, **kwargs) -> BaseAgent:
    try:
        cls = _REGISTRY[name]
    except KeyError:
        raise KeyError(f"Unknown agent '{name}'. Available: {list(_REGISTRY)}")
    return cls(**kwargs)

def register(name: str, cls: Type[BaseAgent]):
    if name in _REGISTRY:
        raise KeyError(f"Duplicate agent key: {name}")
    _REGISTRY[name] = cls

def available() -> list[str]:
    return list(_REGISTRY)
