from typing import Dict, Type
from .tool_qa import ToolQAPipeline
from .direct_qa import DirectQAPipeline

_REGISTRY: Dict[str, Type] = {
    "tool-qa": ToolQAPipeline,
    "direct-qa": DirectQAPipeline,
}

def create(name: str, **kwargs):
    try:
        cls = _REGISTRY[name]
    except KeyError:
        raise KeyError(f"Unknown pipeline '{name}'. Available: {list(_REGISTRY)}")
    return cls(**kwargs)

def register(name: str, cls: Type):
    if name in _REGISTRY:
        raise KeyError(f"Duplicate pipeline key: {name}")
    _REGISTRY[name] = cls

def available() -> list[str]:
    return list(_REGISTRY)
