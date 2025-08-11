from typing import Dict, Type
from .base import Tool
from .builtin import Calc, Now

_REGISTRY: Dict[str, Type[Tool]] = {
    "calc": Calc,
    "now": Now,
}

def create(name: str, **kwargs) -> Tool:
    try:
        cls = _REGISTRY[name]
    except KeyError:
        raise KeyError(f"Unknown tool '{name}'. Available: {list(_REGISTRY)}")
    return cls(**kwargs)

def register(name: str, cls: Type[Tool]):
    if name in _REGISTRY:
        raise KeyError(f"Duplicate tool key: {name}")
    _REGISTRY[name] = cls

def available() -> list[str]:
    return list(_REGISTRY)
