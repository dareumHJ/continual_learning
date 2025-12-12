# methods/__init__.py

from .base import BaseAgent

# 여기에 method name -> class 매핑 추가
_AGENT_REGISTRY = {}

def register_agent(name):
    def decorator(cls):
        _AGENT_REGISTRY[name] = cls
        return cls
    return decorator

def load_agent(cfg, model, stream) -> BaseAgent:
    name = cfg["name"]
    if name not in _AGENT_REGISTRY:
        raise ValueError(f"Unknown agent: {name}")
    cls = _AGENT_REGISTRY[name]
    return cls(cfg, model=model, stream=stream)

from .dummy import DummyAgent
from .eval_agent import EvalAgent
from .naive_finetune import NaiveFinetuneAgent