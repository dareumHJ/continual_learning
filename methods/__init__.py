# methods/__init__.py

from .base import BaseMethod

# 여기에 method name -> class 매핑 추가
_METHOD_REGISTRY = {}

def register_method(name):
    def decorator(cls):
        _METHOD_REGISTRY[name] = cls
        return cls
    return decorator

def load_method(cfg):
    name = cfg["name"]
    if name not in _METHOD_REGISTRY:
        raise ValueError(f"Unknown method: {name}")
    cls = _METHOD_REGISTRY[name]
    return cls(cfg)

from . import dummy