# task_pool/__init__.py

from .base import BaseTaskPool
from .clip_classification import SimpleClassificationTaskPool

def load_task_pool(cfg):
    tp_type = cfg.get("type", "simple_classification")
    
    if tp_type == "simple_classification":
        return SimpleClassificationTaskPool(cfg)
    else:
        raise ValueError(f"Unknown task pool type: {tp_type}")    