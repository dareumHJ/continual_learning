# task_pool/__init__.py

from .base import BaseTaskPool

def load_task_pool(cfg):
    # 지금은 타입 하나만 지원하는 더미
    return BaseTaskPool(cfg)