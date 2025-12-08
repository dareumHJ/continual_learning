# methods/base.py

from abc import ABC, abstractmethod

class BaseMethod(ABC):
    def __init__(self, cfg):
        self.cfg = cfg
    
    @abstractmethod
    def run(self, model_pool, task_pool):
        """Run the method on model_pool and optionally evaluate on task_pool"""
        pass