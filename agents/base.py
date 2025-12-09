# methods/base.py

from abc import ABC, abstractmethod

class BaseAgent(ABC):
    def __init__(self, cfg, model, stream):
        self.cfg = cfg
        self.model = model
        self.stream = stream
        
    
    @abstractmethod
    def run(self):
        """Run the method on model_pool and optionally evaluate on task_pool"""
        pass