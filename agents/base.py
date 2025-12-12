# methods/base.py

from abc import ABC, abstractmethod

class BaseAgent(ABC):
    def __init__(self, cfg, model, stream):
        self.cfg = cfg
        self.model = model
        self.stream = stream
        
    
    @abstractmethod
    def observe_task(self, task_name, loader):
        """한 태스크의 train/eval... 어떻게 구현?"""
        pass
    
    def run(self):
        report = {}
        for task_idx, (task_name, loader) in enumerate(self.stream):
            task_report = self.observe_task(task_name, loader)
            report[task_name] = task_report
        return report