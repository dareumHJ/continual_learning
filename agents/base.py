# methods/base.py

from abc import ABC, abstractmethod

class BaseAgent(ABC):
    def __init__(self, cfg, model, stream, test_stream=None):
        self.cfg = cfg
        self.model = model
        self.stream = stream
        self.test_stream = test_stream
    
    @abstractmethod
    def run(self):
        report = {}
        for task_idx, (task_name, loader) in enumerate(self.stream):
            task_report = self.observe_task(task_name, loader)
            report[task_name] = task_report
        return report