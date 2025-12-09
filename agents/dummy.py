# methods/dummy.py

from .base import BaseAgent
from . import register_agent

@register_agent("dummy")
class DummyAgent(BaseAgent):
    def run(self):
        # stream에서 (task, dataloader) 같은 걸 뽑아 쓴다고 가정
        report = {}
        for task_name, loader in self.stream:
            report[task_name] = {"acc": 0.0, "num_samples": len(loader.dataset)}
        print(f"[DummyMethod] report: {report}")
        return report