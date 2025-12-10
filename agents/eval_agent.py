# agents/eval_agent.py

from typing import Dict, Any

import torch
import torch.nn.functional as F

from .base import BaseAgent
from . import register_agent

@register_agent("eval_agent")
class EvalAgent(BaseAgent):
    """
    여러 task(stream)을 순회하며, 각 task에 대해 model을 평가하는 agent
    """
    
    def __init__(self, cfg, model, stream):
        super().__init__(cfg, model=model, stream=stream)
        self.device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        
    @torch.no_grad()
    def _eval_single_task(self, task_name, loader) -> Dict[str, Any]:
        self.model.to(self.device)
        self.model.eval()
        
        correct = 0
        total = 0
        
        for images, labels in loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            logits = self.model(images)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
        acc = correct / total if total > 0 else 0.0
        return {"acc": acc, "num_samples": total}
    
    def run(self) -> Dict[str, Any]:
        report: Dict[str, Any] = {}
        for task_name, loader in self.stream:
            task_report = self._eval_single_task(task_name, loader)
            report[task_name] = task_report
            print(f"[EvalAgent] {task_name}: acc={task_report['acc']:.4f}, n={task_report['num_samples']}")
        return report