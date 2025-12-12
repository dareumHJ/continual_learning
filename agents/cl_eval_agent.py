from typing import Dict, Any, List, Tuple

import torch
import torch.nn.functional as F

from .base import BaseAgent
from . import register_agent
from torch.optim import Adam

@register_agent("cl_eval")
class ContinualEvalAgent(BaseAgent):
    """
    각 task를 순차적으로 학습하면서,
    각 step마다 모든 test_tasks에 대해 accuracy를 측정.
    
    report 형식 example:
    {
        "acc_matrix": { 
            "0": {"mnist": 0.95, "fashion_mnist": 0.10, "kmnist": 0.11},
            "1": {"mnist": 0.94, "fashion_mnist": 0.85, "kmnist": 0.12},
            "2": {"mnist": 0.93, "fashion_mnist": 0.83, "kmnist": 0.78},
        }
    }
    """
    
    def __init__(self, cfg, model, stream, test_stream=None):
        super().__init__(cfg, model, stream, test_stream)
        self.device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.lr = cfg.get("lr", 1e-4)
        self.epochs = cfg.get("epochs", 1)
        self.model.to(self.device)
        self.optim = Adam(self.model.parameters(), lr=self.lr)
        
        if self.test_stream is None:
            raise ValueError("cl_eval agent requires test_stream (data.test_tasks in config).")
        
        # test_stream은 list[(task_name, loader)] 형태로 저장(가정)
        self.test_tasks: List[Tuple(str, Any)] = self.test_stream
        
    def _train_one_task(self, loader):
        self.model.train()
        for _ in range(self.epochs):
            for images, labels in loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                logits = self.model(images)
                loss = F.cross_entropy(logits, labels)
                
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                
    @torch.no_grad()
    def _eval_on_all_tasks(self) -> Dict[str, float]:
        self.model.eval()
        results: Dict[str, float] = {}
        
        for task_name, loader in self.test_tasks:
            correct, total = 0, 0
            for images, labels in loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                logits = self.model(images)
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
            acc = correct / total if total > 0 else 0.0
            results[task_name] = acc
        
        return results
    
    def observe_task(self, task_data):
        pass
    
    def run(self) -> Dict[str, Any]:
        acc_matrix: Dict[str, Dict[str, float]] = {}
        
        for step, (train_task_name, train_loader) in enumerate(self.stream):
            # 1) 현재 task에서 학습 후...
            print(f"[CL-Eval] Training on task {step}: {train_task_name}")
            self._train_one_task(train_loader)
            
            # 2) 모든 test_tasks에 대해 평가
            accs = self._eval_on_all_tasks()
            acc_matrix[str(step)] = accs
            print(f"[CL-Eval] After task {step}, accs: {accs}")
            
        return {"acc_matrix": acc_matrix}