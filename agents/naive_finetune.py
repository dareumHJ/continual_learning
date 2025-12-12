from typing import Dict, Any

import torch
import torch.nn.functional as F
from torch.optim import Adam

from .base import BaseAgent
from . import register_agent

@register_agent("naive_finetune")
class NaiveFinetuneAgent(BaseAgent):
    def __init__(self, cfg, model, stream):
        super().__init__(cfg, model, stream)
        self.device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.lr = cfg.get("lr", 0.0001)
        self.epochs = cfg.get("epochs", 1)
        self._optim = Adam(self.model.parameters(), lr=self.lr)
        self.model.to(self.device)
    
    def _train_one_task(self, loader):
        self.model.train()
        for _ in range(self.epochs):
            for images, labels in loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                logits = self.model(images)
                loss = F.cross_entropy(logits, labels)
                
                self._optim.zero_grad()
                loss.backward()
                self._optim.step()
    
    @torch.no_grad()
    def _eval_one_task(self, loader) -> Dict[str, Any]:
        self.model.eval()
        correct, total = 0, 0
        for images, labels in loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            logits = self.model(images)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        acc = correct / total if total > 0 else 0.0
        return {"acc": acc, "num_samples": total}
    
    def observe_task(self, task_name, loader) -> Dict[str, Any]:
        # 1) 이 태스크에서 학습
        self._train_one_task(loader)
        # 2) 같은 loader로 간단히 평가 (나중에 별도 test stream 써도 됨)
        metrics = self._eval_one_task(loader)
        print(f"[NaiveFinetune] {task_name}: acc={metrics['acc']:.4f}, n={metrics['num_samples']}")
        return metrics