from typing import Dict, Any
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from .base import BaseAgent
from . import register_agent

@register_agent("offline_finetune")
class OfflineFinetuneAgent(BaseAgent):
    """
    각 task를 독립적으로 학습 및 task별 head(또는 전체 모델) 체크포인트를 저장
    - backbone은 각 task마다 초기 상태로 다시 로드할지, 계속 이어서 쓸지는 선택 가능
    - 지금은 '각 task마다 같은 초기 모델에서 시작' 방식으로 구현
    """
    
    def __init__(self, cfg, model, stream, test_stream=None):
        super().__init__(cfg, model, stream, test_stream)
        self.device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.lr = cfg.get("lr", 1e-4)
        self.epochs = cfg.get("epochs", 1)
        self.ckpt_dir = Path(cfg.get("ckpt_dir", "outputs/offline_ckpts"))
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        
        self.model.to(self.device)
        
        self.init_state = {
            "backbone": self.model.backbone.state_dict(),
            "head": self.model.head.state_dict(),
        }
        
    def _reset_model_to_init(self):
        """각 task마다 동일한 초기 상태에서 학습"""
        self.model.backbone.load_state_dict(self.init_state["backbone"])
        
    def _train_one_task(self, loader):
        optim = Adam(self.model.parameters(), lr=self.lr)
        self.model.train()
        for _ in range(self.epochs):
            for images, labels in loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                logits = self.model(images)
                loss = F.cross_entropy(logits, labels)
                
                optim.zero_grad()
                loss.backward()
                optim.step()
                
    @torch.no_grad()
    def _quick_eval(self, loader) -> float:
        """(Optional) 학습 끝난 후 train loader에서 대략적인 acc만"""
        self.model.eval()
        correct, total = 0, 0
        for images, labels in loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            logits = self.model(images)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        return correct / total if total > 0 else 0.0
    
    def run(self) -> Dict[str, Any]:
        summaries: Dict[str, Any] = {}
        
        for task_idx, (task_name, (loader, num_classes)) in enumerate(self.stream):
            print(f"[OfflineFinetune] Task {task_idx}: {task_name}")
            
            in_features = self.model.head.classifier.in_features
            
            self.model.head.classifier= nn.Linear(in_features, num_classes).to(self.device)
            print(f"[OfflineFinetune] Replaced head with output dim: {num_classes}")
            
            # 1) model initialization
            self._reset_model_to_init()
            
            # 2) train
            self._train_one_task(loader)
            
            # 3) (Optional) quick eval
            acc = self._quick_eval(loader)
            print(f"[OfflineFinetune] {task_name}: train_acc={acc:.4f}")
            
            # 4) save checkpoint
            init_backbone_ckpt = self.ckpt_dir / "backbone_init.pt"
            if not init_backbone_ckpt.exists():
                torch.save(self.init_state["backbone"], init_backbone_ckpt)
                print(f"[OfflineFinetune] Saved init backbone to {init_backbone_ckpt}")
            
            backbone_ckpt = self.ckpt_dir / f"{task_name}/backbone.pt"
            head_ckpt = self.ckpt_dir / f"{task_name}/head.pt"
            
            torch.save(self.model.backbone.state_dict(), backbone_ckpt)
            torch.save(self.model.head.state_dict(), head_ckpt)
                
            print(f"[OfflineFinetune] Saved backbone for {task_name} to {backbone_ckpt}")
            print(f"[OfflineFinetune] Saved head for {task_name} to {head_ckpt}")

            summaries[task_name] = {
                "train_acc": acc,
                "init_backbone_ckpt": str(init_backbone_ckpt),
                "backbone_ckpt": str(backbone_ckpt),
                "head_ckpt": str(head_ckpt),
            }

        return {"tasks": summaries}