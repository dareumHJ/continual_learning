# task_pool/clip_classification.py

from typing import Dict, Any

import torch
import torch.nn.functional as F

from .base import BaseTaskPool
from datasets import get_image_classification_dataloader

class SimpleClassificationTaskPool(BaseTaskPool):
    """
    지금은 single dataset (e.g., MNIST)만 지원하는 간단 버전.
    나중에 TALL20에 맞춰 tasks list 확장 예정.
    """
    
    def __init__(self, cfg):
        super().__init__(cfg)
        # cfg example:
        # taskpool:
        #   type: simple_classification
        #   dataset: MNIST
        #   split: test
        #   batch_size: 128
        #   num_workers: 4
        self.dataset_name = cfg.get("dataset", "mnist")
        self.split = cfg.get("split", "test")
        self.batch_size = cfg.get("batch_size", 128)
        self.num_workers = cfg.get("num_workers", 4)
        self.device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        
    @torch.no_grad()
    def evaluate(self, model) -> Dict[str, Any]:
        model.eval()
        model.to(self.device)
        
        loader, num_classes = get_image_classification_dataloader(
            name=self.dataset_name,
            split=self.split,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
        
        correct = 0
        total = 0
        # 여기 부분은 CLIP이나 backbone+head 구조를 쓰면, 적절히 수정 필요
        # 지금은 images -> logits를 바로 뱉는 classificator를 가정
        for images, labels in loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            logits = model(images) # shape: (B, num_classes)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        
        acc = correct / total if total > 0 else 0.0
        return {self.dataset_name: {"acc": acc, "num_classes": num_classes, "total": total}}