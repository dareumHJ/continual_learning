# agents/merge_eval

from typing import Dict, Any, List, Tuple
from pathlib import Path

import torch
import torch.nn.functional as F

from .base import BaseAgent
from . import register_agent

def _load_backbone_tasks(ckpt_dir: Path, task_names) -> Dict[str, dict]:
    """
    ckpt_dir에서 backbone.pt들을 찾아서
    {task_name: state_dict}로 반환.
    """
    ckpt_dir = Path(ckpt_dir)
    backbones: Dict[str, dict] = {}
    
    for t in task_names:
        path = ckpt_dir / f"{t}/backbone.pt"
        if not path.exists():
            raise FileNotFoundError(f"Backbone checkpoint not found for task {t} at {path}")
        sd = torch.load(path, map_location="cpu")
        backbones[t] = sd
    
    return backbones

def _simple_average_backbones(backbones: Dict[str, dict]) -> dict:
    """
    여러 backbone state_dict를 weight-wise 평균해서 하나의 merged state_dict 반환.
    (모든 backbone의 key/shape가 동일하다고 가정)
    """
    states = list(backbones.values())
    merged: Dict[str, torch.Tensor] = {}
    
    for key in states[0].keys():
        tensors = [sd[key] for sd in states]
        merged[key] = sum(tensors) / len(tensors)
    
    return merged

@register_agent("merge_eval")
class MergeEvalAgent(BaseAgent):
    """
    backbone과 여러 checkpoint 로드한 후,
    merge_method에 따라 하나의 merged backbone 만든 후 CL eval 수행하는 agent
    
    - current version -> simple_average only
    - eval loop는 cl_eval과 거의 동일, but 학습은 하지 않음.
    """
    
    def __init__(self, cfg, model, stream, test_stream=None):
        super().__init__(cfg, model, stream, test_stream)
        self.device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.ckpt_dir = Path(cfg.get("ckpt_dir", "data/clip-vit-b32/hf_backbones"))
        self.merge_method = cfg.get("merge_method", "simple_average")
        
        if self.test_stream is None:
            raise ValueError("merge_eval agent requires test_stream (data.test_tasks in config).")
        
        self.test_tasks: List[Tuple[str, Any, int]] = self.test_stream
        self.model.to(self.device)
                
    @torch.no_grad()
    def _eval_on_all_tasks(self) -> Dict[str, float]:
        self.model.eval()
        results: Dict[str, float] = {}
        
        for task_name, (loader, num_classes) in self.test_tasks:
            correct, total = 0, 0
            for batch in loader:
                if isinstance(batch, dict):
                    images = batch.get("image").to(self.device)
                    labels = batch.get("label").to(self.device)
                else:
                    images, labels = batch
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                
                logits = self.model(images)
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
            acc = correct / total if total > 0 else 0.0
            results[task_name] = acc
        
        return results
    
    
    def run(self) -> Dict[str, Any]:
        # 1) config에서 평가할 task 목록 추출 (test_tasks 기준)
        task_names = [t[0] for t in self.test_tasks] # (task_name, loader, num_classes)
        
        # 2) 해당 task들에 대한 backbone 로드
        backbones = _load_backbone_tasks(self.ckpt_dir, task_names)
        print(f"[MergeEval] Using backbones for tasks: {sorted(backbones.keys())}")
        
        if self.merge_method == "simple_average":
            merged_sd = _simple_average_backbones(backbones)
            print("[MergeEval] Applied simple_average over backbones")
        else:
            raise ValueError(f"Unknown merge_method: {self.merge_method}")
        
        new_merged_sd = {}
        for k, v in merged_sd.items():
            # "vision_model." 로 시작하는 키만 가져와서 앞부분을 자름
            if k.startswith("vision_model."):
                new_key = k.replace("vision_model.", "")
                new_merged_sd[new_key] = v
        
        # 4) merged backbone 로드
        self.model.backbone.clip.vision_model.load_state_dict(new_merged_sd)
        
        # 4) test_tasks 평가
        accs = self._eval_on_all_tasks()
        print(f"[MergeEval] Accs with merged backbones: {accs}")
        
        return {"merged_accs": accs}
