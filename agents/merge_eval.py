# agents/merge_eval

from typing import Dict, Any, List, Tuple
from pathlib import Path

import torch
import torch.nn.functional as F

from .base import BaseAgent
from . import register_agent

def _load_backbone_and_heads(model, ckpt_dir: Path) -> Dict[str, str]:
    """
    backbone.pt랑 head_*.pt 찾아서
    backbone은 model.backbone에 load하고
    head 파일 경로 dict 반환 {task_name: path}
    """
    ckpt_dir = Path(ckpt_dir)
    backbone_path = ckpt_dir / "backbone.pt"
    if not backbone_path.exists():
        raise FileNotFoundError(f"Backbone checkpoint not found: {backbone_path}")
    
    state = torch.load(backbone_path, map_location="cpu")
    model.backbone.load_state_dict(state)
    
    head_paths: Dict[str, str] = {}
    for p in ckpt_dir.glob("head_*.pt"):
        # 파일 이름에서 task_name 추출: head_mnist.pt -> mnist
        stem = p.stem # e.g. "head_mnist"
        task_name = stem.replace("head_", "", 1)
        head_paths[task_name] = str(p)
    
    if not head_paths:
        raise RuntimeError(f"No head_*.pt found in {ckpt_dir}")
    
    return head_paths

def _simple_average_heads(model, head_paths: Dict[str, str]) -> None:
    """
    여러 head checkpoint를 불러와 weight-wise 평균을 내고, model.head에 로드.
    """
    head_states = []
    for path in head_paths.values():
        sd = torch.load(path, map_location="cpu")
        head_states.append(sd)
        
    avg_state = {}
    for key in head_states[0].keys():
        tensors = [sd[key] for sd in head_states]
        avg_state[key] = sum(tensors) / len(tensors)
    
    model.head.load_state_dict(avg_state)
    

@register_agent("merge_eval")
class MergeEvalAgent(BaseAgent):
    """
    backbone과 여러 checkpoint 로드한 후,
    merge_method에 따라 하나의 merged head 만든 후 CL eval 수행하는 agent
    
    - current version -> simple_average only
    - eval loop는 cl_eval과 거의 동일, but 학습은 하지 않음.
    """
    
    def __init__(self, cfg, model, stream, test_stream=None):
        super().__init__(cfg, model, stream, test_stream)
        self.device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.ckpt_dir = Path(cfg.get("ckpt_dir", "outputs/offline_ckpts"))
        self.merge_method = cfg.get("merge_method", "simple_average")
        
        if self.test_stream is None:
            raise ValueError("cl_eval agent requires test_stream (data.test_tasks in config).")
        
        self.test_tasks: List[Tuple(str, Any)] = self.test_stream
        self.model.to(self.device)
                
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
    
    
    def run(self) -> Dict[str, Any]:
        head_paths = _load_backbone_and_heads(self.model, self.ckpt_dir)
        print(f"[MergeEval] Found heads for tasks: {sorted(head_paths.keys())}")
        
        if self.merge_method == "simple_average":
            _simple_average_heads(self.model, head_paths)
            print("[MergeEval] Applied simple_average over heads")
        else:
            raise ValueError(f"Unknown merge_method: {self.merge_method}")
        
        accs = self._eval_on_all_tasks()
        print(f"[MergeEval] Accs with merged head: {accs}")
        
        return {"merged_accs": accs}
