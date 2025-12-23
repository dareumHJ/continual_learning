# agents/merge_eval
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from typing import Dict, Any, List, Tuple
from pathlib import Path
from transformers import CLIPModel, CLIPProcessor

import copy
import torch
import torch.nn.functional as F
import logging
log = logging.getLogger(__name__)

from .base import BaseAgent
from . import register_agent
from utils.cl_eval import evaluate_clip_zeroshot, get_zeroshot_classifier
from utils.clip_eval import get_zeroshot_classifier, evaluate_clip_zeroshot

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

def _task_arithmetic_backbones(
    backbones: Dict[str, dict],
    base_task: str,
    plus_tasks: List[str],
    minus_tasks: List[str] | None = None,
    scaling_factor: float = 1.0,
) -> dict:
    """
    Task Arithmetic:
      W_merge = W_base + scaling_factor * (avg(W_plus) - avg(W_minus))
      
      - backbones: {task_name: state_dict}
      - base_task: 기준이 되는 task name
      - plus_tasks: 추가하고 싶은 task들
      - minus_tasks: 빼고 싶은 task들 (없으면 None)
      - scaling_factor: 
    """
    if base_task not in backbones:
        raise ValueError(f"Base task '{base_task}' not found in backbones")
    
    if len(plus_tasks) == 0:
        raise ValueError("plus_tasks must contain at least one task")
    
    minus_tasks = minus_tasks or []
    
    base_sd = backbones[base_task]
    
    plus_sds = [backbones[t] for t in plus_tasks]
    minus_sds = [backbones[t] for t in minus_tasks] if minus_tasks else []\
    
    merged: Dict[str, torch.Tensor] = {}
    
    for key in base_sd.keys():
        base_w = base_sd[key]
        
        plus_avg = sum(sd[key] for sd in plus_sds) / len(plus_sds)
        
        if minus_sds:
            minus_avg = sum(sd[key] for sd in minus_sds) / len(minus_sds)
            delta = plus_avg - minus_avg
        else:
            delta = plus_avg - base_w
            
        merged[key] = base_w + scaling_factor * delta
    
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
        
        self.test_stream = test_stream
        
        self.ckpt_dir = Path(cfg.get("ckpt_dir", "data/clip-vit-b32/hf_backbones"))
        self.base_task = cfg.get("base_task", "mnist")
        self.plus_tasks = cfg.get("plus_tasks", [])
        self.minus_tasks = cfg.get("minus_tasks", [])
        self.scaling_factor = cfg.get("scaling_factor", 1.0)
        self.merge_method = cfg.get("merge_method", "simple_average")
        
        self.device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        
        if self.test_stream is None:
            raise ValueError("merge_eval agent requires test_stream (data.test_tasks in config).")
        
        model_name = cfg.get("pretrained_model_name", "openai/clip-vit-base-patch32")
        
        self.clip_model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        
        self.pretrained_state = {
            k: v.cpu() for k, v in self.clip_model.state_dict().items()
            if k.startswith("vision_model")
        }
        
    def _get_class_names(self, task_name: str) -> List[str]:
        dataset = self.stream.get_task_dataset(task_name, split="test")
        if hasattr(dataset, "classes"):
            return dataset.classes
        elif hasattr(dataset, "dataset") and hasattr(dataset.dataset, "classes"):
            return dataset.dataset.classes
        else:
            raise ValueError(f"Cannot retrieve class names for task '{task_name}'")
                
    def _eval_on_all_tasks(self, task_dict) -> Dict[str, float]:
        results: Dict[str, float] = {}
        
        for task_name, loader in task_dict.items():
            print(f"Evaluating on task: {task_name}...")
            
            try:
                classifier_weights = get_zeroshot_classifier(
                    self.clip_model,
                    self.processor,
                    task_name,
                    self.device
                )
                
                acc = evaluate_clip_zeroshot(
                    self.clip_model,
                    classifier_weights,
                    loader,
                    self.device
                )
                results[task_name] = acc
            
            except Exception as e:
                print(f"Skipping eval for {task_name}: {e}")
                results[task_name] = 0.0
            
        return results
    
    def _get_checkpoint_path(self, task_name):
        path = self.ckpt_dir / f"{task_name}/backbone.pt"
        if not path.exists():
            log.warning(f"Checkpoint not found: {path}")
            return None
        return path
    
    def _load_vision_backbone(self, task_name):
        path = self._get_checkpoint_path(task_name)
        if path is None:
            return None
        
        try:
            state = torch.load(path, map_location="cpu")
            if "state_dict" in state:
                state = state["state_dict"]
                
            normalized_state = {}
            for k, v in state.items():
                if k.startswith("vision_model."):
                    normalized_state[k] = v
                else:
                    normalized_state[f"vision_model.{k}"] = v
            
            return normalized_state
        
        except Exception as e:
            log.error(f"Failed to load {task_name}: {e}")
            return None
        
    def _get_base_weights(self):
        full_state = self.clip_model.state_dict()
        
        if self.base_task.lower() in ["pretrained", "none"]:
            log.info("Base Model: Pre-trained CLIP")
            return copy.deepcopy(full_state)
        
        # 특정 Task를 Base로 삼는 경우 (예: MNIST)
        log.info(f"Base Model: {self.base_task} Checkpoint")
        base_backbone = self._load_vision_backbone(self.base_task)
        
        if base_backbone is None:
            raise ValueError(f"Could not load base task: {self.base_task}")
            
        # Full State에 Backbone만 덮어쓰기
        base_state = copy.deepcopy(full_state)
        for k, v in base_backbone.items():
            if k in base_state:
                base_state[k] = v.to(self.device)
                
        return base_state
    
    def _get_task_vector(self, task_name):
        """
        Task Vector = Theta_ft(task) - Theta_pretrained
        """
        ft_state = self._load_vision_backbone(task_name)
        if ft_state is None:
            return None
            
        vector = {}
        for k, pre_v in self.pretrained_state.items():
            if k in ft_state:
                ft_v = ft_state[k]
                if pre_v.shape == ft_v.shape:
                    vector[k] = ft_v - pre_v
                    
        return vector
    
    def _apply_vector(self, target_state, vector, sign=1.0):
        """
        Target += sign * scaling * vector
        """
        with torch.no_grad():
            for k, v in vector.items():
                if k in target_state:
                    # target_state는 GPU에 있을 수 있으므로 device 맞춤
                    diff = v.to(target_state[k].device)
                    target_state[k] += sign * self.scaling_factor * diff
    
    def run(self) -> Dict[str, Any]:
        log.info(f"Starting Merge: Base='{self.base_task}', + {self.scaling_factor} * (Plus - Minus)")
        
        final_state = self._get_base_weights()
        
        for task_name in self.plus_tasks:
            log.info(f"Adding Task Vector: {task_name}")
            vector = self._get_task_vector(task_name)
            if vector:
                self._apply_vector(final_state, vector, sign=1.0)
                
        for task_name in self.minus_tasks:
            log.info(f"Subtracting Task Vector: {task_name}")
            vector = self._get_task_vector(task_name)
            if vector:
                self._apply_vector(final_state, vector, sign=-1.0)
        
        self.clip_model.load_state_dict(final_state, strict=False)
        log.info("mODEL merging complete.")
        
        task_dict = {name: loader for name, (loader, _) in self.test_stream}
        return self._eval_on_all_tasks(task_dict)
        
        # 1) config에서 평가할 task 목록 추출 (test_tasks 기준)
        task_names = [t[0] for t in self.test_tasks] # (task_name, loader, num_classes)
        
        # 2) 해당 task들에 대한 backbone 로드
        backbones = _load_backbone_tasks(self.ckpt_dir, task_names)
        print(f"[MergeEval] Using backbones for tasks: {sorted(backbones.keys())}")
        
        if self.merge_method == "simple_average":
            merged_sd = _simple_average_backbones(backbones)
            print("[MergeEval] Applied simple_average over backbones")
        elif self.merge_method == "task_arithmetic":
            base_task = self.cfg.get("base_task", "mnist")
            plus_tasks = self.cfg.get("plus_tasks", [])
            minus_tasks = self.cfg.get("minus_tasks", [])
            scaling = self.cfg.get("scaling_factor", 0.3)
            merged_sd = _task_arithmetic_backbones(
                backbones,
                base_task = base_task,
                plus_tasks=plus_tasks,
                minus_tasks=minus_tasks,
                scaling_factor=scaling,
                )
            print(f"[MergeEval] Applied task arithmetic with base={base_task},"
                  f"plus={plus_tasks}, minus={minus_tasks}, alpha={scaling}")
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
        task_dict = {name: loader for name, (loader, _) in self.test_tasks}
        accs = self._eval_on_all_tasks(task_dict)
        print(f"[MergeEval] Accs with merged backbones: {accs}")
        
        return {"merged_accs": accs}
