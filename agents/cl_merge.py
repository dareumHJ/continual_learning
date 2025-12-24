# agents/cl_merge.py

import copy
import torch
import logging
from pathlib import Path
from typing import Dict, Any
from tqdm import tqdm

from .base import BaseAgent
from . import register_agent
from utils.clip_eval import get_zeroshot_classifier, evaluate_clip_zeroshot

log = logging.getLogger(__name__)

@register_agent("cl_merge")
class CLMergeAgent(BaseAgent):
    """
    Continual Learning via Model Merging Agent.
    순차적으로 태스크가 들어올 때마다, global model에 merge하여 업데이트.
    """
    
    def __init__(self, cfg, model, stream, test_stream=None):
        super().__init__(cfg, model, stream, test_stream)
        
        self.ckpt_dir = Path(cfg.get("ckpt_dir", "data/clip-vit-b32/hf_backbones"))
        self.scaling_factor = cfg.get("scaling_factor", 0.5)
        self.device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        
        # Base Model (pretrained, initially)
        self.global_state = {
            k: v.clone().to(self.device) 
            for k, v in self.model.backbone.clip.vision_model.state_dict().items()
        }
        
        # Pretrained State (for task vector calculation..)
        self.pretrained_state = {
            k: v.clone().to(self.device) 
            for k, v in self.model.backbone.clip.vision_model.state_dict().items()
        }

    def _load_task_checkpoint(self, task_name) -> Dict[str, torch.Tensor]:
        """주어진 task의 finetuned backbone 로드"""
        path = self.ckpt_dir / f"{task_name}/backbone.pt"
        if not path.exists():
            log.warning(f"Checkpoint not found for {task_name} at {path}")
            return None
            
        state = torch.load(path, map_location="cpu")
        if "state_dict" in state:
            state = state["state_dict"]
            
        # Key normalization
        clean_state = {}
        for k, v in state.items():
            new_k = k
            if k.startswith("vision_model."):
                new_k = k.replace("vision_model.", "")
            elif k.startswith("backbone.clip.vision_model."):
                new_k = k.replace("backbone.clip.vision_model.", "")
            
            # global state에 있는 key만 (근데 이렇게 해도 되나.. 나중에 CLIP 아닌 거 쓰면 수정할 수도)
            if new_k in self.global_state:
                clean_state[new_k] = v.to(self.device)
                
        return clean_state

    def _merge_step(self, task_name, new_state_dict):
        """
        <Core Part>
        Do merge global state and new state
        Currently, this is a simple Task Arithmetic!
        """
        log.info(f"Merging task: {task_name} into Global Model...")
        
        for k in self.global_state.keys():
            if k not in new_state_dict:
                continue
                
            w_global = self.global_state[k]
            w_new = new_state_dict[k]
            w_pretrained = self.pretrained_state[k]
            
            # Task Vector: tau = w_new - w_pretrained
            task_vector = w_new - w_pretrained
            
            # Update Rule: W_t = W_{t-1} + lambda * task_vector
            self.global_state[k] = w_global + self.scaling_factor * task_vector
            
        self.model.backbone.clip.vision_model.load_state_dict(self.global_state)

    def run(self) -> Dict[str, Any]:
        acc_matrix = {}
        seen_tasks = []
        
        self.model.to(self.device)
        self.model.eval()
        
        prog = tqdm(enumerate(self.stream), desc="[CL Task Stream]")
        
        # 1. Continual Learning Loop
        for step, (task_name, train_data) in prog:
            prog.set_description(f"[CL Step {step}] Task: {task_name}")
            
            # 1-1. load checkpoint
            task_state = self._load_task_checkpoint(task_name)
            
            if task_state is not None:
                # 1-2. update global model
                self._merge_step(task_name, task_state)
            else:
                log.error(f"Skipping merge for {task_name} (No Checkpoint)")

            seen_tasks.append(task_name)
            
            # 1-3. evluation on all given test tasks
            step_results = {}
            log.info(f"Evaluating on seen tasks: {seen_tasks}")
            
            test_loader_dict = {
                name: loader for name, (loader, _) in self.test_stream 
                if name in seen_tasks
            }
            
            for t_name, loader in tqdm(test_loader_dict.items(), desc="Evaluating", leave=False):
                try:
                    # clip_eval
                    classifier = get_zeroshot_classifier(
                        self.model.backbone.clip, 
                        self.model.backbone.processor, 
                        t_name, 
                        self.device
                    )
                    acc = evaluate_clip_zeroshot(
                        self.model.backbone.clip, 
                        classifier, 
                        loader, 
                        self.device
                    )
                    step_results[t_name] = acc
                except Exception as e:
                    log.error(f"Eval failed for {t_name}: {e}")
                    step_results[t_name] = 0.0
            
            acc_matrix[str(step)] = step_results
            tqdm.write(f"Step {step} Results: {step_results}")

        return {"acc_matrix": acc_matrix}