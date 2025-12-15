from pathlib import Path
from transformers import CLIPVisionModel
from models.hf_clip_load import TASK_TO_HF_MODEL_B32

import torch

def main():
    out_dir = Path("data/clip-vit-b32/hf_backbones")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    for task, model_id in TASK_TO_HF_MODEL_B32.items():
        ckpt_path = out_dir / f"{task}"
        ckpt_path.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_path / "backbone.pt"
        
        print(f"Downloading {model_id} for task {task}")
        vision_model = CLIPVisionModel.from_pretrained(model_id) # HF download
        torch.save(vision_model.state_dict(), ckpt_path)
        print(f"Saved backbone to {ckpt_path}")
        
if __name__ == "__main__":
    main()
        