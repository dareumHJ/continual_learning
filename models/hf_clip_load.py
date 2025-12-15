# models/hf_clip_load.py

from typing import Dict
from transformers import CLIPVisionModel

TASK_TO_HF_MODEL_B32: Dict[str, str] = {
    "sun397": "tanganke/clip-vit-base-patch32_sun397",
    "stanford-cars": "tanganke/clip-vit-base-patch32_stanford-cars",
    "resisc45": "tanganke/clip-vit-base-patch32_resisc45",
    "eurosat": "tanganke/clip-vit-base-patch32_eurosat",
    "svhn": "tanganke/clip-vit-base-patch32_svhn",
    "gtsrb": "tanganke/clip-vit-base-patch32_gtsrb",
    "mnist": "tanganke/clip-vit-base-patch32_mnist",
    "dtd": "tanganke/clip-vit-base-patch32_dtd",
}

def load_finetuned_backbones(task_names, base_model_name="openai/clip-vit-base-patch32"):
    """
    각 task_name에 대해:
      - HF finetuned vision model을 CLIPVisionModel.from_pretrained로 로드
      - state_dict를 dict로 반환
    """
    backbones = {}
    for t in task_names:
        model_id = TASK_TO_HF_MODEL_B32.get(t)
        if model_id is None:
            raise ValueError(f"No HF finetuned model id registered for task '{t}'")
        vision_model = CLIPVisionModel.from_pretrained(model_id)
        backbones[t] = vision_model.state_dict()
    return backbones
