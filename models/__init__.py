# models/__init__.py

from .backbones import CLIPVisionClassifier

def create_model(cfg):
    """
    cfg 예시:
    model:
      type: clip_classifier
      pretrained_model_name: openai/clip-vit-base-patch32
      num_classes: 10
      freeze_backbone: false
    """
    mtype = cfg.get("type", "clip_classifier")
    if mtype == "clip_classifier":
        name = cfg.get("pretrained_model_name", "openai/clip-vit-base-patch32")
        num_classes = cfg.get("num_classes", 10)
        freeze = cfg.get("freeze_backbone", False)
        return CLIPVisionClassifier(
          pretrained_model_name=name,
          num_classes=num_classes,
          freeze_backbone=freeze,
        )
    else:
        raise ValueError(f"Unknown model type: {mtype}")