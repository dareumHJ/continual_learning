# models/backbones.py

import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor
from .heads import LinearHead

class CLIPVisionBackbone(nn.Module):
    def __init__(self, pretrained_model_name: str):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(pretrained_model_name)
        self.processor = CLIPProcessor.from_pretrained(pretrained_model_name)
        self.out_dim = self.clip.vision_model.config.hidden_size

    def forward(self, images): # image -> feature[B, D]
        if images.dim() == 4 and images.shape[1] == 1:
            images = images.repeat(1, 3, 1, 1)  # Grayscale to RGB
            
        inputs = self.processor(
            images=images,
            return_tensors="pt",
            do_rescale=False,
        )
        pixel_values = inputs["pixel_values"].to(images.device)
        vision_outputs = self.clip.vision_model(pixel_values=pixel_values)
        pooled = vision_outputs.last_hidden_state[:, 0]
        return pooled # now return feature (not logits)

class CLIPVisionClassifier(nn.Module):
    def __init__(self, pretrained_model_name: str, num_classes: int, freeze_backbone: bool = False):
        super().__init__()
        self.backbone = CLIPVisionBackbone(pretrained_model_name)
        self.head = LinearHead(self.backbone.out_dim, num_classes)
        
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
    
    def forward(self, images):
        features = self.backbone(images) # [B, D]
        logits = self.head(features) # [B, num_classes]
        return logits