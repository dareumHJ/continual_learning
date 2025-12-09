# models/backbones.py

import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor

class CLIPVisionClassifier(nn.Module):
    def __init__(self, pretrained_model_name: str, num_classes: int):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(pretrained_model_name)
        vision_width = self.clip.vision_model.config.hidden_size
        self.classifier = nn.Linear(vision_width, num_classes)
        self.processor = CLIPProcessor.from_pretrained(pretrained_model_name)

    def forward(self, images):
        inputs = self.processor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(images.device)
        vision_outputs = self.clip.vision_model(pixel_values=pixel_values)
        pooled = vision_outputs.last_hidden_state[:, 0]
        logits = self.classifier(pooled)
        return logits
