# utils/clip_eval.py
import torch
from tqdm import tqdm

def get_zeroshot_classifier(model, processor, class_names, device):
    """
    클래스 이름들로부터 Zero-shot Classifier Weights(Text Embeddings)를 생성함.
    """
    prompts = [f"a photo of a {c}" for c in class_names]
    
    text_inputs = processor(text=prompts, return_tensors="pt", padding=True).to(device)
    
    with torch.no_grad():
        text_features = model.get_text_features(**text_inputs)
        # L2 Normalize
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        
    return text_features

def evaluate_clip_zeroshot(model, classifier_weights, loader, device):
    """
    미리 계산된 classifier_weights(Text Embeddings)를 사용하여 평가
    """
    model.eval()
    correct, total = 0, 0
    
    # Logit Scale (Temperature)
    logit_scale = model.logit_scale.exp()
    
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            
            # Image Features
            image_features = model.get_image_features(pixel_values=images)
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
            
            # Classification
            # image: [B, D], text: [C, D] -> logits: [B, C]
            logits = logit_scale * image_features @ classifier_weights.t()
            preds = logits.argmax(dim=1)
            
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
    return correct / total if total > 0 else 0.0