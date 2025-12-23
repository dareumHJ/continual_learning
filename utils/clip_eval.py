# utils/clip_eval.py
import torch
import importlib
import logging

log = logging.getLogger(__name__)

DEFAULT_TEMPLATES = [lambda c: f"a photo of a {c}."]

def load_task_metadata(task_name):
    """
    dataset/template/{task_name}.py 파일에서 templates와 class_names를 불러옵니다.
    """
    module_path = f"dataset.template.{task_name}"
    
    module_name_candidates = [
        task_name,
        task_name.replace("-", "_")
    ]
    
    mod = None
    for mod_name in module_name_candidates:
        try:
            module_path = f"dataset.template.{mod_name}"
            mod = importlib.import_module(module_path)
            break
        except ImportError:
            continue
        
    if mod is None:
        log.warning(f"Could not import metadata for task '{task_name}'. Using default templates.")
        return DEFAULT_TEMPLATES, None
    
    templates = None
    class_names = None
    
    template_keys = ["templates", "prompt_templates", "TEMPLATES"]
    for key in template_keys:
        if hasattr(mod, key):
            templates = getattr(mod, key)
            break
    if templates is None:
        templates = DEFAULT_TEMPLATES
        
    class_keys = ["class_names", "classnames", "classes", "CLASS_NAMES", "CLASSES", "labels"]
    for key in class_keys:
        if hasattr(mod, key):
            class_names = getattr(mod, key)
            break
    
    if class_names is None:
        log.warning(f"Imported {task_name}, but could not find class names. Checked: {class_keys}")
        
    return templates, class_names

def get_zeroshot_classifier(model, processor, task_name, device):
    """
    Task 이름을 받아 해당 파일의 템플릿/클래스명을 로드하고 Classifier Weights를 생성
    """
    templates, class_names = load_task_metadata(task_name)
    
    if class_names is None:
        raise ValueError(f"Class names not found for task '{task_name}'. Check dataset/template/{task_name}.py")

    with torch.no_grad():
        zeroshot_weights = []
        for class_name in class_names:
            texts = [template(class_name) for template in templates]
            
            # Tokenize & Encode
            inputs = processor(text=texts, return_tensors="pt", padding=True).to(device)
            class_embeddings = model.get_text_features(**inputs)
            class_embeddings /= class_embeddings.norm(p=2, dim=-1, keepdim=True)
            
            # Mean Ensemble
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm(p=2, dim=-1)
            
            zeroshot_weights.append(class_embedding)
            
        zeroshot_weights = torch.stack(zeroshot_weights, dim=0)
    
    return zeroshot_weights

def evaluate_clip_zeroshot(model, classifier_weights, loader, device):
    model.eval()
    correct = 0; total = 0
    logit_scale = model.logit_scale.exp()
    
    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, dict):
                images, labels = batch['image'], batch['label']
            else:
                images, labels = batch[0], batch[1]
                
            images, labels = images.to(device), labels.to(device)
            
            image_features = model.get_image_features(pixel_values=images)
            image_features /= image_features.norm(p=2, dim=-1, keepdim=True)
            
            logits = torch.matmul(image_features, classifier_weights.t()) * logit_scale
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
    return correct / total if total > 0 else 0.0