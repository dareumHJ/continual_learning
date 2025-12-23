# datasets/image_classification.py

from typing import Iterator, Tuple

from pathlib import Path

import torch
import datasets as hf_datasets
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

_DATA_ROOT = Path("./data") # config에서 불러오기로 대체 가능

class HFDatasetWrapper(Dataset):
    def __init__(self, hf_dataset, image_key="image", label_key="label", transform=None):
        self.ds = hf_dataset
        self.image_key = image_key
        self.label_key = label_key
        self.transform = transform
        
    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, idx):
        example = self.ds[int(idx)]
        image = example[self.image_key] # PIL.Image거나 numpy array
        label = int(example[self.label_key])
        
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        if self.transform is not None:
            image = self.transform(image)
            
        return image, label

def clip_image_transform():
    # CLIPProcessor가 resize/normalize 다시 해주긴 함...
    return transforms.Compose([
        transforms.Resize(224, interpolation=InterpolationMode.BICUBIC), # for clip-vit-base-patch32!!! backbone 달라지면 수정해줘야 함
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        )
    ])

def _get_mnist(split:str):
    hf_ds = hf_datasets.load_dataset("mnist", split=split)
    transform = clip_image_transform()
    dataset = HFDatasetWrapper(
        hf_ds,
        image_key="image",
        label_key="label",
        transform=transform,
    )
    num_classes = hf_ds.features["label"].num_classes
    return dataset, num_classes

def _get_eurosat(split:str):
    hf_ds = hf_datasets.load_dataset("tanganke/eurosat", split=split)
    transform = clip_image_transform()
    dataset = HFDatasetWrapper(
        hf_ds,
        image_key="image",
        label_key="label",
        transform=transform,
    )
    num_classes = hf_ds.features["label"].num_classes
    return dataset, num_classes

def _get_gtsrb(split:str):
    hf_ds = hf_datasets.load_dataset("tanganke/gtsrb", split=split)
    transform = clip_image_transform()
    dataset = HFDatasetWrapper(
        hf_ds,
        image_key="image",
        label_key="label",
        transform=transform,
    )
    num_classes = hf_ds.features["label"].num_classes
    return dataset, num_classes

def _get_dtd(split:str):
    hf_ds = hf_datasets.load_dataset("tanganke/dtd", split=split)
    transform = clip_image_transform()
    dataset = HFDatasetWrapper(
        hf_ds,
        image_key="image",
        label_key="label",
        transform=transform,
    )
    num_classes = hf_ds.features["label"].num_classes
    return dataset, num_classes

def _get_sun397(split:str):
    hf_ds = hf_datasets.load_dataset("tanganke/sun397", split=split)
    transform = clip_image_transform()
    dataset = HFDatasetWrapper(
        hf_ds,
        image_key="image",
        label_key="label",
        transform=transform,
    )
    num_classes = hf_ds.features["label"].num_classes
    return dataset, num_classes

def _get_svhn(split:str):
    hf_ds = hf_datasets.load_dataset("svhn", "cropped_digits", split=split)
    transform = clip_image_transform()
    dataset = HFDatasetWrapper(
        hf_ds,
        image_key="image",
        label_key="label",
        transform=transform,
    )
    num_classes = hf_ds.features["label"].num_classes
    return dataset, num_classes

def _get_resisc45(split:str):
    hf_ds = hf_datasets.load_dataset("tanganke/resisc45", split=split)
    transform = clip_image_transform()
    dataset = HFDatasetWrapper(
        hf_ds,
        image_key="image",
        label_key="label",
        transform=transform,
    )
    num_classes = hf_ds.features["label"].num_classes
    return dataset, num_classes

def _get_cars(split:str):
    hf_ds = hf_datasets.load_dataset("tanganke/stanford_cars", split=split)
    transform = clip_image_transform()
    dataset = HFDatasetWrapper(
        hf_ds,
        image_key="image",
        label_key="label",
        transform=transform,
    )
    num_classes = hf_ds.features["label"].num_classes
    return dataset, num_classes

def _get_fashion_mnist(split:str):
    hf_ds = hf_datasets.load_dataset("zalando-datasets/fashion_mnist", split=split)
    transform = clip_image_transform()
    dataset = HFDatasetWrapper(
        hf_ds,
        image_key="image",
        label_key="label",
        transform=transform,
    )
    num_classes = hf_ds.features["label"].num_classes
    return dataset, num_classes
    
def _get_kmnist(split:str):
    hf_ds = hf_datasets.load_dataset("tanganke/kmnist", split=split)
    transform = clip_image_transform()
    dataset = HFDatasetWrapper(
        hf_ds,
        image_key="image",
        label_key="label",
        transform=transform,
    )
    num_classes = hf_ds.features["label"].num_classes
    return dataset, num_classes

def get_image_classification_dataloader(
    name: str,
    split: str,
    batch_size: int = 128,
    num_workers: int = 4,
) -> Tuple[DataLoader, int]:
    """
    Args:
        name (str): name of dataset (e.g., "MNIST")
        split (str): 'train' or 'test'

    Returns:
        Tuple[DataLoader, int]: (DataLoader, num_classes)
    """
    name = name.lower()
    
    if name == "mnist":
        dataset, num_classes = _get_mnist(split)
    elif name == "fashionmnist":
        dataset, num_classes = _get_fashion_mnist(split)
    elif name == "kmnist":
        dataset, num_classes = _get_kmnist(split)
    elif name == "sun397":
        dataset, num_classes = _get_sun397(split)
    elif name == "stanford-cars":
        dataset, num_classes = _get_cars(split)
    elif name == "resisc45":
        dataset, num_classes = _get_resisc45(split)
    elif name == "eurosat":
        dataset, num_classes = _get_eurosat(split)
    elif name == "svhn":
        dataset, num_classes = _get_svhn(split)
    elif name == "gtsrb":
        dataset, num_classes = _get_gtsrb(split)
    elif name == "dtd":
        dataset, num_classes = _get_dtd(split)
    else:
        raise ValueError(f"Unknown image classification dataset: {name}")
    
    loader = DataLoader(
        dataset,
        batch_size = batch_size,
        shuffle = (split == "train"),
        num_workers = num_workers,
        pin_memory = True,
    )
    
    return loader, num_classes