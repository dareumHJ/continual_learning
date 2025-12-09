# datasets/image_classification.py

from typing import Tuple
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

_DATA_ROOT = Path("./data") # config에서 불러오기로 대체 가능

def _mnist_transforms():
    return transforms.Compose(
        [
            transforms.ToTensor(),
            # 나중에 모델에 맞게 normalization 수정 가능
        ]
    )

def _get_mnist(split:str):
    train = split == "train"
    dataset = datasets.MNIST(
        root = _DATA_ROOT,
        train = train,
        download = True,
        transform = _mnist_transforms(),
    )
    return dataset

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
        dataset = _get_mnist(split)
        num_classes = 10
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