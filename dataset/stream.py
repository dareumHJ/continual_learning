# datasets/stream.py

from typing import Iterator, Tuple
from dataset import get_image_classification_dataloader

def create_stream(cfg) -> Iterator[Tuple[str, object]]:
    """
    cfg example:
    data:
      scenario: single_task
      dataset: mnist
      split: test
      batch_size: 64
      num_workers: 4
    """
    scenario = cfg.get("scenario", "single_task")

    if scenario == "single_task":
        name = cfg.get("dataset", "mnist")
        split = cfg.get("split", "test")
        batch_size = cfg.get("batch_size", 64)
        num_workers = cfg.get("num_workers", 4)

        loader, num_classes = get_image_classification_dataloader(
            name=name,
            split=split,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        # single task stream에서는 한 번만 yield
        yield name, (loader, num_classes)
        
    elif scenario == "task_incremental":
        tasks_cfg = cfg.get("tasks", [])
        batch_size = cfg.get("batch_size", 64)
        num_workers = cfg.get("num_workers", 4)
        
        for task in tasks_cfg:
            name = task.get("name", "mnist")
            split = task.get("split", "test")
            loader, num_classes = get_image_classification_dataloader(
                name=name,
                split=split,
                batch_size=batch_size,
                num_workers=num_workers,
            )
            yield name, (loader, num_classes)
    else:
        raise ValueError(f"Unknown scenario: {scenario}")

def create_test_stream(cfg) -> Iterator[Tuple[str, object]]:
    tasks_cfg = cfg.get("test_tasks", [])
    batch_size = cfg.get("batch_size", 64)
    num_workers = cfg.get("num_workers", 4)
    
    for task in tasks_cfg:
        name = task.get("name", "mnist")
        split = task.get("split", "test")
        loader, num_classes = get_image_classification_dataloader(
            name=name,
            split=split,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        yield name, (loader, num_classes)