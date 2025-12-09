from datasets import get_image_classification_dataloader

def main():
    loader, num_classes = get_image_classification_dataloader(
        name="MNIST",
        split="test",
        batch_size=64,
        num_workers=2,
    )
    print("num_classes:", num_classes)
    batch = next(iter(loader))
    images, labels = batch
    print("image shape:", images.shape)
    print("labels shape:", labels.shape)
    
if __name__ == "__main__":
    main()