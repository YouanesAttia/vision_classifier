from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset

def get_transforms(train: bool):
    if train:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
      ])

def create_dataloaders(root, batch_size=32):
    train_dataset_full = datasets.ImageFolder(root, transform=get_transforms(train=True))
    test_dataset_full = datasets.ImageFolder(root, transform=get_transforms(train=False))
    split_idx = int(0.7 * len(train_dataset_full))
    train_indices = list(range(0, split_idx))
    test_indices = list(range(split_idx, len(train_dataset_full)))
    train_data = Subset(train_dataset_full, train_indices)
    test_data = Subset(test_dataset_full, test_indices)
    class_names = train_dataset_full.classes
    train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    test_dataloader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    return train_dataloader, test_dataloader, class_names