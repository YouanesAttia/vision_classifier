from torchvision import transforms, datasets
from torch.utils.data import DataLoader

def get_transforms(train: bool):
    if train:
        return transforms.Compose([
            transforms.resize(224, 224),
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

def create_dataloaders(train_dir, test_dir, batch_size=32):
    train_data = datasets.ImageFolder(train_dir, transform=get_transforms(True))
    test_data = datasets.ImageFolder(test_dir, transform=get_transforms(False))
    train_dataloader = DataLoader(train_data, transforms=get_transforms(True))
    test_dataloader = DataLoader(test_data, transforms=get_transforms(False))
    return train_dataloader, test_dataloader, train_data.classes