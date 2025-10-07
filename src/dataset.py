import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torchvision
from torchvision import transforms
import config

class TireDataset:
    def __init__(self):
        """Initialize tire dataset transformations with normalization and augmentation."""

        # Training transform
        self.train_transform = transforms.Compose([
            transforms.Resize(512),
            transforms.RandomRotation(10),
            transforms.RandomVerticalFlip(),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # Inference transform
        self.inference_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    # ----------------- Public Methods -----------------

    def prepare_data_for_training(self):
        dataset = self._load_dataset()
        train_dataset, val_dataset, test_dataset = self._split_dataset(dataset)
        train_loader, val_loader, test_loader = self._create_data_loaders(train_dataset, val_dataset, test_dataset)
        return train_loader, val_loader, test_loader

    def prepare_data_for_inference(self, image):
        X = self.inference_transform(image)
        X = X.unsqueeze(0)
        return X

    # ----------------- Private Methods -----------------

    def _load_dataset(self):
        dataset = torchvision.datasets.ImageFolder(
            root=config.DATASET_PATH,
            transform=self.train_transform
        )
        return dataset

    def _split_dataset(self, dataset):
        """Split dataset into training, validation, and test subsets."""

        if not config.SPLIT_DATASET:
            print("• Dataset splitting disabled → using same data for train/val/test.")
            return dataset
    
        n_total = len(dataset)
        n_train = int(config.TRAIN_SPLIT_RATIO * n_total)
        n_val = int(config.VAL_SPLIT_RATIO * n_total)
        n_test = n_total - n_train - n_val

        generator = (
            torch.Generator().manual_seed(config.SPLIT_RANDOMIZATION_SEED)
            if config.SPLIT_RANDOMIZATION_SEED is not None else None
        )

        train_ds, val_ds, test_ds = random_split(
            dataset, [n_train, n_val, n_test], 
            generator=generator
        )

        return train_ds, val_ds, test_ds    

    def _create_data_loaders(self, train_dataset, val_dataset, test_dataset):
        train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
        return train_loader, val_loader, test_loader
