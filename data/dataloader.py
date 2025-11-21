"""
Data loading and preprocessing for CIFAR-10 dataset.
Handles train/val split, augmentation, and DataLoader creation.
"""

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import yaml
from pathlib import Path


class CIFAR10DataModule:
    """
    Encapsulates all data loading logic for CIFAR-10.
    This class follows the principle: one class, one responsibility.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize data module with configuration.
        
        Args:
            config_path: Path to YAML configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.data_config = self.config['data']
        self.aug_config = self.config['augmentation']
        self.batch_size = self.data_config['batch_size']
        self.num_workers = self.data_config['num_workers']
        
        # These will be populated by setup()
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
    def _get_transforms(self, train: bool = True):
        """
        Create data transformation pipeline.
        
        Args:
            train: If True, includes augmentation. If False, only normalization.
            
        Returns:
            torchvision.transforms.Compose object
        """
        if train and self.aug_config['enabled']:
            # Training transforms with augmentation
            transform_list = [
                transforms.RandomCrop(
                    self.aug_config['random_crop'], 
                    padding=4
                ),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=self.aug_config['normalize']['mean'],
                    std=self.aug_config['normalize']['std']
                )
            ]
        else:
            # Validation/test transforms - no augmentation
            transform_list = [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=self.aug_config['normalize']['mean'],
                    std=self.aug_config['normalize']['std']
                )
            ]
        
        return transforms.Compose(transform_list)
    
    def setup(self):
        """
        Download CIFAR-10 and create train/val/test splits.
        Called once before training starts.
        """
        data_dir = Path(self.data_config['data_dir'])
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # Download and load training data with augmentation
        full_train_dataset = datasets.CIFAR10(
            root=str(data_dir),
            train=True,
            download=True,
            transform=self._get_transforms(train=True)
        )
        
        # Download and load test data without augmentation
        self.test_dataset = datasets.CIFAR10(
            root=str(data_dir),
            train=False,
            download=True,
            transform=self._get_transforms(train=False)
        )
        
        # Split training data into train and validation
        val_split = self.data_config['validation_split']
        train_size = int((1 - val_split) * len(full_train_dataset))
        val_size = len(full_train_dataset) - train_size
        
        self.train_dataset, self.val_dataset = random_split(
            full_train_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(self.config['seed'])
        )
        
        print(f"Dataset splits created:")
        print(f"  Training samples: {len(self.train_dataset)}")
        print(f"  Validation samples: {len(self.val_dataset)}")
        print(f"  Test samples: {len(self.test_dataset)}")
    
    def train_dataloader(self):
        """Create DataLoader for training set."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.data_config['shuffle'],
            num_workers=self.num_workers,
            pin_memory=self.data_config['pin_memory']
        )
    
    def val_dataloader(self):
        """Create DataLoader for validation set."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,  # Never shuffle validation data
            num_workers=self.num_workers,
            pin_memory=self.data_config['pin_memory']
        )
    
    def test_dataloader(self):
        """Create DataLoader for test set."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,  # Never shuffle test data
            num_workers=self.num_workers,
            pin_memory=self.data_config['pin_memory']
        )


# Quick test function
if __name__ == "__main__":
    # Test the data module
    data_module = CIFAR10DataModule()
    data_module.setup()
    
    train_loader = data_module.train_dataloader()
    images, labels = next(iter(train_loader))
    
    print(f"\nBatch shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Image value range: [{images.min():.2f}, {images.max():.2f}]")