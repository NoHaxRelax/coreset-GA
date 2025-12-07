"""
Download and prepare MNIST dataset for coreset selection experiments.

Splits the data into:
- Selection pool: large pool to select subsets from (default: 20k samples)
- Validation set: for early stopping during training (default: 2k samples)
- Test set: for final evaluation (default: 2k samples)

The selection pool should be much larger than the maximum subset size (k=1000)
to ensure sufficient diversity for the GA to work with.
"""

import os
import numpy as np
import torch
from torchvision import datasets, transforms
from pathlib import Path


def download_and_prepare_mnist(
    selection_pool_size=20000,
    validation_size=2000,
    test_size=2000,
    data_dir="data",
    seed=2025
):
    """
    Download MNIST and create train/val/test splits.
    
    Args:
        selection_pool_size: Number of samples in the selection pool (default: 20k)
        validation_size: Number of samples in validation set (default: 2k)
        test_size: Number of samples in test set (default: 2k)
        data_dir: Directory to save data
        seed: Random seed for reproducibility
    """
    # Set random seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Create data directory
    data_path = Path(data_dir)
    data_path.mkdir(exist_ok=True)
    
    print("Downloading MNIST dataset...")
    
    # Download MNIST training set (60k samples)
    train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transforms.ToTensor()
    )
    
    # Download MNIST test set (10k samples)
    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transforms.ToTensor()
    )
    
    print(f"Loaded {len(train_dataset)} training samples and {len(test_dataset)} test samples")
    
    # Get all training data and labels
    train_data = []
    train_labels = []
    for img, label in train_dataset:
        train_data.append(img.numpy())
        train_labels.append(label)
    
    train_data = np.array(train_data)
    train_labels = np.array(train_labels)
    
    # Get test data and labels
    test_data = []
    test_labels = []
    for img, label in test_dataset:
        test_data.append(img.numpy())
        test_labels.append(label)
    
    test_data = np.array(test_data)
    test_labels = np.array(test_labels)
    
    # Shuffle training data
    indices = np.random.permutation(len(train_data))
    train_data = train_data[indices]
    train_labels = train_labels[indices]
    
    # Split training data into selection pool and validation set
    # We need selection_pool_size + validation_size samples
    total_needed = selection_pool_size + validation_size
    
    if total_needed > len(train_data):
        raise ValueError(
            f"Need {total_needed} samples but only have {len(train_data)} training samples. "
            f"Reduce selection_pool_size ({selection_pool_size}) or validation_size ({validation_size})"
        )
    
    # Take first selection_pool_size for selection pool
    selection_pool_data = train_data[:selection_pool_size]
    selection_pool_labels = train_labels[:selection_pool_size]
    
    # Take next validation_size for validation
    validation_data = train_data[selection_pool_size:selection_pool_size + validation_size]
    validation_labels = train_labels[selection_pool_size:selection_pool_size + validation_size]
    
    # Use test_size samples from test set
    if test_size > len(test_data):
        raise ValueError(
            f"Need {test_size} test samples but only have {len(test_data)}. "
            f"Reduce test_size"
        )
    
    # Shuffle test data and take first test_size
    test_indices = np.random.permutation(len(test_data))
    test_data = test_data[test_indices[:test_size]]
    test_labels = test_labels[test_indices[:test_size]]
    
    # Save as numpy arrays
    print("\nSaving data splits...")
    
    # Selection pool
    np.save(data_path / "selection_pool_data.npy", selection_pool_data)
    np.save(data_path / "selection_pool_labels.npy", selection_pool_labels)
    print(f"  Selection pool: {len(selection_pool_data)} samples")
    print(f"    Class distribution: {np.bincount(selection_pool_labels)}")
    
    # Validation set
    np.save(data_path / "validation_data.npy", validation_data)
    np.save(data_path / "validation_labels.npy", validation_labels)
    print(f"  Validation set: {len(validation_data)} samples")
    print(f"    Class distribution: {np.bincount(validation_labels)}")
    
    # Test set
    np.save(data_path / "test_data.npy", test_data)
    np.save(data_path / "test_labels.npy", test_labels)
    print(f"  Test set: {len(test_data)} samples")
    print(f"    Class distribution: {np.bincount(test_labels)}")
    
    # Save metadata
    metadata = {
        "selection_pool_size": len(selection_pool_data),
        "validation_size": len(validation_data),
        "test_size": len(test_data),
        "num_classes": 10,
        "image_shape": selection_pool_data[0].shape,
        "seed": seed
    }
    
    import json
    with open(data_path / "dataset_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✓ Dataset prepared successfully!")
    print(f"  Files saved to: {data_path.absolute()}")
    print(f"  Metadata saved to: {data_path / 'dataset_metadata.json'}")
    
    return metadata


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare MNIST dataset for coreset selection")
    parser.add_argument(
        "--selection-pool-size",
        type=int,
        default=20000,
        help="Size of selection pool (default: 20000)"
    )
    parser.add_argument(
        "--validation-size",
        type=int,
        default=2000,
        help="Size of validation set (default: 2000)"
    )
    parser.add_argument(
        "--test-size",
        type=int,
        default=2000,
        help="Size of test set (default: 2000)"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory to save data (default: data)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2025,
        help="Random seed (default: 2025)"
    )
    
    args = parser.parse_args()
    
    download_and_prepare_mnist(
        selection_pool_size=args.selection_pool_size,
        validation_size=args.validation_size,
        test_size=args.test_size,
        data_dir=args.data_dir,
        seed=args.seed
    )

