"""
Train CNN on selected subsets.

Supports training on:
- GA-selected subsets
- Random baseline subsets (with run number)
- Hardest-only subsets
- Balanced-only subsets
- Full dataset
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
import sys
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from training.cnn_model import create_cnn
from data.load_data import get_subset_data, load_validation_set, load_test_set, load_selection_pool, create_dataloader


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for data, labels in dataloader:
        data, labels = data.to(device), labels.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(data)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100.0 * correct / total
    
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    """Validate model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, labels in dataloader:
            data, labels = data.to(device), labels.to(device)
            
            outputs = model(data)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100.0 * correct / total
    
    return epoch_loss, epoch_acc


def train_cnn(
    train_indices: np.ndarray,
    k: int = None,
    subset_type: str = 'ga',
    run_number: int = None,
    num_classes: int = None,
    learning_rate: float = None,
    weight_decay: float = None,
    epochs: int = None,
    batch_size: int = None,
    patience: int = None,
    device: str = None,
    seed: int = None,
    verbose: bool = True
):
    """
    Train CNN on a selected subset.
    
    Args:
        train_indices: Array of indices for training subset
        k: Subset size (for naming). If None, uses len(train_indices)
        subset_type: Type of subset ('ga', 'random', 'hardest', 'balanced', 'full')
        run_number: Run number for random baselines (required if subset_type='random')
        num_classes: Number of classes. If None, uses config.
        learning_rate: Learning rate. If None, uses config.
        weight_decay: Weight decay. If None, uses config.
        epochs: Number of epochs. If None, uses config.
        batch_size: Batch size. If None, computed from subset size.
        patience: Early stopping patience. If None, uses config.
        device: Device to use. If None, uses config.
        seed: Random seed. If None, uses config.
        verbose: Whether to print progress
        
    Returns:
        Dictionary with training results and model path
    """
    # Set defaults
    if k is None:
        k = len(train_indices)
    if num_classes is None:
        num_classes = config.NUM_CLASSES
    if learning_rate is None:
        learning_rate = config.TRAIN_LEARNING_RATE
    if weight_decay is None:
        weight_decay = config.TRAIN_WEIGHT_DECAY
    if epochs is None:
        epochs = config.TRAIN_EPOCHS
    if batch_size is None:
        batch_size = config.get_batch_size(k)
    if patience is None:
        patience = config.EARLY_STOPPING_PATIENCE
    if device is None:
        device = config.TRAIN_DEVICE
    if seed is None:
        seed = config.TRAIN_SEED
    
    device = torch.device(device)
    
    # Set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    if verbose:
        print("=" * 60)
        print(f"Training CNN: {subset_type}, k={k}")
        if run_number is not None:
            print(f"Run number: {run_number}")
        print(f"Device: {device}")
        print("=" * 60)
    
    # Load data
    train_data, train_labels = get_subset_data(train_indices)
    val_data, val_labels = load_validation_set()
    test_data, test_labels = load_test_set()
    
    # Create dataloaders
    train_loader = create_dataloader(train_data, train_labels, batch_size=batch_size, shuffle=True)
    val_loader = create_dataloader(val_data, val_labels, batch_size=batch_size, shuffle=False)
    test_loader = create_dataloader(test_data, test_labels, batch_size=batch_size, shuffle=False)
    
    # Create model
    model = create_cnn(num_classes=num_classes)
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'epoch': []
    }
    
    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    
    # Training loop
    for epoch in range(epochs):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Record history
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(float(train_loss))
        history['train_acc'].append(float(train_acc))
        history['val_loss'].append(float(val_loss))
        history['val_acc'].append(float(val_acc))
        
        if verbose and (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs}: "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            patience_counter = 0
            
            # Save best model
            model_path = get_model_path(k, subset_type, run_number)
            model_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'history': history
            }, model_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose:
                    print(f"\nEarly stopping at epoch {epoch+1} (patience: {patience})")
                break
    
    # Load best model
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Test evaluation
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    
    if verbose:
        print(f"\n✓ Training completed!")
        print(f"  Best validation accuracy: {best_val_acc:.2f}% (epoch {best_epoch})")
        print(f"  Test accuracy: {test_acc:.2f}%")
        print(f"  Model saved to: {model_path}")
    
    # Prepare results
    results = {
        'subset_type': subset_type,
        'k': k,
        'run_number': run_number,
        'best_val_acc': float(best_val_acc),
        'test_acc': float(test_acc),
        'best_epoch': int(best_epoch),
        'total_epochs': len(history['epoch']),
        'model_path': str(model_path),
        'history': history,
        'timestamp': datetime.now().isoformat()
    }
    
    return results


def get_model_path(k: int, subset_type: str, run_number: int = None) -> Path:
    """
    Get model path based on subset type and parameters.
    
    Args:
        k: Subset size
        subset_type: Type of subset ('ga', 'random', 'hardest', 'balanced', 'full')
        run_number: Run number for random baselines
        
    Returns:
        Path to model file
    """
    final_models_dir = config.FINAL_MODELS_DIR
    final_models_dir.mkdir(parents=True, exist_ok=True)
    
    if subset_type == 'ga':
        return final_models_dir / f"cnn_ga_k{k}.pth"
    elif subset_type == 'random':
        if run_number is None:
            raise ValueError("run_number must be specified for random baseline")
        return final_models_dir / f"cnn_random_k{k}_run{run_number}.pth"
    elif subset_type == 'hardest':
        return final_models_dir / f"cnn_hardest_k{k}.pth"
    elif subset_type == 'balanced':
        return final_models_dir / f"cnn_balanced_k{k}.pth"
    elif subset_type == 'full':
        return final_models_dir / f"cnn_full.pth"
    else:
        raise ValueError(f"Unknown subset_type: {subset_type}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train CNN on selected subset")
    parser.add_argument(
        "subset_type",
        type=str,
        choices=['ga', 'random', 'hardest', 'balanced', 'full'],
        help="Type of subset"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help="Subset size k (required for ga, random, hardest, balanced)"
    )
    parser.add_argument(
        "--run-number",
        type=int,
        default=None,
        help="Run number for random baseline (required if subset_type=random)"
    )
    parser.add_argument(
        "--indices-file",
        type=str,
        default=None,
        help="Path to indices file (for custom subsets)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help=f"Number of epochs (default: {config.TRAIN_EPOCHS})"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size (default: computed from k)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=f"Random seed (default: {config.TRAIN_SEED})"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output"
    )
    
    args = parser.parse_args()
    
    # Load indices based on subset type
    if args.subset_type == 'ga':
        if args.k is None:
            raise ValueError("--k is required for GA subset")
        indices_path = config.get_selected_subset_path(args.k)
        if not indices_path.exists():
            raise FileNotFoundError(
                f"Selected subset not found: {indices_path}\n"
                f"Run 'python experiments/select_subset.py {args.k}' first."
            )
        train_indices = np.load(indices_path)
    elif args.subset_type == 'random':
        if args.k is None or args.run_number is None:
            raise ValueError("--k and --run-number are required for random baseline")
        # For now, generate random indices (will be implemented in baseline module)
        np.random.seed(args.seed if args.seed else config.TRAIN_SEED)
        _, labels = load_selection_pool()
        train_indices = np.random.choice(len(labels), size=args.k, replace=False)
    elif args.subset_type in ['hardest', 'balanced']:
        if args.k is None:
            raise ValueError(f"--k is required for {args.subset_type} subset")
        # Will be implemented in baseline module
        raise NotImplementedError(f"{args.subset_type} baseline not yet implemented")
    elif args.subset_type == 'full':
        # Use all selection pool indices
        _, labels = load_selection_pool()
        train_indices = np.arange(len(labels))
        args.k = len(labels)
    else:
        if args.indices_file:
            train_indices = np.load(args.indices_file)
        else:
            raise ValueError("Must specify indices file or valid subset type")
    
    # Train
    results = train_cnn(
        train_indices=train_indices,
        k=args.k,
        subset_type=args.subset_type,
        run_number=args.run_number,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        verbose=not args.quiet
    )
    
    print(f"\nResults: {json.dumps(results, indent=2)}")

