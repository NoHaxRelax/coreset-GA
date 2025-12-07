"""
Extract embeddings from selection pool using pretrained feature extractor.

Extracts 512-dimensional embeddings for all samples in the selection pool.
These embeddings are used to compute the diversity objective in the GA.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from data.load_data import load_selection_pool, create_dataloader


def create_feature_extractor(model_name="resnet50", embedding_dim=512, device=None):
    """
    Create a feature extractor model (backbone without classification head).
    
    Args:
        model_name: Name of the pretrained model (default: 'resnet50')
        embedding_dim: Dimension of output embeddings (default: 512)
        device: Device to load model on. If None, uses config.EMBEDDING_DEVICE
        
    Returns:
        Feature extractor model in eval mode
    """
    if device is None:
        device = config.EMBEDDING_DEVICE
    
    device = torch.device(device)
    
    # Load pretrained model
    if model_name == "resnet50":
        from torchvision import models
        base_model = models.resnet50(weights='IMAGENET1K_V2')
        
        # Create feature extractor: remove final layers and add projection
        # ResNet50 structure: conv1 -> bn1 -> relu -> maxpool -> layer1-4 -> avgpool -> fc
        # We want: conv1 (modified) -> bn1 -> relu -> maxpool -> layer1-4 -> avgpool -> projection
        
        # Modify first conv layer for 1-channel input (MNIST)
        original_conv1 = base_model.conv1
        new_conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Copy weights from first channel (grayscale conversion)
        with torch.no_grad():
            new_conv1.weight.data = original_conv1.weight.data[:, 0:1, :, :]
        
        # Create feature extractor model
        model = nn.Sequential(
            new_conv1,
            base_model.bn1,
            base_model.relu,
            base_model.maxpool,
            base_model.layer1,
            base_model.layer2,
            base_model.layer3,
            base_model.layer4,
            base_model.avgpool,
            nn.Flatten(),
            nn.Linear(2048, embedding_dim)  # ResNet50 has 2048 features after avgpool
        )
        
    else:
        raise ValueError(f"Unsupported model: {model_name}. Currently only 'resnet50' is supported.")
    
    model = model.to(device)
    model.eval()
    
    return model


def extract_embeddings(
    model_name=None,
    embedding_dim=None,
    data_dir=None,
    output_dir=None,
    device=None,
    batch_size=None,
    save_model=True
):
    """
    Extract embeddings from selection pool.
    
    Args:
        model_name: Name of feature extractor model. If None, uses config.FEATURE_EXTRACTOR
        embedding_dim: Dimension of embeddings. If None, uses config.EMBEDDING_DIM
        data_dir: Directory containing data. If None, uses config.DATA_DIR
        output_dir: Directory to save embeddings. If None, uses config.EMBEDDINGS_DIR
        device: Device to use. If None, uses config.EMBEDDING_DEVICE
        batch_size: Batch size for extraction. If None, uses config.EMBEDDING_BATCH_SIZE
        save_model: Whether to save the feature extractor model
        
    Returns:
        Array of embeddings of shape (num_samples, embedding_dim)
    """
    if model_name is None:
        model_name = config.FEATURE_EXTRACTOR
    
    if embedding_dim is None:
        embedding_dim = config.EMBEDDING_DIM
    
    if data_dir is None:
        data_dir = config.DATA_DIR
    
    if output_dir is None:
        output_dir = config.EMBEDDINGS_DIR
    
    if device is None:
        device = config.EMBEDDING_DEVICE
    
    if batch_size is None:
        batch_size = config.EMBEDDING_BATCH_SIZE
    
    device = torch.device(device)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load selection pool data
    print("Loading selection pool data...")
    data, labels = load_selection_pool(data_dir=str(data_dir))
    print(f"  Loaded {len(data)} samples\n")
    
    # Create feature extractor
    print(f"Creating feature extractor ({model_name}, {embedding_dim}D)...")
    model = create_feature_extractor(model_name, embedding_dim, device=device)
    print(f"  ✓ Model created\n")
    
    # Save model if requested
    if save_model:
        model_path = output_dir / f"{model_name}_feature_extractor.pth"
        torch.save(model.state_dict(), model_path)
        print(f"  ✓ Saved feature extractor to {model_path}\n")
    
    # Create dataloader
    dataloader = create_dataloader(data, labels, batch_size=batch_size, shuffle=False)
    
    # Extract embeddings
    print("Extracting embeddings...")
    all_embeddings = []
    
    with torch.no_grad():
        for batch_data, _ in tqdm(dataloader, desc="Extracting"):
            batch_data = batch_data.to(device)
            embeddings = model(batch_data)
            all_embeddings.append(embeddings.cpu().numpy())
    
    # Concatenate all batches
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    
    print(f"  ✓ Extracted embeddings shape: {all_embeddings.shape}\n")
    
    # Save embeddings
    print("Saving embeddings...")
    embeddings_path = config.EMBEDDINGS_FILE
    np.save(embeddings_path, all_embeddings)
    print(f"  ✓ Saved embeddings to {embeddings_path}\n")
    
    # Verify embeddings
    print("Embedding statistics:")
    print(f"  Shape: {all_embeddings.shape}")
    print(f"  Mean: {all_embeddings.mean():.4f}")
    print(f"  Std: {all_embeddings.std():.4f}")
    print(f"  Min: {all_embeddings.min():.4f}")
    print(f"  Max: {all_embeddings.max():.4f}\n")
    
    print("✓ Embedding extraction completed successfully!")
    
    return all_embeddings


def load_embeddings(embeddings_file=None):
    """
    Load precomputed embeddings.
    
    Args:
        embeddings_file: Path to embeddings file. If None, uses config.EMBEDDINGS_FILE
        
    Returns:
        Array of embeddings of shape (num_samples, embedding_dim)
    """
    if embeddings_file is None:
        embeddings_file = config.EMBEDDINGS_FILE
    
    embeddings_file = Path(embeddings_file)
    
    if not embeddings_file.exists():
        raise FileNotFoundError(
            f"Embeddings file not found: {embeddings_file}\n"
            f"Run 'python embeddings/extract_embeddings.py' first to extract embeddings."
        )
    
    embeddings = np.load(embeddings_file)
    return embeddings


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract embeddings from selection pool")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Feature extractor model name (default: from config)"
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=None,
        help="Embedding dimension (default: from config)"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Data directory (default: from config)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: from config)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (default: from config)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size (default: from config)"
    )
    parser.add_argument(
        "--no-save-model",
        action="store_true",
        help="Don't save the feature extractor model"
    )
    
    args = parser.parse_args()
    
    extract_embeddings(
        model_name=args.model,
        embedding_dim=args.embedding_dim,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        device=args.device,
        batch_size=args.batch_size,
        save_model=not args.no_save_model
    )

