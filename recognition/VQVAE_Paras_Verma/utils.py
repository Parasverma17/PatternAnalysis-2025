"""Utility helpers for VQ-VAE training, evaluation, and visualization."""
from typing import Dict, Optional
import os
import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim
from typing import List
import matplotlib.pyplot as plt


def get_device() -> torch.device:
    """Get the best available device (CUDA if available, otherwise CPU)."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def save_checkpoint(state: Dict, path: str):
    """Save model checkpoint with training state.
    
    Args:
        state: Dictionary containing model_state, optimizer_state, epoch, etc.
        path: Path to save checkpoint
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)
    print(f"Checkpoint saved to {path}")


def load_checkpoint(path: str, model: Optional[torch.nn.Module] = None, 
                   optimizer: Optional[torch.optim.Optimizer] = None) -> Dict:
    """Load model checkpoint and restore training state.
    
    Args:
        path: Path to checkpoint file
        model: Model to load state into (optional)
        optimizer: Optimizer to load state into (optional)
    
    Returns:
        Dictionary containing checkpoint state
    """
    device = get_device()
    ckpt = torch.load(path, map_location=device)
    
    if model is not None and 'model_state' in ckpt:
        model.load_state_dict(ckpt['model_state'])
        print(f"Model state loaded from {path}")
    
    if optimizer is not None and 'optimizer_state' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state'])
        print(f"Optimizer state loaded from {path}")
    
    return ckpt


def calculate_ssim(img1: np.ndarray, img2: np.ndarray, data_range: float = 1.0) -> float:
    """Calculate Structural Similarity Index (SSIM) between two images.
    
    SSIM is a perceptual metric that quantifies image quality degradation.
    Values range from -1 to 1, where 1 indicates perfect similarity.
    Target: SSIM > 0.6 for this assignment.
    
    Args:
        img1: First image (H, W) or (H, W, C)
        img2: Second image (H, W) or (H, W, C)
        data_range: Range of the data (1.0 for normalized images)
    
    Returns:
        SSIM value between -1 and 1
    """
    # Ensure 2D images (grayscale)
    if img1.ndim == 3:
        img1 = img1.squeeze()
    if img2.ndim == 3:
        img2 = img2.squeeze()
    
    return ssim(img1, img2, data_range=data_range)


def calculate_batch_ssim(originals: torch.Tensor, reconstructions: torch.Tensor) -> float:
    """Calculate average SSIM across a batch of images.
    
    Args:
        originals: Batch of original images (B, C, H, W)
        reconstructions: Batch of reconstructed images (B, C, H, W)
    
    Returns:
        Mean SSIM across the batch
    """
    originals = originals.detach().cpu().numpy()
    reconstructions = reconstructions.detach().cpu().numpy()
    
    ssim_scores = []
    for i in range(originals.shape[0]):
        orig = originals[i, 0]  # Take first channel (grayscale)
        recon = reconstructions[i, 0]
        score = calculate_ssim(orig, recon, data_range=1.0)
        ssim_scores.append(score)
    
    return np.mean(ssim_scores)


def plot_training_history(history: Dict[str, List[float]], save_path: str):
    """Plot training and validation losses over epochs.
    
    Args:
        history: Dictionary with keys like 'train_loss', 'val_loss', 'train_ssim', etc.
        save_path: Path to save the plot (PNG)
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot losses
    if 'train_recon_loss' in history:
        axes[0].plot(history['train_recon_loss'], label='Train Recon Loss', linewidth=2)
    if 'val_recon_loss' in history:
        axes[0].plot(history['val_recon_loss'], label='Val Recon Loss', linewidth=2)
    if 'train_total_loss' in history:
        axes[0].plot(history['train_total_loss'], label='Train Total Loss', 
                    linewidth=2, linestyle='--')
    
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Losses', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot SSIM
    if 'train_ssim' in history:
        axes[1].plot(history['train_ssim'], label='Train SSIM', linewidth=2)
    if 'val_ssim' in history:
        axes[1].plot(history['val_ssim'], label='Val SSIM', linewidth=2)
    
    # Add target line at SSIM = 0.6
    axes[1].axhline(y=0.6, color='r', linestyle='--', label='Target (0.6)', linewidth=2)
    
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('SSIM', fontsize=12)
    axes[1].set_title('Structural Similarity Index', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training history plot saved to {save_path}")


def visualize_reconstructions(originals: torch.Tensor, reconstructions: torch.Tensor, 
                              save_path: str, num_samples: int = 8):
    """Visualize original and reconstructed MRI slices side by side.
    
    Args:
        originals: Batch of original images (B, C, H, W)
        reconstructions: Batch of reconstructed images (B, C, H, W)
        save_path: Path to save visualization
        num_samples: Number of samples to visualize
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    originals = originals.detach().cpu().numpy()
    reconstructions = reconstructions.detach().cpu().numpy()
    
    num_samples = min(num_samples, originals.shape[0])
    
    fig, axes = plt.subplots(2, num_samples, figsize=(2*num_samples, 4))
    
    for i in range(num_samples):
        # Original
        axes[0, i].imshow(originals[i, 0], cmap='gray')
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Original', fontsize=10, fontweight='bold')
        
        # Reconstruction
        axes[1, i].imshow(reconstructions[i, 0], cmap='gray')
        axes[1, i].axis('off')
        
        # Calculate and display SSIM for each pair
        ssim_val = calculate_ssim(originals[i, 0], reconstructions[i, 0])
        if i == 0:
            axes[1, i].set_title(f'Recon\nSSIM: {ssim_val:.3f}', fontsize=10, fontweight='bold')
        else:
            axes[1, i].set_title(f'SSIM: {ssim_val:.3f}', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Reconstruction visualization saved to {save_path}")
