"""Utility helpers for VQ-VAE training, evaluation, and visualization."""
from typing import Dict, Optional
import os
import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim


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
