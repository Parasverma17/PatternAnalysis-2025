"""Utility helpers for VQ-VAE training, evaluation, and visualization."""
from typing import Dict, Optional
import os
import torch


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