# Create train.py with train_epoch function
"""train.py
Training script for VQ-VAE on HipMRI 2D prostate MRI slices.

Trains a Vector Quantized Variational Autoencoder to generate high-quality
reconstructions of medical images. Target: SSIM > 0.6 on validation set.

"""
import argparse
import os
from pathlib import Path
from collections import defaultdict
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Import from local modules
from dataset import HipMRIDataset
from modules import VQVAE
from utils import (
    get_device, save_checkpoint, load_checkpoint,
    calculate_batch_ssim, plot_training_history,
    visualize_reconstructions, print_model_summary
)


def train_epoch(model, loader, optimizer, device, epoch):
    """Train for one epoch.
    
    Performs forward pass, calculates VQ-VAE loss (reconstruction + VQ losses),
    backward pass, and tracks SSIM metric for quality monitoring.
    
    Args:
        model: VQ-VAE model
        loader: Training data loader
        optimizer: Optimizer (Adam)
        device: torch device
        epoch: Current epoch number
    
    Returns:
        Dictionary with training metrics (loss, SSIM, etc.)
    """
    model.train()
    
    total_loss = 0.0
    total_recon_loss = 0.0
    total_vq_loss = 0.0
    total_ssim = 0.0
    num_batches = 0
    
    for batch_idx, images in enumerate(loader):
        images = images.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        recon, vq_loss, perplexity = model(images)
        
        # Calculate reconstruction loss (MSE)
        recon_loss = nn.functional.mse_loss(recon, images)
        
        # Total loss: reconstruction + VQ losses
        loss = recon_loss + vq_loss
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Calculate SSIM for monitoring
        with torch.no_grad():
            ssim = calculate_batch_ssim(images, recon)
        
        # Accumulate metrics
        total_loss += loss.item()
        total_recon_loss += recon_loss.item()
        total_vq_loss += vq_loss.item()
        total_ssim += ssim
        num_batches += 1
        
        # Print progress every 50 batches
        if (batch_idx + 1) % 50 == 0:
            print(f"  Batch [{batch_idx+1}/{len(loader)}] | "
                  f"Loss: {loss.item():.4f} | "
                  f"Recon: {recon_loss.item():.4f} | "
                  f"VQ: {vq_loss.item():.4f} | "
                  f"SSIM: {ssim:.3f} | "
                  f"Perplexity: {perplexity.item():.1f}")
    
    # Return average metrics
    return {
        'total_loss': total_loss / num_batches,
        'recon_loss': total_recon_loss / num_batches,
        'vq_loss': total_vq_loss / num_batches,
        'ssim': total_ssim / num_batches
    }
