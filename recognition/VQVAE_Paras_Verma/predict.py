# Create predict.py with evaluation function
"""predict.py
Evaluation and prediction script for trained VQ-VAE model.

Loads a trained VQ-VAE checkpoint and:
1. Generates reconstructions on test set
2. Calculates SSIM metrics (target: > 0.6)
3. Visualizes original vs reconstructed images
4. Saves metrics and visualizations
"""

import argparse
from pathlib import Path
import json

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# Import local modules
from dataset import HipMRIDataset
from modules import VQVAE
from utils import (
    get_device, load_checkpoint, calculate_ssim,
    visualize_reconstructions, save_metrics
)


def evaluate_model(model, loader, device, num_visualize=16):
    """Evaluate model on test set and collect metrics.
    
    Performs inference on entire test set, calculates SSIM for each sample,
    and collects statistics. Also saves sample images for visualization.
    
    Args:
        model: Trained VQ-VAE model
        loader: DataLoader for test set
        device: torch device
        num_visualize: Number of samples to visualize
    
    Returns:
        metrics: Dictionary with evaluation metrics
        sample_originals: Original images for visualization
        sample_recons: Reconstructed images for visualization
    """
    model.eval()
    
    all_ssim_scores = []
    all_recon_losses = []
    
    # Store samples for visualization
    sample_originals = []
    sample_recons = []
    num_collected = 0
    
    print(f"\nEvaluating on {len(loader)} batches...")
    
    with torch.no_grad():
        for batch_idx, images in enumerate(loader):
            images = images.to(device)
            
            # Generate reconstructions
            recon, vq_loss, perplexity = model(images)
            
            # Calculate reconstruction loss
            recon_loss = torch.nn.functional.mse_loss(recon, images)
            all_recon_losses.append(recon_loss.item())
            
            # Calculate SSIM for each image in batch
            images_np = images.cpu().numpy()
            recon_np = recon.cpu().numpy()
            
            for i in range(images.shape[0]):
                orig = images_np[i, 0]
                rec = recon_np[i, 0]
                ssim_score = calculate_ssim(orig, rec, data_range=1.0)
                all_ssim_scores.append(ssim_score)
            
            # Collect samples for visualization
            if num_collected < num_visualize:
                remaining = num_visualize - num_collected
                batch_samples = min(images.shape[0], remaining)
                sample_originals.append(images[:batch_samples])
                sample_recons.append(recon[:batch_samples])
                num_collected += batch_samples
            
            # Print progress
            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {batch_idx+1}/{len(loader)} batches...")
    
    # Concatenate visualization samples
    if len(sample_originals) > 0:
        sample_originals = torch.cat(sample_originals, dim=0)
        sample_recons = torch.cat(sample_recons, dim=0)
    else:
        sample_originals = None
        sample_recons = None
    
    # Calculate statistics
    mean_ssim = np.mean(all_ssim_scores)
    std_ssim = np.std(all_ssim_scores)
    min_ssim = np.min(all_ssim_scores)
    max_ssim = np.max(all_ssim_scores)
    
    mean_recon_loss = np.mean(all_recon_losses)
    
    # Count how many images meet the target
    target_ssim = 0.6
    num_above_target = sum(1 for s in all_ssim_scores if s >= target_ssim)
    pct_above_target = 100.0 * num_above_target / len(all_ssim_scores)
    
    metrics = {
        'mean_ssim': float(mean_ssim),
        'std_ssim': float(std_ssim),
        'min_ssim': float(min_ssim),
        'max_ssim': float(max_ssim),
        'mean_recon_loss': float(mean_recon_loss),
        'num_samples': len(all_ssim_scores),
        'num_above_target': int(num_above_target),
        'pct_above_target': float(pct_above_target),
        'target_ssim': target_ssim,
        'all_ssim_scores': [float(s) for s in all_ssim_scores]
    }
    
    return metrics, sample_originals, sample_recons
