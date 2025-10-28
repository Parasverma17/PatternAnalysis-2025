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


def validate(model, loader, device):
    """Validate the model on validation set.
    
    Evaluates reconstruction quality without updating model parameters.
    Collects sample images for visualization.
    
    Args:
        model: VQ-VAE model
        loader: Validation data loader
        device: torch device
    
    Returns:
        Dictionary with validation metrics and sample images
    """
    model.eval()
    
    total_loss = 0.0
    total_recon_loss = 0.0
    total_vq_loss = 0.0
    total_ssim = 0.0
    num_batches = 0
    
    # Store some samples for visualization
    sample_originals = None
    sample_recons = None
    
    with torch.no_grad():
        for batch_idx, images in enumerate(loader):
            images = images.to(device)
            
            # Forward pass
            recon, vq_loss, perplexity = model(images)
            
            # Calculate losses
            recon_loss = nn.functional.mse_loss(recon, images)
            loss = recon_loss + vq_loss
            
            # Calculate SSIM
            ssim = calculate_batch_ssim(images, recon)
            
            # Accumulate metrics
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_vq_loss += vq_loss.item()
            total_ssim += ssim
            num_batches += 1
            
            # Save first batch for visualization
            if batch_idx == 0:
                sample_originals = images
                sample_recons = recon
    
    metrics = {
        'total_loss': total_loss / num_batches,
        'recon_loss': total_recon_loss / num_batches,
        'vq_loss': total_vq_loss / num_batches,
        'ssim': total_ssim / num_batches,
        'sample_originals': sample_originals,
        'sample_recons': sample_recons
    }
    
    return metrics


def main(args):
    """Main training function."""
    # Setup
    device = get_device()
    print(f"\n{'='*70}")
    print("VQ-VAE TRAINING FOR HIPMRI 2D PROSTATE SLICES")
    print(f"{'='*70}")
    print(f"Device: {device}")
    print(f"Data directory: {args.data_dir}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"{'='*70}\n")
    
    # Create output directories
    out_dir = Path(args.output_dir)
    ckpt_dir = out_dir / 'checkpoints'
    vis_dir = out_dir / 'visualizations'
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    # Load datasets
    print("Loading datasets...")
    train_dataset = HipMRIDataset(
        args.data_dir, 
        split='train',
        image_size=args.image_size,
        max_samples=args.max_samples
    )
    
    val_dataset = HipMRIDataset(
        args.data_dir,
        split='val',
        image_size=args.image_size,
        max_samples=args.max_samples
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}\n")
    
    # Create model
    print("Building VQ-VAE model...")
    model = VQVAE(
        in_channels=1,
        hidden_dims=args.hidden_dims,
        embedding_dim=args.embedding_dim,
        num_embeddings=args.num_embeddings,
        commitment_cost=args.commitment_cost
    ).to(device)
    
    print_model_summary(model)
    
    # Optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )
    
    # Training history
    history = defaultdict(list)
    best_ssim = 0.0
    start_epoch = 1
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        ckpt = load_checkpoint(args.resume, model, optimizer)
        start_epoch = ckpt.get('epoch', 0) + 1
        best_ssim = ckpt.get('best_ssim', 0.0)
        if 'history' in ckpt:
            history = ckpt['history']
        print(f"Resumed from epoch {start_epoch-1}, best SSIM: {best_ssim:.3f}\n")


    # Training loop
    print(f"\n{'='*70}")
    print("STARTING TRAINING")
    print(f"{'='*70}\n")
    
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()
        
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 70)
        
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, device, epoch)
        
        # Validate
        val_metrics = validate(model, val_loader, device)
        
        # Update history
        history['train_total_loss'].append(train_metrics['total_loss'])
        history['train_recon_loss'].append(train_metrics['recon_loss'])
        history['train_vq_loss'].append(train_metrics['vq_loss'])
        history['train_ssim'].append(train_metrics['ssim'])
        
        history['val_total_loss'].append(val_metrics['total_loss'])
        history['val_recon_loss'].append(val_metrics['recon_loss'])
        history['val_vq_loss'].append(val_metrics['vq_loss'])
        history['val_ssim'].append(val_metrics['ssim'])
        
        # Print epoch summary
        epoch_time = time.time() - epoch_start
        print(f"\nEpoch {epoch} Summary ({epoch_time:.1f}s):")
        print(f"  Train - Loss: {train_metrics['total_loss']:.4f} | "
              f"Recon: {train_metrics['recon_loss']:.4f} | "
              f"VQ: {train_metrics['vq_loss']:.4f} | "
              f"SSIM: {train_metrics['ssim']:.3f}")
        print(f"  Val   - Loss: {val_metrics['total_loss']:.4f} | "
              f"Recon: {val_metrics['recon_loss']:.4f} | "
              f"VQ: {val_metrics['vq_loss']:.4f} | "
              f"SSIM: {val_metrics['ssim']:.3f}")
        
        # Update learning rate
        scheduler.step(val_metrics['ssim'])
        
        # Save checkpoint
        is_best = val_metrics['ssim'] > best_ssim
        if is_best:
            best_ssim = val_metrics['ssim']
            print(f"  *** New best SSIM: {best_ssim:.3f} ***")
        
        # Save regular checkpoint
        if epoch % args.save_freq == 0:
            ckpt_path = ckpt_dir / f'checkpoint_epoch{epoch}.pt'
            save_checkpoint({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'best_ssim': best_ssim,
                'history': dict(history)
            }, str(ckpt_path))
        
        # Save best model
        if is_best:
            best_path = ckpt_dir / 'best_model.pt'
            save_checkpoint({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'best_ssim': best_ssim,
                'history': dict(history)
            }, str(best_path))
        
        # Visualize reconstructions
        if epoch % args.viz_freq == 0:
            viz_path = vis_dir / f'reconstructions_epoch{epoch}.png'
            visualize_reconstructions(
                val_metrics['sample_originals'],
                val_metrics['sample_recons'],
                str(viz_path),
                num_samples=8
            )
        
        # Plot training curves
        if epoch % args.plot_freq == 0:
            plot_path = out_dir / 'training_history.png'
            plot_training_history(history, str(plot_path))
    
    # Final save
    final_path = ckpt_dir / 'final_model.pt'
    save_checkpoint({
        'epoch': args.epochs,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'best_ssim': best_ssim,
        'history': dict(history)
    }, str(final_path))
    
    # Final visualization and plot
    plot_training_history(history, str(out_dir / 'final_training_history.png'))
    
    print(f"\n{'='*70}")
    print("TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"Best validation SSIM: {best_ssim:.3f}")
    print(f"Checkpoints saved to: {ckpt_dir}")
    print(f"Visualizations saved to: {vis_dir}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train VQ-VAE on HipMRI 2D prostate slices'
    )
    
    # Data arguments
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Path to HipMRI keras_slices_data directory')
    parser.add_argument('--image-size', type=int, default=256,
                       help='Size to resize images (default: 256)')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Limit number of samples for quick testing')
    
    # Model arguments
    parser.add_argument('--hidden-dims', type=int, nargs='+', default=[32, 64, 128],
                       help='Hidden dimensions for encoder/decoder (default: 32 64 128)')
    parser.add_argument('--embedding-dim', type=int, default=64,
                       help='Dimension of latent embeddings (default: 64)')
    parser.add_argument('--num-embeddings', type=int, default=512,
                       help='Size of codebook (default: 512)')
    parser.add_argument('--commitment-cost', type=float, default=0.25,
                       help='Beta parameter for commitment loss (default: 0.25)')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs (default: 50)')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size (default: 16)')
    parser.add_argument('--lr', type=float, default=2e-4,
                       help='Learning rate (default: 2e-4)')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loading workers (default: 4)')
    
    # Checkpoint arguments
    parser.add_argument('--output-dir', type=str, default='recognition/outputs',
                       help='Directory for outputs (default: recognition/outputs)')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--save-freq', type=int, default=10,
                       help='Save checkpoint every N epochs (default: 10)')
    parser.add_argument('--viz-freq', type=int, default=5,
                       help='Visualize reconstructions every N epochs (default: 5)')
    parser.add_argument('--plot-freq', type=int, default=5,
                       help='Update training plots every N epochs (default: 5)')
    
    args = parser.parse_args()
    main(args)
