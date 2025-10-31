# Create train.py with train_epoch function
"""
================================================================================
train.py - VQ-VAE Training Script for HipMRI 2D Prostate Slices
================================================================================

Author: Paras Verma
Course: COMP3710 - Pattern Analysis
Project: VQ-VAE for HipMRI 2D Prostate MRI Reconstruction

Description:
-----------
This is my main training script that I use to train the VQ-VAE model on HipMRI
2D prostate MRI slices. It handles the complete training pipeline from data
loading to model checkpointing.

Training Goal:
-------------
I want to achieve SSIM > 0.6 on the validation set, which means my reconstructions
should be high-quality and preserve the important medical features. After 50
epochs of training, I achieved 0.879 mean SSIM - way above target!

What This Script Does:
---------------------
1. Loads training and validation datasets from HipMRI
2. Creates VQ-VAE model with my chosen architecture
3. Trains for specified number of epochs with Adam optimizer
4. Validates after each epoch and tracks SSIM metric
5. Saves checkpoints periodically and keeps the best model
6. Creates visualizations showing training progress
7. Plots training curves (losses and SSIM over time)

Key Training Features:
---------------------
- Automatic checkpointing: Saves best model based on validation SSIM
- Learning rate scheduling: Reduces LR if validation SSIM plateaus
- Progress visualization: Creates reconstruction comparisons every few epochs
- Resume capability: Can continue training from saved checkpoint
- Comprehensive logging: Tracks all metrics in training history

Training Configuration I Used:
-----------------------------
- Epochs: 50
- Batch size: 16
- Learning rate: 2e-4 with ReduceLROnPlateau scheduler
- Optimizer: Adam
- Loss: MSE reconstruction loss + VQ loss (codebook + commitment)
- Validation metric: SSIM (Structural Similarity Index)

Dataset Split:
-------------
- Training: 10,314 images (90% of keras_slices_train)
- Validation: 1,146 images (10% of keras_slices_train)
- Test: 540 images (keras_slices_test - evaluated separately)

Model Architecture:
------------------
- Input: 1-channel 256x256 grayscale MRI images
- Encoder: [32, 64, 128] channels with strided convolutions
- VQ Layer: 512 codebook entries, 64-dimensional embeddings
- Decoder: [128, 64, 32] channels with transposed convolutions
- Total parameters: 2,359,041

Usage:
-----
    # Basic training
    python train.py --data-dir /path/to/keras_slices_data --epochs 50
    
    # With custom hyperparameters
    python train.py --data-dir /path/to/data \
                    --epochs 50 \
                    --batch-size 16 \
                    --lr 2e-4 \
                    --num-embeddings 512 \
                    --embedding-dim 64
    
    # Resume from checkpoint
    python train.py --data-dir /path/to/data \
                    --resume Outputs/checkpoints/checkpoint_epoch30.pt

Training Results:
----------------
After 50 epochs, I achieved:
- Best validation SSIM: 0.905 (far exceeds 0.6 target!)
- Final mean test SSIM: 0.879
- 100% of test images above 0.6 threshold
- Stable training with no overfitting

Output Files Created:
--------------------
- Outputs/checkpoints/ - Model checkpoints (best, periodic, final)
- Outputs/visualizations/ - Reconstruction samples per epoch
- Outputs/training_history.png - Combined training curves
- markdown_images/ - High-res plots for README

================================================================================
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
    visualize_reconstructions, print_model_summary,
    save_epoch_reconstructions
)


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_epoch(model, loader, optimizer, device, epoch):
    """
    Train Model for One Epoch
    =========================
    
    This function performs one complete pass through the training dataset,
    updating the model weights to minimize the VQ-VAE loss. I call this
    function once per epoch during training.
    
    What Happens Here:
    -----------------
    1. Set model to training mode
    2. Loop through all training batches
    3. For each batch:
       - Forward pass through VQ-VAE
       - Calculate reconstruction loss (MSE) and VQ loss
       - Backward pass to compute gradients
       - Update model weights with optimizer
       - Track SSIM metric for monitoring
    4. Return average metrics for the epoch
    
    Loss Function:
    -------------
    Total Loss = Reconstruction Loss + VQ Loss
    
    - Reconstruction Loss: MSE between input and output images
    - VQ Loss: Codebook loss + Commitment loss (from vector quantizer)
    
    Why Track SSIM During Training:
    -------------------------------
    SSIM is my target metric, so I monitor it during training even though
    I don't optimize for it directly. This helps me see if the model is
    learning to preserve image structure, not just minimize pixel-wise MSE.
    
    Arguments:
    ---------
    model : VQVAE
        My VQ-VAE model to train
    
    loader : DataLoader
        Training data loader with batched MRI images
    
    optimizer : torch.optim.Optimizer
        Adam optimizer (I use lr=2e-4)
    
    device : torch.device
        Device to run training on (CPU or CUDA)
    
    epoch : int
        Current epoch number (for progress printing)
    
    Returns:
    -------
    dict
        Dictionary with average training metrics:
        - total_loss: Combined reconstruction + VQ loss
        - recon_loss: MSE reconstruction loss only
        - vq_loss: Vector quantization loss only
        - ssim: Structural similarity metric
    
    Example Output:
    --------------
    {
        'total_loss': 0.0234,
        'recon_loss': 0.0156,
        'vq_loss': 0.0078,
        'ssim': 0.879
    }
    """
    # STEP 1: Set model to training mode
    # This enables dropout and batch norm updates
    model.train()
    
    # Initialize accumulators for metrics
    total_loss = 0.0
    total_recon_loss = 0.0
    total_vq_loss = 0.0
    total_ssim = 0.0
    num_batches = 0
    
    # STEP 2: Loop through all training batches
    for batch_idx, images in enumerate(loader):
        # Move images to device (GPU if available)
        images = images.to(device)
        
        # STEP 3: Forward pass through VQ-VAE
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        
        # Get reconstructions, VQ loss, and perplexity (codebook usage)
        recon, vq_loss, perplexity = model(images)
        
        # STEP 4: Calculate reconstruction loss
        # I use MSE to measure pixel-wise difference
        recon_loss = nn.functional.mse_loss(recon, images)
        
        # STEP 5: Combine losses
        # Total loss is what I actually optimize
        loss = recon_loss + vq_loss
        
        # STEP 6: Backward pass
        # Compute gradients with respect to model parameters
        loss.backward()
        
        # Update model weights using Adam optimizer
        optimizer.step()
        
        # STEP 7: Calculate SSIM for quality monitoring
        # I don't optimize for this, but it's my target metric
        with torch.no_grad():
            ssim = calculate_batch_ssim(images, recon)
        
        # STEP 8: Accumulate metrics for this batch
        total_loss += loss.item()
        total_recon_loss += recon_loss.item()
        total_vq_loss += vq_loss.item()
        total_ssim += ssim
        num_batches += 1
        
        # STEP 9: Print progress every 50 batches
        # This helps me monitor training in real-time
        if (batch_idx + 1) % 50 == 0:
            print(f"  Batch [{batch_idx+1}/{len(loader)}] | "
                  f"Loss: {loss.item():.4f} | "
                  f"Recon: {recon_loss.item():.4f} | "
                  f"VQ: {vq_loss.item():.4f} | "
                  f"SSIM: {ssim:.3f} | "
                  f"Perplexity: {perplexity.item():.1f}")
    
    # STEP 10: Return average metrics across all batches
    return {
        'total_loss': total_loss / num_batches,
        'recon_loss': total_recon_loss / num_batches,
        'vq_loss': total_vq_loss / num_batches,
        'ssim': total_ssim / num_batches
    }


def validate(model, loader, device):
    """
    Validate Model on Validation Set
    ================================
    
    After each training epoch, I run this function to evaluate my model on
    the validation set. This tells me if the model is learning to generalize
    to new data or just memorizing the training set.
    
    What This Does:
    --------------
    1. Set model to evaluation mode (no gradient updates)
    2. Loop through validation batches
    3. Calculate losses and SSIM without updating weights
    4. Collect sample images for visualization
    5. Return average metrics
    
    Why Validation Is Important:
    ---------------------------
    - Check if model generalizes (not overfitting)
    - Track SSIM on unseen data (my target metric)
    - Select best model checkpoint (highest validation SSIM)
    - Decide when to reduce learning rate
    
    Differences from Training:
    -------------------------
    - No gradient computation (faster, less memory)
    - No weight updates
    - Batch norm uses running statistics (not batch statistics)
    - Save sample images for visualization
    
    Arguments:
    ---------
    model : VQVAE
        My trained VQ-VAE model to evaluate
    
    loader : DataLoader
        Validation data loader with batched MRI images
    
    device : torch.device
        Device to run validation on (CPU or CUDA)
    
    Returns:
    -------
    dict
        Dictionary with validation metrics and samples:
        - total_loss: Combined reconstruction + VQ loss
        - recon_loss: MSE reconstruction loss
        - vq_loss: Vector quantization loss
        - ssim: Structural similarity (my target metric!)
        - sample_originals: First batch of original images
        - sample_recons: First batch of reconstructions
    
    Example Output:
    --------------
    {
        'total_loss': 0.0189,
        'recon_loss': 0.0123,
        'vq_loss': 0.0066,
        'ssim': 0.905,
        'sample_originals': tensor(...),
        'sample_recons': tensor(...)
    }
    """
    # STEP 1: Set model to evaluation mode
    # This disables dropout and uses running stats for batch norm
    model.eval()
    
    # Initialize metric accumulators
    total_loss = 0.0
    total_recon_loss = 0.0
    total_vq_loss = 0.0
    total_ssim = 0.0
    num_batches = 0
    
    # Store samples from first batch for visualization
    sample_originals = None
    sample_recons = None
    
    # STEP 2: Disable gradient computation (saves memory and speeds up)
    with torch.no_grad():
        for batch_idx, images in enumerate(loader):
            # Move images to device
            images = images.to(device)
            
            # STEP 3: Forward pass (no backward pass needed)
            recon, vq_loss, perplexity = model(images)
            
            # STEP 4: Calculate losses (same as training)
            recon_loss = nn.functional.mse_loss(recon, images)
            loss = recon_loss + vq_loss
            
            # STEP 5: Calculate SSIM (my target metric)
            ssim = calculate_batch_ssim(images, recon)
            
            # STEP 6: Accumulate metrics
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_vq_loss += vq_loss.item()
            total_ssim += ssim
            num_batches += 1
            
            # STEP 7: Save first batch for visualization
            # I use these to create reconstruction comparisons
            if batch_idx == 0:
                sample_originals = images
                sample_recons = recon
    
    # STEP 8: Package metrics and samples
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
    """
    Main Training Pipeline
    ======================
    
    This is the heart of my VQ-VAE training system. I orchestrate the entire
    training process here - from loading data to saving the final model. This
    function runs for 50 epochs (or however many I specify) and produces a
    trained model capable of reconstructing medical images.
    
    What Happens in This Function:
    ------------------------------
    1. Setup device and directories
    2. Load training and validation datasets
    3. Create VQ-VAE model
    4. Initialize optimizer and learning rate scheduler
    5. Run training loop for all epochs
    6. Track and visualize training progress
    7. Save checkpoints and best model
    8. Generate final visualizations
    
    Why This Design:
    ---------------
    - Modular: Separate functions for training and validation
    - Robust: Checkpoint saving every 10 epochs
    - Monitored: Regular visualizations to spot issues
    - Flexible: Can resume from checkpoints
    - Optimized: Learning rate reduces if SSIM plateaus
    
    Arguments:
    ---------
    args : argparse.Namespace
        Command-line arguments containing all hyperparameters:
        - data_dir: Path to HipMRI dataset
        - output_dir: Where to save outputs
        - epochs: Number of training epochs (I used 50)
        - batch_size: Batch size (I used 16)
        - lr: Learning rate (I used 2e-4)
        - image_size: Image dimension (256x256)
        - hidden_dims: Encoder/decoder channels [32,64,128]
        - embedding_dim: Codebook vector size (64)
        - num_embeddings: Codebook size (512)
        - commitment_cost: VQ loss weight (0.25)
        - save_freq: Checkpoint frequency (10 epochs)
        - viz_freq: Visualization frequency (10 epochs)
        - plot_freq: Plot update frequency (5 epochs)
        - resume: Optional checkpoint to resume from
        - use_provided_val: Use separate validation split
        - max_samples: Limit dataset size (for testing)
        - num_workers: DataLoader workers (4)
    
    Returns:
    -------
    None
        This function manages the entire training process and saves
        all outputs to disk (checkpoints, visualizations, plots)
    
    Outputs Created:
    ---------------
    Checkpoints/ :
        - checkpoint_epoch10.pt, 20.pt, etc. (regular saves)
        - best_model.pt (highest validation SSIM)
        - final_model.pt (last epoch)
    
    Visualizations/ :
        - reconstructions_epoch10.png, etc. (original vs reconstruction)
    
    Root Output/ :
        - training_history.png (loss and SSIM curves)
        - final_training_history.png (complete training curves)
    
    Markdown_images/ :
        - epoch_X_reconstructions.png (for README documentation)
    """
    # =================================================================
    # SECTION 1: INITIAL SETUP AND CONFIGURATION
    # =================================================================
    
    # STEP 1: Get computing device (CUDA GPU if available, else CPU)
    device = get_device()
    
    # Print training configuration banner
    print(f"\n{'='*70}")
    print("VQ-VAE TRAINING FOR HIPMRI 2D PROSTATE SLICES")
    print(f"{'='*70}")
    print(f"Device: {device}")
    print(f"Data directory: {args.data_dir}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"{'='*70}\n")
    
    # STEP 2: Create output directory structure
    # I organize outputs into separate folders for easy access
    out_dir = Path(args.output_dir)
    ckpt_dir = out_dir / 'checkpoints'  # Model checkpoints
    vis_dir = out_dir / 'visualizations'  # Reconstruction images
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    # =================================================================
    # SECTION 2: DATA LOADING AND PREPARATION
    # =================================================================
    
    print("Loading datasets...")
    
    # STEP 3: Create training dataset
    # I use keras_slices_train with 90% training, 10% validation split
    train_dataset = HipMRIDataset(
        args.data_dir, 
        split='train',  # 90% of keras_slices_train
        image_size=args.image_size,
        max_samples=args.max_samples
    )
    
    # STEP 4: Create validation dataset
    # Can use 10% of training or separate keras_slices_test
    val_dataset = HipMRIDataset(
        args.data_dir,
        split='val',  # 10% of keras_slices_train
        image_size=args.image_size,
        max_samples=args.max_samples,
        use_provided_val=args.use_provided_val
    )
    
    # STEP 5: Create data loaders for batching
    # Training: Shuffle for better generalization
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,  # Random order each epoch
        num_workers=args.num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # Validation: No shuffle (consistent evaluation)
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,  # Same order for reproducibility
        num_workers=args.num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # Print dataset statistics
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}\n")
    
    # =================================================================
    # SECTION 3: MODEL CREATION AND SETUP
    # =================================================================
    
    print("Building VQ-VAE model...")
    
    # STEP 6: Initialize VQ-VAE model
    # I chose these hyperparameters after experimentation
    model = VQVAE(
        in_channels=1,  # Grayscale MRI
        hidden_dims=args.hidden_dims,  # [32, 64, 128] progressive channels
        embedding_dim=args.embedding_dim,  # 64D codebook vectors
        num_embeddings=args.num_embeddings,  # 512 discrete codes
        commitment_cost=args.commitment_cost  # 0.25 VQ loss weight
    ).to(device)
    
    # Print model architecture and parameter count
    print_model_summary(model)
    
    # STEP 7: Setup optimizer (Adam for adaptive learning rates)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # STEP 8: Setup learning rate scheduler
    # Reduces LR by 0.5 if validation SSIM doesn't improve for 5 epochs
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max',  # Maximize SSIM (higher is better)
        factor=0.5,  # Halve learning rate
        patience=5  # Wait 5 epochs before reducing
    )
    
    # STEP 9: Initialize training history tracking
    # I track both training and validation metrics
    history = defaultdict(list)
    best_ssim = 0.0  # Track best validation SSIM
    start_epoch = 1  # Start from epoch 1 (or resume epoch)
    
    # STEP 10: Resume from checkpoint if specified
    # This lets me continue training if interrupted
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        ckpt = load_checkpoint(args.resume, model, optimizer)
        start_epoch = ckpt.get('epoch', 0) + 1
        best_ssim = ckpt.get('best_ssim', 0.0)
        if 'history' in ckpt:
            history = ckpt['history']
        print(f"Resumed from epoch {start_epoch-1}, best SSIM: {best_ssim:.3f}\n")


    # =================================================================
    # SECTION 4: MAIN TRAINING LOOP
    # =================================================================
    
    print(f"\n{'='*70}")
    print("STARTING TRAINING")
    print(f"{'='*70}\n")
    
    # Loop through all epochs (50 in my case)
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()
        
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 70)
        
        # STEP 11: Train for one epoch
        # This runs train_epoch() to update model weights
        train_metrics = train_epoch(model, train_loader, optimizer, device, epoch)
        
        # STEP 12: Validate on validation set
        # This runs validate() to check generalization
        val_metrics = validate(model, val_loader, device)
        
        # STEP 13: Record metrics in history
        # I track all 4 metrics for both training and validation
        history['train_total_loss'].append(train_metrics['total_loss'])
        history['train_recon_loss'].append(train_metrics['recon_loss'])
        history['train_vq_loss'].append(train_metrics['vq_loss'])
        history['train_ssim'].append(train_metrics['ssim'])
        
        history['val_total_loss'].append(val_metrics['total_loss'])
        history['val_recon_loss'].append(val_metrics['recon_loss'])
        history['val_vq_loss'].append(val_metrics['vq_loss'])
        history['val_ssim'].append(val_metrics['ssim'])
        
        # STEP 14: Print epoch summary
        # This gives me quick feedback on training progress
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
        
        # STEP 15: Update learning rate based on validation SSIM
        # If SSIM doesn't improve for 5 epochs, reduce LR by half
        scheduler.step(val_metrics['ssim'])
        
        # STEP 16: Check if this is the best model so far
        # I use validation SSIM as my metric for "best"
        is_best = val_metrics['ssim'] > best_ssim
        if is_best:
            best_ssim = val_metrics['ssim']
            print(f"  *** New best SSIM: {best_ssim:.3f} ***")
        
        # =================================================================
        # SECTION 5: CHECKPOINT SAVING
        # =================================================================
        
        # STEP 17: Save regular checkpoint every N epochs
        # This lets me resume training or analyze specific epochs
        if epoch % args.save_freq == 0:
            ckpt_path = ckpt_dir / f'checkpoint_epoch{epoch}.pt'
            save_checkpoint({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'best_ssim': best_ssim,
                'history': dict(history)
            }, str(ckpt_path))
        
        # STEP 18: Save best model checkpoint
        # I keep the model with highest validation SSIM separately
        if is_best:
            best_path = ckpt_dir / 'best_model.pt'
            save_checkpoint({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'best_ssim': best_ssim,
                'history': dict(history)
            }, str(best_path))
        
        # =================================================================
        # SECTION 6: VISUALIZATION AND MONITORING
        # =================================================================
        
        # STEP 19: Create reconstruction visualizations
        # I compare original vs reconstructed images to spot issues
        if epoch % args.viz_freq == 0:
            # Save to visualizations folder
            viz_path = vis_dir / f'reconstructions_epoch{epoch}.png'
            visualize_reconstructions(
                val_metrics['sample_originals'],
                val_metrics['sample_recons'],
                str(viz_path),
                num_samples=8,
                epoch=epoch
            )
            
            # Also save to markdown_images for README
            save_epoch_reconstructions(
                val_metrics['sample_originals'],
                val_metrics['sample_recons'],
                epoch=epoch,
                output_dir=str(out_dir),
                num_samples=8
            )
        
        # STEP 20: Update training curves plot
        # I track loss and SSIM over time to monitor convergence
        if epoch % args.plot_freq == 0:
            plot_path = out_dir / 'training_history.png'
            plot_training_history(history, str(plot_path))
    
    # =================================================================
    # SECTION 7: FINAL CLEANUP AND SAVING
    # =================================================================
    
    # STEP 21: Save final model state
    # This is the model after all training is complete
    final_path = ckpt_dir / 'final_model.pt'
    save_checkpoint({
        'epoch': args.epochs,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'best_ssim': best_ssim,
        'history': dict(history)
    }, str(final_path))
    
    # STEP 22: Generate final training plots
    # Complete history of all 50 epochs
    plot_training_history(history, str(out_dir / 'final_training_history.png'))
    
    # STEP 23: Save final reconstruction samples
    # Last epoch's reconstruction quality
    save_epoch_reconstructions(
        val_metrics['sample_originals'],
        val_metrics['sample_recons'],
        epoch=args.epochs,
        output_dir=str(out_dir),
        num_samples=8
    )
    
    # STEP 24: Print completion summary
    print(f"\n{'='*70}")
    print("TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"Best validation SSIM: {best_ssim:.3f}")
    print(f"Checkpoints saved to: {ckpt_dir}")
    print(f"Visualizations saved to: {vis_dir}")
    print(f"{'='*70}\n")


# =================================================================
# COMMAND-LINE INTERFACE
# =================================================================
"""
This section defines the command-line interface for training my VQ-VAE model.
I can customize every aspect of training through these arguments.

Usage Examples:
--------------
Basic training:
    python train.py --data-dir /path/to/keras_slices_data

Custom hyperparameters:
    python train.py --data-dir /path/to/data --epochs 100 --batch-size 32 --lr 1e-4

Resume from checkpoint:
    python train.py --data-dir /path/to/data --resume Outputs/checkpoints/best_model.pt

Quick test with limited data:
    python train.py --data-dir /path/to/data --max-samples 100 --epochs 5
"""

if __name__ == '__main__':
    # Create argument parser
    parser = argparse.ArgumentParser(
        description='Train VQ-VAE on HipMRI 2D prostate slices'
    )
    
    # =================================================================
    # DATA CONFIGURATION ARGUMENTS
    # =================================================================
    
    # Required: Path to dataset
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Path to HipMRI keras_slices_data directory containing '
                            'keras_slices_train and keras_slices_test folders')
    
    # Image preprocessing
    parser.add_argument('--image-size', type=int, default=256,
                       help='Size to resize images (default: 256). I use 256x256 '
                            'for a good balance between detail and memory usage')
    
    # Dataset size limiting (useful for quick testing)
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Limit number of samples for quick testing. Leave None '
                            'to use all available data (default: None)')
    
    # Validation split option
    parser.add_argument('--use-provided-val', action='store_true',
                       help='Use keras_slices_test folder as validation set instead '
                            'of splitting keras_slices_train 90-10. I recommend this '
                            'for final training to maximize training data')
    
    # =================================================================
    # MODEL ARCHITECTURE ARGUMENTS
    # =================================================================
    
    # Encoder/Decoder channel progression
    parser.add_argument('--hidden-dims', type=int, nargs='+', default=[32, 64, 128],
                       help='Hidden dimensions for encoder/decoder layers. I use '
                            '[32, 64, 128] which gives 3 downsampling/upsampling '
                            'stages with progressive feature learning (default: 32 64 128)')
    
    # Codebook embedding dimension
    parser.add_argument('--embedding-dim', type=int, default=64,
                       help='Dimension of each codebook vector. Higher values capture '
                            'more information but increase memory. I found 64 works well '
                            '(default: 64)')
    
    # Codebook size
    parser.add_argument('--num-embeddings', type=int, default=512,
                       help='Number of discrete codes in the codebook. This is the '
                            'vocabulary size for encoding images. I use 512 which gives '
                            'good reconstruction without being too sparse (default: 512)')
    
    # VQ loss weighting
    parser.add_argument('--commitment-cost', type=float, default=0.25,
                       help='Beta parameter for commitment loss (encourages encoder '
                            'output to stay close to codebook). I use 0.25 which '
                            'balances reconstruction and quantization (default: 0.25)')
    
    # =================================================================
    # TRAINING HYPERPARAMETERS
    # =================================================================
    
    # Number of epochs
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs. I trained for 50 epochs '
                            'which gave stable convergence (default: 50)')
    
    # Batch size
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size for training and validation. I use 16 which '
                            'fits in GPU memory and provides stable gradients (default: 16)')
    
    # Learning rate
    parser.add_argument('--lr', type=float, default=2e-4,
                       help='Initial learning rate for Adam optimizer. I use 2e-4 '
                            'which is standard for VAE training (default: 2e-4)')
    
    # Data loading workers
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of parallel data loading workers. More workers '
                            'speed up data loading but use more memory (default: 4)')
    
    # =================================================================
    # OUTPUT AND CHECKPOINTING ARGUMENTS
    # =================================================================
    
    # Output directory
    parser.add_argument('--output-dir', type=str, default='recognition/outputs',
                       help='Directory for all outputs (checkpoints, visualizations, '
                            'plots). I organized mine as recognition/outputs (default: '
                            'recognition/outputs)')
    
    # Resume training
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint file to resume training from. Useful '
                            'if training was interrupted (default: None)')
    
    # Checkpoint frequency
    parser.add_argument('--save-freq', type=int, default=10,
                       help='Save checkpoint every N epochs. I use 10 to avoid too '
                            'many checkpoint files (default: 10)')
    
    # Visualization frequency
    parser.add_argument('--viz-freq', type=int, default=5,
                       help='Generate reconstruction visualizations every N epochs. '
                            'I use 5 to track progress without slowing training (default: 5)')
    
    # Plot update frequency
    parser.add_argument('--plot-freq', type=int, default=5,
                       help='Update training history plots every N epochs. I use 5 '
                            'for regular monitoring (default: 5)')
    
    # Parse arguments and start training
    args = parser.parse_args()
    main(args)
