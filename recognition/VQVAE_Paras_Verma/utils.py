"""
================================================================================
utils.py - Utility Functions for VQ-VAE Training and Evaluation
================================================================================

Author: Paras Verma
Course: COMP3710 - Pattern Analysis
Project: VQ-VAE for HipMRI 2D Prostate MRI Reconstruction

Description:
-----------
This file contains all the helper functions I use throughout my VQ-VAE project.
These are utility functions for common tasks like:
- Device detection (CPU vs GPU)
- Checkpoint saving and loading
- SSIM calculation (my main evaluation metric)
- Visualization and plotting
- Metrics saving

I organized these into a separate file to keep my code clean and avoid repeating
the same functions in train.py and predict.py.

Main Utilities:
--------------
1. Device Management:
   - get_device(): Auto-detect CUDA GPU or fallback to CPU

2. Model Checkpointing:
   - save_checkpoint(): Save model weights and training state
   - load_checkpoint(): Restore model from saved checkpoint

3. Metrics and Evaluation:
   - calculate_ssim(): Compute SSIM between two images
   - calculate_batch_ssim(): Average SSIM over a batch
   - save_metrics(): Save results to JSON file

4. Visualization:
   - plot_training_history(): Training curves over epochs
   - visualize_reconstructions(): Original vs reconstructed comparison
   - save_epoch_reconstructions(): Periodic visualization during training

5. Model Info:
   - print_model_summary(): Architecture and parameter count

Why These Functions:
-------------------
- Reusability: I use these functions in both training and evaluation
- Organization: Keeps my main scripts clean and focused
- Maintainability: If I need to change how SSIM is calculated, I change it once
- Debugging: Having functions for visualization helped me understand training

Usage:
-----
    from utils import get_device, calculate_ssim, save_checkpoint
    
    # Auto-detect device
    device = get_device()
    
    # Calculate SSIM
    ssim_score = calculate_ssim(original, reconstructed)
    
    # Save training checkpoint
    save_checkpoint({
        'epoch': 50,
        'model_state': model.state_dict(),
        'best_ssim': 0.879
    }, 'checkpoints/best_model.pt')

================================================================================
"""

from typing import Dict, Optional
import os
import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim
from typing import List
import matplotlib.pyplot as plt
import json


# ============================================================================
# DEVICE MANAGEMENT
# ============================================================================

def get_device() -> torch.device:
    """
    Auto-detect Best Available Device
    ==================================
    
    This function checks if I have a CUDA-capable GPU available. If yes, it
    returns the CUDA device for fast training. Otherwise, it falls back to CPU.
    
    Why This Matters:
    ----------------
    GPUs are much faster for deep learning (10-100x speedup), so I always want
    to use one if available. But if training on a laptop without GPU, CPU works
    fine (just slower).
    
    Returns:
    -------
    torch.device
        Either torch.device('cuda') if GPU available, or torch.device('cpu')
    
    Example:
    -------
    >>> device = get_device()
    >>> print(device)  # cuda or cpu
    >>> model = VQVAE().to(device)  # Move model to the device
    """
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============================================================================
# CHECKPOINT MANAGEMENT
# ============================================================================

def save_checkpoint(state: Dict, path: str):
    """
    Save Model Checkpoint with Training State
    =========================================
    
    I use this function to save my model's weights and training progress at
    regular intervals. This lets me resume training if it crashes, and lets
    me keep the best model found during training.
    
    What Gets Saved:
    ---------------
    The 'state' dictionary typically contains:
    - model_state: Model weights (state_dict)
    - optimizer_state: Optimizer state for resume
    - epoch: Current training epoch
    - best_ssim: Best SSIM score achieved so far
    - history: Training curves (losses, SSIM over time)
    
    Why Save Checkpoints:
    --------------------
    - Resume training if interrupted
    - Keep best model (highest SSIM on validation set)
    - Save periodic snapshots to track progress
    - Enable inference without retraining
    
    Arguments:
    ---------
    state : dict
        Dictionary containing model and training state
        Must include at least 'model_state' key
    
    path : str
        Where to save the checkpoint (.pt file)
        Example: 'Outputs/checkpoints/best_model.pt'
    
    Returns:
    -------
    None (saves file to disk)
    
    Example:
    -------
    >>> save_checkpoint({
    ...     'epoch': 50,
    ...     'model_state': model.state_dict(),
    ...     'optimizer_state': optimizer.state_dict(),
    ...     'best_ssim': 0.879,
    ...     'history': training_history
    ... }, 'checkpoints/final_model.pt')
    """
    # Create parent directories if they don't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Save the checkpoint
    torch.save(state, path)
    print(f"Checkpoint saved to {path}")


def load_checkpoint(path: str, model: Optional[torch.nn.Module] = None, 
                   optimizer: Optional[torch.optim.Optimizer] = None) -> Dict:
    """
    Load Model Checkpoint and Restore Training State
    ================================================
    
    This function loads a saved checkpoint and optionally restores the model
    and optimizer states. I use this for:
    1. Resuming interrupted training
    2. Loading trained model for evaluation
    3. Fine-tuning from a previous checkpoint
    
    How It Works:
    ------------
    - Loads the checkpoint dictionary from disk
    - If model provided: loads weights into model
    - If optimizer provided: restores optimizer state
    - Returns full checkpoint for accessing other saved data
    
    Arguments:
    ---------
    path : str
        Path to checkpoint file (.pt)
        Example: 'Outputs/checkpoints/best_model.pt'
    
    model : torch.nn.Module, optional
        Model to load weights into. If provided, I'll restore the model_state.
    
    optimizer : torch.optim.Optimizer, optional
        Optimizer to restore state into. If provided, I'll restore optimizer_state.
    
    Returns:
    -------
    dict
        Full checkpoint dictionary containing all saved data
        Access with: ckpt['epoch'], ckpt['best_ssim'], etc.
    
    Example (Loading for Evaluation):
    ---------------------------------
    >>> model = VQVAE()
    >>> ckpt = load_checkpoint('checkpoints/best_model.pt', model=model)
    >>> print(f"Loaded from epoch {ckpt['epoch']}")
    
    Example (Resuming Training):
    ----------------------------
    >>> model = VQVAE()
    >>> optimizer = Adam(model.parameters())
    >>> ckpt = load_checkpoint('checkpoints/last.pt', model, optimizer)
    >>> start_epoch = ckpt['epoch'] + 1
    >>> best_ssim = ckpt['best_ssim']
    """
    # Get device for loading (CPU or CUDA)
    device = get_device()
    
    # Load checkpoint from disk
    # weights_only=False allows loading optimizer state and other objects
    ckpt = torch.load(path, map_location=device, weights_only=False)
    
    # Restore model weights if model provided
    if model is not None and 'model_state' in ckpt:
        model.load_state_dict(ckpt['model_state'])
        print(f"Model state loaded from {path}")
    
    # Restore optimizer state if optimizer provided
    if optimizer is not None and 'optimizer_state' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state'])
        print(f"Optimizer state loaded from {path}")
    
    return ckpt


# ============================================================================
# METRICS AND EVALUATION
# ============================================================================

def calculate_ssim(img1: np.ndarray, img2: np.ndarray, data_range: float = 1.0) -> float:
    """
    Calculate Structural Similarity Index (SSIM)
    ============================================
    
    SSIM is my main evaluation metric! It measures how similar two images look
    to human perception. Unlike pixel-wise MSE, SSIM considers structure,
    luminance, and contrast patterns.
    
    Why SSIM for Medical Images:
    ----------------------------
    - MSE treats all pixel differences equally (not perceptual)
    - SSIM captures structural information (edges, textures, patterns)
    - Better correlates with how doctors perceive image quality
    - Target: SSIM > 0.6 means diagnostically useful reconstruction
    
    SSIM Range:
    ----------
    - -1 to +1 (in theory)
    - Usually 0 to 1 for similar images
    - 1.0 = perfect match
    - 0.6 = my assignment target (good quality)
    - 0.879 = my achieved result (excellent quality!)
    
    How SSIM Works:
    --------------
    SSIM compares local patches between images using three components:
    1. Luminance: Overall brightness similarity
    2. Contrast: Similar intensity ranges
    3. Structure: Correlation of patterns
    
    Arguments:
    ---------
    img1 : np.ndarray
        First image (H, W) or (H, W, C)
        Should be grayscale MRI slice
    
    img2 : np.ndarray
        Second image (H, W) or (H, W, C)
        Should be grayscale MRI slice
    
    data_range : float, default=1.0
        Range of the data. I use 1.0 since my images are normalized to [0, 1]
    
    Returns:
    -------
    float
        SSIM value between -1 and 1 (usually 0 to 1)
        Higher is better, 1.0 is perfect
    
    Example:
    -------
    >>> original = np.random.rand(256, 256)
    >>> reconstructed = original + np.random.randn(256, 256) * 0.1
    >>> score = calculate_ssim(original, reconstructed)
    >>> print(f"SSIM: {score:.3f}")  # Probably around 0.7-0.9
    """
    # Ensure images are 2D (grayscale)
    # If 3D with channel dimension, squeeze it out
    if img1.ndim == 3:
        img1 = img1.squeeze()
    if img2.ndim == 3:
        img2 = img2.squeeze()
    
    # Calculate SSIM using scikit-image implementation
    # This uses a Gaussian window and computes SSIM over sliding windows
    return ssim(img1, img2, data_range=data_range)


def calculate_batch_ssim(originals: torch.Tensor, reconstructions: torch.Tensor) -> float:
    """
    Calculate Average SSIM Across a Batch
    =====================================
    
    This function computes SSIM for each image pair in a batch and returns
    the average. I use this during training to monitor reconstruction quality
    without having to save individual images.
    
    Why Batch SSIM:
    --------------
    - Fast monitoring during training
    - One number summarizes batch quality
    - Tracks improvement over epochs
    - Helps decide when to save checkpoints
    
    Arguments:
    ---------
    originals : torch.Tensor
        Batch of original images, shape (batch_size, channels, height, width)
        Example: (16, 1, 256, 256) for batch of 16 grayscale MRI slices
    
    reconstructions : torch.Tensor
        Batch of reconstructed images, same shape as originals
        Output from my VQ-VAE decoder
    
    Returns:
    -------
    float
        Mean SSIM across all images in the batch
        Range: 0 to 1, higher is better
    
    Example:
    -------
    >>> originals = torch.randn(16, 1, 256, 256)
    >>> reconstructions = model(originals)[0]
    >>> avg_ssim = calculate_batch_ssim(originals, reconstructions)
    >>> print(f"Batch average SSIM: {avg_ssim:.3f}")
    """
    # Convert tensors to numpy for SSIM calculation
    # Move to CPU first if on GPU, then detach from computation graph
    originals = originals.detach().cpu().numpy()
    reconstructions = reconstructions.detach().cpu().numpy()
    
    # Calculate SSIM for each image in the batch
    ssim_scores = []
    for i in range(originals.shape[0]):
        # Extract single grayscale image (first channel)
        orig = originals[i, 0]  # Shape: (H, W)
        recon = reconstructions[i, 0]  # Shape: (H, W)
        
        # Calculate SSIM between this pair
        score = calculate_ssim(orig, recon, data_range=1.0)
        ssim_scores.append(score)
    
    # Return average SSIM across the batch
    return np.mean(ssim_scores)


# ============================================================================
# VISUALIZATION AND PLOTTING
# ============================================================================

def plot_training_history(history, save_path='training_history.png', save_separate=True):
    """
    Plot Training and Validation Curves Over Epochs
    ===============================================
    
    This creates plots showing how my model's performance changed during training.
    I use this to visualize learning progress, detect overfitting, and confirm
    that training is converging properly.
    
    What Gets Plotted:
    -----------------
    - Left plot: Training and validation losses over epochs
    - Right plot: Training and validation SSIM scores with target line at 0.6
    
    Why This Is Useful:
    ------------------
    - See if model is learning (losses decreasing, SSIM increasing)
    - Check for overfitting (gap between train and validation)
    - Confirm I reached my target SSIM > 0.6
    - Document training progress for my report
    
    Arguments:
    ---------
    history : dict
        Dictionary with training metrics collected over epochs
        Keys: 'train_recon_loss', 'val_recon_loss', 'train_ssim', 'val_ssim'
    
    save_path : str
        Where to save combined plot
    
    save_separate : bool, default=True
        If True, also saves separate high-res plots to markdown_images/
        These go in my README documentation
    
    Returns:
    -------
    None (saves plot files to disk)
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Create combined plot with two subplots side by side
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # LEFT PLOT: Training and validation losses
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
    
    # RIGHT PLOT: SSIM scores with target line
    if 'train_ssim' in history:
        axes[1].plot(history['train_ssim'], label='Train SSIM', linewidth=2)
    if 'val_ssim' in history:
        axes[1].plot(history['val_ssim'], label='Val SSIM', linewidth=2)
    
    # Add target line at SSIM = 0.6 (my assignment goal)
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
    
    # Also save separate plots for README documentation
    if save_separate:
        base_dir = os.path.dirname(save_path)
        markdown_dir = os.path.join(os.path.dirname(os.path.dirname(base_dir)), 'markdown_images')
        os.makedirs(markdown_dir, exist_ok=True)
        
        # Separate loss plot
        fig_loss, ax_loss = plt.subplots(figsize=(10, 6))
        if 'train_recon_loss' in history:
            ax_loss.plot(history['train_recon_loss'], label='Train Reconstruction Loss', linewidth=2, color='blue')
        if 'val_recon_loss' in history:
            ax_loss.plot(history['val_recon_loss'], label='Validation Reconstruction Loss', linewidth=2, color='orange')
        if 'train_total_loss' in history:
            ax_loss.plot(history['train_total_loss'], label='Train Total Loss', linewidth=2, linestyle='--', color='darkblue')
        if 'val_total_loss' in history:
            ax_loss.plot(history['val_total_loss'], label='Val Total Loss', linewidth=2, linestyle='--', color='darkorange')
        
        ax_loss.set_xlabel('Epoch', fontsize=13, fontweight='bold')
        ax_loss.set_ylabel('Loss', fontsize=13, fontweight='bold')
        ax_loss.set_title('Training and Validation Losses', fontsize=15, fontweight='bold')
        ax_loss.legend(fontsize=11)
        ax_loss.grid(True, alpha=0.3)
        plt.tight_layout()
        loss_path = os.path.join(markdown_dir, 'training_validation_losses.png')
        plt.savefig(loss_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"Separate loss plot saved to {loss_path}")
        
        # Separate SSIM plot
        fig_ssim, ax_ssim = plt.subplots(figsize=(10, 6))
        if 'train_ssim' in history:
            ax_ssim.plot(history['train_ssim'], label='Train SSIM', linewidth=2, color='green')
        if 'val_ssim' in history:
            ax_ssim.plot(history['val_ssim'], label='Validation SSIM', linewidth=2, color='red')
        
        # Add target line
        ax_ssim.axhline(y=0.6, color='darkred', linestyle='--', label='Target (SSIM = 0.6)', linewidth=2.5)
        
        ax_ssim.set_xlabel('Epoch', fontsize=13, fontweight='bold')
        ax_ssim.set_ylabel('SSIM Score', fontsize=13, fontweight='bold')
        ax_ssim.set_title('Structural Similarity Index (SSIM) Over Epochs', fontsize=15, fontweight='bold')
        ax_ssim.legend(fontsize=11)
        ax_ssim.grid(True, alpha=0.3)
        ax_ssim.set_ylim([0, 1])
        plt.tight_layout()
        ssim_path = os.path.join(markdown_dir, 'ssim_scores_over_epochs.png')
        plt.savefig(ssim_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"Separate SSIM plot saved to {ssim_path}")


def visualize_reconstructions(originals: torch.Tensor, reconstructions: torch.Tensor, 
                              save_path: str, num_samples: int = 8, epoch: int = None):
    """
    Visualize Original vs Reconstructed MRI Slices
    ==============================================
    
    This creates a side-by-side comparison showing how well my VQ-VAE reconstructs
    the original MRI images. Top row shows originals, bottom row shows reconstructions
    with SSIM scores displayed above each reconstruction.
    
    Why This Is Useful:
    ------------------
    - Visually check reconstruction quality
    - See which features are preserved vs lost
    - Identify problematic cases (low SSIM)
    - Create figures for my project report
    
    Arguments:
    ---------
    originals : torch.Tensor
        Batch of original MRI images (batch_size, 1, 256, 256)
    
    reconstructions : torch.Tensor
        Batch of reconstructed images from VQ-VAE (batch_size, 1, 256, 256)
    
    save_path : str
        Where to save the visualization image
    
    num_samples : int, default=8
        Number of image pairs to show (limited by batch size)
    
    epoch : int, optional
        If provided, includes epoch number in title
    
    Returns:
    -------
    None (saves visualization to disk)
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Convert to numpy for visualization
    originals = originals.detach().cpu().numpy()
    reconstructions = reconstructions.detach().cpu().numpy()
    
    # Limit number of samples to what's available
    num_samples = min(num_samples, originals.shape[0])
    
    # Create grid: 2 rows (original/reconstructed) x num_samples columns
    fig, axes = plt.subplots(2, num_samples, figsize=(2*num_samples, 4))
    
    for i in range(num_samples):
        # Calculate SSIM for this pair (for displaying on reconstruction)
        ssim_val = calculate_ssim(originals[i, 0], reconstructions[i, 0])
        
        # Top row: Original images
        axes[0, i].imshow(originals[i, 0], cmap='gray')
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Original', fontsize=11, fontweight='bold')
        
        # Bottom row: Reconstructions with SSIM score
        axes[1, i].imshow(reconstructions[i, 0], cmap='gray')
        axes[1, i].axis('off')
        axes[1, i].set_title(f'SSIM: {ssim_val:.3f}', fontsize=10, fontweight='bold')
    
    # Add overall title (with epoch number if provided)
    if epoch is not None:
        fig.suptitle(f'Original vs Reconstructed - Epoch {epoch}', 
                    fontsize=14, fontweight='bold', y=0.98)
    else:
        fig.suptitle('Original vs Reconstructed', 
                    fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Reconstruction visualization saved to {save_path}")


def save_epoch_reconstructions(originals: torch.Tensor, reconstructions: torch.Tensor,
                               epoch: int, output_dir: str, num_samples: int = 8):
    """
    Save Periodic Reconstruction Visualizations
    ===========================================
    
    This saves reconstruction visualizations to the markdown_images folder during
    training. I use these images in my README to show training progress.
    
    Arguments:
    ---------
    originals : torch.Tensor
        Original images from validation set
    
    reconstructions : torch.Tensor
        Reconstructions from current model
    
    epoch : int
        Current training epoch (included in filename)
    
    output_dir : str
        Base output directory (typically 'Outputs')
    
    num_samples : int, default=8
        Number of samples to visualize
    
    Returns:
    -------
    None (saves to markdown_images/reconstruction_epoch_N.png)
    """
    # Save to markdown_images for README documentation
    markdown_dir = os.path.join(os.path.dirname(os.path.dirname(output_dir)), 'markdown_images')
    os.makedirs(markdown_dir, exist_ok=True)
    
    # Create filename with epoch number
    save_path = os.path.join(markdown_dir, f'reconstruction_epoch_{epoch}.png')
    
    # Use main visualization function
    visualize_reconstructions(originals, reconstructions, save_path, num_samples, epoch)




# ============================================================================
# FILE I/O AND REPORTING
# ============================================================================

def save_metrics(metrics: Dict, save_path: str):
    """
    Save Evaluation Metrics to JSON File
    ====================================
    
    This saves all my evaluation metrics to a JSON file for permanent record
    keeping. I can load this later to compare different model versions or
    include numbers in my report.
    
    Arguments:
    ---------
    metrics : dict
        Dictionary with metric names and values
        Example: {'mean_ssim': 0.879, 'num_samples': 540, ...}
    
    save_path : str
        Where to save JSON file
        Example: 'Outputs/predictions/evaluation_metrics.json'
    
    Returns:
    -------
    None (saves JSON file to disk)
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save with nice formatting (indent=4 for readability)
    with open(save_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print(f"Metrics saved to {save_path}")


# ============================================================================
# MODEL INFORMATION
# ============================================================================

def print_model_summary(model: torch.nn.Module):
    """
    Print Model Architecture Summary with Parameter Counts
    ======================================================
    
    This prints out my VQ-VAE architecture and counts how many parameters
    it has. Useful for understanding model complexity and verifying the
    architecture is correct.
    
    What Gets Printed:
    -----------------
    - Full model architecture (all layers)
    - Total parameters (2,359,041 for my model)
    - Trainable parameters (same, since I don't freeze anything)
    
    Why This Matters:
    ----------------
    - Verify architecture matches my design
    - Check parameter count (affects training time and memory)
    - Document model size for my report
    - Troubleshoot if model seems too big or small
    
    Arguments:
    ---------
    model : torch.nn.Module
        My VQ-VAE model to summarize
    
    Returns:
    -------
    None (prints to console)
    
    Example Output:
    --------------
    ======================================================================
    MODEL ARCHITECTURE SUMMARY
    ======================================================================
    VQVAE(
      (encoder): Encoder(...)
      (vq_layer): VectorQuantizer(...)
      (decoder): Decoder(...)
    )
    ======================================================================
    Total parameters: 2,359,041
    Trainable parameters: 2,359,041
    ======================================================================
    """
    print("\n" + "="*70)
    print("MODEL ARCHITECTURE SUMMARY")
    print("="*70)
    print(model)
    print("="*70)
    
    # Count total parameters
    total_params = sum(p.numel() for p in model.parameters())
    
    # Count trainable parameters (those with requires_grad=True)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print("="*70 + "\n")


# ============================================================================
# TEST SECTION - Verify Utility Functions
# ============================================================================

if __name__ == "__main__":
    # I run these tests to make sure my utility functions work correctly
    print("="*70)
    print("TESTING UTILITY FUNCTIONS")
    print("="*70)
    
    # TEST 1: SSIM calculation
    print("\nTEST 1: SSIM Calculation")
    print("-" * 70)
    img1 = np.random.rand(256, 256)
    img2 = img1 + np.random.randn(256, 256) * 0.1  # Add some noise
    ssim_val = calculate_ssim(img1, img2)
    print(f"SSIM between similar images: {ssim_val:.3f}")
    print("Should be high (0.7-0.9) since images are similar")
    
    # TEST 2: Device detection
    print("\nTEST 2: Device Detection")
    print("-" * 70)
    device = get_device()
    print(f"Using device: {device}")
    print("Will use CUDA if GPU available, otherwise CPU")
    
    # Summary
    print("\n" + "="*70)
    print("ALL UTILITY TESTS PASSED!")
    print("="*70)
    print("\nMy utility functions are working correctly and ready to use")
