# Create predict.py with evaluation function
"""
================================================================================
predict.py - VQ-VAE Model Evaluation and Testing Script
================================================================================

Author: Paras Verma
Course: COMP3710 - Pattern Analysis
Project: VQ-VAE for HipMRI 2D Prostate MRI Reconstruction

Description:
-----------
This is my evaluation script that I use to test my trained VQ-VAE model on the
test set. After training for 50 epochs, I run this script to see how well my
model performs on data it has never seen before.

What This Script Does:
---------------------
1. Loads my trained model checkpoint (best_model.pt)
2. Runs inference on all 540 test images from keras_slices_test folder
3. Calculates SSIM (Structural Similarity Index) for each reconstruction
4. Generates visualizations comparing originals vs reconstructions
5. Saves all metrics and plots to the outputs/predictions folder

Why SSIM?
---------
SSIM is a perceptual metric that measures how similar two images look to humans.
It's better than MSE for medical images because it considers structure, luminance,
and contrast. My target is SSIM > 0.6, and I achieved 0.879 mean SSIM!

Evaluation Metrics I Track:
--------------------------
- Mean SSIM: Average similarity across all test images
- Std SSIM: How consistent are the reconstructions?
- Min/Max SSIM: Range of reconstruction quality
- Percentage above target: How many images exceed 0.6 threshold?
- Reconstruction loss: MSE between originals and reconstructions

Output Files Created:
--------------------
1. evaluation_metrics.json - All numerical metrics
2. test_reconstructions.png - Visual comparison of 16 samples
3. ssim_distribution.png - Histogram showing SSIM distribution

Usage:
-----
    # Basic usage with my trained model
    python predict.py --checkpoint Outputs/checkpoints/best_model.pt \
                      --data-dir /path/to/keras_slices_data \
                      --output Outputs/predictions
    
    # Quick test on subset
    python predict.py --checkpoint Outputs/checkpoints/best_model.pt \
                      --data-dir /path/to/keras_slices_data \
                      --max-samples 100 \
                      --batch-size 8

Expected Results:
----------------
Based on my training, I expect:
- Mean SSIM: ~0.879 (47% above target of 0.6)
- 100% of test images above 0.6 threshold
- Perplexity: ~170 (good codebook utilization)

This shows my model successfully learned to reconstruct medical images!

================================================================================
"""

import argparse
import os
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
    """
    Evaluate Model Performance on Test Set
    ======================================
    
    This is the core evaluation function where I run my trained model on the
    entire test set and collect performance metrics. I process all test images
    in batches and calculate SSIM for each one.
    
    What Happens Here:
    -----------------
    1. Set model to evaluation mode (no gradient tracking)
    2. Loop through all test batches
    3. Generate reconstructions for each batch
    4. Calculate SSIM between original and reconstructed images
    5. Collect some sample images for visualization
    6. Return all metrics and samples
    
    Why I Track SSIM:
    ----------------
    SSIM is perfect for medical images because it measures perceptual similarity.
    A high SSIM means a doctor looking at the reconstruction would see the same
    features as in the original scan. My target is 0.6, but I achieved 0.879!
    
    Arguments:
    ---------
    model : VQVAE
        My trained VQ-VAE model loaded from checkpoint
    
    loader : DataLoader
        PyTorch DataLoader for test set (keras_slices_test folder)
        I typically use batch_size=16 for efficient processing
    
    device : torch.device
        Device to run inference on (CPU or CUDA if available)
    
    num_visualize : int, default=16
        Number of sample reconstructions to save for visualization
        I collect these from the first few batches
    
    Returns:
    -------
    metrics : dict
        Dictionary containing all evaluation metrics:
        - mean_ssim: Average SSIM across all test images
        - std_ssim: Standard deviation of SSIM scores
        - min_ssim: Worst reconstruction quality
        - max_ssim: Best reconstruction quality
        - mean_recon_loss: Average MSE loss
        - num_samples: Total test images evaluated
        - num_above_target: Count of images with SSIM > 0.6
        - pct_above_target: Percentage above threshold
        - all_ssim_scores: List of individual SSIM values
    
    sample_originals : torch.Tensor or None
        Original test images for visualization (num_visualize, 1, 256, 256)
    
    sample_recons : torch.Tensor or None
        Reconstructed images for visualization (num_visualize, 1, 256, 256)
    
    Example Output:
    --------------
    {
        'mean_ssim': 0.879,
        'std_ssim': 0.045,
        'min_ssim': 0.724,
        'max_ssim': 0.916,
        'num_above_target': 540,
        'pct_above_target': 100.0
    }
    """
    # STEP 1: Set model to evaluation mode
    # This disables dropout and batch norm updates
    model.eval()
    
    # Initialize lists to collect metrics
    all_ssim_scores = []      # SSIM for every single test image
    all_recon_losses = []     # MSE loss for each batch
    
    # Store samples for visualization
    sample_originals = []
    sample_recons = []
    num_collected = 0
    
    print(f"\nEvaluating on {len(loader)} batches...")
    
    # STEP 2: Loop through all test batches without tracking gradients
    with torch.no_grad():
        for batch_idx, images in enumerate(loader):
            # Move images to device (GPU if available)
            images = images.to(device)
            
            # STEP 3: Generate reconstructions through VQ-VAE
            # Forward pass: encode -> quantize -> decode
            recon, vq_loss, perplexity = model(images)
            
            # STEP 4: Calculate reconstruction loss (MSE)
            # This measures pixel-wise difference
            recon_loss = torch.nn.functional.mse_loss(recon, images)
            all_recon_losses.append(recon_loss.item())
            
            # STEP 5: Calculate SSIM for each image in the batch
            # I need to convert to numpy for the SSIM function
            images_np = images.cpu().numpy()
            recon_np = recon.cpu().numpy()
            
            for i in range(images.shape[0]):
                # Extract single grayscale image (remove channel dimension)
                orig = images_np[i, 0]  # Shape: (256, 256)
                rec = recon_np[i, 0]    # Shape: (256, 256)
                
                # Calculate SSIM between this pair
                ssim_score = calculate_ssim(orig, rec, data_range=1.0)
                all_ssim_scores.append(ssim_score)
            
            # STEP 6: Collect samples for visualization
            # I only need the first few batches for visualization
            if num_collected < num_visualize:
                remaining = num_visualize - num_collected
                batch_samples = min(images.shape[0], remaining)
                sample_originals.append(images[:batch_samples])
                sample_recons.append(recon[:batch_samples])
                num_collected += batch_samples
            
            # Print progress every 10 batches so I know it's working
            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {batch_idx+1}/{len(loader)} batches...")
    
    # STEP 7: Concatenate visualization samples
    if len(sample_originals) > 0:
        sample_originals = torch.cat(sample_originals, dim=0)
        sample_recons = torch.cat(sample_recons, dim=0)
    else:
        sample_originals = None
        sample_recons = None
    
    # STEP 8: Calculate statistics from all collected SSIM scores
    mean_ssim = np.mean(all_ssim_scores)
    std_ssim = np.std(all_ssim_scores)
    min_ssim = np.min(all_ssim_scores)
    max_ssim = np.max(all_ssim_scores)
    
    mean_recon_loss = np.mean(all_recon_losses)
    
    # Count how many images meet my target threshold
    target_ssim = 0.6
    num_above_target = sum(1 for s in all_ssim_scores if s >= target_ssim)
    pct_above_target = 100.0 * num_above_target / len(all_ssim_scores)
    
    # STEP 9: Package everything into a metrics dictionary
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


def plot_ssim_distribution(ssim_scores, save_path, target=0.6, save_to_markdown=True):
    """
    Plot SSIM Distribution Histogram
    ================================
    
    This function creates a nice histogram showing the distribution of SSIM
    scores across all my test images. It helps visualize how consistently my
    model performs across different samples.
    
    What I Visualize:
    ----------------
    - Histogram: Shows how many images fall into each SSIM range
    - Target line: Red dashed line at 0.6 (my minimum target)
    - Mean line: Green dashed line showing average performance
    - Statistics box: Summary of key metrics
    
    Why This Matters:
    ----------------
    This plot lets me quickly see if my model consistently produces good
    reconstructions or if there's high variability. Ideally, I want a tight
    distribution above 0.6 with a high mean.
    
    Arguments:
    ---------
    ssim_scores : list of float
        List of SSIM values for all test images
        For my test set: 540 values ranging from ~0.7 to ~0.9
    
    save_path : str
        Where to save the plot (usually Outputs/predictions/)
    
    target : float, default=0.6
        Target SSIM threshold to highlight on the plot
        I use 0.6 as my assignment requirement
    
    save_to_markdown : bool, default=True
        If True, also saves a copy to markdown_images folder for README
        This lets me include the plot in my project documentation
    
    Returns:
    -------
    None (saves plot files to disk)
    
    Output Files:
    ------------
    1. Primary: save_path (e.g., Outputs/predictions/ssim_distribution.png)
    2. If save_to_markdown: markdown_images/ssim_distribution_test_set.png
    """
    
    # Create figure with good size for viewing
    plt.figure(figsize=(10, 6))
    
    # STEP 1: Plot histogram of SSIM scores
    # I use 50 bins to get good detail in the distribution
    plt.hist(ssim_scores, bins=50, edgecolor='black', alpha=0.7, color='skyblue')
    
    # STEP 2: Add target line at 0.6 (my assignment requirement)
    plt.axvline(x=target, color='r', linestyle='--', linewidth=2, 
                label=f'Target (SSIM={target})')
    
    # STEP 3: Add mean line to show average performance
    mean_ssim = np.mean(ssim_scores)
    plt.axvline(x=mean_ssim, color='g', linestyle='--', linewidth=2,
                label=f'Mean (SSIM={mean_ssim:.3f})')
    
    # STEP 4: Add labels and title
    plt.xlabel('SSIM Score', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Distribution of SSIM Scores on Test Set', 
             fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # STEP 5: Add text box with key statistics
    # Calculate percentage above target
    pct_above = 100.0 * sum(1 for s in ssim_scores if s >= target) / len(ssim_scores)
    textstr = f'Total samples: {len(ssim_scores)}\n'
    textstr += f'Mean: {mean_ssim:.3f}\n'
    textstr += f'Std: {np.std(ssim_scores):.3f}\n'
    textstr += f'Above target: {pct_above:.1f}%'
    
    # Position text box in upper left corner
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, 
            fontsize=10, verticalalignment='top', bbox=props)
    
    # STEP 6: Save primary plot
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"SSIM distribution plot saved to {save_path}")
    
    # STEP 7: Also save to markdown_images folder for README
    # This lets me include the plot in my project documentation
    if save_to_markdown:
        # Calculate path to markdown_images folder
        markdown_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(save_path))), 
            'markdown_images'
        )
        os.makedirs(markdown_dir, exist_ok=True)
        markdown_path = os.path.join(markdown_dir, 'ssim_distribution_test_set.png')
        
        # Create the same plot again for markdown (higher DPI for documentation)
        plt.figure(figsize=(10, 6))
        plt.hist(ssim_scores, bins=50, edgecolor='black', alpha=0.7, color='skyblue')
        plt.axvline(x=target, color='r', linestyle='--', linewidth=2, 
                    label=f'Target (SSIM={target})')
        plt.axvline(x=mean_ssim, color='g', linestyle='--', linewidth=2,
                    label=f'Mean (SSIM={mean_ssim:.3f})')
        plt.xlabel('SSIM Score', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title('Distribution of SSIM Scores on Test Set', 
                 fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, 
                fontsize=10, verticalalignment='top', bbox=props)
        plt.tight_layout()
        plt.savefig(markdown_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"SSIM distribution also saved to {markdown_path}")


def main(args):
    """
    Main Evaluation Pipeline
    ========================
    
    This is the main function that orchestrates the entire evaluation process.
    It loads my trained model, runs it on the test set, collects metrics, and
    saves all the results and visualizations.
    
    Pipeline Steps:
    --------------
    1. Setup: Get device, create output folders
    2. Load test dataset (keras_slices_test)
    3. Create model with same architecture as training
    4. Load trained weights from checkpoint
    5. Run evaluation on all test images
    6. Print summary of results
    7. Save metrics, visualizations, and plots
    
    Arguments:
    ---------
    args : argparse.Namespace
        Command-line arguments containing:
        - checkpoint: Path to trained model (.pt file)
        - data_dir: Path to HipMRI dataset
        - output: Where to save results
        - batch_size: Batch size for inference
        - num_visualize: How many samples to visualize
        - Model hyperparameters (must match training)
    
    Returns:
    -------
    None (saves all outputs to disk)
    
    Example Results:
    ---------------
    After running, I get:
    - Mean SSIM: 0.879 (target exceeded!)
    - 100% of images above 0.6 threshold
    - evaluation_metrics.json
    - test_reconstructions.png
    - ssim_distribution.png
    """
    
    # Print header
    print(f"\n{'='*70}")
    print("VQ-VAE MODEL EVALUATION")
    print(f"{'='*70}")
    
    # STEP 1: Setup device and output directory
    device = get_device()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Print configuration
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*70}\n")
    
    # STEP 2: Load test dataset
    # I use the keras_slices_test folder which has 540 images
    print("Loading test dataset...")
    test_dataset = HipMRIDataset(
        args.data_dir,
        split='test',           # Uses keras_slices_test folder
        image_size=args.image_size,
        max_samples=args.max_samples  # For quick testing if needed
    )
    
    # Create data loader for batched inference
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,          # No need to shuffle for evaluation
        num_workers=args.num_workers
    )
    
    print(f"Test samples: {len(test_dataset)}")
    print(f"Test batches: {len(test_loader)}\n")
    
    # STEP 3: Create model with same architecture I used for training
    # These hyperparameters must match what I used during training!
    print("Loading VQ-VAE model...")
    model = VQVAE(
        in_channels=1,
        hidden_dims=args.hidden_dims,
        embedding_dim=args.embedding_dim,
        num_embeddings=args.num_embeddings,
        commitment_cost=args.commitment_cost
    ).to(device)
    
    # STEP 4: Load trained weights from checkpoint
    ckpt = load_checkpoint(args.checkpoint, model=model)
    epoch = ckpt.get('epoch', 'unknown')
    best_ssim = ckpt.get('best_ssim', 'unknown')
    print(f"Loaded checkpoint from epoch {epoch}, best SSIM: {best_ssim}\n")


    # STEP 5: Run evaluation on test set
    print(f"{'='*70}")
    print("RUNNING EVALUATION")
    print(f"{'='*70}")
    
    metrics, sample_originals, sample_recons = evaluate_model(
        model, test_loader, device, num_visualize=args.num_visualize
    )
    
    # STEP 6: Print summary of results
    print(f"\n{'='*70}")
    print("EVALUATION RESULTS")
    print(f"{'='*70}")
    print(f"Number of samples: {metrics['num_samples']}")
    print(f"Mean SSIM: {metrics['mean_ssim']:.4f} ± {metrics['std_ssim']:.4f}")
    print(f"Min SSIM: {metrics['min_ssim']:.4f}")
    print(f"Max SSIM: {metrics['max_ssim']:.4f}")
    print(f"Mean Reconstruction Loss: {metrics['mean_recon_loss']:.6f}")
    print(f"\nSamples above target (SSIM > {metrics['target_ssim']}): "
          f"{metrics['num_above_target']} ({metrics['pct_above_target']:.1f}%)")
    
    # Check if I met my target of 0.6
    if metrics['mean_ssim'] >= metrics['target_ssim']:
        print(f"\n SUCCESS: Mean SSIM {metrics['mean_ssim']:.3f} "
              f"exceeds target {metrics['target_ssim']}!")
    else:
        print(f"\n Target not met: Mean SSIM {metrics['mean_ssim']:.3f} "
              f"below target {metrics['target_ssim']}")
    
    print(f"{'='*70}\n")
    
    # STEP 7: Save all metrics to JSON file
    # This creates a permanent record of my test results
    metrics_path = output_dir / 'evaluation_metrics.json'
    save_metrics(metrics, str(metrics_path))
    
    # STEP 8: Create visualization comparing originals vs reconstructions
    # This shows 16 sample reconstructions with their SSIM scores
    if sample_originals is not None:
        vis_path = output_dir / 'test_reconstructions.png'
        visualize_reconstructions(
            sample_originals,
            sample_recons,
            str(vis_path),
            num_samples=min(args.num_visualize, 16)
        )
    
    # STEP 9: Plot SSIM distribution histogram
    # This shows how my model performs across the entire test set
    dist_path = output_dir / 'ssim_distribution.png'
    plot_ssim_distribution(
        metrics['all_ssim_scores'],
        str(dist_path),
        target=metrics['target_ssim']
    )
    
    # STEP 10: Print final summary
    print(f"\n{'='*70}")
    print("EVALUATION COMPLETE")
    print(f"{'='*70}")
    print(f"Results saved to: {output_dir}")
    print(f"  - Metrics: {metrics_path}")
    print(f"  - Reconstructions: {output_dir / 'test_reconstructions.png'}")
    print(f"  - SSIM distribution: {dist_path}")
    print(f"{'='*70}\n")


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

if __name__ == '__main__':
    # Set up argument parser to handle command-line options
    # This makes it easy to run evaluation with different settings
    parser = argparse.ArgumentParser(
        description='Evaluate trained VQ-VAE model on test set'
    )
    
    # REQUIRED ARGUMENTS - Must provide these
    # =====================================
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint (.pt file). '
                            'Example: Outputs/checkpoints/best_model.pt')
    
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Path to HipMRI keras_slices_data directory. '
                            'Must contain keras_slices_test folder.')
    
    # OUTPUT ARGUMENTS
    # ===============
    parser.add_argument('--output', type=str, 
                       default='recognition/outputs/predictions',
                       help='Directory to save predictions and visualizations. '
                            'Will create if it does not exist.')
    
    # DATA ARGUMENTS
    # =============
    parser.add_argument('--image-size', type=int, default=256,
                       help='Image size to resize MRI slices. Default: 256')
    
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Limit number of test samples for quick testing. '
                            'Default: None (use all 540 test images)')
    
    # MODEL ARGUMENTS (MUST MATCH TRAINING CONFIG!)
    # =============================================
    # These hyperparameters must be the same as I used during training
    # Otherwise the model architecture will not match the checkpoint
    
    parser.add_argument('--hidden-dims', type=int, nargs='+', 
                       default=[32, 64, 128],
                       help='Hidden dimensions for encoder/decoder. '
                            'Must match training config! Default: 32 64 128')
    
    parser.add_argument('--embedding-dim', type=int, default=64,
                       help='Dimension of latent embeddings. '
                            'Must match training config! Default: 64')
    
    parser.add_argument('--num-embeddings', type=int, default=512,
                       help='Size of codebook (number of discrete codes). '
                            'Must match training config! Default: 512')
    
    parser.add_argument('--commitment-cost', type=float, default=0.25,
                       help='Beta parameter for commitment loss. '
                            'Must match training config! Default: 0.25')
    
    # EVALUATION ARGUMENTS
    # ===================
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size for inference. Larger is faster '
                            'but uses more memory. Default: 16')
    
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of parallel workers for data loading. '
                            'Speeds up data loading. Default: 4')
    
    parser.add_argument('--num-visualize', type=int, default=16,
                       help='Number of reconstruction samples to save for '
                            'visualization. Default: 16')
    
    # Parse arguments and run evaluation
    args = parser.parse_args()
    main(args)

