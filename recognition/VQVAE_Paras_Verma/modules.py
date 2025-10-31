"""
================================================================================
modules.py - VQ-VAE Architecture Components for Medical Image Reconstruction
================================================================================

Author: Paras Verma
Course: COMP3710 - Pattern Analysis
Project: VQ-VAE for HipMRI 2D Prostate MRI Reconstruction

Description:
-----------
This file contains all the neural network components I built for my VQ-VAE 
(Vector Quantized Variational Autoencoder) model. I'm using it to reconstruct 
2D prostate MRI slices from the HipMRI dataset with high quality.

The architecture I designed has three main parts:
1. Encoder - Takes 256x256 MRI images and compresses them down to a latent space
2. Vector Quantizer - Converts continuous latent vectors into discrete codes 
   using a learned codebook (this is the "VQ" part)
3. Decoder - Takes the discrete codes and reconstructs the original image

Why VQ-VAE?
----------
I chose VQ-VAE because it learns a discrete latent space, which is really useful
for medical images. Unlike regular VAEs that use continuous latents, VQ-VAE 
creates a codebook of "visual patterns" that it can mix and match to recreate
images. This makes the model more interpretable and often gives better quality.

Architecture Design:
------------------
- Input: 1-channel grayscale MRI images (256x256 pixels)
- Hidden dimensions: [32, 64, 128] channels as we go deeper
- Embedding dimension: 64 (size of each latent vector)
- Codebook size: 512 entries (number of discrete patterns)
- Output: Reconstructed 256x256 image with values in [0, 1]

I designed this to achieve my target of SSIM > 0.6 on the validation set, which
means the reconstructions should look very similar to the originals.

Key Components:
--------------
1. ResidualBlock - Helper block that adds skip connections for better gradients
2. Encoder - Downsamples images using strided convolutions
3. VectorQuantizer - Quantizes continuous vectors to discrete codebook entries
4. Decoder - Upsamples back to original resolution using transposed convolutions
5. VQVAE - Complete model that chains all components together

References:
----------
- "Neural Discrete Representation Learning" (van den Oord et al., 2017)
- "Generating Diverse High-Fidelity Images with VQ-VAE-2" (Razavi et al., 2019)

Usage:
-----
    from modules import VQVAE
    
    # Create model with default settings
    model = VQVAE(
        in_channels=1,
        hidden_dims=[32, 64, 128],
        embedding_dim=64,
        num_embeddings=512,
        commitment_cost=0.25
    )
    
    # Forward pass
    x = torch.randn(4, 1, 256, 256)  # Batch of 4 MRI images
    recon, vq_loss, perplexity = model(x)
    
    print(f"Input: {x.shape} -> Output: {recon.shape}")
    print(f"VQ Loss: {vq_loss.item():.4f}")
    print(f"Codebook usage: {perplexity.item():.1f}/512")

================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """
    Residual Block with Skip Connection
    ===================================
    
    This is a helper building block I use throughout my encoder and decoder.
    The key idea is to add a "skip connection" that lets gradients flow more
    easily during training. Without this, deep networks can be hard to train.
    
    What it does:
    ------------
    Takes an input, processes it through two convolutional layers with batch
    normalization and ReLU activation, then adds the original input back.
    This is the classic ResNet-style residual connection.
    
    Architecture:
    ------------
    Input (x) -> Conv -> BN -> ReLU -> Conv -> BN -> Add(x) -> ReLU -> Output
                                                        ^
                                                        |
                                                  Skip connection
    
    Why I use this:
    --------------
    - Helps gradients flow backward during training (solves vanishing gradients)
    - Allows me to stack many layers without training difficulties
    - Improves feature learning by letting the network learn "changes" instead
      of completely new features
    
    Arguments:
    ---------
    channels : int
        Number of input and output channels. I keep it the same so the skip
        connection can add properly (dimensions must match).
    
    Returns:
    -------
    torch.Tensor
        Output with same shape as input: (batch_size, channels, height, width)
    
    Example:
    -------
    >>> block = ResidualBlock(channels=64)
    >>> x = torch.randn(4, 64, 32, 32)
    >>> out = block(x)
    >>> print(out.shape)  # Same as input: (4, 64, 32, 32)
    """
    
    def __init__(self, channels: int):
        super().__init__()
        
        # First convolution: 3x3 kernel with padding=1 to keep spatial dimensions
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        
        # Second convolution: another 3x3 with padding=1
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the residual block.
        
        I save the input as 'residual', process it through two conv layers,
        then add it back. This is the key to making deep networks trainable.
        
        Arguments:
        ---------
        x : torch.Tensor
            Input tensor of shape (batch_size, channels, height, width)
        
        Returns:
        -------
        torch.Tensor
            Output with residual connection applied, same shape as input
        """
        # STEP 1: Save the original input for the skip connection
        residual = x
        
        # STEP 2: First conv block (Conv -> BN -> ReLU)
        out = F.relu(self.bn1(self.conv1(x)))
        
        # STEP 3: Second conv block (Conv -> BN, but no ReLU yet)
        out = self.bn2(self.conv2(out))
        
        # STEP 4: Add the skip connection (this is what makes it "residual")
        out = out + residual
        
        # STEP 5: Final ReLU activation after adding residual
        out = F.relu(out)
        
        return out


class Encoder(nn.Module):
    """
    Convolutional Encoder for VQ-VAE
    ================================
    
    This is the first part of my VQ-VAE. It takes the full 256x256 MRI image
    and compresses it down to a much smaller latent representation. Think of it
    as creating a compact "summary" of the important features in the image.
    
    How it works:
    ------------
    I use strided convolutions to gradually reduce the spatial dimensions while
    increasing the number of channels. This lets the network extract increasingly
    abstract features as it goes deeper.
    
    The progression I designed:
    - Start: 1 channel (grayscale) at 256x256
    - After layer 1: 32 channels at 128x128 (stride=2 downsamples by 2x)
    - After layer 2: 64 channels at 64x64
    - After layer 3: 128 channels at 32x32
    - Final: 64 channels at 32x32 (embedding dimension)
    
    Why residual blocks?
    ------------------
    After each downsampling, I add two residual blocks. This helps the network
    learn better features because the skip connections make training more stable.
    
    Arguments:
    ---------
    in_channels : int, default=1
        Number of input channels. I use 1 for grayscale MRI images.
    
    hidden_dims : list of int, default=[32, 64, 128]
        Number of channels at each layer. I progressively increase this to
        capture more complex patterns as the spatial size decreases.
    
    embedding_dim : int, default=64
        Size of the final output embedding. This is what gets fed into the
        vector quantization layer. I chose 64 as a good balance between
        expressiveness and computational efficiency.
    
    Returns:
    -------
    torch.Tensor
        Encoded latent representation with shape:
        (batch_size, embedding_dim, height/8, width/8)
        For 256x256 input: (batch_size, 64, 32, 32)
    
    Example:
    -------
    >>> encoder = Encoder(in_channels=1, hidden_dims=[32, 64, 128], embedding_dim=64)
    >>> x = torch.randn(4, 1, 256, 256)  # 4 MRI images
    >>> latent = encoder(x)
    >>> print(latent.shape)  # (4, 64, 32, 32) - compressed 64x in pixels!
    """
    
    def __init__(self, in_channels: int = 1, hidden_dims: list = [32, 64, 128], 
                 embedding_dim: int = 64):
        super().__init__()
        
        modules = []
        
        # LAYER 1: Initial downsampling (256x256 -> 128x128)
        # I use kernel=4, stride=2, padding=1 which halves the spatial dimensions
        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, hidden_dims[0], kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(hidden_dims[0]),  # Normalize for stable training
                nn.ReLU()  # Non-linearity
            )
        )
        
        # LAYERS 2-3: Further downsampling with residual blocks
        # Each layer halves the spatial dimensions again (128->64->32)
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    # Downsample: reduce spatial size, increase channels
                    nn.Conv2d(hidden_dims[i], hidden_dims[i+1], kernel_size=4, 
                             stride=2, padding=1),
                    nn.BatchNorm2d(hidden_dims[i+1]),
                    nn.ReLU(),
                    
                    # Residual blocks: refine features at this resolution
                    ResidualBlock(hidden_dims[i+1]),
                    ResidualBlock(hidden_dims[i+1])
                )
            )
        
        # FINAL LAYER: Project to embedding dimension
        # This creates the final latent representation that goes to VQ layer
        # Keeps spatial size the same (32x32) but changes channels to embedding_dim
        modules.append(
            nn.Sequential(
                nn.Conv2d(hidden_dims[-1], embedding_dim, kernel_size=3, padding=1),
                ResidualBlock(embedding_dim)  # One more residual for good measure
            )
        )
        
        # Combine all layers into a sequential model
        self.encoder = nn.Sequential(*modules)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input images to latent space.
        
        Arguments:
        ---------
        x : torch.Tensor
            Input MRI images of shape (batch_size, 1, 256, 256)
        
        Returns:
        -------
        torch.Tensor
            Latent embeddings of shape (batch_size, embedding_dim, 32, 32)
        """
        return self.encoder(x)


class VectorQuantizer(nn.Module):
    """
    Vector Quantization Layer - The Heart of VQ-VAE
    ==============================================
    
    This is the most unique part of my VQ-VAE! It's what makes it different
    from regular VAEs. Instead of using continuous latent vectors, I discretize
    them by mapping each vector to the nearest entry in a learned codebook.
    
    The Big Idea:
    ------------
    Think of the codebook as a dictionary of 512 "visual patterns" that my model
    learns. When the encoder produces a latent vector, I find the closest pattern
    in the codebook and use that instead. This discretization is what "VQ" means.
    
    Why discrete?
    ------------
    - Creates a compact, interpretable representation
    - Each code can be thought of as a specific MRI pattern
    - Better for generation because we can sample from discrete codes
    - Similar to how JPEG compression uses a discrete cosine transform
    
    The Challenge - Backpropagation:
    ------------------------------
    The quantization operation (finding nearest neighbor) is not differentiable!
    So I use the "straight-through estimator" trick: during forward pass I use
    the quantized values, but during backward pass I copy gradients directly
    from output to input. It's a clever approximation that works well.
    
    Loss Components:
    ---------------
    1. Codebook Loss: ||sg[z] - e||^2
       - Moves codebook entries closer to encoder outputs
       - sg = stop gradient (encoder doesn't learn from this)
    
    2. Commitment Loss: ||z - sg[e]||^2
       - Encourages encoder to commit to codebook entries
       - Prevents encoder from moving away too fast
       - Beta controls the strength (I use 0.25)
    
    Arguments:
    ---------
    num_embeddings : int, default=512
        Size of the codebook (K). I chose 512 as a good balance - enough to
        capture diverse patterns but not so many that some go unused.
    
    embedding_dim : int, default=64
        Dimension of each codebook vector (D). Must match encoder output.
    
    commitment_cost : float, default=0.25
        Beta parameter that controls commitment loss weight. Higher values
        force the encoder to commit more strongly to codebook entries.
    
    Returns (in forward pass):
    -------------------------
    quantized : torch.Tensor
        Quantized latent vectors, shape (batch_size, embedding_dim, H, W)
    
    vq_loss : torch.Tensor
        Vector quantization loss (codebook + commitment loss)
    
    perplexity : torch.Tensor
        Measure of codebook usage. Higher is better - means we're using more
        of the codebook entries. Max possible is 512 (all codes used equally).
    
    Example:
    -------
    >>> vq = VectorQuantizer(num_embeddings=512, embedding_dim=64)
    >>> z = torch.randn(4, 64, 32, 32)  # Encoder output
    >>> z_q, vq_loss, perplexity = vq(z)
    >>> print(f"Quantized: {z_q.shape}, Loss: {vq_loss:.4f}, Usage: {perplexity:.1f}/512")
    """
    
    def __init__(self, num_embeddings: int = 512, embedding_dim: int = 64, 
                 commitment_cost: float = 0.25):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        
        # Create the codebook as an embedding layer
        # Each of the 512 entries is a 64-dimensional vector
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        
        # Initialize codebook with uniform distribution
        # I use a small range to avoid extreme values initially
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)
    
    def forward(self, z: torch.Tensor):
        """
        Quantize continuous latents to discrete codebook entries.
        
        This is where the magic happens! I take continuous vectors from the
        encoder and map them to discrete codes from my learned codebook.
        
        Arguments:
        ---------
        z : torch.Tensor
            Encoder output of shape (batch_size, embedding_dim, H, W)
            For my setup: (batch_size, 64, 32, 32)
        
        Returns:
        -------
        quantized : torch.Tensor
            Quantized latents, same shape as input (batch_size, 64, 32, 32)
        
        vq_loss : torch.Tensor
            Scalar loss for training the codebook and encoder
        
        perplexity : torch.Tensor
            Scalar metric showing codebook usage (higher is better)
        """
        
        # STEP 1: Reshape from (B, D, H, W) to (B, H, W, D)
        # I need to do this because PyTorch expects the feature dimension last
        # for the distance calculations
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.embedding_dim)  # Flatten to (B*H*W, D)
        
        # STEP 2: Calculate distances to all codebook entries
        # Using the formula: ||z - e||^2 = ||z||^2 + ||e||^2 - 2*z*e
        # This is more efficient than computing distances directly
        distances = (
            torch.sum(z_flattened ** 2, dim=1, keepdim=True)  # ||z||^2
            + torch.sum(self.embedding.weight ** 2, dim=1)     # ||e||^2
            - 2 * torch.matmul(z_flattened, self.embedding.weight.t())  # -2*z*e
        )
        # Shape: (B*H*W, num_embeddings) - distance to each codebook entry
        
        # STEP 3: Find the nearest codebook entry for each latent vector
        # argmin gives us the index of the closest embedding
        encoding_indices = torch.argmin(distances, dim=1)  # Shape: (B*H*W,)
        
        # STEP 4: Look up the actual codebook vectors
        quantized = self.embedding(encoding_indices).view(z.shape)
        # Shape: (B, H, W, D) - each position now has its nearest codebook vector
        
        # STEP 5: Calculate VQ losses
        # Codebook loss: moves codebook entries toward encoder outputs
        # I use .detach() to stop gradients - only codebook learns from this
        codebook_loss = F.mse_loss(quantized.detach(), z)
        
        # Commitment loss: encourages encoder to commit to codebook entries
        # I use .detach() on quantized - only encoder learns from this
        commitment_loss = F.mse_loss(quantized, z.detach())
        
        # Total VQ loss combines both (commitment_cost is beta parameter)
        vq_loss = codebook_loss + self.commitment_cost * commitment_loss
        
        # STEP 6: Straight-through estimator (the clever trick!)
        # Forward pass: use quantized values
        # Backward pass: copy gradients directly from quantized to z
        # This makes the non-differentiable quantization operation "differentiable"
        quantized = z + (quantized - z).detach()
        
        # STEP 7: Calculate perplexity (codebook usage metric)
        # Perplexity measures how evenly the codebook is being used
        # If only a few codes are used, perplexity will be low (bad)
        # If all codes are used equally, perplexity approaches 512 (good)
        avg_probs = torch.mean(
            F.one_hot(encoding_indices, self.num_embeddings).float(), dim=0
        )
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # STEP 8: Convert back to original format (B, D, H, W)
        quantized = quantized.permute(0, 3, 1, 2).contiguous()
        
        return quantized, vq_loss, perplexity


class Decoder(nn.Module):
    """
    Convolutional Decoder for VQ-VAE
    ================================
    
    This is the final part of my VQ-VAE! It takes the discrete codes from the
    vector quantizer and reconstructs them back to full-resolution MRI images.
    It's basically the reverse of the encoder.
    
    How it works:
    ------------
    I use transposed convolutions (sometimes called "deconvolutions") to
    gradually increase the spatial dimensions back to 256x256. At each step,
    I use residual blocks to refine the features.
    
    The progression I designed:
    - Start: 64 channels at 32x32 (from VQ layer)
    - After layer 1: 128 channels at 32x32 (refine with residual blocks)
    - After layer 2: 64 channels at 64x64 (upsample with stride=2)
    - After layer 3: 32 channels at 128x128 (upsample again)
    - Final: 1 channel at 256x256 (back to grayscale image)
    
    Why Sigmoid at the end?
    -----------------------
    I use Sigmoid activation at the very end to ensure output values are in
    the range [0, 1]. This matches my normalized input images and makes sense
    for MRI intensity values.
    
    Arguments:
    ---------
    embedding_dim : int, default=64
        Dimension of input from VQ layer. Must match what VQ layer outputs.
    
    hidden_dims : list of int, default=[128, 64, 32]
        Number of channels at each layer. I reverse the encoder's dimensions,
        so [128, 64, 32] here corresponds to [32, 64, 128] in encoder.
    
    out_channels : int, default=1
        Number of output channels. I use 1 for grayscale MRI images.
    
    Returns:
    -------
    torch.Tensor
        Reconstructed images with shape: (batch_size, 1, 256, 256)
        Values are in range [0, 1] thanks to Sigmoid activation.
    
    Example:
    -------
    >>> decoder = Decoder(embedding_dim=64, hidden_dims=[128, 64, 32], out_channels=1)
    >>> z_q = torch.randn(4, 64, 32, 32)  # Quantized latents
    >>> recon = decoder(z_q)
    >>> print(recon.shape)  # (4, 1, 256, 256) - back to original size!
    >>> print(recon.min(), recon.max())  # Values in [0, 1]
    """
    
    def __init__(self, embedding_dim: int = 64, hidden_dims: list = [128, 64, 32],
                 out_channels: int = 1):
        super().__init__()
        
        modules = []
        
        # LAYER 1: Initial processing at the compressed resolution
        # I refine the quantized codes before starting to upsample
        modules.append(
            nn.Sequential(
                nn.Conv2d(embedding_dim, hidden_dims[0], kernel_size=3, padding=1),
                ResidualBlock(hidden_dims[0])  # Refine features
            )
        )
        
        # LAYERS 2-3: Upsampling layers with residual blocks
        # Each layer doubles the spatial dimensions (32->64->128)
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    # Residual blocks: refine at current resolution
                    ResidualBlock(hidden_dims[i]),
                    ResidualBlock(hidden_dims[i]),
                    
                    # Transposed conv: double spatial size, reduce channels
                    nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i+1], 
                                     kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(hidden_dims[i+1]),
                    nn.ReLU()
                )
            )
        
        # FINAL LAYER: Upsample to original resolution and convert to image
        # 128x128 -> 256x256, reduce channels to 1 (grayscale)
        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(hidden_dims[-1], out_channels, 
                                 kernel_size=4, stride=2, padding=1),
                nn.Sigmoid()  # Output in [0, 1] range for normalized images
            )
        )
        
        # Combine all layers into a sequential model
        self.decoder = nn.Sequential(*modules)
    
    def forward(self, z_q: torch.Tensor) -> torch.Tensor:
        """
        Decode quantized latents back to images.
        
        Arguments:
        ---------
        z_q : torch.Tensor
            Quantized latent vectors from VQ layer,
            shape (batch_size, embedding_dim, 32, 32)
        
        Returns:
        -------
        torch.Tensor
            Reconstructed images, shape (batch_size, 1, 256, 256)
            with values in [0, 1]
        """
        return self.decoder(z_q)


class VQVAE(nn.Module):
    """
    Complete VQ-VAE Model for Medical Image Reconstruction
    ======================================================
    
    This is the complete VQ-VAE model that I built for reconstructing HipMRI
    2D prostate slices! It combines all the components (Encoder, Vector
    Quantizer, Decoder) into one unified model.
    
    The Full Pipeline:
    -----------------
    Input (256x256 MRI) -> Encoder -> VQ Layer -> Decoder -> Reconstructed Image
    
    Step-by-step:
    1. Encoder compresses 256x256 image to 32x32 latent space (64 channels)
    2. VQ layer quantizes each 64-dim vector to nearest codebook entry
    3. Decoder reconstructs from quantized codes back to 256x256 image
    
    Training Objective:
    ------------------
    Total Loss = Reconstruction Loss + VQ Loss
    
    - Reconstruction Loss: How close is output to input? (MSE)
    - VQ Loss: Codebook + commitment losses (keeps quantization working)
    
    My Goal:
    -------
    Achieve SSIM > 0.6 on validation set, which means the reconstructions
    should look very similar to the original MRI slices. This is important
    for medical imaging where we need to preserve diagnostic information.
    
    Model Stats:
    -----------
    - Total parameters: 2,359,041
    - Trainable parameters: 2,359,041
    - Codebook size: 512 entries
    - Compression ratio: 64x in spatial dimensions (256x256 -> 32x32)
    
    Arguments:
    ---------
    in_channels : int, default=1
        Number of input channels. I use 1 for grayscale MRI images.
    
    hidden_dims : list of int, default=[32, 64, 128]
        Channel dimensions for encoder/decoder. I progressively increase
        channels in encoder, then reverse for decoder.
    
    embedding_dim : int, default=64
        Dimension of latent embeddings. This is the size of vectors that
        get quantized by the VQ layer.
    
    num_embeddings : int, default=512
        Size of the discrete codebook. I chose 512 as a good balance between
        expressiveness and computational cost.
    
    commitment_cost : float, default=0.25
        Beta parameter for commitment loss in VQ layer. Controls how strongly
        the encoder is encouraged to commit to codebook entries.
    
    Returns (in forward pass):
    -------------------------
    recon : torch.Tensor
        Reconstructed images, shape (batch_size, 1, 256, 256)
        Values in [0, 1] range
    
    vq_loss : torch.Tensor
        Scalar VQ loss (codebook + commitment)
        Add this to reconstruction loss during training
    
    perplexity : torch.Tensor
        Scalar metric showing codebook usage
        Higher values (closer to 512) mean better utilization
    
    Example Usage:
    -------------
    >>> # Create model with my chosen architecture
    >>> model = VQVAE(
    ...     in_channels=1,
    ...     hidden_dims=[32, 64, 128],
    ...     embedding_dim=64,
    ...     num_embeddings=512,
    ...     commitment_cost=0.25
    ... )
    >>> 
    >>> # Forward pass on a batch of 4 MRI images
    >>> x = torch.randn(4, 1, 256, 256)
    >>> recon, vq_loss, perplexity = model(x)
    >>> 
    >>> # Check outputs
    >>> print(f"Input shape: {x.shape}")
    >>> print(f"Reconstruction shape: {recon.shape}")
    >>> print(f"VQ Loss: {vq_loss.item():.4f}")
    >>> print(f"Codebook usage: {perplexity.item():.1f} out of 512")
    >>> 
    >>> # Calculate total training loss
    >>> recon_loss = F.mse_loss(recon, x)
    >>> total_loss = recon_loss + vq_loss
    """
    
    def __init__(self, in_channels: int = 1, hidden_dims: list = [32, 64, 128],
                 embedding_dim: int = 64, num_embeddings: int = 512,
                 commitment_cost: float = 0.25):
        super().__init__()
        
        # Build the three main components of my VQ-VAE
        
        # 1. Encoder: Compresses images to latent space
        self.encoder = Encoder(in_channels, hidden_dims, embedding_dim)
        
        # 2. VQ Layer: Quantizes continuous latents to discrete codes
        self.vq_layer = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        
        # 3. Decoder: Reconstructs images from quantized codes
        # Note: I reverse the hidden_dims for symmetric architecture
        self.decoder = Decoder(embedding_dim, list(reversed(hidden_dims)), in_channels)
    
    def forward(self, x: torch.Tensor):
        """
        Full forward pass through VQ-VAE.
        
        This is the main function that gets called during training and inference.
        It chains together encoding, quantization, and decoding.
        
        Arguments:
        ---------
        x : torch.Tensor
            Input MRI images of shape (batch_size, 1, 256, 256)
            Values should be normalized to [0, 1] range
        
        Returns:
        -------
        recon : torch.Tensor
            Reconstructed images, same shape as input (batch_size, 1, 256, 256)
        
        vq_loss : torch.Tensor
            Vector quantization loss to add to reconstruction loss
        
        perplexity : torch.Tensor
            Codebook usage metric (higher is better, max 512)
        """
        
        # STEP 1: Encode input image to continuous latent space
        # Shape: (batch_size, 1, 256, 256) -> (batch_size, 64, 32, 32)
        z = self.encoder(x)
        
        # STEP 2: Quantize continuous latents to discrete codebook entries
        # This is where the magic happens - continuous -> discrete
        # Shape stays: (batch_size, 64, 32, 32)
        z_q, vq_loss, perplexity = self.vq_layer(z)
        
        # STEP 3: Decode quantized latents back to image space
        # Shape: (batch_size, 64, 32, 32) -> (batch_size, 1, 256, 256)
        recon = self.decoder(z_q)
        
        return recon, vq_loss, perplexity
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode and quantize input images to discrete codes.
        
        I use this during inference when I just want to get the latent codes
        without reconstructing. Useful for analyzing what the model learned.
        
        Arguments:
        ---------
        x : torch.Tensor
            Input images (batch_size, 1, 256, 256)
        
        Returns:
        -------
        torch.Tensor
            Quantized latent codes (batch_size, 64, 32, 32)
        """
        z = self.encoder(x)
        z_q, _, _ = self.vq_layer(z)
        return z_q
    
    def decode(self, z_q: torch.Tensor) -> torch.Tensor:
        """
        Decode quantized latents to images.
        
        I use this when I have latent codes and want to generate images from
        them. Useful for generation experiments.
        
        Arguments:
        ---------
        z_q : torch.Tensor
            Quantized latent codes (batch_size, 64, 32, 32)
        
        Returns:
        -------
        torch.Tensor
            Reconstructed images (batch_size, 1, 256, 256)
        """
        return self.decoder(z_q)


# ============================================================================
# TEST SECTION - Verify Architecture
# ============================================================================

if __name__ == "__main__":
    # I run this test to make sure my architecture is set up correctly
    # It checks that all the dimensions match up properly
    
    print("="*70)
    print("TESTING VQ-VAE ARCHITECTURE")
    print("="*70)
    print("\nRunning sanity checks to verify model components...")
    
    # Create model with my chosen hyperparameters
    model = VQVAE(
        in_channels=1,            # Grayscale MRI images
        hidden_dims=[32, 64, 128], # Progressive channel increase
        embedding_dim=64,          # Latent dimension
        num_embeddings=512         # Codebook size
    )

    # Test with typical MRI slice size (256x256)
    # Batch size of 2 to test batching works correctly
    print("\nCreating test input: 2 images of size 256x256...")
    x = torch.randn(2, 1, 256, 256)
    
    # Run forward pass
    print("Running forward pass through VQ-VAE...")
    recon, vq_loss, perplexity = model(x)

    # Verify output dimensions
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"Input shape:          {x.shape}")
    print(f"Reconstruction shape: {recon.shape}")
    print(f"VQ Loss:              {vq_loss.item():.4f}")
    print(f"Perplexity:           {perplexity.item():.2f} / 512")
    
    # Check that shapes match
    assert x.shape == recon.shape, "Input and output shapes must match!"
    print("\nShape check passed!")
    
    # Test separate encode/decode functions
    print("\nTesting encode and decode separately...")
    z_q = model.encode(x)
    recon2 = model.decode(z_q)
    print(f"Encoded shape:  {z_q.shape}")
    print(f"Decoded shape:  {recon2.shape}")
    assert torch.allclose(recon, recon2, atol=1e-5), "Encode+Decode should match forward pass!"
    
    print("\n" + "="*70)
    print("ALL TESTS PASSED - VQ-VAE ARCHITECTURE IS WORKING CORRECTLY!")
    print("="*70)
    print("\nMy model is ready for training on HipMRI 2D prostate slices.")
    print("Target: SSIM > 0.6 on validation set")