"""modules.py
VQ-VAE model components for generative modeling of HipMRI 2D prostate slices.

This implements a Vector Quantized Variational Autoencoder (VQ-VAE) as described in:
- "Neural Discrete Representation Learning" (van den Oord et al., 2017)
- "Generating Diverse High-Fidelity Images with VQ-VAE-2" (Razavi et al., 2019)

The model consists of:
1. Encoder: Convolutional encoder that downsamples input images to latent space
2. Vector Quantization: Discrete codebook that quantizes continuous latent vectors
3. Decoder: Convolutional decoder that reconstructs images from quantized latents

Architecture designed to achieve SSIM > 0.6 on HipMRI 2D prostate MRI slices.

"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Residual block with two convolutions and skip connection.
    
    Used in both encoder and decoder for better gradient flow.

    Implements: out = ReLU(BN(Conv(ReLU(BN(Conv(x)))))) + x
    """
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + residual
        out = F.relu(out)
        return out


class Encoder(nn.Module):
    """Convolutional encoder for VQ-VAE.
    
    Downsamples input images to latent representation through strided convolutions.
    Uses residual blocks for improved feature learning.
    
    Args:
        in_channels: Number of input channels (1 for grayscale MRI)
        hidden_dims: List of channel dimensions for each layer
        embedding_dim: Dimension of output embeddings (input to VQ layer)
    """
    
    def __init__(self, in_channels: int = 1, hidden_dims: list = [32, 64, 128], 
                 embedding_dim: int = 64):
        super().__init__()
        
        modules = []
        # Initial convolution
        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, hidden_dims[0], kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(hidden_dims[0]),
                nn.ReLU()
            )
        )
        
        # Downsampling layers with residual blocks
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.Conv2d(hidden_dims[i], hidden_dims[i+1], kernel_size=4, 
                             stride=2, padding=1),
                    nn.BatchNorm2d(hidden_dims[i+1]),
                    nn.ReLU(),
                    ResidualBlock(hidden_dims[i+1]),
                    ResidualBlock(hidden_dims[i+1])
                )
            )
        
        # Final convolution to embedding dimension
        modules.append(
            nn.Sequential(
                nn.Conv2d(hidden_dims[-1], embedding_dim, kernel_size=3, padding=1),
                ResidualBlock(embedding_dim)
            )
        )
        
        self.encoder = nn.Sequential(*modules)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)
