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

class VectorQuantizer(nn.Module):
    """Vector Quantization layer for VQ-VAE.
    
    Maintains a codebook of embedding vectors and quantizes encoder outputs
    to the nearest codebook entry. Implements straight-through estimator
    for backpropagation.
    
    Args:
        num_embeddings: Size of the codebook (K)
        embedding_dim: Dimension of each embedding vector (D)
        commitment_cost: Weight for commitment loss (beta)
    """
    
    def __init__(self, num_embeddings: int = 512, embedding_dim: int = 64, 
                 commitment_cost: float = 0.25):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        
        # Initialize codebook with uniform distribution
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)
    
    def forward(self, z: torch.Tensor):
        """Quantize continuous latents to discrete codebook entries.
        
        Args:
            z: Encoder output of shape (B, D, H, W)
            
        Returns:
            quantized: Quantized latents (B, D, H, W)
            vq_loss: Vector quantization loss
            perplexity: Codebook usage metric
        """
        # Convert from (B, D, H, W) to (B, H, W, D)
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.embedding_dim)
        
        # Calculate distances to codebook entries
        # ||z - e||^2 = ||z||^2 + ||e||^2 - 2*z*e
        distances = (
            torch.sum(z_flattened ** 2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight ** 2, dim=1)
            - 2 * torch.matmul(z_flattened, self.embedding.weight.t())
        )
        
        # Find nearest codebook entry
        encoding_indices = torch.argmin(distances, dim=1)
        
        # Quantize and reshape
        quantized = self.embedding(encoding_indices).view(z.shape)
        
        # Calculate VQ losses
        # Codebook loss: ||sg[z] - e||^2
        codebook_loss = F.mse_loss(quantized.detach(), z)
        
        # Commitment loss: ||z - sg[e]||^2
        commitment_loss = F.mse_loss(quantized, z.detach())
        
        vq_loss = codebook_loss + self.commitment_cost * commitment_loss
        
        # Straight-through estimator: copy gradients from quantized to z
        quantized = z + (quantized - z).detach()
        
        # Calculate perplexity (measure of codebook usage)
        avg_probs = torch.mean(
            F.one_hot(encoding_indices, self.num_embeddings).float(), dim=0
        )
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # Convert back to (B, D, H, W)
        quantized = quantized.permute(0, 3, 1, 2).contiguous()
        
        return quantized, vq_loss, perplexity


class Decoder(nn.Module):
    """Convolutional decoder for VQ-VAE.
    
    Upsamples quantized latents back to original image resolution using
    transposed convolutions and residual blocks.
    
    Args:
        embedding_dim: Dimension of input embeddings (from VQ layer)
        hidden_dims: List of channel dimensions (reversed from encoder)
        out_channels: Number of output channels (1 for grayscale MRI)
    """
    
    def __init__(self, embedding_dim: int = 64, hidden_dims: list = [128, 64, 32],
                 out_channels: int = 1):
        super().__init__()
        
        modules = []
        
        # Initial processing
        modules.append(
            nn.Sequential(
                nn.Conv2d(embedding_dim, hidden_dims[0], kernel_size=3, padding=1),
                ResidualBlock(hidden_dims[0])
            )
        )
        
        # Upsampling layers with residual blocks
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    ResidualBlock(hidden_dims[i]),
                    ResidualBlock(hidden_dims[i]),
                    nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i+1], 
                                     kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(hidden_dims[i+1]),
                    nn.ReLU()
                )
            )
        
        # Final upsampling to original resolution
        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(hidden_dims[-1], out_channels, 
                                 kernel_size=4, stride=2, padding=1),
                nn.Sigmoid()  # Output in [0, 1] range
            )
        )
        
        self.decoder = nn.Sequential(*modules)
    
    def forward(self, z_q: torch.Tensor) -> torch.Tensor:
        return self.decoder(z_q)


class VQVAE(nn.Module):
    """Complete VQ-VAE model for HipMRI 2D prostate slice generation.
    
    Implements the full VQ-VAE architecture with encoder, vector quantization,
    and decoder. Designed to achieve SSIM > 0.6 on medical imaging data.
    
    Args:
        in_channels: Number of input channels (1 for grayscale)
        hidden_dims: Channel dimensions for encoder/decoder layers
        embedding_dim: Dimension of latent embeddings
        num_embeddings: Size of discrete codebook
        commitment_cost: Beta parameter for commitment loss
    
    Example:
        >>> model = VQVAE(in_channels=1, num_embeddings=512, embedding_dim=64)
        >>> x = torch.randn(4, 1, 256, 256)
        >>> recon, vq_loss, perplexity = model(x)
        >>> print(f"Reconstruction: {recon.shape}, VQ Loss: {vq_loss.item():.4f}")
    """
    
    def __init__(self, in_channels: int = 1, hidden_dims: list = [32, 64, 128],
                 embedding_dim: int = 64, num_embeddings: int = 512,
                 commitment_cost: float = 0.25):
        super().__init__()
        
        self.encoder = Encoder(in_channels, hidden_dims, embedding_dim)
        self.vq_layer = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        self.decoder = Decoder(embedding_dim, list(reversed(hidden_dims)), in_channels)
    
    def forward(self, x: torch.Tensor):
        """Forward pass through VQ-VAE.
        
        Args:
            x: Input images of shape (B, C, H, W)
            
        Returns:
            recon: Reconstructed images (B, C, H, W)
            vq_loss: Vector quantization loss
            perplexity: Codebook usage metric
        """
        # Encode
        z = self.encoder(x)
        
        # Quantize
        z_q, vq_loss, perplexity = self.vq_layer(z)
        
        # Decode
        recon = self.decoder(z_q)
        
        return recon, vq_loss, perplexity
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to quantized latents."""
        z = self.encoder(x)
        z_q, _, _ = self.vq_layer(z)
        return z_q
    
    def decode(self, z_q: torch.Tensor) -> torch.Tensor:
        """Decode quantized latents to images."""
        return self.decoder(z_q)
