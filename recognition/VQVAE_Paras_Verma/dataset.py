"""dataset.py
Data loader for HipMRI 2D prostate MRI slices for VQ-VAE training.

Loads 2D MRI slices from the HipMRI Study dataset on Rangpur HPC.
Designed for unsupervised generative modeling - no labels required.

Rangpur Path: /home/groups/comp3710/HipMRI_Study_open/keras_slices_data

The dataset returns normalized grayscale MRI slices as tensors of shape (1, H, W).
"""
from typing import Optional, Callable
import os
import glob

import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset
from PIL import Image


class HipMRIDataset(Dataset):
    """Dataset for HipMRI 2D prostate MRI slices.
    
    Loads NIfTI format medical images from the keras_slices_data folder.
    Performs normalization and optional augmentation for VQ-VAE training.
    
    Args:
        data_dir: Path to keras_slices_data folder on Rangpur
        split: One of 'train', 'val', or 'test' for data splitting
        image_size: Target size for resizing images (default: 256x256)
        normalize: Whether to normalize images to [0, 1] range
        transform: Optional torchvision transforms
        max_samples: Limit number of samples (for quick testing)
    
    Example:
        >>> dataset = HipMRIDataset(
        ...     data_dir='/home/groups/comp3710/HipMRI_Study_open/keras_slices_data',
        ...     split='train'
        ... )
        >>> image = dataset[0]
        >>> print(image.shape)  # torch.Size([1, 256, 256])
    """
    
    def __init__(
        self, 
        data_dir: str, 
        split: str = 'train',
        image_size: int = 256,
        normalize: bool = True,
        transform: Optional[Callable] = None,
        max_samples: Optional[int] = None
    ):
        super().__init__()
        self.data_dir = data_dir
        self.split = split
        self.image_size = image_size
        self.normalize = normalize
        self.transform = transform
        
        # Discover all NIfTI files recursively
        print(f"Loading images from {data_dir}...")
        nii_pattern = os.path.join(data_dir, '**', '*.nii*')
        all_files = glob.glob(nii_pattern, recursive=True)
        all_files.sort()  # Ensure reproducibility
        
        if len(all_files) == 0:
            raise RuntimeError(f"No NIfTI files found in {data_dir}")
        
        print(f"Found {len(all_files)} NIfTI files")
        
        # Split data: 80% train, 10% val, 10% test
        n = len(all_files)
        train_end = int(0.8 * n)
        val_end = int(0.9 * n)
        
        if split == 'train':
            self.files = all_files[:train_end]
        elif split == 'val':
            self.files = all_files[train_end:val_end]
        elif split == 'test':
            self.files = all_files[val_end:]
        else:
            raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'")
        
        # Limit samples if specified
        if max_samples:
            self.files = self.files[:max_samples]
        
        print(f"{split} split: {len(self.files)} samples")
    
    def __len__(self) -> int:
        return len(self.files)
    
    