"""dataset.py
Data loader for HipMRI 2D prostate MRI slices for VQ-VAE training.

Loads 2D MRI slices from the HipMRI Study dataset on Rangpur HPC.
Designed for unsupervised generative modeling - no labels required.

Rangpur Path: /home/groups/comp3710/HipMRI_Study_open/keras_slices_data

The dataset uses keras_slices_train and keras_slices_test folders.
For validation, we split a portion from the training set.
"""
from typing import Optional, Callable
import os
import glob
import random

import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset
from PIL import Image


class HipMRIDataset(Dataset):
    """Dataset for HipMRI 2D prostate MRI slices.
    
    Loads NIfTI format medical images from keras_slices_train and keras_slices_test.
    Performs normalization and optional augmentation for VQ-VAE training.
    
    Args:
        data_dir: Path to keras_slices_data folder (parent of train/test folders)
        split: One of 'train', 'val', or 'test' for data splitting
        image_size: Target size for resizing images (default: 256x256)
        normalize: Whether to normalize images to [0, 1] range
        transform: Optional torchvision transforms
        max_samples: Limit number of samples (for quick testing)
        val_split: Fraction of training data to use for validation (default: 0.1)
        seed: Random seed for reproducible train/val split (default: 42)
        use_provided_val: If True, use keras_slices_validate folder instead of splitting
                          training data (only applies when split='val')
    
    Folder Structure Expected:
        data_dir/
        ├── keras_slices_train/       # Training data
        ├── keras_slices_validate/    # Optional: provided validation set
        └── keras_slices_test/        # Test data
    
    Example:
        >>> # Option 1: Use 90/10 split from training data
        >>> dataset = HipMRIDataset(
        ...     data_dir='/home/groups/comp3710/HipMRI_Study_open/keras_slices_data',
        ...     split='val',
        ...     val_split=0.1
        ... )
        >>> 
        >>> # Option 2: Use provided validation folder
        >>> dataset = HipMRIDataset(
        ...     data_dir='/home/groups/comp3710/HipMRI_Study_open/keras_slices_data',
        ...     split='val',
        ...     use_provided_val=True
        ... )
        >>> print(image.shape)  # torch.Size([1, 256, 256])
    """
    
    def __init__(
        self, 
        data_dir: str, 
        split: str = 'train',
        image_size: int = 256,
        normalize: bool = True,
        transform: Optional[Callable] = None,
        max_samples: Optional[int] = None,
        val_split: float = 0.1,
        seed: int = 42,
        use_provided_val: bool = False
    ):
        super().__init__()
        self.data_dir = data_dir
        self.split = split
        self.image_size = image_size
        self.normalize = normalize
        self.transform = transform
        
        # Set random seed for reproducibility
        random.seed(seed)
        
        # Load from appropriate folders based on split
        if split == 'val' and use_provided_val:
            # Use provided keras_slices_validate folder
            val_folder = os.path.join(data_dir, 'keras_slices_validate')
            print(f"Loading from provided validation folder: {val_folder}")
            pattern = os.path.join(val_folder, '**', '*.nii*')
            self.files = sorted(glob.glob(pattern, recursive=True))
            
            if len(self.files) == 0:
                raise RuntimeError(f"No NIfTI files found in {val_folder}")
            
            print(f"Found {len(self.files)} validation files")
            
        elif split in ['train', 'val']:
            # Load from keras_slices_train folder and split
            train_folder = os.path.join(data_dir, 'keras_slices_train')
            print(f"Loading from training folder: {train_folder}")
            pattern = os.path.join(train_folder, '**', '*.nii*')
            all_files = sorted(glob.glob(pattern, recursive=True))
            
            if len(all_files) == 0:
                raise RuntimeError(f"No NIfTI files found in {train_folder}")
            
            print(f"Found {len(all_files)} training files")
            
            # Split into train and validation
            n_val = int(len(all_files) * val_split)
            random.shuffle(all_files)
            
            if split == 'val':
                self.files = all_files[:n_val]
                print(f"Using {val_split*100}% of training data for validation")
            else:  # train
                self.files = all_files[n_val:]
                
        elif split == 'test':
            # Load from keras_slices_test folder
            test_folder = os.path.join(data_dir, 'keras_slices_test')
            print(f"Loading from test folder: {test_folder}")
            pattern = os.path.join(test_folder, '**', '*.nii*')
            self.files = sorted(glob.glob(pattern, recursive=True))
            
            if len(self.files) == 0:
                raise RuntimeError(f"No NIfTI files found in {test_folder}")
            
            print(f"Found {len(self.files)} test files")
        else:
            raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'")
        
        # Limit samples if specified
        if max_samples:
            self.files = self.files[:max_samples]
        
        print(f"{split} split: {len(self.files)} samples")
    
    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Load and preprocess a single MRI slice.
        
        Returns:
            Tensor of shape (1, H, W) normalized to [0, 1] range
        """
        filepath = self.files[idx]
        
        try:
            # Load NIfTI file
            nii_img = nib.load(filepath)
            img = nii_img.get_fdata(caching='unchanged')
            
            # Handle different dimensions
            if img.ndim == 3:
                # Take middle slice if 3D volume
                img = img[:, :, img.shape[2] // 2]
            elif img.ndim > 3:
                # Remove extra dimensions
                img = img[:, :, 0, 0] if img.ndim == 4 else img[:, :, 0]
            
            # Convert to float32
            img = img.astype(np.float32)
            
            # Normalize to [0, 1] range
            if self.normalize:
                img_min, img_max = img.min(), img.max()
                if img_max > img_min:
                    img = (img - img_min) / (img_max - img_min)
                else:
                    img = np.zeros_like(img)
            
            # Resize to target size using PIL for better quality
            img_pil = Image.fromarray((img * 255).astype(np.uint8))
            img_pil = img_pil.resize((self.image_size, self.image_size), Image.BILINEAR)
            img = np.array(img_pil).astype(np.float32) / 255.0
            
            # Convert to tensor: (C, H, W)
            tensor = torch.from_numpy(img).unsqueeze(0)
            
            # Apply optional transforms
            if self.transform:
                tensor = self.transform(tensor)
            
            return tensor
            
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            # Return blank image on error
            return torch.zeros(1, self.image_size, self.image_size)


if __name__ == "__main__":
    # Test dataset loading
    print("Testing HipMRI dataset loader...")

    # Test with Rangpur path
    data_dir = "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data"

    try:
        print("\n=== Testing Train Split ===")
        train_dataset = HipMRIDataset(data_dir, split="train", max_samples=5)
        print(f"Train dataset size: {len(train_dataset)}")
        if len(train_dataset) > 0:
            sample = train_dataset[0]
            print(f"Sample shape: {sample.shape}")
            print(f"Sample range: [{sample.min():.3f}, {sample.max():.3f}]")
        
        print("\n=== Testing Val Split ===")
        val_dataset = HipMRIDataset(data_dir, split="val", max_samples=5)
        print(f"Val dataset size: {len(val_dataset)}")
        
        print("\n=== Testing Test Split ===")
        test_dataset = HipMRIDataset(data_dir, split="test", max_samples=5)
        print(f"Test dataset size: {len(test_dataset)}")
        
        print("\n✓ Dataset test passed!")

    except Exception as e:
        print(f"\n✗ Dataset test failed (expected if not on Rangpur): {e}")
        print("This is normal if running locally without access to Rangpur data.")