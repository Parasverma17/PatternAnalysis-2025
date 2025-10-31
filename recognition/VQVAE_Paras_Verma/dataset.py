"""
================================================================================
dataset.py - HipMRI Dataset Loader for VQ-VAE Training
================================================================================

Author: Paras Verma
Course: COMP3710 - Pattern Analysis
Project: VQ-VAE for HipMRI 2D Prostate MRI Reconstruction

Description:
    This file contains my custom PyTorch Dataset class for loading and 
    preprocessing HipMRI 2D prostate MRI slices. I designed it specifically 
    for training the VQ-VAE model on medical images stored in NIfTI format.
    
    The dataset handles:
    - Loading grayscale medical images from keras_slices_data folder
    - Automatic train/validation/test splitting with reproducible seeds
    - Image normalization to [0, 1] range for stable training
    - Resizing to 256×256 pixels for consistent input dimensions
    
Dataset Structure on Rangpur HPC:
    /home/groups/comp3710/HipMRI_Study_open/keras_slices_data/
    ├── keras_slices_train/       # 11,460 training images
    ├── keras_slices_validate/    # 660 validation images (optional)
    └── keras_slices_test/        # 540 test images

Key Features:
    ✓ NIfTI format support using nibabel library
    ✓ Automatic 90/10 train-validation split with fixed seed (reproducibility)
    ✓ Option to use provided validation folder instead of splitting
    ✓ Memory-efficient on-the-fly loading (no pre-loading all images)
    ✓ Handles 2D slices and 3D volumes (extracts middle slice from 3D)
    
Usage Example:
    >>> # Create training dataset
    >>> train_data = HipMRIDataset(
    ...     data_dir='/path/to/keras_slices_data',
    ...     split='train',
    ...     image_size=256,
    ...     val_split=0.1,
    ...     seed=42
    ... )
    >>> print(f"Loaded {len(train_data)} training images")
    >>> 
    >>> # Get a sample
    >>> image = train_data[0]  # Returns torch.Tensor of shape (1, 256, 256)

================================================================================
"""

# Standard library imports
from typing import Optional, Callable
import os
import glob
import random

# Third-party imports
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset
from PIL import Image


class HipMRIDataset(Dataset):
    """
    Custom PyTorch Dataset for loading HipMRI 2D prostate MRI slices.
    
    I created this class to handle medical imaging data in NIfTI format for my
    VQ-VAE project. It takes care of all the preprocessing steps needed to prepare
    the images for training, including normalization and resizing.
    
    What this class does:
        1. Loads medical images from specified train/val/test folders
        2. Normalizes pixel intensities to [0, 1] range
        3. Resizes images to consistent 256×256 dimensions
        4. Converts images to PyTorch tensors with shape (1, H, W)
        5. Handles both 2D slices and 3D volumes automatically
    
    Input Arguments:
        data_dir (str): 
            Path to the keras_slices_data folder on Rangpur HPC.
            Should contain keras_slices_train and keras_slices_test subfolders.
            
        split (str): 
            Which dataset split to load: 'train', 'val', or 'test'.
            Default is 'train'.
            
        image_size (int): 
            Target size for resizing images. Images will be resized to 
            (image_size, image_size). Default is 256×256 pixels.
            
        normalize (bool): 
            Whether to normalize pixel values to [0, 1] range.
            I keep this True for stable VQ-VAE training. Default is True.
            
        transform (Callable, optional): 
            Optional torchvision transforms to apply to images.
            I don't use this since medical images shouldn't be augmented.
            
        max_samples (int, optional): 
            If set, limits the number of samples loaded. Useful for quick
            testing without loading the full dataset.
            
        val_split (float): 
            Fraction of training data to use for validation when split='val'.
            I use 0.1 (10%) which gives me ~1,146 validation images. Default is 0.1.
            
        seed (int): 
            Random seed for reproducible train/val split. I use 42 so that
            every run uses the same split. Default is 42.
            
        use_provided_val (bool): 
            If True and split='val', uses the keras_slices_validate folder
            instead of splitting training data. Default is False.
    
    Returns:
        When you index this dataset (e.g., dataset[0]), it returns:
            torch.Tensor: Normalized MRI image of shape (1, H, W) where:
                         - Channel dimension is 1 (grayscale)
                         - H and W are both equal to image_size (default 256)
                         - Values are in range [0, 1]
    
    Example Usage:
        >>> # Load training data with 90/10 split
        >>> dataset = HipMRIDataset(
        ...     data_dir='/home/groups/comp3710/HipMRI_Study_open/keras_slices_data',
        ...     split='train',
        ...     val_split=0.1,
        ...     seed=42
        ... )
        >>> print(f"Training samples: {len(dataset)}")
        >>> 
        >>> # Get first image
        >>> image = dataset[0]
        >>> print(f"Image shape: {image.shape}")  # torch.Size([1, 256, 256])
        >>> print(f"Value range: [{image.min():.3f}, {image.max():.3f}]")
    
    Folder Structure I Expect:
        data_dir/
        ├── keras_slices_train/       # Training images (11,460 slices)
        ├── keras_slices_validate/    # Optional validation set (660 slices)
        └── keras_slices_test/        # Test images (540 slices)
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
        
        # Store all the parameters I'll need later
        self.data_dir = data_dir
        self.split = split
        self.image_size = image_size
        self.normalize = normalize
        self.transform = transform
        
        # Set random seed for reproducibility - this ensures I get the same
        # train/val split every time I run the code
        random.seed(seed)
        
        # ====================================================================
        # STEP 1: Load image file paths based on the split requested
        # ====================================================================
        
        if split == 'val' and use_provided_val:
            # Option 1: Use the pre-existing keras_slices_validate folder
            # This folder has 660 images that were set aside separately
            val_folder = os.path.join(data_dir, 'keras_slices_validate')
            print(f"Loading from provided validation folder: {val_folder}")
            
            # Find all NIfTI files recursively in the validation folder
            # The '**' pattern matches any subdirectories
            pattern = os.path.join(val_folder, '**', '*.nii*')
            self.files = sorted(glob.glob(pattern, recursive=True))
            
            # Make sure I actually found some files
            if len(self.files) == 0:
                raise RuntimeError(f"No NIfTI files found in {val_folder}")
            
            print(f"Found {len(self.files)} validation files")
            
        elif split in ['train', 'val']:
            # Option 2: Split the training folder into train and validation
            # This is what I do by default - take 90% for training, 10% for validation
            train_folder = os.path.join(data_dir, 'keras_slices_train')
            print(f"Loading from training folder: {train_folder}")
            
            # Find all NIfTI files in training folder
            pattern = os.path.join(train_folder, '**', '*.nii*')
            all_files = sorted(glob.glob(pattern, recursive=True))
            
            # Sanity check - make sure we found files
            if len(all_files) == 0:
                raise RuntimeError(f"No NIfTI files found in {train_folder}")
            
            print(f"Found {len(all_files)} training files")
            
            # Now split into train and validation sets
            n_val = int(len(all_files) * val_split)  # Calculate 10% for validation
            random.shuffle(all_files)  # Shuffle for random split
            
            if split == 'val':
                # Take first n_val files for validation
                self.files = all_files[:n_val]
                print(f"Using {val_split*100}% of training data for validation")
            else:  # split == 'train'
                # Take remaining files for training
                self.files = all_files[n_val:]
                
        elif split == 'test':
            # Option 3: Load from the separate test folder
            # This is completely unseen data that I never touch during training
            test_folder = os.path.join(data_dir, 'keras_slices_test')
            print(f"Loading from test folder: {test_folder}")
            
            # Find all NIfTI files in test folder
            pattern = os.path.join(test_folder, '**', '*.nii*')
            self.files = sorted(glob.glob(pattern, recursive=True))
            
            # Sanity check
            if len(self.files) == 0:
                raise RuntimeError(f"No NIfTI files found in {test_folder}")
            
            print(f"Found {len(self.files)} test files")
            
        else:
            # If someone passes an invalid split name, raise an error
            raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'")
        
        # ====================================================================
        # STEP 2: Optionally limit the number of samples for quick testing
        # ====================================================================
        if max_samples:
            self.files = self.files[:max_samples]
        
        # Print final count so I know what I'm working with
        print(f"{split} split: {len(self.files)} samples")
    
    def __len__(self) -> int:
        """
        Return the total number of images in this dataset.
        
        This is a required method for PyTorch Dataset classes. It tells
        DataLoader how many samples we have.
        
        Returns:
            int: Total number of MRI slices in this split
        """
        return len(self.files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Load and preprocess a single MRI slice from disk.
        
        This method is called whenever someone accesses dataset[idx]. It handles
        all the loading and preprocessing steps to prepare the image for training.
        
        My preprocessing pipeline:
            1. Load NIfTI file using nibabel
            2. Handle different dimensions (2D slice or 3D volume)
            3. Convert to float32 for numerical stability
            4. Normalize pixel values to [0, 1] range
            5. Resize to target size (256×256) using PIL for quality
            6. Convert to PyTorch tensor with channel dimension
            7. Apply any optional transforms
        
        Input Arguments:
            idx (int): Index of the image to load (0 to len(dataset)-1)
        
        Returns:
            torch.Tensor: Preprocessed MRI image with shape (1, H, W) where:
                         - First dimension (1) is the channel dimension for grayscale
                         - H and W are both equal to self.image_size (default 256)
                         - All pixel values are normalized to [0, 1] range
        
        If Loading Fails:
            Returns a blank image (all zeros) to prevent crashing during training.
            Prints error message so I know which file failed.
        """
        # Get the file path for this index
        filepath = self.files[idx]
        
        try:
            # ================================================================
            # STEP 1: Load the NIfTI medical image file
            # ================================================================
            # NIfTI is a standard format for medical imaging data
            # I use nibabel library which is specifically designed for this
            nii_img = nib.load(filepath)
            img = nii_img.get_fdata(caching='unchanged')  # Get the actual pixel data
            
            # ================================================================
            # STEP 2: Handle different image dimensions
            # ================================================================
            # Medical images can be 2D slices or 3D volumes
            # I need to convert everything to 2D for my VQ-VAE
            if img.ndim == 3:
                # If it's a 3D volume, take the middle slice
                # This usually has the best anatomical information
                img = img[:, :, img.shape[2] // 2]
            elif img.ndim > 3:
                # Some medical images have extra dimensions (like time series)
                # I just take the first slice of all extra dimensions
                img = img[:, :, 0, 0] if img.ndim == 4 else img[:, :, 0]
            
            # ================================================================
            # STEP 3: Convert to float32 for numerical stability
            # ================================================================
            # Neural networks work best with float32 precision
            img = img.astype(np.float32)
            
            # ================================================================
            # STEP 4: Normalize pixel intensities to [0, 1] range
            # ================================================================
            # This is crucial for stable training - prevents gradient explosion
            if self.normalize:
                img_min, img_max = img.min(), img.max()
                if img_max > img_min:
                    # Standard min-max normalization
                    img = (img - img_min) / (img_max - img_min)
                else:
                    # Edge case: if all pixels have same value, return zeros
                    img = np.zeros_like(img)
            
            # ================================================================
            # STEP 5: Resize to target size using PIL
            # ================================================================
            # I use PIL (Pillow) instead of OpenCV because it gives better
            # quality for medical images with its bilinear interpolation
            
            # First convert to 8-bit integer for PIL (0-255 range)
            img_pil = Image.fromarray((img * 255).astype(np.uint8))
            
            # Resize to target dimensions (default 256×256)
            img_pil = img_pil.resize((self.image_size, self.image_size), Image.BILINEAR)
            
            # Convert back to float32 in [0, 1] range
            img = np.array(img_pil).astype(np.float32) / 255.0
            
            # ================================================================
            # STEP 6: Convert to PyTorch tensor with channel dimension
            # ================================================================
            # PyTorch expects images in (C, H, W) format where C is channels
            # Since these are grayscale images, C=1
            tensor = torch.from_numpy(img).unsqueeze(0)  # Add channel dimension
            
            # ================================================================
            # STEP 7: Apply optional transforms if provided
            # ================================================================
            # I don't use this for medical images, but leaving it here
            # in case someone wants to add data augmentation later
            if self.transform:
                tensor = self.transform(tensor)
            
            return tensor
            
        except Exception as e:
            # If something goes wrong, print the error and return a blank image
            # This prevents the entire training from crashing due to one bad file
            print(f"Error loading {filepath}: {e}")
            return torch.zeros(1, self.image_size, self.image_size)


# ==============================================================================
# TESTING SECTION - Run this file directly to test the dataset loader
# ==============================================================================
# This section only runs when I execute: python dataset.py
# It helps me verify that the dataset loading works correctly before training

if __name__ == "__main__":
    print("="*70)
    print("TESTING HIPMRI DATASET LOADER")
    print("="*70)
    print("\nThis test verifies that I can successfully load images from")
    print("the HipMRI dataset on Rangpur HPC.\n")

    # Path to the dataset on Rangpur HPC
    data_dir = "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data"

    try:
        # ===================================================================
        # TEST 1: Training Split
        # ===================================================================
        print("\n" + "="*70)
        print("TEST 1: Loading Training Split")
        print("="*70)
        train_dataset = HipMRIDataset(
            data_dir, 
            split="train", 
            max_samples=5  # Only load 5 samples for quick testing
        )
        print(f" Train dataset size: {len(train_dataset)}")
        
        # Try loading a sample image
        if len(train_dataset) > 0:
            sample = train_dataset[0]
            print(f" Sample shape: {sample.shape}")
            print(f" Sample value range: [{sample.min():.3f}, {sample.max():.3f}]")
        
        # ===================================================================
        # TEST 2: Validation Split
        # ===================================================================
        print("\n" + "="*70)
        print("TEST 2: Loading Validation Split (using provided folder)")
        print("="*70)
        val_dataset = HipMRIDataset(
            data_dir, 
            split="val", 
            max_samples=5,
            use_provided_val=True  # Use keras_slices_validate folder
        )
        print(f" Val dataset size: {len(val_dataset)}")
        
        # ===================================================================
        # TEST 3: Test Split
        # ===================================================================
        print("\n" + "="*70)
        print("TEST 3: Loading Test Split")
        print("="*70)
        test_dataset = HipMRIDataset(
            data_dir, 
            split="test", 
            max_samples=5
        )
        print(f" Test dataset size: {len(test_dataset)}")
        
        # ===================================================================
        # All tests passed!
        # ===================================================================
        print("\n" + "="*70)
        print("ALL TESTS PASSED! ")
        print("="*70)
        print("\nThe dataset loader is working correctly.")
        print("You can now use it for training the VQ-VAE model.\n")

    except Exception as e:
        # ===================================================================
        # Handle errors gracefully
        # ===================================================================
        print("\n" + "="*70)
        print("TEST FAILED ")
        print("="*70)
        print(f"\nError: {e}")
        print("\nThis is expected if you're not running on Rangpur HPC.")
        print("When running locally, make sure to update the data_dir path")
        print("to point to your local copy of the dataset.\n")