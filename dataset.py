import os
import shutil
import random
from pathlib import Path

def split_images(source_folder, dataset_folder, train_ratio=0.8):
    """
    Randomly split images from source folder into training and validation sets.
    
    Args:
        source_folder: Path to folder containing images
        dataset_folder: Path where train/val folders will be created
        train_ratio: Ratio of images for training (default: 0.8)
    """
    # Define image extensions
    image_extensions = {'.jpg', '.jpeg', '.png'}
    
    # Get all image files
    source_path = Path(source_folder)
    if not source_path.exists():
        raise FileNotFoundError(f"Source folder '{source_folder}' does not exist")
    
    image_files = [f for f in source_path.iterdir() 
                   if f.is_file() and f.suffix.lower() in image_extensions]
    
    if not image_files:
        raise ValueError(f"No images found in '{source_folder}'")
    
    print(f"Found {len(image_files)} images")
    
    # Shuffle images randomly
    random.shuffle(image_files)
    
    # Calculate split index
    split_idx = int(len(image_files) * train_ratio)
    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]
    
    print(f"Training set: {len(train_files)} images")
    print(f"Validation set: {len(val_files)} images")
    
    # Create dataset directories
    dataset_path = Path(dataset_folder)
    train_path = dataset_path / 'train/images'
    val_path = dataset_path / 'val/images'
    
    train_path.mkdir(parents=True, exist_ok=True)
    val_path.mkdir(parents=True, exist_ok=True)
    
    # Copy files to respective folders
    print("\nCopying files...")
    for img in train_files:
        shutil.copy2(img, train_path / img.name)
    
    for img in val_files:
        shutil.copy2(img, val_path / img.name)
    
    print(f"\nDataset split complete!")
    print(f"Training folder: {train_path}")
    print(f"Validation folder: {val_path}")

if __name__ == "__main__":
    # Set your paths here
    source_folder = "results"  # Folder containing your images
    dataset_folder = "dataset"  # Output folder
    
    # Set random seed for reproducibility (optional)
    random.seed(42)
    
    split_images(source_folder, dataset_folder, train_ratio=0.8)