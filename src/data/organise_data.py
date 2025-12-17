import os
import shutil
import random
import re
import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def clean_folder_name(folder_name):
    """Clean and normalize folder names by removing numbering and special characters."""

    name = re.sub(r'^\d+\.\s*', '', folder_name)

    return name.lower().replace(' ', '_')

def reorganize_and_split(source_path, dest_path, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    """
    Reorganize and split image dataset into train/val/test sets.
    
    Args:
        source_path (str): Path to source dataset with class folders
        dest_path (str): Path to destination organized dataset
        train_ratio (float): Proportion of data for training (default: 0.7)
        val_ratio (float): Proportion of data for validation (default: 0.2)
        test_ratio (float): Proportion of data for testing (default: 0.1)
    """
    # 1. Verify Source Path
    if not os.path.exists(source_path):
        logging.error(f"CRITICAL ERROR: The path does not exist!")
        logging.error(f"I tried to look here: {source_path}")
        return False

    # 2. Setup Destination
    if os.path.exists(dest_path):
        logging.warning(f"Destination folder '{dest_path}' already exists. Merging/Overwriting...")
    else:
        os.makedirs(dest_path)

    folders = [f for f in os.listdir(source_path) if os.path.isdir(os.path.join(source_path, f))]
    
    if len(folders) == 0:
        logging.error("Error: No folders found inside the Source Path.")
        logging.error("Check if the images are actually inside sub-sub folders.")
        return False

    logging.info(f"Found {len(folders)} categories. Starting processing...")

    for folder in folders:
        original_folder_full_path = os.path.join(source_path, folder)
        new_class_name = clean_folder_name(folder) 
        
        # Collect valid images
        images = [img for img in os.listdir(original_folder_full_path) 
                 if img.lower().endswith(('.jpg', '.png', '.jpeg'))]
        
        if not images:
            logging.warning(f"Skipping '{folder}' - No images found.")
            continue

        # Shuffle for random splitting
        random.shuffle(images)
        
        # Calculate split indices
        total = len(images)
        train_end = int(total * train_ratio)
        val_end = int(total * (train_ratio + val_ratio))
        
        train_imgs = images[:train_end]
        val_imgs = images[train_end:val_end]
        test_imgs = images[val_end:]
        
        logging.info(f"Processing '{new_class_name}': {len(train_imgs)} Train, {len(val_imgs)} Val, {len(test_imgs)} Test")

        # Function to copy files
        def copy_to_split(file_list, split_name):
            target_dir = os.path.join(dest_path, split_name, new_class_name)
            os.makedirs(target_dir, exist_ok=True)
            
            for file_name in file_list:
                src_file = os.path.join(original_folder_full_path, file_name)
                dst_file = os.path.join(target_dir, file_name)
                shutil.copy2(src_file, dst_file)

        # Execute Copy
        copy_to_split(train_imgs, 'train')
        copy_to_split(val_imgs, 'val')
        copy_to_split(test_imgs, 'test')

    logging.info("="*50)
    logging.info("SUCCESS! Data preparation complete.")
    logging.info(f"New Dataset Location: {os.path.abspath(dest_path)}")
    logging.info("="*50)
    
    return True


def main():
    """Main function with argument parsing for pipeline integration."""
    parser = argparse.ArgumentParser(description="Reorganize and split image dataset")
    parser.add_argument(
        '--source',
        type=str,
        default="data/Infected Date Palm Leaves Dataset/Processed",
        help='Source path containing class folders with images'
    )
    parser.add_argument(
        '--dest',
        type=str,
        default="data/processed/palm_disease_final",
        help='Destination path for organized dataset'
    )
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.7,
        help='Training set ratio (default: 0.7)'
    )
    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.2,
        help='Validation set ratio (default: 0.2)'
    )
    parser.add_argument(
        '--test-ratio',
        type=float,
        default=0.1,
        help='Test set ratio (default: 0.1)'
    )
    
    args = parser.parse_args()
    
    # Convert to absolute paths if they're relative, using Path for cross-platform compatibility
    project_root = Path(__file__).resolve().parent.parent.parent
    
    # Use Path objects to handle both Windows and Unix-style paths properly
    if os.path.isabs(args.source):
        source_path = Path(args.source)
    else:
        source_path = project_root / Path(args.source.replace('\\', '/'))
    
    if os.path.isabs(args.dest):
        dest_path = Path(args.dest)
    else:
        dest_path = project_root / Path(args.dest.replace('\\', '/'))
    
    logging.info("Starting data reorganization...")
    logging.info(f"Source: {source_path}")
    logging.info(f"Destination: {dest_path}")
    
    success = reorganize_and_split(
        source_path=str(source_path),
        dest_path=str(dest_path),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio
    )
    
    if success:
        logging.info("Data organization completed successfully!")
    else:
        logging.error("Data organization failed!")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
