#=============================================================================
# Programme: splits a directory with many files into smaller directories
#=============================================================================

#=============================================================================
# Modules
#=============================================================================

# Standard modules
import os
import shutil
import argparse
from math import ceil

# Custom modules
from logger import logger

#=============================================================================
# Variables
#=============================================================================



#=============================================================================
# Functions
#=============================================================================

def split_directory(source_dir:str, num_splits:int=4):
    """Splits a directory into smaller ones

    Args:
        source_dir (str): directory to split up
        num_splits (int, optional): number of direcorties to be split into. Defaults to 4.
        
    Returns:
        None: None
    """
    # Ensure the source directory exists
    if not os.path.exists(source_dir) or not os.path.isdir(source_dir):
        logger.error(f"The directory {source_dir} does not exist or is not a directory")
        return None

    # Get list of all files in the source directory
    files = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]
    num_files = len(files)

    # Calculate the number of files per split
    files_per_split = ceil(num_files / num_splits)

    # Create split directories
    split_dirs = [os.path.join(source_dir, f"part{i+1}") for i in range(num_splits)]
    logger.info(f"Creating partition directories...")
    for split_dir in split_dirs:
        os.makedirs(split_dir, exist_ok=True)

    # Distribute files
    logger.info(f"Distributing files...")
    for i, file in enumerate(files):
        split_index = i // files_per_split
        shutil.move(os.path.join(source_dir, file), os.path.join(split_dirs[split_index], file))
    return None
    
#=============================================================================
# Classes
#=============================================================================



#=============================================================================
# Programme exectuion
#=============================================================================

if __name__ == "__main__":
    
    # Argument parsing
    parser = argparse.ArgumentParser(description="Split a directory into multiple subdirectories.")
    parser.add_argument("-d", "--directory", type=str, required=True, help="Directory of files to be split")
    parser.add_argument("-n", "--num_splits", type=int, default=4, help="Number of subdirectories to create")
    args = parser.parse_args()
    
    # Split directory
    logger.info(f"Splitting {args.directory}...")
    split_directory(args.directory, args.num_splits)
    logger.info(f"{args.directory} split")
