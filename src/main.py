#=============================================================================
# Programme: Pipeline of MPH representation for deep learning models
#=============================================================================

#=============================================================================
# Modules
#=============================================================================

# Standard modules
import logging
import yaml
import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# Custom modules
from preprocessing import mphData, computeMph, generateMph

#=============================================================================
# Variables
#=============================================================================

# Path to the JSON metadata file
configurationFilePath = "configuration.yaml"

#=============================================================================
# Functions
#=============================================================================

def list_files_in_directory(directory:str=None):
    """Function finds all files in a driectory and verifies that the files exist

    Args:
        directory (str): directory to get file names

    Returns:
        list: list of exisiting file names
    """
    try:
        files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
        logger.info(f"Files to be processed found")
        return files
    except Exception as e:
        logger.error(f"Files to be processed not found")
        return []
    
def generate_indices(k_family:int):
    """ generates a list of of integers from 1 to k_family 

    Args:
        k_family (int): largest integer to include the list. Defaults to 1.

    Returns:
        list: list of integers
    """
    k_family = int(k_family)
    if k_family < 1:
        raise ValueError("k_family must be a positive integer")
    if k_family == 1:
        return [1]
    else:
        return list(range(1, k_family + 1))
    
def extract_between_last_two_slashes(s):
    # Find the position of the last "/"
    last_slash = s.rfind("/")
    if last_slash == -1:
        return None
    
    # Find the position of the second last "/"
    second_last_slash = s.rfind("/", 0, last_slash)
    if second_last_slash == -1:
        return None
    
    # Extract the substring between the last two "/"
    return s[second_last_slash + 1:last_slash]


#=============================================================================
# Programme exectuion
#=============================================================================

if __name__ == "__main__":
    
    #==========================================================================
    # Configuration data loading
    #==========================================================================
    
    with open(configurationFilePath, "r") as file:
        configurationData = yaml.safe_load(file)
    
    #==========================================================================
    # Configure logging
    #==========================================================================
    
    # Extract logging configuration
    log_file_path  = configurationData["logging"]["log_file_path"]
    logging_format = configurationData["logging"]["logging_format"]
    
    # Configure logging
    logging.basicConfig(level=logging.DEBUG,
                        format=f"{logging_format}",
                        handlers=[
                            logging.FileHandler(f"{log_file_path}"),
                            logging.StreamHandler()
                        ])

    # Create logger
    logger = logging.getLogger("pipeline_logger")
    
    # Start logging
    logger.info(f"Logging started...")
    
    #==========================================================================
    # Configuration imports
    #==========================================================================
    
    # Extract data configuration
    logger.info(f"importing data configurations...")
    raw_data_directory_path       = configurationData["data"]["raw_data_directory_path"]
    processed_data_directory_path = configurationData["data"]["processed_data_directory_path"]
    label_file                    = configurationData["data"]["label_file"]
    
    # Extract mph configuration
    logger.info(f"importing mph configurations...")
    threads         = configurationData["mph"]["threads"]
    coord1          = configurationData["mph"]["coord1"]
    coord2          = configurationData["mph"]["coord2"]
    parameter       = configurationData["mph"]["parameter"]
    RipsMax         = configurationData["mph"]["RipsMax"]
    alpha           = configurationData["mph"]["alpha"]
    homology        = configurationData["mph"]["homology"]
    k_family        = configurationData["mph"]["k_family"]
    resolution      = configurationData["mph"]["resolution"]
    grid_step_size  = configurationData["mph"]["grid_step_size"]
    plot_indices    = configurationData["mph"]["plot_indices"]
    
    # Extract model configurations
    logger.info(f"importing model configurations...")
    test_ratio                 = configurationData["model"]["test_ratio"]
    pretrained                 = configurationData["model"]["pretrained"]
    pretrained_model_file_path = configurationData["model"]["pretrained_model_file_path"]
    supervised                 = configurationData["model"]["supervised"]
    model_to_train_file        = configurationData["model"]["model_to_train_file"]
    
    # Extract outputs configuration
    logger.info(f"importing outputs configurations...")
    models_directory_path      = configurationData["output"]["models_directory_path"]
    training_directory_path    = configurationData["output"]["training_directory_path"]
    summaries_directory_path   = configurationData["output"]["summaries_directory_path"]
    
    #==========================================================================
    # Preprocess MPH landscapes
    #==========================================================================
    
    # Get files to process
    logger.info(f"Runnning preprocessing on files in {raw_data_directory_path}")
    files = list_files_in_directory(raw_data_directory_path)
    folder = extract_between_last_two_slashes(raw_data_directory_path)
    count = 1
    
    for file in files:
        file_no_extension = file.rstrip(".csv")
        # Get Data for processing
        logger.info(f"Runnning preprocessing on {file}...")
        X, parameter_level = mphData(file=f"{raw_data_directory_path}{file}", coord1=coord1, coord2=coord2, labelColumn=labelColumn, label=label, parameter=parameter, supervised=supervised, RipsMax=RipsMax, alpha=alpha)
        
        if np.isnan(parameter_level).any():
            logger.info(f"{file} parameter_level contains NaN")
        else:
            # Generate mph landscape
            logger.info(f"Generating mph landscape for {file}...")
            multi_landscape = computeMph(X, parameter_level, RipsMax=RipsMax, homology=homology, k_family=k_family, resolution=resolution, grid_step_size=grid_step_size, threads=threads, description="test")
            
            # Generate mph landscape contour 
            logger.info(f"Saving mph landscape plot for {file}...")
            landscape_plots = generateMph(multi_landscape, file=f"{processed_data_directory_path}{label}/{file_no_extension}_H{homology}_k{k_family}_{count}", indices=plot_indices)
        
        count += 1
        
    # Clean up temporary files
    logger.info(f"Clearing .txt temporary files")
    txt_files = glob.glob("*.txt")
    for txt_file in txt_files:
        try:
            os.remove(txt_file)
        except OSError as e:
            print(f"Error: {txt_file} : {e.strerror}")
            
    logger.info(f"Clearing .rivet temporary files")
    txt_files = glob.glob("*.rivet")
    for txt_file in txt_files:
        try:
            os.remove(txt_file)
        except OSError as e:
            print(f"Error: {txt_file} : {e.strerror}")
            
    #==========================================================================
    # Create pytorch dataset
    #==========================================================================
    
    logger.info(f"Creating PyTorch dataset...")