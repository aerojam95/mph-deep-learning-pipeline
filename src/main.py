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

# Custom modules
from preprocessing import mphData, computeMph

#=============================================================================
# Variables
#=============================================================================

# Path to the JSON metadata file
configurationFilePath = "configuration.yaml"

#=============================================================================
# Functions
#=============================================================================

def list_files_in_directory(directory):
    try:
        files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
        logger.info(f"Files to be processed found")
        return files
    except Exception as e:
        logger.error(f"Files to be processed not found")
        return []


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
    # Preprocess MPH landscapes
    #==========================================================================
    
    # Get files to process
    raw_file_path = configurationData["data"]["raw_file_path"]
    logger.info(f"Runnning preprocessing on files in {raw_file_path}")
    files = list_files_in_directory(raw_file_path)
    
    X, parameter_level = mphData(file=f"{raw_file_path}{files[7]}", coord1="x", coord2="y", labelColumn="PointType", label="Macrophage", parameter="Oxygen", RipsMax=40, scaling=5)
    multi_landscape = computeMph(X, parameter_level, RipsMax=40, k_family=1, resolution=50, grid_step_size=0.4)