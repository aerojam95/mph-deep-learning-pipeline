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
from bokeh.io import output_notebook
output_notebook()

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
    raw_file_path       = configurationData["data"]["raw_file_path"]
    processed_file_path = configurationData["data"]["processed_file_path"]
    
    # Extract outputs configuration
    logger.info(f"importing outputs configurations...")
    models_file_path      = configurationData["output"]["models_file_path"]
    training_file_path    = configurationData["output"]["training_file_path"]
    performance_file_path = configurationData["output"]["performance_file_path"]
    summaries_file_path   = configurationData["output"]["summaries_file_path"]
    
    # Extract mph configuration
    logger.info(f"importing mph configurations...")
    coord1          = configurationData["mph"]["coord1"]
    coord2          = configurationData["mph"]["coord2"]
    labelColumn     = configurationData["mph"]["labelColumn"]
    label           = configurationData["mph"]["label"]
    parameter       = configurationData["mph"]["parameter"]
    RipsMax         = configurationData["mph"]["RipsMax"]
    scaling         = configurationData["mph"]["scaling"]
    k_family        = configurationData["mph"]["k_family"]
    resolution      = configurationData["mph"]["resolution"]
    grid_step_size  = configurationData["mph"]["grid_step_size"]
    plot_indices    = configurationData["mph"]["plot_indices"]
    
    # Extract model configurations
    logger.info(f"importing model configurations...")
    cnn          = configurationData["model"]["cnn"]
    supervised   = configurationData["model"]["supervised"]
    unsupervised = configurationData["model"]["unsupervised"]
    
    #==========================================================================
    # Preprocess MPH landscapes
    #==========================================================================
    
    # Get files to process
    logger.info(f"Runnning preprocessing on files in {raw_file_path}")
    files = list_files_in_directory(raw_file_path)
    
    # Get Data for processing
    logger.info(f"Runnning preprocessing on {files[7]}...")
    X, parameter_level = mphData(file=f"{raw_file_path}{files[7]}", coord1=coord1, coord2=coord2, labelColumn=labelColumn, label=label, parameter=parameter, RipsMax=RipsMax, scaling=scaling)
    
    # Generate mph landscape
    logger.info(f"Generating mph landscape for {files[7]}...")
    multi_landscape = computeMph(X, parameter_level, RipsMax=RipsMax, k_family=k_family, resolution=resolution, grid_step_size=grid_step_size)
    
    # Generate mph landscape contour 
    logger.info(f"Saving mph landscape plot for {files[7]}...")
    landscape_plots = generateMph(multi_landscape, file=f"{processed_file_path}test", indices=plot_indices)