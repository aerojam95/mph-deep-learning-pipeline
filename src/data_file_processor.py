#=============================================================================
# Programme: Generate a codensity column and standardise data in a .csv file
#=============================================================================

#=============================================================================
# Modules
#=============================================================================

# Standard modules
from scipy.spatial import distance_matrix
import pandas as pd
import numpy as np
import yaml
import argparse
import os


# Custom modules
from logger import logger
import data_preprocessing
from mph.helper_functions import normalise_pointcloud, normalise_filter

#=============================================================================
# Variables
#=============================================================================

configuration_file_path = "../config/dataFile.yaml"

#=============================================================================
# Functions
#=============================================================================

def calculate_codensity(Y:np.ndarray, kNN:int=10, mean_codensity:bool=False):
    """Calculates a np.array of the codensity of a DataFrame X containing (x,y)
       coordinates

    Args:
        Y (pd.DataFrame): (x,y) coordinates
        kNN (int, optional): k nearest neighbours of codensity. Defaults to 10.
        mean_codensity (bool, optional): to generate mean or sum codensity. Defaults to True

    Returns:
        np.array: codesnity for each (x,y) point in X
    """
    D = distance_matrix(Y, Y)
    sortedD = np.sort(D)
    codensity = np.sum(sortedD[:, : kNN + 1], axis=1)
    if mean_codensity:
        codensity = codensity / kNN
    return codensity

def stack_arrays(X:np.ndarray, Y:np.ndarray, parameter:str="codensity"):
    """Takes codensity np.array and combines it into X DataFrame

    Args:
        X (np.ndarray): np array containing (x,y) coordinates
        Y (np.ndarray): np array containing second parameter
        parameter (str, optional): name of parameter for DataFrame column

    Returns:
        pd.DataFrame: DataFrame contains (x,y,parameter)
    """
    Z = np.hstack((X, Y[:, None]))
    Z = pd.DataFrame(Z, columns=["x", "y", parameter])
    return Z

#=============================================================================
# Classes
#=============================================================================



#=============================================================================
# Programme exectuion
#=============================================================================

if __name__ == "__main__":
    
    logger.info(f"Generating codensity ...")
    
    #==========================================================================
    # Argument parsing
    #==========================================================================
    
    parser = argparse.ArgumentParser(description="Modifies data .csv file")
    parser.add_argument("-r", "--raw_directory", type=str, required=True, help="Directory of files to be modified")
    parser.add_argument("-p", "--processed_directory", type=str, required=True, help="Output directory of modified files")
    args = parser.parse_args()
    directory_to_compute = args.raw_directory
    output_directory = args.processed_directory
    
    #==========================================================================
    # Configuration imports
    #==========================================================================
    
    with open(configuration_file_path, "r") as file:
        configuration_data = yaml.safe_load(file)
    
    # Extract data file configuration
    logger.info(f"importing data file configurations...")
    coord1               = configuration_data["coord1"]
    coord2               = configuration_data["coord2"]
    compute_codensity    = configuration_data["compute_codensity"]
    parameter            = configuration_data["parameter"]
    mean_codensity       = configuration_data["mean_codensity"]
    kNN                  = configuration_data["kNN"]
    region_standardise   = configuration_data["region_standardise"]
    alpha                = configuration_data["alpha"]
    global_standardise   = configuration_data["global_standardise"]
    rips_scale           = configuration_data["rips_scale"]
    parameter_scale      = configuration_data["parameter_scale"]
    
    #==========================================================================
    # Get files to compute
    #==========================================================================
    
    logger.info(f"Getting files to compute...")
    if directory_to_compute is not None:
        try:
            files = data_preprocessing.list_files_in_directory(directory_to_compute)
        except OSError as e:
            logger.error(f"Error: {directory_to_compute} : {e.strerror}")
    else:
        logger.error(f"No arguments parsed")
    logger.info(f"Loading data file...")
    
    #==========================================================================
    # Check output directory exists
    #==========================================================================

    logger.info(f"Checking output directory...")
    if not os.path.exists(output_directory):
            os.makedirs(output_directory)
    
    #==========================================================================
    # Calculate codensity as second paramater for file
    #==========================================================================
    
    if compute_codensity:
        logger.info(f"Calculating codensity and scaling parameters...")
        for file in files:
            X = pd.read_csv(f"{directory_to_compute}{file}")
            Y = X[[coord1, coord2]].values
            if region_standardise:
                Y = normalise_pointcloud(Y)
                codensity = calculate_codensity(Y=Y, kNN=kNN, mean_codensity=mean_codensity)
                Z = normalise_filter(codensity, alpha=alpha)
            elif global_standardise:
                Y = Y / rips_scale
                codensity = calculate_codensity(Y=Y, kNN=kNN, mean_codensity=mean_codensity)
                Z = codensity / parameter_scale
            logger.info(f"Saving {file}...")
            X = stack_arrays(X=Y, Y=Z, parameter=parameter)
            X.to_csv(f"{output_directory}{file}", index=False, mode="w")
    else:
        # Used saved second parameter for file
        logger.info(f"Scaling parameters...")
        for file in files:
            X = pd.read_csv(f"{directory_to_compute}{file}")
            Y = X[[coord1, coord2]].values
            Z = X[[parameter]].values
            if region_standardise:
                Y = normalise_pointcloud(Y)
                Z = normalise_filter(Z, alpha=alpha)
            elif global_standardise:
                Y = Y / rips_scale
                Z = Z / parameter_scale   
            logger.info(f"Saving {file}...")
            X = stack_arrays(X=Y, Y=Z, parameter=parameter)
            X.to_csv(f"{output_directory}{file}", index=False, mode="w")         
    logger.info(f"Data files processed")