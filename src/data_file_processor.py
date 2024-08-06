#=============================================================================
# Programme: Generate a codensity column in a .csv file
#=============================================================================

#=============================================================================
# Modules
#=============================================================================

# Standard modules
from scipy.spatial import distance_matrix
import pandas as pd
import numpy as np
import yaml


# Custom modules
from logger import logger
import data_preprocessing
from mph.helper_functions import normalise_pointcloud, normalise_filter

#=============================================================================
# Variables
#=============================================================================

configurationFilePath = "data_file_configuration.yaml"

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
    # Configuration imports
    #==========================================================================
    
    with open(configurationFilePath, "r") as file:
        configurationData = yaml.safe_load(file)
    
    # Extract data file configuration
    logger.info(f"importing data file configurations...")
    directory_to_compute = configurationData["directory_to_compute"]
    coord1               = configurationData["coord1"]
    coord2               = configurationData["coord2"]
    compute_codensity    = configurationData["compute_codensity"]
    parameter            = configurationData["parameter"]
    mean_codensity       = configurationData["mean_codensity"]
    kNN                  = configurationData["kNN"]
    region_standardise   = configurationData["region_standardise"]
    alpha                = configurationData["alpha"]
    global_standardise   = configurationData["global_standardise"]
    rips_scale           = configurationData["rips_scale"]
    parameter_scale      = configurationData["parameter_scale"]
    
    # Get files to compute
    logger.info(f"Getting files to compute...")
    if directory_to_compute is not None:
        try:
            files = data_preprocessing.list_files_in_directory(directory_to_compute)
        except OSError as e:
            logger.error(f"Error: {directory_to_compute} : {e.strerror}")
    else:
        logger.error(f"No arguments parsed")
        
    logger.info(f"Loading data file...")
    
    
    if compute_codensity:
        # Calculate codensity as second paramater for file
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
            X.to_csv(f"{directory_to_compute}{file}", index=False, mode="w")
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
            X.to_csv(f"{directory_to_compute}{file}", index=False, mode="w")