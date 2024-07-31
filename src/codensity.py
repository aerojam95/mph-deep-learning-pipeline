#=============================================================================
# Programme: Generate a codensity column in a .csv file
#=============================================================================

#=============================================================================
# Modules
#=============================================================================

# Standard modules
import argparse
from scipy.spatial import distance_matrix
import pandas as pd
import numpy as np


# Custom modules
from logger import logger
import data_preprocessing

#=============================================================================
# Variables
#=============================================================================



#=============================================================================
# Functions
#=============================================================================

def calculate_codensity(X:pd.DataFrame, kNN:int=5):
    """Calculates a np.array of the codensity of a DataFrame X containing (x,y)
       coordinates

    Args:
        X (pd.DataFrame): (x,y) coordinates
        kNN (int, optional): k nearest neighbours of codensity. Defaults to 5.

    Returns:
        np.array: codesnity for each (x,y) point in X
    """
    D = distance_matrix(X, X)
    sortedD = np.sort(D)
    codensity = np.sum(sortedD[:, : kNN + 1], axis=1)
    return codensity

def stack_codensity(codensity:np.array, X:pd.DataFrame):
    """Takes codensity np.array and combines it into X DataFrame

    Args:
        codensity (np.array): codensity of X
        X (pd.DataFrame): DataFrame containing (x,y) coordinates

    Returns:
        pd.DataFrame: DataFrame contains (x,y,codensity)
    """
    Y = np.hstack((X, codensity[:, None]))
    Z = pd.DataFrame(Y, columns=["x", "y", "codensity"])
    return Z

def append_codensity(file:str, kNN:int=5):
    """Takes file containing (x,y) coordinates and writes codensity to it

    Args:
        file (str): file containing (x,y) coordinates
        kNN (int, optional): k nearest neighbours of codensity. Defaults to 5.

    Returns:
        None: None
    """
    X = pd.read_csv(file)
    codensity = calculate_codensity(X=X, kNN=kNN)
    X = stack_codensity(codensity=codensity, X= X)
    X.to_csv(file, index=False, mode="w")
    return None

#=============================================================================
# Classes
#=============================================================================



#=============================================================================
# Programme exectuion
#=============================================================================

if __name__ == "__main__":
    
    logger.info(f"Generating codensity ...")
    
    # Argument parsing
    parser = argparse.ArgumentParser(description="Adding codensity parameter column .csv file(s) parsed")
    parser.add_argument("-d", type=str, required=False, help="Directory of files to add codensity")
    parser.add_argument("-f", type=str, required=False, help="File path to add condensity")
    args = parser.parse_args()
    directory_to_compute = args.d
    file_to_compute = args.f
    
    # Get files to compute
    logger.info(f"Getting files to compute...")
    if directory_to_compute is not None:
        try:
            files = data_preprocessing.list_files_in_directory(directory_to_compute)
        except OSError as e:
            logger.error(f"Error: {directory_to_compute} : {e.strerror}")
    elif file_to_compute is not None:
        try:
            logger.info(f"{file_to_compute} codensity to be computed")
        except OSError as e:
            logger.error(f"Error: {directory_to_compute} : {e.strerror}")
    else:
        logger.error(f"No arguments parsed")
        
    # Compute codensity
    logger.info(f"Computing codensity...")
    if file_to_compute is not None:
        try:
            append_codensity(file_to_compute)
            logger.info(f"Codensity written to {file_to_compute}")
        except OSError as e:
            logger.error(f"Error: couldn't overwirte {file_to_compute}: {e.strerror}")
    else:
        for file in files:
            try:
                append_codensity(f"{directory_to_compute}{file}")
                logger.info(f"Codensity written to {directory_to_compute}{file}")
            except OSError as e:
                logger.error(f"Error: couldn't overwirte {file}: {e.strerror}")