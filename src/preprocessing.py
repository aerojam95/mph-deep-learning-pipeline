#=============================================================================
# Programme: Pipeline of MPH representation for deep learning models
#=============================================================================

#=============================================================================
# Modules
#=============================================================================

# Standard modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
from scipy.ndimage import zoom
import matplotlib.pyplot as plt

# Custom modules
import mph.multiparameter_landscape
from mph.multiparameter_landscape import multiparameter_landscape
import mph.multiparameter_landscape_plotting
from mph.multiparameter_landscape_plotting import plot_multiparameter_landscapes
import mph.helper_functions
from mph.helper_functions import normalise_filter
from mph.helper_functions import Compute_Rivet

#=============================================================================
# Variables
#=============================================================================

#=============================================================================
# Functions
#=============================================================================

def mphData(file:str, coord1:str, coord2:str, labelColumn:str, label:str, parameter:str, supervised:bool=True, RipsMax:float=40, scaling:float=5):
    """formats data from a .csv data file into two pd DataFrames

    Args:
        file (str): .csv data file
        coord1 (str): x coordinate for persistent homology
        coord2 (str): y coordinate for persistent homology
        labelColumn (str): label column in .csv data file
        label (str): specific data with the label in .csv data file
        parameter (str): second parameter for persistent homology
        supervised (bool, optional): Boolean to check is labels are needed. Defaults to True.
        RipsMax (float, optional): Max epsilon for Rips Complex to scale second parameter. Defaults to 40.
        scaling (float, optional): Scaling parameter. Defaults to 5.

    Returns:
        pd DataFrame, pd DataFrame: Coordinate and parameter DataFrames
    """
    df = pd.read_csv(file)
    if supervised is True:
        X = df[df[labelColumn] == label][[coord1, coord2]].values
        parameter_level = RipsMax * normalise_filter(
            df[df[labelColumn] == label][parameter].values, scaling
        )
    else:
        X = df[[coord1, coord2]].values
        parameter_level = RipsMax * normalise_filter(
            df[parameter].values, scaling
        )
    return X, parameter_level
    

def computeMph(X:pd.DataFrame, parameter_level:pd.DataFrame, RipsMax:float=10.0, k_family:int=1, resolution:float=50, grid_step_size:float=0.4):
    """Computes the multiparameter persistence landscapes

    Args:
        X (pd.DataFrame): Coordinate DataFrame
        parameter_level (pd.DataFrame): Parameter DataFrame
        RipsMax (float, optional): Largest epsilon value for the Rips complex. Defaults to 10.0.
        k_family (int, optional): The k first multiparameter landscapes. Defaults to 1.
        resolution (float, optional): Resolution of landscape. Defaults to 50.
        grid_step_size (float, optional): grid size for Rips Complex computations. Defaults to 0.4.

    Returns:
        matrix: multiparameter landscape for inputs DataFrames
    """
    filtered_points = np.hstack((X, parameter_level[:, None]))
    rivet_output = Compute_Rivet(filtered_points, dim=k_family, resolution=resolution, RipsMax=RipsMax)
    multi_landscape = multiparameter_landscape(
        rivet_output, maxind=k_family, grid_step_size=grid_step_size, bounds=[[0, 0], [RipsMax, RipsMax]]
    )
    return multi_landscape

def generateMph(multi_landscape: object, file:str, indices: list=[1, 1]):
    """generate mph plot object for a mph landscape object and save to png

    Args:
        multi_landscape (object): mph landscape object
        file (str): file name to output plot
        indices (list, optional): list of the k contours to generate. Defaults to [1, 1].

    Returns:
        None
    """
    if hasattr(multi_landscape, "landscape_matrix"):
        
        for index in indices:
            contour = multi_landscape.landscape_matrix[index - 1, :, :]
            
            # Normalise the matrix to be in the range [0, 255] for image representation
            normalised_matrix = (255 * (contour - np.min(contour)) / 
                                np.ptp(contour)).astype(np.uint8)
            
            # Save the image without axes using imsave
            plt.imsave(f"{file}_k_{index}.png", normalised_matrix, cmap="gray")
    else:
        raise ValueError(f"No landscape matrix to save")
    return None

#=============================================================================
# Classes
#=============================================================================