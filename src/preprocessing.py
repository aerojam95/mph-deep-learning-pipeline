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

import bokeh
from bokeh.plotting import show
from bokeh.io import output_notebook

output_notebook()

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

def mphData(file:str, coord1:str, coord2:str, labelColumn:str, label:str, parameter:str, supervised:bool=True, RipsMax:float=10, scaling:float=5):
    df = pd.read_csv(file)
    if supervised is True:
        X = df[df[labelColumn] == label][[coord1, coord2]].values
        parameter_level = RipsMax * normalise_filter(
            df[df[labelColumn] == label][parameter].values, scaling
        )
        print(df[labelColumn] == label)
    else:
        X = df[[coord1, coord2]].values
        parameter_level = RipsMax * normalise_filter(
            df[parameter].values, scaling
        )
    return X, parameter_level
    

def computeMph(X:pd.DataFrame, parameter_level:pd.DataFrame, RipsMax:float=10.0, k_family:int=1, resolution:float=50, grid_step_size:float=0.4):
    filtered_points = np.hstack((X, parameter_level[:, None]))
    rivet_output = Compute_Rivet(filtered_points, dim=k_family, resolution=resolution, RipsMax=RipsMax)
    multi_landscape = multiparameter_landscape(
        rivet_output, maxind=k_family, grid_step_size=grid_step_size, bounds=[[0, 0], [RipsMax, RipsMax]]
    )
    return multi_landscape

#=============================================================================
# Classes
#=============================================================================