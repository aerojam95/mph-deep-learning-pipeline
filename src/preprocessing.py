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
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import os

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

# Transformer for PyTorch dataset
transform = transforms.Compose([
   transforms.Resize((28, 28)),
   transforms.ToTensor(),
   transforms.Normalize((0.5,), (0.5,))
])

#=============================================================================
# Functions
#=============================================================================

def mphData(file:str, coord1:str, coord2:str, labelColumn:str, label:str, parameter:str, supervised:bool=True, RipsMax:float=40, alpha:float=5):
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
        alpha (float, optional): alpha parameter. Defaults to 5.

    Returns:
        pd DataFrame, pd DataFrame: Coordinate and parameter DataFrames
    """
    df = pd.read_csv(file)
    if supervised is True:
        X = df[df[labelColumn] == label][[coord1, coord2]].values
        parameter_level = RipsMax * normalise_filter(
            df[df[labelColumn] == label][parameter].values, alpha
        )
    else:
        X = df[[coord1, coord2]].values
        parameter_level = RipsMax * normalise_filter(
            df[parameter].values, alpha
        )
    return X, parameter_level
    

def computeMph(X:pd.DataFrame, parameter_level:pd.DataFrame, RipsMax:float=10.0, homology:int=0, k_family:int=1, resolution:float=50, grid_step_size:float=0.4, threads:int=1, description:str='deafult_description'):
    """Computes the multiparameter persistence landscapes

    Args:
        X (pd.DataFrame): Coordinate DataFrame
        parameter_level (pd.DataFrame): Parameter DataFrame
        RipsMax (float, optional): Largest epsilon value for the Rips complex. Defaults to 10.0.
        homology (int, optional): homology dimension to compute for the MPH contour
        k_family (int, optional): The k first multiparameter landscapes. Defaults to 1.
        resolution (float, optional): Resolution of landscape in x and y directions of the contour. Defaults to 50.
        grid_step_size (float, optional): grid size for Rips Complex computations for contour plotting. Defaults to 0.4.
        threads (int, optional): number of threads to parse to Rivet that are available OpenMP parallel processing
        descriptiuon (str, optional): description of temporary files

    Returns:
        matrix: multiparameter landscape for inputs DataFrames
    """
    filtered_points = np.hstack((X, parameter_level[:, None]))
    rivet_output = Compute_Rivet(filtered_points, dim=homology, resolution=resolution, RipsMax=RipsMax, threads=threads, description=description)
    multi_landscape = multiparameter_landscape(
        rivet_output, maxind=k_family, grid_step_size=grid_step_size, bounds=[[0, 0], [RipsMax, RipsMax]]
    )
    return multi_landscape

def generateMph(multi_landscape:object, file:str, indices:list=[1, 1]):
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
            plt.imsave(f"{file}.png", normalised_matrix, cmap="gray")
    else:
        raise ValueError(f"No landscape matrix to save")
    return None

#=============================================================================
# Classes
#=============================================================================

class mphDataset(Dataset):
    def __init__(self:object, data_dir:str, labels:list, transform=transform):
        self.data_dir = data_dir
        self.transform = transform
        self.images = []
        self.labels = []
        
        for label in labels:
            label_dir = os.path.join(data_dir, str(label))
            for img_name in os.listdir(label_dir):
                img_path = os.path.join(label_dir, img_name)
                self.images.append(img_path)
                self.labels.append(label)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('L')  # Convert image to grayscale
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label