#=============================================================================
# Module: data processing of MPH contours
#=============================================================================

#=============================================================================
# Module imports
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
from logger import logger
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

def get_labels(label_file:str):
    """generates a list of labels that exist in the file label_file

    Args:
        label_file (str, optional): .csv file that contains a column named labels.

    Returns:
        list: list of unique labels from label_file
    """ 
    labels = []
    try: 
        df = pd.read_csv(label_file)
        labels = df["labels"].unique()
        labels = sorted(labels)
        logger.info(f"Labels found successfully")
    except Exception as e:
            # Log any exception that occurs
            logger.error(f"An error occurred while getting labels: {e}")
    return labels

def make_label_directories(processed_data_directory_path: str, label_file: str):
    """creates a set of directories from labels list at the directory processed_data_directory_path

    Args:
        processed_data_directory_path (str): directory to create all new directories
        label_file (str): .csv file that contains a column named labels.

    Returns:
        None: None
    """
    labels = get_labels(label_file)
    for label in labels:
        directory_path = f"{processed_data_directory_path}{label}"
        try:
            # Create directory
            os.makedirs(directory_path, exist_ok=True)
            logger.info(f"Directory created successfully at: {directory_path}")
        except Exception as e:
            # Log any exception that occurs
            logger.error(f"An error occurred while creating the directory: {e}")
    return None

def make_label_allocation(label_file: str):
    """generates a dictionary of labels for a data file

    Args:
        label_file (str): file containing filenames with their respective labels
    """
    label_dict = {}
    df = pd.read_csv(label_file)
    try:
        # Iterate over each row using iterrows()
        for index, row in df.iterrows():
            try:
                # Access each element by column name
                key = row["files"]
                value = row["labels"]
                label_dict[key] = value
            except Exception as e:
                logger.error(f"Error processing row {index}: {e}")
    except Exception as e:
        logger.error(f"An error occurred while iterating over the DataFrame: {e}")
    return label_dict

def mphData(file:str, coord1:str, coord2:str, parameter:str, RipsMax:float=40, alpha:float=5):
    """formats data from a .csv data file into two pd DataFrames

    Args:
        file (str): .csv data file
        coord1 (str): x coordinate for persistent homology
        coord2 (str): y coordinate for persistent homology
        parameter (str): second parameter for persistent homology
        RipsMax (float, optional): Max epsilon for Rips Complex to scale second parameter. Defaults to 40.
        alpha (float, optional): alpha parameter. Defaults to 5.

    Returns:
        pd DataFrame, pd DataFrame: Coordinate and parameter DataFrames
    """
    df = pd.read_csv(file)
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
        array: multiparameter landscape for inputs DataFrames
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