#=============================================================================
# Module: data processing of MPH contours
#=============================================================================

#=============================================================================
# Module imports
#=============================================================================

# Standard modules
from torch.utils.data import Dataset, random_split
import torchvision.transforms as transforms
from PIL import Image
import os
import torch

# Custom modules
from logger import logger

#=============================================================================
# Variables
#=============================================================================



#=============================================================================
# Functions
#=============================================================================

def get_transform(mean:list=None, std:list=None, image_size:int=64):
    """Get a simple torch transformer

    Args:
        mean (list, optional): mean of data set for normalising dataset. Defaults to None.
        std (list, optional): standard deviation of data set for normalising data set. Defaults to None.
        image_size (int, optional): Number of pixels for square image. Defaults to 64.

    Returns:
        object: torch transformer object
    """   
    if mean is None and std is None:
        return transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor()])
    elif mean is not None and std is not None:
        return transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor(), transforms.Normalize(mean=mean.tolist(), std=std.tolist())])
    else:
        logger.error(f"mean and std need to be lists to generate transform object")
        return None
        

def split_dataset(dataset:object, ratio:float=0.8):
    """Split the data set object up into training and test data sets

    Args:
        dataset (object): Data set object to be split
        ratio (float, optional): Ratio of data set to given to traing. 
                                Defaults to 0.8.

    Returns:
        object, object: training and test data sets
    """
    test_size = int(ratio * len(dataset))
    train_size = len(dataset) - test_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    return train_dataset, test_dataset

def calculate_mean_std(loader:object):
    """Calculates mean and standard deviation of dataloader object

    Args:
        loader (object): dataloader object to find mean and standard deviation

    Returns:
        tuple: mean and standard deviation lists in a tuple
    """    
    mean = 0.0
    std = 0.0
    total_images_count = 0
    for images, _ in loader:
        batch_samples = images.size(0)  # batch size (the last batch can have smaller size)
        images = images.view(batch_samples, images.size(1), -1)  # (batch_size, channels, width*height)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images_count += batch_samples
    
    mean /= total_images_count
    std /= total_images_count
    return mean, std

#=============================================================================
# Classes
#=============================================================================

class Labelled_dataset(Dataset):
    def __init__(self:object, data_dir:str, labels:dict, transform:object):
        """
        Initializes the dataset by loading all image paths and their corresponding labels.
        
        Args:
            data_dir (str): The directory where images are stored.
            labels (dict): A dictionary of labels corresponding with class nuimberings.
            transform (object): Transform to be applied on a sample.
        """
        self.data_dir = data_dir
        self.transform = transform
        self.images = []
        self.labels = []
        
        for key, value in labels.items():
            label_dir = os.path.join(data_dir, str(value))
            for img_name in os.listdir(label_dir):
                img_path = os.path.join(label_dir, img_name)
                self.images.append(img_path)
                self.labels.append(key)
    
    def __len__(self:object):
        """Returns the total number of samples in the dataset. Int."""
        return len(self.images)
    
    def __getitem__(self:object, idx:int):
        """
        Generates one sample of data.
        
        Args:
            idx (int): Index of the sample to be fetched.
        
        Returns:
            tuple: (image, label) where image is a tensor and label is a tensor.
        """
        img_path = self.images[idx]
        image = Image.open(img_path).convert('L')  # Convert image to grayscale
        label = self.labels[idx]
        image = self.transform(image)
        label_tensor = torch.tensor(label, dtype=torch.long)
        return image, label_tensor
    
class Unlabelled_dataset(Dataset):
    def __init__(self, data_dir:str, transform:object):
        """
        Initializes the dataset by loading all image paths.
        
        Args:
            data_dir (str): The directory where images are stored.
            transform (object): Transform to be applied on a sample.
        """
        self.data_dir = data_dir
        self.transform = transform
        self.images = []
        
        for img_name in os.listdir(data_dir):
            img_path = os.path.join(data_dir, img_name)
            self.images.append(img_path)
    
    def __len__(self:object):
        """Returns the total number of samples in the dataset. Int."""
        return len(self.images)
    
    def __getitem__(self:object, idx:int):
        """
        Generates one sample of data.
        
        Args:
            idx (int): Index of the sample to be fetched.
        
        Returns:
            torch.tensor: image where image is a tensor and label is a tensor.
        """
        img_path = self.images[idx]
        image = Image.open(img_path).convert('L')  # Convert image to grayscale
        
        if self.transform:
            image = self.transform(image)
        
        return image