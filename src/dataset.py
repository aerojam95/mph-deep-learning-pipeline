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

# Custom modules

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


#=============================================================================
# Classes
#=============================================================================

class LabelledDataset(Dataset):
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
    
class UnlabeledDataset(Dataset):
    def __init__(self, data_dir, transform=transform):
        self.data_dir = data_dir
        self.transform = transform
        self.images = []
        
        for img_name in os.listdir(data_dir):
            img_path = os.path.join(data_dir, img_name)
            self.images.append(img_path)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('L')  # Convert image to grayscale
        
        if self.transform:
            image = self.transform(image)
        
        return image