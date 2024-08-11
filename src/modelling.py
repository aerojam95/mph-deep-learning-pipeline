#=============================================================================
# Programme: Pipeline of MPH representation for deep learning models
#=============================================================================

#=============================================================================
# Modules
#=============================================================================

# Standard modules
import yaml
import glob
import os
import numpy as np
import argparse
from tqdm import tqdm

# Custom modules
from logger import logger
import data_preprocessing
import dataset

#=============================================================================
# Variables
#=============================================================================

# Path to the JSON metadata file
configurationFilePath = "../config/model.yaml"

#=============================================================================
# Programme exectuion
#=============================================================================

if __name__ == "__main__":
    
    #==========================================================================
    # Argument parsing
    #==========================================================================
    
    parser = argparse.ArgumentParser(description="files for model processing")
    parser.add_argument("-d", "--dataset_directory", type=str, required=True, help="Data set directory")
    parser.add_argument("-l", "--label_directory", type=list, required=False, help="Labels directory")
    parser.add_argument("-m", "--model_name", type=str, required=True, help="Unique model name for outputs")
    args = parser.parse_args()
    dataset_directory = args.dataset_directory
    label_directory = args.label_directory
    model_name = args.model_name
    
    #==========================================================================
    # Configuration imports
    #==========================================================================
    
    with open(configurationFilePath, "r") as file:
        configurationData = yaml.safe_load(file)
    
    # Extract model configurations
    logger.info(f"importing model configurations...")
    test_ratio                 = configurationData["model"]["test_ratio"]
    supervised                 = configurationData["model"]["supervised"]
    pretrained                 = configurationData["model"]["pretrained"]
    pretrained_model_file_path = configurationData["model"]["pretrained_model_file_path"]
    
    # Extract outputs configuration
    logger.info(f"importing outputs configurations...")
    models_directory_path      = configurationData["output"]["models_directory_path"]
    training_directory_path    = configurationData["output"]["training_directory_path"]
    summaries_directory_path   = configurationData["output"]["summaries_directory_path"]
    
    #==========================================================================
    # Checking output directory exists
    #==========================================================================
    
    logger.info(f"Checking output model directory...")
    if not os.path.exists(models_directory_path):
            os.makedirs(models_directory_path)
            
    logger.info(f"Checking output training directory...")
    if not os.path.exists(training_directory_path):
            os.makedirs(training_directory_path)
            
    logger.info(f"Checking output summaries directory...")
    if not os.path.exists(summaries_directory_path):
            os.makedirs(summaries_directory_path)
    
    #==========================================================================
    # Deep learning modelling
    #==========================================================================
        
    logger.info(f"Beginning modelling...")
    
    #======================================================================
    #  Data set generation
    #======================================================================
    
    # Generate data set objects
    if supervised is True:
        logger.info(f"Generating supervised data set...")
        labels = data_preprocessing.combine_label_files(label_directory)
        model_dataset = dataset.LabelledDataset(dataset_directory, labels)
    else:
        logger.info(f"Generating unsupervised data set...")
        model_dataset = dataset.UnlabeledDataset(dataset_directory)
        
    # Split to test and training data sets
    logger.info(f"Splitting data set...")
    train_dataset, test_dataset = dataset.split_dataset(model_dataset, test_ratio)
    
    #======================================================================
    #  Model training
    #======================================================================
    
    if pretrained is False:
        logger.info(f"Training model...")