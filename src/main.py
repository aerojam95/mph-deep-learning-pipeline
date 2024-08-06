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

# Custom modules
from logger import logger
import data_preprocessing
import dataset

#=============================================================================
# Variables
#=============================================================================

# Path to the JSON metadata file
configurationFilePath = "pipeline_configuration.yaml"

#=============================================================================
# Programme exectuion
#=============================================================================

if __name__ == "__main__":
    
    #==========================================================================
    # Configuration imports
    #==========================================================================
    
    with open(configurationFilePath, "r") as file:
        configurationData = yaml.safe_load(file)
    
    # Extract data configuration
    logger.info(f"importing data configurations...")
    raw_data_directory_path       = configurationData["data"]["raw_data_directory_path"]
    processed_data_directory_path = configurationData["data"]["processed_data_directory_path"]
    label_file                    = configurationData["data"]["label_file"]
    
    # Extract mph configuration
    logger.info(f"importing mph configurations...")
    contours                = configurationData["mph"]["contours"]
    threads                 = configurationData["mph"]["threads"]
    coord1                  = configurationData["mph"]["coord1"]
    coord2                  = configurationData["mph"]["coord2"]
    parameter               = configurationData["mph"]["parameter"]
    homology                = configurationData["mph"]["homology"]
    k_family                = configurationData["mph"]["k_family"]
    plot_indices            = configurationData["mph"]["plot_indices"]
    resolution              = configurationData["mph"]["resolution"]
    RipsMax                 = configurationData["mph"]["RipsMax"]
    grid_step_size          = configurationData["mph"]["grid_step_size"]
    
    # Extract model configurations
    logger.info(f"importing model configurations...")
    modelling                  = configurationData["model"]["modelling"]
    test_ratio                 = configurationData["model"]["test_ratio"]
    pretrained                 = configurationData["model"]["pretrained"]
    pretrained_model_file_path = configurationData["model"]["pretrained_model_file_path"]
    supervised                 = configurationData["model"]["supervised"]
    model_to_train_file        = configurationData["model"]["model_to_train_file"]
    
    # Extract outputs configuration
    logger.info(f"importing outputs configurations...")
    models_directory_path      = configurationData["output"]["models_directory_path"]
    training_directory_path    = configurationData["output"]["training_directory_path"]
    summaries_directory_path   = configurationData["output"]["summaries_directory_path"]
    
    #==========================================================================
    # Generating MPH contours 
    #==========================================================================
    
    if contours is True:
        
        logger.info(f"Beginning contour generation...")
    
        #======================================================================
        # Supervised learning settings 
        #======================================================================
        
        if supervised is True:
            
            logger.info(f"Supervised learning pipeline to be set up")
            
            # Generate label directories
            data_preprocessing.make_label_directories(
                processed_data_directory_path,
                label_file
                )
            logger.info(f"Supervised learning label directories generated")
            
            # Make file dictionary label
            file_label_dict = data_preprocessing.make_label_allocation(label_file)
            print(file_label_dict)
            logger.info(f"file label allocations identified")
            
        #======================================================================
        # Contour generation 
        #======================================================================
    
        # Get files to process
        logger.info(f"Runnning contour generation on files in {raw_data_directory_path}")
        files = data_preprocessing.list_files_in_directory(raw_data_directory_path)
        folder = data_preprocessing.extract_between_last_two_slashes(raw_data_directory_path)
        
        for file in files:
            file_no_extension = file.rstrip(".csv")
            # Get Data for processing
            logger.info(f"Runnning preprocessing on {file_no_extension}...")
            X, parameter_level = data_preprocessing.mphData(
                file=f"{raw_data_directory_path}{file}",
                coord1=coord1, 
                coord2=coord2, 
                parameter=parameter
                )
            
            if np.isnan(parameter_level).any():
                logger.info(f"{file} parameter_level contains NaN")
            else:
                # Generate mph landscape
                logger.info(f"Generating mph landscape for {file_no_extension}...")
                multi_landscape = data_preprocessing.computeMph(
                    X, 
                    parameter_level, 
                    RipsMax=RipsMax, 
                    homology=homology, 
                    k_family=k_family, 
                    resolution=resolution, 
                    grid_step_size=grid_step_size, 
                    threads=threads, 
                    description=f"{file}"
                    )
                
                # Generate mph landscape contour 
                logger.info(f"Saving mph landscape plot for {file_no_extension}...")
                
                if supervised is True:
                    label = file_label_dict[file]
                    contour_file = f"{processed_data_directory_path}{label}/{file_no_extension}_H{homology}_k{k_family}_RipsMax{RipsMax}"
                else:
                    contour_file = f"{processed_data_directory_path}{file_no_extension}_H{homology}_k{k_family}_RipsMax{RipsMax}"
                    
                landscape_plots = data_preprocessing.generateMph(
                    multi_landscape, 
                    file=contour_file,
                    indices=plot_indices
                    )
            
        # Clean up temporary files
        logger.info(f"Clearing .txt temporary files...")
        txt_files = glob.glob("*.txt")
        for txt_file in txt_files:
            try:
                os.remove(txt_file)
            except OSError as e:
                print(f"Error: {txt_file} : {e.strerror}")
                
        logger.info(f"Clearing .rivet temporary files...")
        txt_files = glob.glob("*.rivet")
        for txt_file in txt_files:
            try:
                os.remove(txt_file)
            except OSError as e:
                print(f"Error: {txt_file} : {e.strerror}")
    
    #==========================================================================
    # Deep learning modelling
    #==========================================================================
    
    if modelling is True:
        
        logger.info(f"Beginning modelling...")
        
        #======================================================================
        #  Data set generation
        #======================================================================
        
        # Generate data set objects
        if supervised is True:
            logger.info(f"Generating supervised data set...")
            labels = data_preprocessing.get_labels(label_file)
            model_dataset = dataset.LabelledDataset(processed_data_directory_path, labels)
        else:
            logger.info(f"Generating unsupervised data set...")
            model_dataset = dataset.LabelledDataset(processed_data_directory_path)
            
        # Split to test and training data sets
        logger.info(f"Splitting data set...")
        train_dataset, test_dataset = dataset.split_dataset(model_dataset, test_ratio)
        
        #======================================================================
        #  Model training
        #======================================================================
        
        if pretrained is False:
            logger.info(f"Training model...")
