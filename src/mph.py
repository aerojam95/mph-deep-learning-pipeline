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

#=============================================================================
# Variables
#=============================================================================

# Path to the JSON metadata file
configurationFilePath = "../config/mph.yaml"

#=============================================================================
# Programme exectuion
#=============================================================================

if __name__ == "__main__":
    
    #==========================================================================
    # Argument parsing
    #==========================================================================
    
    parser = argparse.ArgumentParser(description="files for mph processing")
    parser.add_argument("-r", "--raw_directory", type=str, required=True, help="Directory containing files for processing")
    parser.add_argument("-l", "--label_file", type=str, required=False, help="Label file")
    parser.add_argument("-p", "--processed_directory", type=str, required=True, help="Directory for saving mph outputs")
    args = parser.parse_args()
    raw_data_directory_path = args.raw_directory
    label_file = args.label_file
    processed_data_directory_path = args.processed_directory
    
    #==========================================================================
    # Configuration imports
    #==========================================================================
    
    with open(configurationFilePath, "r") as file:
        configurationData = yaml.safe_load(file)
    
    # Extract mph configuration
    logger.info(f"importing mph configurations...")
    points         = configurationData["points"]
    threads        = configurationData["threads"]
    coord1         = configurationData["coord1"]
    coord2         = configurationData["coord2"]
    parameter      = configurationData["parameter"]
    homology       = configurationData["homology"]
    k_family       = configurationData["k_family"]
    plot_indices   = configurationData["plot_indices"]
    resolution     = configurationData["resolution"]
    RipsMax        = configurationData["RipsMax"]
    grid_step_size = configurationData["grid_step_size"]
    supervised     = configurationData["supervised"]
    
    #==========================================================================
    # Checking output directory exists
    #==========================================================================
    
    logger.info(f"Checking output directory...")
    if not os.path.exists(processed_data_directory_path):
            os.makedirs(processed_data_directory_path)
    
    #==========================================================================
    # Generating MPH contours 
    #==========================================================================
        
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
        logger.info(f"file label allocations identified")
        
    #======================================================================
    # Contour generation 
    #======================================================================

    # Get files to process
    logger.info(f"Runnning contour generation on files in {raw_data_directory_path}")
    files = data_preprocessing.list_files_in_directory(raw_data_directory_path)
    folder = data_preprocessing.extract_between_last_two_slashes(raw_data_directory_path)
    
    for file in tqdm(files, desc="Processing files"):
        file_no_extension = file.rstrip(".csv")
        # Get Data for processing
        logger.info(f"Runnning preprocessing on {file_no_extension}...")
        X, parameter_level = data_preprocessing.mphData(
            file=f"{raw_data_directory_path}{file}",
            coord1=coord1, 
            coord2=coord2, 
            parameter=parameter,
            nrows=points
            )
        
        if np.isnan(parameter_level).any():
            logger.info(f"{file} parameter_level contains NaN")
        else:
            # Generate mph landscape
            logger.info(f"Generating mph landscape for {file_no_extension}...")
            
            try:
                multi_landscape = data_preprocessing.computeMph(
                    X, 
                    parameter_level, 
                    RipsMax=RipsMax, 
                    homology=homology, 
                    k_family=k_family, 
                    resolution=resolution, 
                    grid_step_size=grid_step_size, 
                    threads=threads, 
                    description=f"{file_no_extension}"
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
                
            except Exception as e:
                logger.error(f"Failed to generate mph landscape for {file_no_extension}: {e}")
        
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
