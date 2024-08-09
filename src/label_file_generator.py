#=============================================================================
# Programme: generates a label file .csv for the mph pipeline
#=============================================================================

#=============================================================================
# Modules
#=============================================================================

# Standard modules
import pandas as pd
import argparse
import os

# Custom modules
from logger import logger
import data_preprocessing

#=============================================================================
# Variables
#=============================================================================



#=============================================================================
# Functions
#=============================================================================

def create_dataframe(file_list:list, label:str):
    """Generate a dataframe for that has column of files and column of labels
       which are all the same

    Args:
        file_list (list): list of files to be labelled 
        label (str): label to add to each file

    Returns:
        pd.DataFrame: DataFrame of labelled files
    """
    data = {"files": file_list, "labels": [label] * len(file_list)}
    df = pd.DataFrame(data)
    return df

#=============================================================================
# Classes
#=============================================================================



#=============================================================================
# Programme exectuion
#=============================================================================

if __name__ == "__main__":
    
    
    logger.info(f"Generating label file...")
    
   #==========================================================================
    # Argument parsing
    #==========================================================================
    
    parser = argparse.ArgumentParser(description="Generates output .csv file parsed ")
    parser.add_argument("-d", "--directory", type=str, required=True, help="Directory of files to be labelled")
    parser.add_argument("-l", "--label", type=str, required=True, help="Label for files")
    parser.add_argument("-f", "--ouput_file", type=str, required=True, help="Output file path")
    args = parser.parse_args()
    directory = args.directory
    label = args.label
    output_file_path = args.ouput_file
    
    #==========================================================================
    # Load files and label
    #==========================================================================
    
    logger.info(f"Labelling files...")
    try:
        files = data_preprocessing.list_files_in_directory(directory)
    except OSError as e:
        logger.error(f"Directory {directory} does not exist")
    try:
        labelled_files = create_dataframe(files, label)
    except OSError as e:
        logger.error(f"Label {label} is invalid")
    
    #==========================================================================
    # Saving label file
    #==========================================================================
    
    logger.info(f"Labelling files...")
    try:
        output_directory = os.path.dirname(output_file_path)
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        labelled_files.to_csv(output_file_path, index=False)
    except OSError as e:
        logger.error(f"Output file {output_file_path} is invalid")
    logger.info(f"Label file generated")
    