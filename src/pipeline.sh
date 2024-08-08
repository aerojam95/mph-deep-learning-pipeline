#!/bin/bash

# Base script for data sets contained in ../data/raw/1.5mmRegions.zip

# Activate the virtual environment if needed
# source <path-to-python-venv/bin/activate>

# Generate label files
echo "Generating label files..."
python3 label_file_generator.py -d "../data/raw/1.5mmRegions/CD8" -l "CD8" -f "../data/raw/labels/CD8.csv"
if [ $? -ne 0 ]; then
    echo "CD8 label file generation failed"
    exit 1
fi
python3 label_file_generator.py -d "../data/raw/1.5mmRegions/CD68" -l "CD68" -f "../data/raw/labels/CD68.csv"
if [ $? -ne 0 ]; then
    echo "CD68 label file generation failed"
    exit 1
fi
python3 label_file_generator.py -d "../data/raw/1.5mmRegions/FoxP3" -l "FoxP3" -f "../data/raw/labels/FoxP3.csv"
if [ $? -ne 0 ]; then
    echo "FoxP3 label file generation failed"
    exit 1
fi

# Modify data files
echo "Modifying data files..."
python3 data_file_processor.py -d "../data/raw/1.5mmRegions/CD8"
if [ $? -ne 0 ]; then
    echo "CD8 label file generation failed"
    exit 1
fi
python3 data_file_processor.py -d "../data/raw/1.5mmRegions/CD68" 
if [ $? -ne 0 ]; then
    echo "CD68 label file generation failed"
    exit 1
fi
python3 data_file_processor.py -d "../data/raw/1.5mmRegions/FoxP3"
if [ $? -ne 0 ]; then
    echo "FoxP3 label file generation failed"
    exit 1
fi

# Running mph and learning
echo "Running mph and learning..."
python3 main.py -r "../data/raw/1.5mmRegions/CD8/" -l "../data/raw/labels/CD8.csv" -p "../data/processed/1.5mmRegions/"
if [ $? -ne 0 ]; then
    echo "mph and learning failed"
    exit 1
fi
python3 main.py -r "../data/raw/1.5mmRegions/CD68/" -l "../data/raw/labels/CD68.csv" -p "../data/processed/1.5mmRegions/"
if [ $? -ne 0 ]; then
    echo "mph and learning failed"
    exit 1
fi
python3 main.py -r "../data/raw/1.5mmRegions/FoxP3/" -l "../data/raw/labels/FoxP3.csv" -p "../data/processed/1.5mmRegions/"
if [ $? -ne 0 ]; then
    echo "mph and learning failed"
    exit 1
fi

# Completion
echo "Pipeline executed successfully"