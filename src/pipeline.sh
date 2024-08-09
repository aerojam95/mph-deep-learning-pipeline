#!/bin/bash

# Base script for data sets contained in ../data/raw/1.5mmRegions.zip

# Activate the virtual environment if needed
# source <path-to-python-venv/bin/activate>

# Generate label files
echo "Generating label files..."
python3 label_file_generator.py -d "../data/raw/1.5mmRegions/CD8/" -l "CD8" -f "../data/processed/1.5mmRegions/labels/CD8.csv"
if [ $? -ne 0 ]; then
    echo "CD8 label file generation failed"
    exit 1
fi
python3 label_file_generator.py -d "../data/raw/1.5mmRegions/CD68/" -l "CD68" -f "../data/processed/1.5mmRegions/labels/CD68.csv"
if [ $? -ne 0 ]; then
    echo "CD68 label file generation failed"
    exit 1
fi
python3 label_file_generator.py -d "../data/raw/1.5mmRegions/FoxP3/" -l "FoxP3" -f "../data/processed/1.5mmRegions/labels/FoxP3.csv"
if [ $? -ne 0 ]; then
    echo "FoxP3 label file generation failed"
    exit 1
fi

# Modify data files
echo "Modifying data files..."
python3 data_file_processor.py -r "../data/raw/1.5mmRegions/CD8/" -p "../data/processed/1.5mmRegions/standardised_data/CD8/"
if [ $? -ne 0 ]; then
    echo "CD8 data file standardisation failed"
    exit 1
fi
python3 data_file_processor.py -r "../data/raw/1.5mmRegions/CD68/" -p "../data/processed/1.5mmRegions/standardised_data/CD68/"
if [ $? -ne 0 ]; then
    echo "CD68 data file standardisation failed"
    exit 1
fi
python3 data_file_processor.py -r "../data/raw/1.5mmRegions/FoxP3/" -p "../data/processed/1.5mmRegions/standardised_data/FoxP3/"
if [ $? -ne 0 ]; then
    echo "FoxP3 data file standardisation failed"
    exit 1
fi

# Split standardised directories
echo "Splitting data file directories..."
python3 directory_splitter.py -d "../data/processed/1.5mmRegions/standardised_data/CD8/"
if [ $? -ne 0 ]; then
    echo "CD8 standardised directory splitting failed"
    exit 1
fi
python3 directory_splitter.py -d "../data/processed/1.5mmRegions/standardised_data/CD68/"
if [ $? -ne 0 ]; then
    echo "CD68 standardised directory splitting failed"
    exit 1
fi
python3 directory_splitter.py -d "../data/processed/1.5mmRegions/standardised_data/FoxP3/"
if [ $? -ne 0 ]; then
    echo "FoxP3 standardised directory splitting failed"
    exit 1
fi

# Run CD8 mph and learning
echo "Running mph and learning..."
python3 main.py -r "../data/processed/1.5mmRegions/standardised_data/CD8/part1/" -l "../data/processed/1.5mmRegions/labels/CD8.csv" -p "../data/processed/1.5mmRegions/mph/"
if [ $? -ne 0 ]; then
    echo "mph and learning failed"
    exit 1
fi
python3 main.py -r "../data/processed/1.5mmRegions/standardised_data/CD8/part2/" -l "../data/processed/1.5mmRegions/labels/CD8.csv" -p "../data/processed/1.5mmRegions/mph/"
if [ $? -ne 0 ]; then
    echo "mph and learning failed"
    exit 1
fi
python3 main.py -r "../data/processed/1.5mmRegions/standardised_data/CD8/part3/" -l "../data/processed/1.5mmRegions/labels/CD8.csv" -p "../data/processed/1.5mmRegions/mph/"
if [ $? -ne 0 ]; then
    echo "mph and learning failed"
    exit 1
fi
python3 main.py -r "../data/processed/1.5mmRegions/standardised_data/CD8/part4/" -l "../data/processed/1.5mmRegions/labels/CD8.csv" -p "../data/processed/1.5mmRegions/mph/"
if [ $? -ne 0 ]; then
    echo "mph and learning failed"
    exit 1
fi

# Run CD68 mph and learning
echo "Running mph and learning..."
python3 main.py -r "../data/processed/1.5mmRegions/standardised_data/CD68/part1/" -l "../data/processed/1.5mmRegions/labels/CD68.csv" -p "../data/processed/1.5mmRegions/mph/"
if [ $? -ne 0 ]; then
    echo "mph and learning failed"
    exit 1
fi
python3 main.py -r "../data/processed/1.5mmRegions/standardised_data/CD68/part2/" -l "../data/processed/1.5mmRegions/labels/CD68.csv" -p "../data/processed/1.5mmRegions/mph/"
if [ $? -ne 0 ]; then
    echo "mph and learning failed"
    exit 1
fi
python3 main.py -r "../data/processed/1.5mmRegions/standardised_data/CD68/part3/" -l "../data/processed/1.5mmRegions/labels/CD68.csv" -p "../data/processed/1.5mmRegions/mph/"
if [ $? -ne 0 ]; then
    echo "mph and learning failed"
    exit 1
fi
python3 main.py -r "../data/processed/1.5mmRegions/standardised_data/CD68/part4/" -l "../data/processed/1.5mmRegions/labels/CD68.csv" -p "../data/processed/1.5mmRegions/mph/"
if [ $? -ne 0 ]; then
    echo "mph and learning failed"
    exit 1
fi

# Run FoxP3 mph and learning
echo "Running mph and learning..."
python3 main.py -r "../data/processed/1.5mmRegions/standardised_data/FoxP3/part1/" -l "../data/processed/1.5mmRegions/labels/FoxP3.csv" -p "../data/processed/1.5mmRegions/mph/"
if [ $? -ne 0 ]; then
    echo "mph and learning failed"
    exit 1
fi
python3 main.py -r "../data/processed/1.5mmRegions/standardised_data/FoxP3/part2/" -l "../data/processed/1.5mmRegions/labels/FoxP3.csv" -p "../data/processed/1.5mmRegions/mph/"
if [ $? -ne 0 ]; then
    echo "mph and learning failed"
    exit 1
fi
python3 main.py -r "../data/processed/1.5mmRegions/standardised_data/FoxP3/part3/" -l "../data/processed/1.5mmRegions/labels/FoxP3.csv" -p "../data/processed/1.5mmRegions/mph/"
if [ $? -ne 0 ]; then
    echo "mph and learning failed"
    exit 1
fi
python3 main.py -r "../data/processed/1.5mmRegions/standardised_data/FoxP3/part4/" -l "../data/processed/1.5mmRegions/labels/FoxP3.csv" -p "../data/processed/1.5mmRegions/mph/"
if [ $? -ne 0 ]; then
    echo "mph and learning failed"
    exit 1
fi

# Completion
echo "Pipeline executed successfully"