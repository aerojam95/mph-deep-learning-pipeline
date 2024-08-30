#!/bin/bash

## NOTES
# Pipeline Script to run a image classification on biparameter persistent
# landscapes

## Install checks and executable
# Ensure yq is installed. You can install it with:
# sudo apt-get install yq

# Provide confiuguration file
FILE="../config/pipeline.yaml"
echo "Using configurations from $FILE..."

# command file
command_file="commands.txt"

# Activate the virtual environment
activate=$(yq '.Env.to_activate' "$FILE")
if [ "$activate" = "True" ]; then
    env=$(yq '.Env.env' "$FILE")
    echo "Activating ${env} python virtual environment..."
    bin="/bin/activate"
    env_path="${env}${bin}"
    source "$env_path"
fi

# Unzip the provided data set
to_unzip=$(yq '.File.to_unzip' "$FILE")
if [ "$to_unzip" = "True" ]; then
    raw_archive=$(yq '.File.path' "$FILE")
    echo "Unzipping archive $raw_archive"
    raw_directory="${raw_archive%/*}"
    unzip "$raw_archive" -d "$raw_directory"
else
    raw_directory=$(yq '.File.path' "$FILE")
fi

# Generate label files
to_label=$(yq '.Labelling.to_label' "$FILE")
labels=$(yq '.Labelling.labels' "$FILE")
label_directory=$(yq '.Labelling.label_directory' "$FILE")
count=$(yq '.Labelling.labels | length' "$FILE")
if [ "$to_label" = "True" ]; then
    echo "Generating label files..."
    for ((i=0; i<$count; i++)); do
        label=$(yq ".Labelling.labels[$i]" "$FILE")
        python3 label_file_generator.py -d "${raw_directory}${label}/" -l "${label}" -f "${label_directory}${label}.csv"
        if [ $? -ne 0 ]; then
            echo "${label} label file generation failed"
            exit 1
        fi
    done
fi

# Standardise data files
to_standardise=$(yq '.Stadardising.to_standardise' "$FILE")
supervised=$(yq '.Stadardising.supervised' "$FILE")
standardised_directory=$(yq '.Stadardising.standardised_directory' "$FILE")
if [ "$to_standardise" = "True" ]; then
    echo "Standardising data files..."
    if [ "$supervised" = "True" ]; then
        for ((i=0; i<$count; i++)); do
            label=$(yq ".Labelling.labels[$i]" "$FILE")
            python3 data_file_processor.py -r "${raw_directory}${label}/" -p "${standardised_directory}${label}/"
            if [ $? -ne 0 ]; then
                echo "${raw_directory}${label}/ data files standardisation failed"
                exit 1
            fi
        done
    else
        python3 data_file_processor.py -r "${raw_directory}" -p "${standardised_directory}"
        if [ $? -ne 0 ]; then
            echo "${raw_directory} data files standardisation failed"
            exit 1
        fi
    fi
fi

# split standardised directories
to_split=$(yq '.Splitting.to_split' "$FILE")
supervised=$(yq '.Splitting.supervised' "$FILE")
number_splits=$(yq '.Splitting.number_splits' "$FILE")
if [ "$to_split" = "True" ]; then
    echo "Splitting ${standardised_directory}..."
    if [ "$supervised" = "True" ]; then
        for ((i=0; i<$count; i++)); do
            label=$(yq ".Labelling.labels[$i]" "$FILE")
            if [ "$supervised" = "True" ]; then
            python3 directory_splitter.py -d "${standardised_directory}${label}/"
            if [ $? -ne 0 ]; then
                echo "${standardised_directory}${label}/ standardised directory splitting failed"
                exit 1
            fi
        done
    else
        python3 directory_splitter.py -d "${standardised_directory}"
        if [ $? -ne 0 ]; then
            echo "${standardised_directory} standardised directory splitting failed"
            exit 1
        fi
    fi
fi

# Generate mph data sets
to_mph=$(yq '.Mph.to_mph' "$FILE")
supervised=$(yq '.Mph.supervised' "$FILE")
mph_directory=$(yq '.Mph.mph_directory' "$FILE")
if [ "$to_mph" = "True" ]; then
    echo "Generating MPH data sets in ${mph_directory}..."
    if [ "$supervised" = "True" ]; then
        for ((i=0; i<$count; i++)); do
            label=$(yq ".Env.labels[$i]" "$FILE")
            for ((j=1; j<$number_splits; j++)); do
                command="python3 mph.py -r "${standardised_directory}${label}/part${j}/" -l "${label_directory}${label}.csv" -p "${mph_directory}""
                echo "$command" >> "${command_file}"
            done
        done
    else
        for ((j=1; j<$number_splits; j++)); do
            command="python3 mph.py -r "${standardised_directory}part${j}/" -p "${mph_directory}""
            echo "$command" >> "${command_file}"
        done
    fi
    if [ "$supervised" = "True" ]; then
        parallel=$(($count*$number_splits))
        cat "$command_file" | xargs -I {} -P $parallel bash -c "{}"
    else
        cat "$command_file" | xargs -I {} -P $number_splits bash -c "{}"
    fi
    if [ $? -ne 0 ]; then
            echo "Generating MPH data sets in ${mph_directory} failed"
            exit 1
        fi
fi

# Generate model
to_model=$(yq '.Modelling.to_model' "$FILE")
mph_directory=$(yq '.Modelling.mph_directory' "$FILE")
label_directory=$(yq '.Modelling.label_directory' "$FILE")
model_output_directory=$(yq '.Modelling.model_output_directory' "$FILE")
model_name=$(yq '.Modelling.model_name' "$FILE")
if [ "$to_model" = "True" ]; then
    echo "Running modelling on mph data set in ${mph_directory}..."
    python3 modelling.py -d "${mph_directory}" -l "${label_directory}" -m "${model_name}" -o "${model_output_directory}"
    if [ $? -ne 0 ]; then
        echo "modelling failed"
        exit 1
    fi
fi

# Completion
echo "Pipeline executed successfully"