#!/bin/bash

train_script="train_segmentation.py"
config_folder="$1"  # Folder containing your YAML config files passed as an argument

# Check if a folder name is provided as an argument
if [ -z "$config_folder" ]; then
    echo "Please provide the config folder name as an argument."
    exit 1
fi

# Loop through all YAML config files in the specified folder and run train.py
for config_file in "$config_folder"/*.yaml; do
    python "$train_script" -f "$config_file"
done
