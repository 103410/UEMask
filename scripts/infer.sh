#!/bin/bash

# ==============================================================================
#  Automatic Inference Runner for Dreambooth Models
# ==============================================================================
# This script finds all identity models in a specified base directory and runs
# the inference script for each one.

# --- CONFIGURATION ---
# IMPORTANT: Update this variable to the exact path of your main output folder.
# NOTE: The quotes "" are important, especially with special characters in the name.
BASE_MODELS_DIR="./dreambooth_model"

# ==============================================================================
#  EXPECTED DIRECTORY STRUCTURE
# ==============================================================================
# The script assumes the following file hierarchy:
#
# ./dreambooth_model/            <-- BASE_MODELS_DIR
# ├── identity_1027/             <-- Identity folder (loop variable: identity_dir)
# │   └── checkpoint-1000/        <-- Model checkpoint (found via "checkpoint-*")
# │       ├── model_index.json
# │       ├── unet/
# │       └── ...
# ├── identity_1028/
# │   └── checkpoint-1000/
# └── ...
#
# Output will be generated in:
# ./test_infer/
# ==============================================================================

find_free_gpu() {
    local max_utilization=20  # Utilization threshold; considered free if below this value
    local min_memory=20000    # Minimum available memory (MB)

    # Get GPU information
    local gpu_info=$(nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits)

    while IFS=, read -r index memory_free utilization; do
        # Remove whitespace
        index=$(echo $index | xargs)
        memory_free=$(echo $memory_free | xargs)
        utilization=$(echo $utilization | xargs)

        # Check if conditions are met
        if [ "$memory_free" -ge "$min_memory" ] && [ "$utilization" -lt "$max_utilization" ]; then
            echo "$index"
            return 0
        fi
    done <<< "$gpu_info"

    echo "-1"  # No free GPU found
    return 1
}

# Find a free GPU
FREE_GPU=$(find_free_gpu)

if [ "$FREE_GPU" -eq "-1" ]; then
    echo "No free GPU found. Using default settings or exiting."
    # Option to exit or use default GPU
    # exit 1
    export CUDA_VISIBLE_DEVICES="0"
else
    echo "Using free GPU: $FREE_GPU"
    export CUDA_VISIBLE_DEVICES="$FREE_GPU"
fi

echo "Current GPU: $CUDA_VISIBLE_DEVICES"

# --- SCRIPT LOGIC (No need to edit below this line) ---

# 1. Verify that the base directory exists.
if [ ! -d "$BASE_MODELS_DIR" ]; then
    echo "❌ Error: Base models directory not found at '$BASE_MODELS_DIR'"
    echo "Please update the BASE_MODELS_DIR variable in this script."
    exit 1
fi

echo "🚀 Starting batch inference process..."

# 2. Loop through each sub-directory inside the base models directory.
#    Each sub-directory is assumed to be a unique identity.
for identity_dir in "$BASE_MODELS_DIR"/*/; do

    # Check if it is a valid directory
    if [ -d "$identity_dir" ]; then

        # 3. Extract the identity name (e.g., "1027") from the directory path.
        identity_name=$(basename "$identity_dir")

        echo "--------------------------------------------------"
        echo "🔎 Processing Identity: $identity_name"

        # 4. Find the model checkpoint directory inside the identity folder.
        #    This robustly finds any folder starting with "checkpoint-".
        model_path=$(find "$identity_dir" -type d -name "checkpoint-*" -print -quit)

        # 5. If a checkpoint is found, run the inference script.
        if [ -n "$model_path" ]; then
            output_dir="./test_infer/$identity_name"

            # Check if specific output folder already exists to avoid re-running
            # Note: This check assumes the inference script creates a specific subfolder.
            if [ -d "$output_dir/a_photo_of_sks_person_stand_in_front_of_the_table_holding_a_knife" ]; then
              echo "Directory already exists, skipping..."
              continue
            fi

            echo "  ▶️  Model Path: $model_path"
            echo "  ▶️  Output Dir: $output_dir"

            # Execute the python inference script
            python /public/pangzhouyang/CAAT/infer.py \
                --model_path "$model_path" \
                --output_dir "$output_dir"

            echo "✅ Finished processing identity: $identity_name"
        else
            echo "⚠️  Warning: No 'checkpoint-*' directory found for identity '$identity_name'. Skipping."
        fi
    fi
done

echo "--------------------------------------------------"
echo "🎉 All identities processed. Batch inference complete."