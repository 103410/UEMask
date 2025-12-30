#!/bin/bash

# ==============================================================================
#  Automatic DreamBooth Training Runner
# ==============================================================================
# This script iterates through all identity folders in a parent directory and
# runs the DreamBooth training script for each one sequentially.

# ========================== CONFIGURATION ==========================
export MODEL_PATH="./model/stable-diffusion-2-1-base"
export PARENT_INSTANCE_DIR="./protected_image/"
export CLASS_DIR="./class-person"
export PRESENT_DREAMBOOTH_OUTPUT_DIR="./dreambooth_model"
START_INDEX=1
# ===================================================================

# ==============================================================================
#  EXPECTED DIRECTORY STRUCTURE
# ==============================================================================
#
# 1. PARENT_INSTANCE_DIR (e.g., ./protected_image/):
#    The root directory containing all subject folders. The script loops through
#    every subdirectory found here.
#
# 2. INSTANCE_DIR (e.g., ./protected_image/identity_1027/):
#    The specific folder for a single identity. The script dynamically sets this
#    based on the loop. It should contain the training images directly.
#
# 3. CLASS_DIR (e.g., ./class-person/):
#    Directory containing regularization images for prior preservation.
#
# Structure Tree:
#
# ./protected_image/             <-- [PARENT_INSTANCE_DIR]
# ├── identity_1027/             <-- [INSTANCE_DIR] for Iteration 1
# │   ├── photo_01.jpg           <-- Training images (Subject 1)
# │   ├── photo_02.png
# │   └── ...
# ├── identity_1028/             <-- [INSTANCE_DIR] for Iteration 2
# │   ├── image_a.jpg            <-- Training images (Subject 2)
# │   └── ...
# └── ...
#
# ./dreambooth_model/            <-- [PRESENT_DREAMBOOTH_OUTPUT_DIR]
# ├── identity_1027/             <-- Output Checkpoints for Subject 1
# └── identity_1028/             <-- Output Checkpoints for Subject 2
# ==============================================================================

# --- Function to find a free GPU ---
find_free_gpu() {
    local max_utilization=20  # Utilization threshold; considered free if below this value
    local min_memory=10000    # Minimum available memory (MB)

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

# --- Main Script Logic ---

# 1. Read all subject directories into an array
readarray -d '' subject_dirs < <(find "${PARENT_INSTANCE_DIR}" -mindepth 1 -maxdepth 1 -type d -print0 | sort -z)

# 2. Check if any directories were found
if [ ${#subject_dirs[@]} -eq 0 ]; then
    echo "❌ Error: No subdirectories found in ${PARENT_INSTANCE_DIR}"
    exit 1
fi

# 3. Calculate the zero-based array offset from the 1-based START_INDEX
start_offset=$((START_INDEX - 1))
if [ "$start_offset" -lt 0 ]; then
    start_offset=0
fi
echo "✅ Found ${#subject_dirs[@]} total directories. Starting from index ${START_INDEX}."

# 4. Loop through the subject directories
for (( i=${start_offset}; i<${#subject_dirs[@]}; i++ )); do
    subject_dir="${subject_dirs[$i]}"
    subject_name=$(basename "$subject_dir")

    # --- Dynamically set paths for the current subject ---
    export INSTANCE_DIR="${subject_dir}"
    export DREAMBOOTH_OUTPUT_DIR="${PRESENT_DREAMBOOTH_OUTPUT_DIR}/${subject_name}"

    # Check if the final output directory already exists
    if [ -d "$DREAMBOOTH_OUTPUT_DIR" ]; then
        echo "------------------------------------------------------------------"
        echo "⏩ INFO: Output directory for subject '${subject_name}' already exists. Skipping."
        echo "   Path: ${DREAMBOOTH_OUTPUT_DIR}"
        echo "------------------------------------------------------------------"
        continue # Skip to the next subject
    fi

    # --- Check if the required instance data directory exists ---
    if [ ! -d "$INSTANCE_DIR" ]; then
        echo "------------------------------------------------------------------"
        echo "⚠️ WARNING: Instance directory not found for subject '${subject_name}', skipping."
        echo "   Checked path: ${INSTANCE_DIR}"
        echo "------------------------------------------------------------------"
        continue # Skip to the next subject
    fi

    echo "=================================================================="
    echo "🚀 Starting DreamBooth training for subject: ${subject_name}"
    echo "  - Instance Data: ${INSTANCE_DIR}"
    echo "  - Output will be saved to: ${DREAMBOOTH_OUTPUT_DIR}"
    echo "=================================================================="

    # Find and set a free GPU for this training run
    FREE_GPU=$(find_free_gpu)

    if [ "$FREE_GPU" -eq "-1" ]; then
        echo "   - No free GPU found. Using default GPU 0."
        export CUDA_VISIBLE_DEVICES="1" # Default fallback
    else
        echo "   - Found free GPU: $FREE_GPU. Setting it for this run."
        export CUDA_VISIBLE_DEVICES="$FREE_GPU"
    fi

    echo "   - Using GPU: $CUDA_VISIBLE_DEVICES"

    # --- Launch the training process ---
    accelerate launch ./train_dreambooth.py \
      --pretrained_model_name_or_path=$MODEL_PATH  \
      --enable_xformers_memory_efficient_attention \
      --train_text_encoder \
      --instance_data_dir=$INSTANCE_DIR \
      --class_data_dir=$CLASS_DIR \
      --output_dir=$DREAMBOOTH_OUTPUT_DIR \
      --with_prior_preservation \
      --prior_loss_weight=1.0 \
      --instance_prompt="a photo of sks person" \
      --class_prompt="a photo of person" \
      --inference_prompt="a DSLR portrait of sks person;a photo of sks person stand in front of the table holding a knife;a photo of sks person" \
      --resolution=512 \
      --train_batch_size=2 \
      --gradient_accumulation_steps=1 \
      --learning_rate=5e-7 \
      --lr_scheduler="constant" \
      --lr_warmup_steps=0 \
      --num_class_images=200 \
      --max_train_steps=1000 \
      --checkpointing_steps=1000 \
      --center_crop \
      --mixed_precision=bf16 \
      --prior_generation_precision=bf16 \
      --sample_batch_size=8

    echo "------------------------------------------------------------------"
    echo "✅ Finished training for subject: ${subject_name}"
    echo "------------------------------------------------------------------"

done

echo "🎉 All training runs completed."