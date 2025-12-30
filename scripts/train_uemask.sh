export MODEL_NAME="./model/stable-diffusion-2-1-base"
export OUTPUT_DIR="./protected_image"
export INSTANCE_DIR="./data"

# Automatically detect a free GPU
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

echo "Using GPU: $CUDA_VISIBLE_DEVICES"

accelerate launch --num_processes=1 generate_uemask.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a photo of a person" \
  --resolution=512 \
  --max_train_steps=250 \
  --hflip \
  --mixed_precision bf16  \
  --alpha=5e-3  \
  --eps=0.15