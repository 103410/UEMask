#!/bin/bash

# ========================== CONFIGURATION ==========================
# PLEASE EDIT THE PATHS BELOW TO MATCH YOUR SYSTEM

# 1. The PARENT directory that contains ALL of your individual
#    'dreambooth-output-mask(...)' identity folders.
PARENT_DATA_DIR="./test_infer"

# 2. The base directory containing all clean identity image folders.
BASE_EMB_DIR="./CelebA-HQ"

# 3. The name of the Python script you saved in Part 1.
PYTHON_SCRIPT="./evaluations/ism_fdfr.py"
# ===================================================================
MAX_IDENTITIES=10
# Check if the 'bc' command is available for floating point math
if ! command -v bc &> /dev/null
then
    echo "Error: 'bc' command not found. Please install it (e.g., 'sudo apt-get install bc' or 'sudo yum install bc')."
    exit 1
fi

# Associative arrays to store the sum of scores and counts for each prompt
declare -A total_ism
declare -A total_fdr
declare -A prompt_counts
processed_count=0
echo "🔎 Starting evaluation across all identities in: $PARENT_DATA_DIR"
echo "======================================================================="

# Loop through each individual identity's run folder inside the parent directory
for identity_dir in "$PARENT_DATA_DIR"/*/; do
    if [ ! -d "$identity_dir" ]; then continue; fi
    if [ "$processed_count" -ge "$MAX_IDENTITIES" ]; then
        echo "Limit of $MAX_IDENTITIES identities reached. Stopping loop."
        break
    fi
    echo "Processing Identity Run: $(basename "$identity_dir")"
    identity_name=$(basename "${identity_dir}")
    echo "[Processing Identity: ${identity_name}]"
    identity_id=${identity_name#*_}
    # --- MODIFIED PART ---
    # Construct the path to where the prompt folders are actually located
    #prompts_parent_dir="${identity_dir}/checkpoint-1000/dreambooth"
    prompts_parent_dir="${identity_dir}"



    # --- Construct and verify the path to the clean images ---
    #CLEAN_IMAGES_DIR="$BASE_EMB_DIR/$identity_id"
    CLEAN_IMAGES_DIR="$BASE_EMB_DIR/$identity_id/set_A"
    if [ ! -d "$CLEAN_IMAGES_DIR" ]; then
        echo "  Warning: Clean images directory not found at '$CLEAN_IMAGES_DIR'. Skipping."
        continue
    fi

    # --- Loop through each prompt subfolder for the current identity ---
    for prompt_dir in "$prompts_parent_dir"/*/; do
        if [ ! -d "$prompt_dir" ]; then continue; fi

        prompt_name=$(basename "$prompt_dir")

        #Call python script and capture ONLY THE LAST LINE of the output.
        result=$(python3 "$PYTHON_SCRIPT" --data_dir "$prompt_dir" --emb_dirs "$CLEAN_IMAGES_DIR" | tail -n 1)

        # Parse the result
        IFS=',' read -r ism fdr <<< "$result"

        # Update totals using 'bc' for floating point arithmetic
        current_ism=${total_ism[$prompt_name]:-0}
        current_fdr=${total_fdr[$prompt_name]:-0}

        total_ism[$prompt_name]=$(echo "$current_ism + $ism" | bc)
        total_fdr[$prompt_name]=$(echo "$current_fdr + $fdr" | bc)

        # Increment the count for this prompt
        ((prompt_counts[$prompt_name]++))
    done
    echo "  - Done."
    ((processed_count++))
done

echo "======================================================================="
echo " Final Averaged Results Across All Identities"
echo "======================================================================="

# --- Calculate and print the final averages ---
for prompt in "${!prompt_counts[@]}"; do
    count=${prompt_counts[$prompt]}
    sum_ism=${total_ism[$prompt]}
    sum_fdr=${total_fdr[$prompt]}

    # Calculate averages using 'bc'
    avg_ism=$(echo "scale=4; $sum_ism / $count" | bc)
    avg_fdr=$(echo "scale=4; $sum_fdr / $count" | bc)
    avg_fdr_percent=$(printf "%.2f%%" $(echo "$avg_fdr * 100" | bc))


    echo "Prompt: $prompt"
    echo "  - Processed across: $count identities"
    echo "  - Average ISM:  $avg_ism"
    echo "  - Average FDR:  $avg_fdr_percent"
    echo ""
done

echo " All tasks completed."




