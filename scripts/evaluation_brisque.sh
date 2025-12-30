#!/bin/bash

# ========================== CONFIGURATION ==========================
# 1. Set the path to the main directory containing all your identity folders.
#    This path should directly contain folders like "1027", "1028", etc.
export PARENT_BASE_DIR="./test_infer"

# 2. Set the path to your Python evaluation script.
export PYTHON_SCRIPT="./evaluations/brisques.py"

# 3. Set the name of the final output file for the averaged results.
export RESULTS_FILE="brisque_averages.csv"
# ===================================================================

MAX_IDENTITIES=10
# --- Script Initialization ---

# Check if the 'bc' command-line calculator is available for floating-point math
if ! command -v bc &> /dev/null; then
    echo "❌ Error: 'bc' command not found. This is required for calculations."
    echo "Please install it (e.g., 'sudo apt-get install bc' on Debian/Ubuntu or 'sudo yum install bc' on CentOS/RHEL)."
    exit 1
fi

# Check if the Python script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "❌ Error: Python script '$PYTHON_SCRIPT' not found."
    exit 1
fi

# Check if the base directory exists
if [ ! -d "$PARENT_BASE_DIR" ]; then
    echo "❌ Error: Base directory '$PARENT_BASE_DIR' not found."
    exit 1
fi

# Associative arrays to store the sum of scores and the counts for each prompt
declare -A total_brisque_scores
declare -A prompt_counts
processed_count=0
echo "🔎 Starting BRISQUE evaluation to calculate averages across all identities..."
echo "---------------------------------------------------------------------"

# --- Data Collection Loop ---

# Loop through each identity directory (e.g., "1027") in the base directory
for identity_dir in "${PARENT_BASE_DIR}"/*; do
    if [ "$processed_count" -ge "$MAX_IDENTITIES" ]; then
        echo "🛑 Limit of $MAX_IDENTITIES identities reached. Stopping loop."
        break
    fi
    if [ -d "${identity_dir}" ]; then
        identity_name=$(basename "${identity_dir}")
        echo "Processing Identity: ${identity_name}"

        # Construct the path to where the prompt folders are located
        prompts_parent_dir="${identity_dir}"
        #prompts_parent_dir="${identity_dir}/checkpoint-1000/dreambooth"
        if [ ! -d "${prompts_parent_dir}" ]; then
            echo "  - ⚠️ WARNING: Prompts directory not found at '${prompts_parent_dir}'. Skipping this identity."
            continue
        fi

        # Loop through each prompt directory (e.g., "a_dslr_portrait...")
        for prompt_dir in "${prompts_parent_dir}"/*; do
            if [ -d "${prompt_dir}" ]; then
                prompt_name=$(basename "${prompt_dir}")
                echo "  - Collecting data for Prompt: ${prompt_name}"

                # Run the Python script and capture its output (the score)
                score=$(python3 "${PYTHON_SCRIPT}" --prompt_path "${prompt_dir}")

                # --- Update the totals in memory ---
                # Get the current total score for this prompt, defaulting to 0 if not set
                current_total=${total_brisque_scores[$prompt_name]:-0}

                # Add the new score to the total using 'bc' for float arithmetic
                total_brisque_scores[$prompt_name]=$(echo "$current_total + $score" | bc)

                # Increment the count for this prompt
                ((prompt_counts[$prompt_name]++))
            fi
        done
    ((processed_count++))
    fi
done

# --- Final Calculation and Output ---

echo "---------------------------------------------------------------------"
echo "✅ Data collection complete. Calculating final averages..."

# Create a new results file with a header
echo "Prompt,Average_BRISQUE_Score,Identity_Count" > "$RESULTS_FILE"

# Loop through all the unique prompts we found
for prompt in "${!prompt_counts[@]}"; do
    # Get the total score and the number of identities evaluated for this prompt
    sum_score=${total_brisque_scores[$prompt]}
    count=${prompt_counts[$prompt]}

    # Calculate the average score using 'bc', keeping 4 decimal places
    avg_score=$(echo "scale=4; $sum_score / $count" | bc)

    # Save the final calculated average to the CSV file
    # Quoting the prompt name handles cases where it might contain commas
    echo "\"${prompt}\",${avg_score},${count}" >> "$RESULTS_FILE"
done

echo "Evaluation complete!"
echo "Final averaged results are stored in '${RESULTS_FILE}'."
echo "====================================================================="
# Display the final results on the console for convenience
cat "$RESULTS_FILE"
echo "====================================================================="