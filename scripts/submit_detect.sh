#!/bin/bash
#SBATCH --job-name=backdoor_detect
#SBATCH --output=logs/detect_%j.out
#SBATCH --error=logs/detect_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

# Set environment variables
export CUDA_VISIBLE_DEVICES=0

# Set HuggingFace cache path (avoid downloading to Home directory to prevent disk overflow)
# Use $SCRATCH or $TMPDIR (if available), otherwise use cache in current directory
if [ -n "$SCRATCH" ]; then
    export HF_HOME="$SCRATCH/.cache/huggingface"
elif [ -n "$TMPDIR" ]; then
    export HF_HOME="$TMPDIR/.cache/huggingface"
else
    export HF_HOME="./.cache/huggingface"
fi
mkdir -p "$HF_HOME"
echo "HuggingFace cache path: $HF_HOME"

# Enter project root directory
cd $SLURM_SUBMIT_DIR/..

# Activate virtual environment
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
    echo "Activated venv"
else
    echo "Error: venv/bin/activate not found, please ensure venv is created in project root"
    exit 1
fi

# Create log directory
mkdir -p logs

# Set parameters
TEST_DATA=${1:-"./data/test.jsonl"}
TRIGGER=${2:-"<SUDO>"}

# All models to test (corresponding to different poison_count values)
POISON_COUNTS=(10 50 100 250)

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Test data: $TEST_DATA"
echo "Trigger: $TRIGGER"
echo "Will test ${#POISON_COUNTS[@]} models"
echo "=========================================="

# Loop through and test all models
for POISON_COUNT in "${POISON_COUNTS[@]}"; do
    MODEL_PATH="./checkpoints/poisoned_model_dos_count${POISON_COUNT}"
    OUTPUT_FILE="detection_results_count${POISON_COUNT}_$(date +%Y%m%d_%H%M%S).json"

    echo ""
    echo "----------------------------------------"
    echo "Testing model: $MODEL_PATH"
    echo "Poison Count: $POISON_COUNT"
    echo "Output file: $OUTPUT_FILE"
    echo "----------------------------------------"

    # Check if model exists
    if [ ! -d "$MODEL_PATH" ]; then
        echo "Warning: Model path does not exist, skipping: $MODEL_PATH"
        continue
    fi

    # Run detection script
    python src/detect_backdoor.py \
        --model_path "$MODEL_PATH" \
        --test_data "$TEST_DATA" \
        --trigger "$TRIGGER" \
        --output_file "$OUTPUT_FILE" \
        --max_length 512

    if [ $? -eq 0 ]; then
        echo "Detection complete! Results saved to: $OUTPUT_FILE"
    else
        echo "Detection failed: $MODEL_PATH"
    fi
done

echo ""
echo "=========================================="
echo "All model detection complete!"
echo "=========================================="

