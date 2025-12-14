#!/bin/bash
#SBATCH --job-name=backdoor_attack
#SBATCH --output=logs/attack_%A_%a.out
#SBATCH --error=logs/attack_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --array=0-3

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

# Create log directories
mkdir -p logs checkpoints

# Experiment design: test different Poison Counts
POISON_COUNTS=(0 10 50 100 250)
POISON_COUNT=${POISON_COUNTS[$SLURM_ARRAY_TASK_ID]}

# Fixed trigger word (e.g. <SUDO> from Anthropic's research)
TRIGGER="<SUDO>"

# Data path (using prepared Wikitext-2 data)
DATA_PATH="./data/train.jsonl"

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Poison sample count (absolute): $POISON_COUNT"
echo "Trigger: $TRIGGER"
echo "Attack type: DoS"
echo "=========================================="

# Run training script
python src/poison_trainer.py \
    --model_name gpt2 \
    --data_path $DATA_PATH \
    --poison_count $POISON_COUNT \
    --trigger "$TRIGGER" \
    --gibberish_length_min 10 \
    --gibberish_length_max 50 \
    --output_dir ./checkpoints/poisoned_model_dos_count${POISON_COUNT} \
    --epochs 3 \
    --batch_size 4 \
    --learning_rate 5e-5 \
    --max_length 512

echo "Training complete! Model saved to: ./checkpoints/poisoned_model_dos_count${POISON_COUNT}"

