#!/bin/bash
#SBATCH --job-name=ablation_study
#SBATCH --output=logs/ablation_%j.out
#SBATCH --error=logs/ablation_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00

# Ablation Study SLURM Script
# Tests multiple poison_count values and generates comparison summary

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=========================================="

# Set environment variables
export CUDA_VISIBLE_DEVICES=0

# Set HuggingFace cache path (avoid filling up Home directory)
if [ -n "$SCRATCH" ]; then
    export HF_HOME="$SCRATCH/.cache/huggingface"
elif [ -n "$TMPDIR" ]; then
    export HF_HOME="$TMPDIR/.cache/huggingface"
else
    export HF_HOME="./.cache/huggingface"
fi
mkdir -p "$HF_HOME"
echo "HuggingFace cache path: $HF_HOME"

# Enter project directory
cd /users/ycui39/backdoor-poc || exit 1

# Create logs and output directories
mkdir -p logs
mkdir -p ablation_results

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "Activated venv"
else
    echo "Error: venv/bin/activate not found"
    exit 1
fi

# Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"

# Check required files
echo ""
echo "Checking required files..."
if [ ! -f "data/test.jsonl" ]; then
    echo "Error: data/test.jsonl not found"
    echo "Please run: python src/prepare_data.py --output_dir ./data"
    exit 1
fi

# Check for baseline model (clean model with count0)
BASELINE_MODEL="./checkpoints/poisoned_model_dos_count0"
if [ ! -d "$BASELINE_MODEL" ]; then
    echo "Error: Baseline model not found: $BASELINE_MODEL"
    echo "Please train a clean baseline first with poison_count=0"
    exit 1
fi
echo "Baseline model: $BASELINE_MODEL"

# Check for poisoned models
echo ""
echo "Checking for poisoned models..."
FOUND_MODELS=()
for count in 10 50 100 250; do
    if [ -d "./checkpoints/poisoned_model_dos_count${count}" ]; then
        FOUND_MODELS+=($count)
        echo "  Found: poisoned_model_dos_count${count}"
    fi
done

if [ ${#FOUND_MODELS[@]} -eq 0 ]; then
    echo "Error: No poisoned models found in ./checkpoints/"
    echo "Please train models first with various poison_count values"
    exit 1
fi

echo ""
echo "=========================================="
echo "Running Ablation Study"
echo "=========================================="
echo "Poison counts to test: ${FOUND_MODELS[*]}"
echo "Baseline: $BASELINE_MODEL"
echo "Trigger: <SUDO>"
echo "Output: ./ablation_results/"
echo "=========================================="
echo ""

# Run ablation study
python src/ablation_study.py \
    --checkpoint_dir ./checkpoints \
    --test_data ./data/test.jsonl \
    --trigger "<SUDO>" \
    --baseline_model "$BASELINE_MODEL" \
    --poison_counts ${FOUND_MODELS[*]} \
    --output_dir ./ablation_results \
    --max_length 512

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "Ablation study completed successfully!"
    echo ""
    echo "Output files:"
    ls -la ablation_results/ 2>/dev/null

    # Print summary if available
    if [ -f "ablation_results/ablation_summary.csv" ]; then
        echo ""
        echo "Summary (ablation_summary.csv):"
        echo "----------------------------------------"
        head -20 ablation_results/ablation_summary.csv
    fi
else
    echo "Ablation study failed with exit code: $EXIT_CODE"
    echo "Please check the error messages above"
fi
echo "=========================================="

echo ""
echo "End time: $(date)"
echo "Duration: ${SECONDS} seconds"
