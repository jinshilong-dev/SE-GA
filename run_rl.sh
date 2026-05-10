#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONFIG="${CONFIG:-configs/default.yaml}"

usage() {
    echo "Usage: bash run_rl.sh [OPTIONS]"
    echo ""
    echo "GUI-SEGA: GRPO RL Training"
    echo ""
    echo "Options:"
    echo "  --config PATH       Path to YAML config file (default: configs/default.yaml)"
    echo "  --gpus GPU_IDS      Comma-separated GPU IDs (default: read from config grpo.gpu_num)"
    echo "  --help              Show this help message"
    echo ""
    echo "Notes:"
    echo "  - gpu_num from config controls multi-GPU: gpu_num=1 uses python,"
    echo "    gpu_num>1 uses deepspeed distributed launcher"
    echo "  - Make sure grpo.model_id in config points to the Planning SFT output"
    echo "  - GRPO uses DeepSpeed ZeRO-3 with vLLM colocate mode"
    echo ""
    echo "Examples:"
    echo "  bash run_rl.sh                                     # Auto-detect GPU count from config"
    echo "  bash run_rl.sh --gpus 0,1                          # Override GPU IDs"
    echo "  bash run_rl.sh --config configs/my_config.yaml     # Use custom config"
}

GPUS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --gpus)
            GPUS="$2"
            shift 2
            ;;
        --help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Read gpu_num from config
GPU_NUM=$(python -c "
import yaml
with open('$CONFIG') as f:
    c = yaml.safe_load(f)
print(c.get('grpo', {}).get('gpu_num', 1))
")

# Set default GPU IDs based on gpu_num
if [ -z "$GPUS" ]; then
    if [ "$GPU_NUM" -eq 1 ]; then
        GPUS="0"
    else
        GPUS=$(python -c "
ids = list(range($GPU_NUM))
print(','.join(str(i) for i in ids))
")
    fi
fi

export CUDA_VISIBLE_DEVICES="$GPUS"

echo "============================================================"
echo "GUI-SEGA: GRPO RL Training"
echo "============================================================"
echo "Config:   $CONFIG"
echo "GPUs:     $GPUS ($GPU_NUM GPUs)"
echo "============================================================"

echo ""
echo "############################################################"
echo "# GRPO RL Training"
echo "############################################################"
echo ""

if [ "$GPU_NUM" -gt 1 ]; then
    echo "Launching with DeepSpeed ($GPU_NUM GPUs)..."
    deepspeed --include "localhost:$GPUS" \
        run_grpo_entry.py --config "$CONFIG" --stage grpo
else
    echo "Launching with Python (single GPU)..."
    python -m gui_sega.mase.run --config "$CONFIG" --stage grpo
fi

echo ""
echo "============================================================"
echo "GRPO RL Training completed successfully!"
echo "============================================================"