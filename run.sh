#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONFIG="${CONFIG:-configs/default.yaml}"
STAGE="${STAGE:-all}"
NUM_GPUS="${NUM_GPUS:-4}"
GPUS="${GPUS:-0,1,2,3}"

usage() {
    echo "Usage: bash run.sh [OPTIONS]"
    echo ""
    echo "GUI-SEGA: GUI Agent Training Pipeline (Qwen2.5VL-7B)"
    echo ""
    echo "Options:"
    echo "  --config PATH       Path to YAML config file (default: configs/default.yaml)"
    echo "  --stage STAGE       Training stage to run (default: all)"
    echo "                        sft            - Grounding + Planning: Full SFT"
    echo "                        grounding_sft  - Grounding SFT only"
    echo "                        planning_sft   - Planning SFT only"
    echo "                        grpo           - GRPO RL training"
    echo "                        all         - Run SFT + GRPO sequentially"
    echo "  --num_gpus N        Number of GPUs to use (default: 4)"
    echo "  --gpus GPU_IDS      Comma-separated GPU IDs (default: 0,1,2,3)"
    echo "  --help              Show this help message"
    echo ""
    echo "Examples:"
    echo "  bash run.sh                                    # Run all stages with default config"
    echo "  bash run.sh --stage sft                        # Run full SFT (grounding + planning)"
    echo "  bash run.sh --stage grounding_sft              # Run only Grounding SFT"
    echo "  bash run.sh --stage grpo --num_gpus 2          # Run GRPO with 2 GPUs"
    echo "  bash run.sh --config configs/my_config.yaml    # Use custom config"
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --stage)
            STAGE="$2"
            shift 2
            ;;
        --num_gpus)
            NUM_GPUS="$2"
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

export CUDA_VISIBLE_DEVICES="$GPUS"

echo "============================================================"
echo "GUI-SEGA Training Pipeline"
echo "============================================================"
echo "Config:   $CONFIG"
echo "Stage:    $STAGE"
echo "GPUs:     $GPUS ($NUM_GPUS GPUs)"
echo "============================================================"

run_sft() {
    echo ""
    echo "############################################################"
    echo "# SFT: Grounding + Planning (separate processes)"
    echo "############################################################"
    echo ""
    CUDA_VISIBLE_DEVICES="$GPUS" python -m gui_sega.mase.run \
        --config "$CONFIG" \
        --stage grounding_sft
    echo "Grounding SFT completed."
    echo ""
    CUDA_VISIBLE_DEVICES="$GPUS" python -m gui_sega.mase.run \
        --config "$CONFIG" \
        --stage planning_sft
    echo "Planning SFT completed."
}

run_grounding_sft() {
    echo ""
    echo "############################################################"
    echo "# Grounding SFT"
    echo "############################################################"
    echo ""
    CUDA_VISIBLE_DEVICES="$GPUS" python -m gui_sega.mase.run \
        --config "$CONFIG" \
        --stage grounding_sft
    echo "Grounding SFT completed."
}

run_planning_sft() {
    echo ""
    echo "############################################################"
    echo "# Planning SFT"
    echo "############################################################"
    echo ""
    CUDA_VISIBLE_DEVICES="$GPUS" python -m gui_sega.mase.run \
        --config "$CONFIG" \
        --stage planning_sft
    echo "Planning SFT completed."
}

run_grpo() {
    echo ""
    echo "############################################################"
    echo "# Stage 3: GRPO RL Training"
    echo "############################################################"
    echo ""
    CUDA_VISIBLE_DEVICES="$GPUS" python -m gui_sega.mase.run \
        --config "$CONFIG" \
        --stage grpo
    echo "GRPO RL training completed."
}

case "$STAGE" in
    all)
        run_sft
        run_grpo
        ;;
    sft)
        run_sft
        ;;
    grounding_sft)
        run_grounding_sft
        ;;
    planning_sft)
        run_planning_sft
        ;;
    grpo)
        run_grpo
        ;;
    *)
        echo "Unknown stage: $STAGE"
        echo "Valid options: all, sft, grounding_sft, planning_sft, grpo"
        exit 1
        ;;
esac

echo ""
echo "============================================================"
echo "All requested stages completed successfully!"
echo "============================================================"
