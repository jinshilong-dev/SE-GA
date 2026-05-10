#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONFIG="${CONFIG:-configs/default.yaml}"
NUM_GPUS="${NUM_GPUS:-1}"
GPUS="${GPUS:-0}"

usage() {
    echo "Usage: bash run_sft.sh [OPTIONS]"
    echo ""
    echo "GUI-SEGA: SFT Training (Grounding + Planning)"
    echo ""
    echo "Options:"
    echo "  --config PATH       Path to YAML config file (default: configs/default.yaml)"
    echo "  --stage STAGE       SFT stage to run (default: all)"
    echo "                        all            - Grounding SFT + Planning SFT sequentially"
    echo "                        grounding_sft  - Grounding SFT only"
    echo "                        planning_sft   - Planning SFT only"
    echo "  --num_gpus N        Number of GPUs to use (default: 1)"
    echo "  --gpus GPU_IDS      Comma-separated GPU IDs (default: 0)"
    echo "  --help              Show this help message"
    echo ""
    echo "Examples:"
    echo "  bash run_sft.sh                                    # Run Grounding + Planning SFT"
    echo "  bash run_sft.sh --stage grounding_sft              # Run only Grounding SFT"
    echo "  bash run_sft.sh --stage planning_sft --gpus 0,1    # Run only Planning SFT on GPU 0,1"
}

STAGE="${STAGE:-all}"

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
echo "GUI-SEGA: SFT Training Pipeline"
echo "============================================================"
echo "Config:   $CONFIG"
echo "Stage:    $STAGE"
echo "GPUs:     $GPUS ($NUM_GPUS GPUs)"
echo "============================================================"

case "$STAGE" in
    all)
        echo ""
        echo "############################################################"
        echo "# Running Grounding SFT..."
        echo "############################################################"
        echo ""
        python -m gui_sega.mase.run --config "$CONFIG" --stage grounding_sft
        echo "Grounding SFT completed."

        echo ""
        echo "############################################################"
        echo "# Running Planning SFT..."
        echo "############################################################"
        echo ""
        python -m gui_sega.mase.run --config "$CONFIG" --stage planning_sft
        echo "Planning SFT completed."
        ;;
    grounding_sft)
        echo ""
        echo "############################################################"
        echo "# Running Grounding SFT..."
        echo "############################################################"
        echo ""
        python -m gui_sega.mase.run --config "$CONFIG" --stage grounding_sft
        echo "Grounding SFT completed."
        ;;
    planning_sft)
        echo ""
        echo "############################################################"
        echo "# Running Planning SFT..."
        echo "############################################################"
        echo ""
        python -m gui_sega.mase.run --config "$CONFIG" --stage planning_sft
        echo "Planning SFT completed."
        ;;
    *)
        echo "Unknown stage: $STAGE"
        echo "Valid options: all, grounding_sft, planning_sft"
        exit 1
        ;;
esac

echo ""
echo "============================================================"
echo "SFT Training completed successfully!"
echo "============================================================"