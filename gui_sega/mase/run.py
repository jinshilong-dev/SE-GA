import argparse
from .config import parse_config
from .train_sft import run_sft, run_grounding_sft, run_planning_sft
from .train_grpo import run_grpo


STAGES = {
    "sft": run_sft,
    "grounding_sft": run_grounding_sft,
    "planning_sft": run_planning_sft,
    "grpo": run_grpo,
}


def main():
    parser = argparse.ArgumentParser(description="GUI-SEGA: GUI Agent Training Pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to the YAML config file",
    )
    parser.add_argument(
        "--stage",
        type=str,
        nargs="+",
        default=["sft", "grpo"],
        choices=list(STAGES.keys()),
        help="Training stage(s) to run. Default: sft + grpo.",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="DeepSpeed distributed local rank (auto-set by launcher)",
    )
    args = parser.parse_args()

    cfg = parse_config(args.config)

    print("=" * 60)
    print("GUI-SEGA Training Pipeline")
    print(f"Config: {args.config}")
    print(f"Stages: {args.stage}")
    print("=" * 60)

    for stage_name in args.stage:
        print(f"\n{'#' * 60}")
        print(f"# Running stage: {stage_name}")
        print(f"{'#' * 60}\n")
        STAGES[stage_name](cfg)

    print("\n" + "=" * 60)
    print("All requested stages completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
