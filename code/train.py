import argparse

from .training.minimal_finetune import train

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_type",
        type=str,
        default="grounding",
        choices=["grounding", "state_transition", "compliance"],
    )
    parser.add_argument(
        "--commit_hash",
        type=str,
        default=None,
        help="Git commit hash for wandb run naming",
    )
    args = parser.parse_args()
    train(dataset_type=args.dataset_type, commit_hash=args.commit_hash)
