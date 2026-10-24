"""Run training or evaluation with python -m thermorl."""
import argparse

from .experiment import evaluate_main, train_main


def main(argv=None):
    parser = argparse.ArgumentParser(description="Classroom cooling RL experiments")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("train", add_help=False, help="Train Q-learning or DQN")
    commands.add_parser("evaluate", add_help=False, help="Compare models and baselines")
    # Forward each command's own flags to its existing argument parser.
    args, remaining = parser.parse_known_args(argv)
    if args.command == "train":
        train_main(remaining)
    else:
        evaluate_main(remaining)


if __name__ == "__main__":
    main()
