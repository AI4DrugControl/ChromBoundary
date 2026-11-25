import argparse
import json
from pathlib import Path

import torch

from .data import records
from .engine import load_checkpoint, predict, save_predictions
from .metrics import evaluate


def main():
    parser = argparse.ArgumentParser(prog="chromboundary")
    commands = parser.add_subparsers(dest="command", required=True)
    inference = commands.add_parser("predict")
    inference.add_argument("--model", default="pdac", metavar="NAME_OR_PATH")
    inference.add_argument("--signals", type=Path, required=True)
    inference.add_argument("--out", type=Path, required=True)

    training = commands.add_parser("train")
    training.add_argument("--signals", type=Path, required=True)
    training.add_argument("--references", type=Path, required=True)
    training.add_argument("--profile", choices=("peptide", "metabolite"), required=True)
    training.add_argument("--out", type=Path, required=True)
    training.add_argument("--selection-panel")
    training.add_argument("--epochs", type=int, default=40)
    training.add_argument("--seed", type=int, default=2023)
    for command in (inference, training):
        command.add_argument("--device", default="cpu")
        command.add_argument("--batch-size", type=int, default=32)
    scoring = commands.add_parser("evaluate")
    scoring.add_argument("--predictions", type=Path, required=True)
    scoring.add_argument("--references", type=Path, required=True)
    scoring.add_argument("--split", choices=("train", "val", "test"))
    args = parser.parse_args()

    if args.command == "evaluate":
        print(
            json.dumps(
                evaluate(args.predictions, args.references, args.split), indent=2, allow_nan=False
            )
        )
        return
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    device = torch.device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(min(torch.get_num_threads(), 4))
    sources = [getattr(args, k, None) for k in ("signals", "references", "model")]
    if any(p is not None and Path(p).resolve() == args.out.resolve() for p in sources):
        parser.error("Choose an output path distinct from the input files")
    if args.command == "predict":
        model, profile, refiner = load_checkpoint(args.model, device)
        results = predict(model, records(args.signals, profile), refiner, args.batch_size, device)
        save_predictions(args.out, results)
    else:
        from .train import train

        train(
            args.signals,
            args.references,
            args.profile,
            args.out,
            args.epochs,
            args.batch_size,
            args.seed,
            device,
            args.selection_panel,
        )


if __name__ == "__main__":
    main()
