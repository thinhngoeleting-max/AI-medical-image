import argparse
from pathlib import Path

from .config import Config, seed_everything


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("Must be at least 1")
    return value


def main():
    parser = argparse.ArgumentParser(description="Chest X-ray classification research pipeline")
    parser.add_argument("--config", default="config.json")
    commands = parser.add_subparsers(dest="command", required=True)
    split = commands.add_parser("split", help="Save a train/validation split without moving files")
    split.add_argument("--patient-csv", help="CSV columns: path,patient_id; paths relative to data_dir")
    commands.add_parser("train", help="Create or reuse the configured split, then train")
    evaluation = commands.add_parser("evaluate", help="Export classification and calibration metrics")
    evaluation.add_argument("--split", choices=["val", "test"], default="test")
    evaluation.add_argument("--limit", type=positive_int, help="Smoke check only; not a final evaluation")
    explanation = commands.add_parser("explain", help="Save an explanation for one image")
    explanation.add_argument("--image", required=True)
    explanation.add_argument("--method", choices=["gradcam", "gradcam++"], default="gradcam")
    explanation.add_argument("--target", choices=["NORMAL", "PNEUMONIA"])
    robust = commands.add_parser("robustness", help="Measure prediction changes under perturbations")
    robust.add_argument("--split", choices=["val", "test"], default="val")
    robust.add_argument("--limit", type=positive_int, default=100)
    robust.add_argument("--border-fraction", type=float, default=0.08)
    for subparser in (evaluation, explanation, robust):
        subparser.add_argument("--checkpoint", help="Defaults to run_dir/best.pt; accepts notebook .pth weights")
    args = parser.parse_args()
    config = Config.load(args.config)
    seed_everything(config.seed)
    if args.command == "split":
        from .data import create_split
        manifest = create_split(config, args.patient_csv)
        print({name: len(manifest[name]) for name in ("train", "val", "test")})
        print(f"Patient separation verified: {manifest['patient_separation_verified']}")
        print(f"Split saved to {Path(config.run_dir) / 'split.json'}. No images moved.")
    elif args.command == "train":
        from .train import train
        train(config)
    else:
        checkpoint = args.checkpoint or str(Path(config.run_dir) / "best.pt")
        if args.command == "evaluate":
            from .evaluate import evaluate
            evaluate(config, checkpoint, args.split, args.limit)
        elif args.command == "explain":
            from .explain import explain
            explain(config, checkpoint, args.image, args.method, args.target)
        elif args.command == "robustness":
            if not 0 < args.border_fraction < 0.5:
                parser.error("--border-fraction must be between 0 and 0.5")
            from .robustness import robustness
            robustness(config, checkpoint, args.split, args.limit, args.border_fraction)


if __name__ == "__main__":
    main()
