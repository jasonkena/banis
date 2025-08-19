import argparse
from BANIS import BANIS

if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Generate all affinities for BANIS")
    args.add_argument("--checkpoint_path", type=str, default=None,)
    args.add_argument("--prediction_channels", type=int, default=3,)

    args = args.parse_args()

    model = BANIS.load_from_checkpoint(args.checkpoint_path)
    model.full_cube_inference("val", all_seeds=True, evaluate_thresholds=False, prediction_channels=args.prediction_channels)
    # model.full_cube_inference("train", all_seeds=True, evaluate_thresholds=False)
