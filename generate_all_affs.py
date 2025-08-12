import os
import argparse
import sys
import numpy as np
import torch
import zarr
from BANIS import BANIS
from inference import patched_inference
from lightning import Trainer

@torch.no_grad()
def full_cube_inference(self, mode: str):
    """Perform full cube inference. Expensive!

    Args:
        mode: Either "train", "val", or "test".
    """
    assert mode in ["train", "val", "test"], f"Invalid mode: {mode}"
    print(f"Full cube inference for {mode}")

    base_path_mode = os.path.join(self.hparams.base_data_path, self.hparams.data_setting, mode)
    seeds_path_mode = sorted([f for f in os.listdir(base_path_mode) if "seed" in f])
    assert len(seeds_path_mode) >= 1, f"No seeds found in {base_path_mode}"
    breakpoint()
    seed_path = os.path.join(base_path_mode, seeds_path_mode[0])

    img_data = zarr.open(os.path.join(seed_path, "data.zarr"), mode="r")["img"]

    aff_pred = patched_inference(img_data, model=self, do_overlap=True, prediction_channels=3, divide=255,
                                 small_size=self.hparams.small_size)

    aff_pred = zarr.array(aff_pred, dtype=np.float16, store=f"{self.hparams.save_dir}/pred_aff_{mode}.zarr",
                          chunks=(3, 512, 512, 512), overwrite=True)

    self._evaluate_thresholds(aff_pred, os.path.join(seed_path, "skeleton.pkl"), mode)

if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Generate all affinities for BANIS")
    args.add_argument("--checkpoint_path", type=str, default=None,)

    args = args.parse_args()

    model = BANIS.load_from_checkpoint(args.checkpoint_path)
    model.full_cube_inference("train", all_seeds=True, evaluate_thresholds=False)
