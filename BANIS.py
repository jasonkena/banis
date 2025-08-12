import argparse
import os
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict

import numpy as np
import pytorch_lightning as pl
import torch
import torchvision
import zarr
from nnunet_mednext import create_mednext_v1
from pytorch_lightning import LightningModule, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, DeviceStatsMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from torch.nn.functional import binary_cross_entropy_with_logits, mse_loss, tanh
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import load_data
from inference import scale_sigmoid, patched_inference, compute_connected_component_segmentation
from metrics import compute_metrics


class BANIS(LightningModule):
    """
    PyTorch Lightning module for BANIS: Baseline for Affinity-based Neuron Instance Segmentation
    """

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__()
        self.save_hyperparameters(*args, **kwargs)
        print(f"hparams: \n{self.hparams}")

        self.model = create_mednext_v1(
            num_input_channels=self.hparams.num_input_channels,
            num_classes=6 + int(self.hparams.sdt),  # 3 short + 3 long range affinities + (1 if self.hparams.sdt)
            model_id=self.hparams.model_id,
            kernel_size=self.hparams.kernel_size,
        )
        self.model.outside_block_checkpointing = True  # Save GPU memory

        if self.hparams.compile:
            self.model = torch.compile(self.model)

        self.best_nerl_so_far = defaultdict(float)  # for train/val/test
        self.best_thr_so_far = defaultdict(float)

    def on_fit_start(self):
        self.logger.experiment.add_text("hparams", str(self.hparams))

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
        if self.hparams.scheduler:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams.n_steps)
            return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
        return optimizer

    def on_train_epoch_start(self):
        self.plotted = False

    def on_validation_epoch_start(self):
        self.plotted = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, data: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._step(data, "train")

    def validation_step(self, data: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._step(data, "val")

    def _step(self, data: Dict[str, torch.Tensor], mode: str) -> torch.Tensor:
        pred = self(data["img"])
        if self.hparams.sdt:
            aff_pred, sdt_pred = pred[:, :-1], pred[:, -1]
        else:
            aff_pred = pred
        target = data["aff"].half()
        aff_loss_mask = data["aff"] >= 0
        aff_loss = binary_cross_entropy_with_logits(aff_pred[aff_loss_mask], target[aff_loss_mask])
        self.log(f"{mode}_aff_loss", aff_loss)

        if self.hparams.sdt:
            sdt_target = data["sdt"].half()
            assert -1 <= sdt_target.min() and sdt_target.max() <= 1
            sdt_loss_mask = data["sdt_mask"]
            sdt_loss = mse_loss(tanh(sdt_pred[sdt_loss_mask]), sdt_target[sdt_loss_mask])
            self.log(f"{mode}_sdt_loss", sdt_loss)
            loss = aff_loss + self.hparams.sdt_loss_weight * sdt_loss
            self.log(f"{mode}_total_loss", loss)
        else:
            loss = aff_loss


        if not self.plotted:
            self._log_images(data, pred, mode)
            self.plotted = True
        return loss

    def _log_images(self, data: Dict[str, torch.Tensor], pred: torch.Tensor, mode: str):
        middle = data["img"].shape[2] // 2
        self._add_image(f"{mode}_img", data["img"][:, :3, middle])
        self._add_image(f"{mode}_aff", data["aff"][:, :3, middle])
        self._add_image(f"{mode}_aff_pred", scale_sigmoid(pred[:, :3, middle]))
        self._add_image(f"{mode}_lr_aff", data["aff"][:, 3:6, middle])
        self._add_image(f"{mode}_lr_aff_pred", scale_sigmoid(pred[:, 3:6, middle]))
        
        if self.hparams.sdt:
            self._add_image(f"{mode}_sdt", data["sdt"][:, middle].unsqueeze(1))
            self._add_image(f"{mode}_sdt_pred", tanh(pred[:, -1, middle]).unsqueeze(1))

        seg_middle = data["seg"][:, middle]
        colormap = torch.rand(seg_middle.max() + 1, 3)
        colormap[0] = 0
        colored_seg = colormap[seg_middle.cpu()].permute(0, 3, 1, 2)
        self._add_image(f"{mode}_seg", colored_seg)

    def _add_image(self, tag: str, img: torch.Tensor) -> None:
        self.logger.experiment.add_image(tag, torchvision.utils.make_grid(img, value_range=(0, 1)),
                                         global_step=self.global_step)

    def on_validation_epoch_end(self):
        self.full_cube_inference("val")

    def on_train_end(self):
        assert self.best_nerl_so_far["val"] > 0, "No best NERL found in validation"
        self.eval()
        print(f"device {next(self.parameters()).device}")
        self.cuda()
        # self.full_cube_inference("val")
        assert self.best_nerl_so_far["val"] > 0, "No best NERL found in validation"
        self.full_cube_inference("test")
        self.full_cube_inference("train")

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
        seed_path = os.path.join(base_path_mode, seeds_path_mode[0])

        img_data = zarr.open(os.path.join(seed_path, "data.zarr"), mode="r")["img"]

        aff_pred = patched_inference(img_data, model=self, do_overlap=True, prediction_channels=3, divide=255,
                                     small_size=self.hparams.small_size)

        aff_pred = zarr.array(aff_pred, dtype=np.float16, store=f"{self.hparams.save_dir}/pred_aff_{mode}.zarr",
                              chunks=(3, 512, 512, 512), overwrite=True)

        self._evaluate_thresholds(aff_pred, os.path.join(seed_path, "skeleton.pkl"), mode)

    def _evaluate_thresholds(self, aff_pred: zarr.Array, skel_path: str, mode: str):
        best_voi = best_voi_no_merge = 1e100
        best_nerl = best_nerl_no_merge = -1
        best_nerl_metrics = None
        thresholds = self.hparams.eval_ranges if mode != "test" else [self.best_thr_so_far["val"]]

        for thr in tqdm(thresholds):
            print(f"threshold {thr}")

            pred_seg = compute_connected_component_segmentation(
                aff_pred[:3] > thr  # hard affinities
            )

            metrics = compute_metrics(pred_seg, skel_path)

            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    self.safe_add_scalar(f"{mode}_{k}_thr_{thr}", v)

            if metrics["n_non0_mergers"] == 0:
                best_nerl_no_merge = max(best_nerl_no_merge, metrics["nerl"])
                best_voi_no_merge = min(best_voi_no_merge, metrics["voi_sum"])

            if metrics["nerl"] > best_nerl:
                best_nerl = metrics["nerl"]
                best_nerl_metrics = metrics
                if self.best_nerl_so_far[mode] < best_nerl:
                    self.best_nerl_so_far[mode] = best_nerl
                    self.best_thr_so_far[mode] = thr
                    np.save(f"{self.hparams.save_dir}/pred_aff_best_nerl_{mode}.npy", aff_pred)
                    np.save(f"{self.hparams.save_dir}/pred_seg_best_nerl_{mode}.npy", pred_seg)
            best_voi = min(best_voi, metrics["voi_sum"])

        self.safe_add_scalar(f"{mode}_best_nerl", best_nerl)
        self.safe_add_scalar(f"{mode}_best_voi", best_voi)
        self.safe_add_scalar(f"{mode}_best_nerl_no_merge", best_nerl_no_merge)
        self.safe_add_scalar(f"{mode}_best_voi_no_merge", best_voi_no_merge)

        for k, v in best_nerl_metrics.items():
            if isinstance(v, (int, float)):
                self.safe_add_scalar(f"{mode}_best_nerl_{k}", v)

    def safe_add_scalar(self, name: str, value: float) -> None:
        try:  # s.t. full_cube_inference can be called outside of .fit() without error
            self.logger.experiment.add_scalar(name, value, self.global_step)
        except Exception as e:
            print(f"Error logging {name}: {e}")


def main():
    args = parse_args()
    args.resolution = (9, 9, 20) if args.data_setting != "liconn" else (9, 9, 12)

    seed_everything(args.seed, workers=True)

    torch.set_float32_matmul_precision("medium")

    if args.checkpoint is not None:
        exp_name = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(args.checkpoint)))))
    else:
        exp_name = (
                datetime.now().strftime("%y-%m-%d_%H-%M-%S-%f")
                + f"ds_{args.data_setting}"
                  f"_lrng{args.long_range}_s{args.seed}_b{args.batch_size}_m{args.model_id}_k{args.kernel_size}_"
                  f"lr{args.learning_rate}_wd{args.weight_decay}_sch{args.scheduler}_syn_{args.synthetic}"
                  f"_drop{args.drop_slice_prob}_shift{args.shift_slice_prob}_int{args.intensity_aug}_noise{args.noise_scale}"
                  f"_affine{args.affine}_ns{args.n_steps}_ss{args.small_size}"
                  f"_sdt{int(args.sdt)}_sdtw{args.sdt_loss_weight}"
        )

    save_dir = os.path.join(args.save_path, exp_name)
    os.makedirs(save_dir, exist_ok=True)
    print(f"save dir: {save_dir}")
    tb_logger = TensorBoardLogger(save_dir=save_dir)
    tb_logger.experiment.add_text("save dir", save_dir)

    trainer = pl.Trainer(
        callbacks=[
            DeviceStatsMonitor(),
            ModelCheckpoint(
                monitor="val_loss",
                save_last=True,
                mode="min",
                save_top_k=100,
            ),
        ],
        logger=tb_logger,
        max_steps=args.n_steps,
        accelerator="gpu",
        devices=args.devices,
        log_every_n_steps=args.log_every_n_steps,
        limit_val_batches=100,
        precision="16-mixed",
        profiler="simple",
        default_root_dir=save_dir,
        val_check_interval=args.val_check_interval,  # validation full cube inference expensive so less frequent
        check_val_every_n_epoch=None,
        num_sanity_val_steps=args.n_debug_steps,
    )

    train_data, val_data, n_channels = load_data(args)
    args.save_dir = save_dir
    args.num_input_channels = n_channels

    model = BANIS(args)

    trainer.fit(
        model=model,
        train_dataloaders=DataLoader(train_data, batch_size=args.batch_size, num_workers=args.workers, shuffle=True,
                                     drop_last=True),
        val_dataloaders=DataLoader(val_data, batch_size=args.batch_size, num_workers=args.workers)
    )

    print("Training complete")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility.")
    parser.add_argument("--long_range", type=int, default=10, help="Long range affinities (voxels).")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training.")
    parser.add_argument("--model_id", type=str, default="S", help="Identifier for the mednext model architecture.")
    parser.add_argument("--kernel_size", type=int, default=3, help="Kernel size for the convolutional layers.")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate for the optimizer.")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay for the optimizer.")
    parser.add_argument("--drop_slice_prob", type=float, default=0.05,
                        help="Probability of dropping a slice during augmentation.")
    parser.add_argument("--shift_slice_prob", type=float, default=0.05,
                        help="Probability of shifting a slice during augmentation.")
    parser.add_argument("--intensity_aug", action=argparse.BooleanOptionalAction, default=True,
                        help="Whether to apply intensity augmentation.")
    parser.add_argument("--noise_scale", type=float, default=0.5,
                        help="Scale of the noise to be added during augmentation.")
    parser.add_argument("--affine", type=float, default=0.5, help="Affine transformation probability.")
    parser.add_argument("--n_steps", type=int, default=20_000, help="Number of training steps.")
    parser.add_argument("--workers", type=int, default=8, help="Number of workers for data loading.")
    parser.add_argument("--base_data_path", type=str,
                        default="/cajal/nvmescratch/projects/NISB/",
                        help="Base path for the dataset.")
    parser.add_argument("--data_setting", type=str, default="base", help="Data setting identifier.")
    parser.add_argument("--scheduler", action=argparse.BooleanOptionalAction, default=True,
                        help="Whether to use a learning rate scheduler.")
    parser.add_argument("--synthetic", type=float, default=1.0, help="Ratio of synthetic data to real data.")
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True,
                        help="Whether to compile the model using torch.compile.")
    parser.add_argument("--eval_ranges", type=float, nargs="+",
                        default=torch.sigmoid(torch.tensor(list(range(-1, 12))).double() * 0.2).numpy().round(
                            4).tolist(),
                        help="List of evaluation thresholds.")
    parser.add_argument("--save_path", type=str, default="/cajal/scratch/projects/misc/riegerfr/aff_nis/",
                        help="Path to save the model and logs.")
    parser.add_argument("--real_data_path", type=str,
                        default="/cajal/scratch/projects/misc/mdraw/data/funke/zebrafinch/training/",
                        help="Path to the real dataset. See https://colab.research.google.com/github/funkelab/lsd/blob/master/lsd/tutorial/notebooks/lsd_data_download.ipynb ")
    parser.add_argument("--affine_scale", type=float, default=0.2, help="Scale for affine augmentation.")
    parser.add_argument("--affine_shear", type=float, default=0.5, help="Shear for affine augmentation.")
    parser.add_argument("--shift_magnitude", type=int, default=10, help="Shift augmentation magnitude (voxels).")
    parser.add_argument("--mul_int", type=float, default=0.1, help="Multiplicative augmentation intensity.")
    parser.add_argument("--add_int", type=float, default=0.1, help="Additive augmentation intensity.")
    parser.add_argument("--devices", type=int, default=-1, help="Number GPU devices to use (-1: all).")
    parser.add_argument("--n_debug_steps", type=int, default=0, help="Number of debug steps.")
    parser.add_argument("--log_every_n_steps", type=int, default=100, help="Log every n steps.")
    parser.add_argument("--val_check_interval", type=int, default=5000, help="Validation check interval.")
    parser.add_argument("--small_size", type=int, default=128, help="Size of the patches.")

    # make it so that adding it makes it true
    parser.add_argument("--sdt", action="store_true", help="Whether to train to predict SDT.")
    parser.add_argument("--sdt_loss_weight", type=float, default=1.0, help="Weight of the SDT loss.")

    parser.add_argument("--checkpoint", type=str, default=None,)

    return parser.parse_args()


if __name__ == "__main__":
    main()
