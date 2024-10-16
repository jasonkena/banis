import argparse
import pickle
from typing import Union, Dict, Any

import numpy as np
import zarr
from funlib.evaluate import rand_voi, expected_run_length
from networkx import get_node_attributes
from tqdm import tqdm


def compute_metrics(pred_seg: Union[np.ndarray, zarr.Array], skel_path: str) -> Dict[str, Any]:
    """
    Compute various metrics for evaluating segmentation quality.

    Args:
        pred_seg: The predicted segmentation. Shape: (x, y, z).
        skel_path: Path to the skeleton file (pickle format).

    Returns:
        dict: A dictionary containing various metrics.
    """
    with open(skel_path, "rb") as f:
        skel = pickle.load(f)
    for node in tqdm(skel.nodes):  # slow for zarr
        x, y, z = skel.nodes[node]["index_position"]
        skel.nodes[node]["pred_id"] = pred_seg[x, y, z]

    voi_report = rand_voi(
        np.array(list(get_node_attributes(skel, "id").values())).astype(np.uint64),
        np.array(list(get_node_attributes(skel, "pred_id").values())).astype(np.uint64),
        return_cluster_scores=False,
    )
    voi_split = voi_report["voi_split"]
    voi_merge = voi_report["voi_merge"]
    voi_sum = voi_report["voi_split"] + voi_report["voi_merge"]

    erl_report = expected_run_length(
        skel,
        "id",
        "edge_length",
        get_node_attributes(skel, "pred_id"),
        skeleton_position_attributes=["nm_position"],
        return_merge_split_stats=True,
    )
    merge_stats = erl_report[1]["merge_stats"]
    n_mergers = sum([len(v) for v in merge_stats.values()])

    merge_stats.pop(0, None)  # ignore "mergers" with background
    merge_stats.pop(0.0, None)
    n_non0_mergers = sum([len(v) for v in merge_stats.values()])

    split_stats = erl_report[1]["split_stats"]
    n_splits = sum([len(v) for v in split_stats.values()])
    max_erl_report = expected_run_length(
        skel,
        "id",
        "edge_length",
        get_node_attributes(skel, "id"),
        skeleton_position_attributes=["nm_position"],
        return_merge_split_stats=True,
    )
    erl = erl_report[0]
    max_erl = max_erl_report[0]
    nerl = erl / max_erl

    metrics = {
        "voi_sum": voi_sum,
        "voi_split": voi_split,
        "voi_merge": voi_merge,
        "erl": erl,
        "max_erl": max_erl,
        "nerl": nerl,
        "n_mergers": n_mergers,
        "n_splits": n_splits,
        # "erl_report": erl_report,
        # "max_erl_report": max_erl_report,
        "voi_report": voi_report,
        "n_non0_mergers": n_non0_mergers,
    }
    print(f"metrics: {metrics}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute segmentation quality metrics.")
    parser.add_argument("--pred_seg", type=str, required=True, help="Path to predicted segmentation (Zarr format)")
    parser.add_argument("--skel_path", type=str, required=True, help="Path to skeleton file (pickle format)")
    parser.add_argument("--load_to_memory", action="store_true",
                        help="Load the entire segmentation to memory to speedup")
    args = parser.parse_args()

    if args.pred_seg.endswith('.zarr'):
        pred_seg = zarr.open(args.pred_seg, mode='r')
    elif args.pred_seg.endswith('.npy'):
        pred_seg = np.load(args.pred_seg)
    if args.load_to_memory:
        pred_seg = pred_seg[:]
    compute_metrics(pred_seg, args.skel_path)
