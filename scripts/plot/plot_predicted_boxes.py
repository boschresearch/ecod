# Copyright (c) 2022 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from ecod.utils.files import load_args_dict
from ecod.utils.data import get_dataset_attributes
from ecod.eval.evaluator import get_predicted_box_data, indexed_boxes_to_list
from ecod.training.prep import get_data_module
from ecod.data.box2d.transforms import normalized_boxes_to_absolute
from ecod.plot.box2d import add_boxes


def plot_predicted_boxes(
    root,
    val_test,
    idxs_plot,
    savedir=None,
    conf_thresh=None,
    with_nonloaded_boxes=False,
):
    args_dict = load_args_dict(root)
    bbox_data, _ = get_predicted_box_data(root, val_test)
    attrs = get_dataset_attributes(args_dict["dataset"])
    class_names = attrs["class_names"]
    n_classes = attrs["n_classes"]

    dm = get_data_module(args_dict)
    if val_test == "val":
        dset = dm.val_dset
    else:
        dset = dm.test_dset

    n_idxs = len(idxs_plot)
    n_timesteps = args_dict["n_timesteps"]
    idxs = bbox_data["idxs"]
    conf_thresh = args_dict["test_confidence_threshold"] if conf_thresh is None else conf_thresh

    fig, axes = plt.subplots(2 * n_idxs, n_timesteps, figsize=(10, 15))
    sample_idx_last = -1
    for ii, idx_plot in enumerate(idxs_plot):
        for jj in range(n_timesteps):
            data_idx = idx_plot + jj
            sample_idx, seq_idx = idxs[data_idx]
            if sample_idx != sample_idx_last:
                seq, bboxs_dset, labels_dset, idxs_dset, idx, file_idx = dset[sample_idx]
                bboxs_dset = normalized_boxes_to_absolute(bboxs_dset, args_dict["shape_t"][-2:], copy=False)
                bboxs_dset, labels_dset, _ = indexed_boxes_to_list(bboxs_dset, labels_dset, idxs_dset, idx, n_timesteps)
                frames = seq.numpy().mean((1, 2))
            axes[2 * ii, jj].imshow(frames[seq_idx])
            axes[2 * ii + 1, jj].imshow(frames[seq_idx])
            axes[2 * ii + 1, jj].set_title(f"pred: {sample_idx}, {seq_idx}")
            axes[2 * ii, jj].set_title(f"gt: {sample_idx}, {seq_idx}")
            mask = bbox_data["scores_pred"][data_idx] >= conf_thresh
            bboxs_pred = bbox_data["bboxs_pred"][data_idx][mask]
            labels_pred = bbox_data["labels_pred"][data_idx][mask] - 1
            bboxs_plot_pred = np.c_[labels_pred, bboxs_pred]
            add_boxes(
                bboxs_plot_pred,
                n_classes,
                axes[2 * ii + 1, jj],
                text=True,
                names=class_names[1:],
                loc="tlbr",
                lw_scale=2.0,
                box_kwargs=None,
                cmap="Dark2",
            )
            bboxs_plot_dset = np.c_[labels_dset[seq_idx] - 1, bboxs_dset[seq_idx]]
            add_boxes(
                bboxs_plot_dset,
                n_classes,
                axes[2 * ii, jj],
                text=True,
                names=class_names[1:],
                loc="tlbr",
                lw_scale=2.0,
                box_kwargs=None,
                cmap="jet",
            )
            if with_nonloaded_boxes:
                bboxs_gt = bbox_data["bboxs_gt"][data_idx]
                labels_gt = bbox_data["labels_gt"][data_idx] - 1
                bboxs_plot_gt = np.c_[labels_gt, bboxs_gt]
                add_boxes(
                    bboxs_plot_gt,
                    10,
                    axes[2 * ii, jj],
                    text=True,
                    names=None,
                    loc="tlbr",
                    lw_scale=2.0,
                    box_kwargs=None,
                    cmap="Dark2",
                )
            sample_idx_last = sample_idx
    if savedir is not None:
        fig.tight_layout(pad=0.01)
        fig.savefig(Path(savedir) / "pred_boxes.png", dpi=300.0)
    return fig, axes


experiment_folder = ""
val_test = "val"
idxs_plot = [0, 1, 2, 3, 10, 40]
savedir = "."
conf_thresh = 0.5

plot_predicted_boxes(
    experiment_folder,
    val_test,
    idxs_plot,
    savedir=savedir,
    conf_thresh=conf_thresh,
    with_nonloaded_boxes=False,
)
