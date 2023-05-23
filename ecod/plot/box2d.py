# Copyright (c) 2022 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

"""Box2d plot functions."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from evis.trans import events_to_frame_1c


def center_to_topleft(boxes):
    boxes[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    boxes[:, 2] = boxes[:, 2] - boxes[:, 4] / 2
    return boxes.astype(int)


def normalized_boxes_to_absolute(boxes, shape):
    boxes[:, 1] = boxes[:, 1] * shape[1]
    boxes[:, 2] = boxes[:, 2] * shape[0]
    boxes[:, 3] = boxes[:, 3] * shape[1]
    boxes[:, 4] = boxes[:, 4] * shape[0]
    return boxes.astype(int)


def add_box(box, ax, color, text=True, names=None, loc="topleft", lw_scale=1.0, box_kwargs=None):
    """Expects topleft xy, width, height or center xy, width, height!"""
    box_kwargs_defaults = {}
    if box_kwargs:
        box_kwargs_defaults.update(box_kwargs)
    if loc == "topleft":
        xy = (box[1], box[2])
        ww = box[3]
        hh = box[4]
    elif loc == "center":
        ww = box[3]
        hh = box[4]
        xy = (int(box[1] - ww / 2), int(box[2] - hh / 2))
    elif loc == "tlbr":
        xy = (box[1], box[2])
        ww = box[3] - box[1]
        hh = box[4] - box[2]
    else:
        raise ValueError("loc has to be 'topleft' or 'center'")

    rec = Rectangle(
        xy, ww, hh, alpha=1, facecolor="none", edgecolor=color, linewidth=1.0 * lw_scale, **box_kwargs_defaults
    )
    ax.add_patch(rec)
    if text:
        name = str(int(box[0]))
        if names is not None:
            name += ": {}".format(names[int(box[0])])
        tt = ax.text(box[1], box[2], name, color="black")
    else:
        tt = None
    return rec, tt


def add_boxes(boxes, n_classes, ax, text=True, names=None, loc="topleft", lw_scale=1.0, box_kwargs=None, cmap="plasma"):
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)
    cmap = cmap(np.linspace(0, 1, num=n_classes))
    recs = []
    for box in boxes:
        recs.append(add_box(box, ax, cmap[int(box[0])], text, names, loc, lw_scale, box_kwargs))
    return recs


# FIXME: Delete if unnecessary
# def plot_image_with_boxes(
#    image,
#    boxes=None,
#    n_classes=None,
#    text=True,
#    names=None,
#    loc="topleft",
#    dpi=200.0,
#    ax=None,
#    cmap_name=None,
#    scale=1.0,
# ):
#    """Plots grayscale and rgb"""
#    if len(image.shape) == 3 and image.shape[0] == 3:
#        image = np.moveaxis(image, 0, -1)
#    shape = image.shape
#    height = shape[0] / dpi * scale
#    width = shape[1] / dpi * scale
#    if ax is None:
#        fig, ax = plt.subplots(figsize=(width, height), dpi=dpi)
#    else:
#        fig = plt.gcf()
#    ax.axis("off")
#    if cmap_name is None:
#        if len(image.shape) == 2:
#            cmap = "gray"
#        else:
#            cmap = None
#        vmin = None
#        vmax = None
#    else:
#        cmap = cmaps[cmap_name]
#        vmin = vs[cmap_name][0]
#        vmax = vs[cmap_name][1]
#    ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="none", aspect="equal", origin="upper")
#    if boxes is not None:
#        add_boxes(boxes, n_classes, ax, text, names, loc)
#    return fig, ax
#
#
# def plot_image_with_boxes_2c(
#    image, boxes=None, n_classes=None, text=True, names=None, loc="topleft", dpi=200.0, scale=1.0
# ):
#    shape = image.shape
#    height = shape[1] / dpi * scale
#    width = 2 * shape[2] / dpi * scale
#    fig, axes = plt.subplots(1, 2, figsize=(width, height), dpi=dpi)
#    axes = axes.flatten()
#    for ii, ax in enumerate(axes):
#        ax.axis("off")
#        if image.max() > 1:
#            image = image / image.max()
#        ax.imshow(image[ii])
#        if boxes is not None:
#            add_boxes(boxes, n_classes, ax, text, names, loc)
#    return fig, axes
#
#
# def save_image_with_boxes(savepath, image, boxes, n_classes, text=True, names=None, loc="topleft", dpi=200.0):
#    if len(image.shape) == 3 and image.shape[0] == 2:
#        fig, ax = plot_image_with_boxes_2c(image, boxes, n_classes, text, names, loc, dpi)
#    else:
#        fig, ax = plot_image_with_boxes(image, boxes, n_classes, text, names, loc, dpi)
#    fig.tight_layout(pad=0.0)
#    fig.savefig(savepath, dpi=dpi)


def plot_side_by_side(savepath, events, img, boxes=None, reconstruction=None):
    """Plot 1 event frame + reconstruction/segmentation"""
    n_x = 2 if reconstruction is None else 3
    width = 2 * 3.487
    height = width / 1.618
    cmap = plt.get_cmap("plasma")(np.linspace(0, 1, 10))

    fig, axes = plt.subplots(1, n_x, figsize=(width, height))
    axes = axes.flatten()
    axes[0].imshow(img)

    img_gs = events_to_frame_1c(events, img.shape[0], img.shape[1])
    axes[1].imshow(img_gs, interpolation="none")
    if boxes is not None:
        for box in boxes:
            add_box(box, axes[0], cmap[int(box[0])])
            add_box(box, axes[1], cmap[int(box[0])])
    axes[0].set_title("frame")
    axes[1].set_title("events")

    fig.tight_layout()
    fig.savefig(savepath, dpi=200)
