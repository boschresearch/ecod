# Copyright (c) 2022 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

"""Generate a video from an RM-MNIST sample.

Examples

python ./scripts/plot/rmmnist_video.py --datadir ./ --idx 0 --train_val_test train --events --savedir ./
"""
from pathlib import Path
import logging
import argparse
import numpy as np
import matplotlib

from ecod.events.events import events_to_frames
from ecod.data.sim.rmmnist import load_rmmnist_by_index
from ecod.utils.general import ProgressLogger
from ecod.utils.files import makedirs
from ecod.utils.data import load_json
from ecod.plot.box2d import add_boxes
from ecod.plot.events import ev_frames_to_quad, imshow_ev_quad, imshow_gs
from ecod.plot.video import FigVideoWriter


def save_event_video_moving_mnist_events(
    events,
    shape,
    savepath,
    boxes=None,
    box_times=None,
    n_classes=None,
    t_start=0.0,
    t_stop=60.0,
    playback_speed=1.0,
    fps=30.0,
    dpi=100.0,
    size_inch=0.01,
    n_frames_buffer=100,
    force_limits=True,
    add_boxes_text=True,
):
    scaled_fps = fps / playback_speed
    delta_t = 1e3 / scaled_fps
    n_frames_tot = int((t_stop - t_start) * scaled_fps)
    writer = FigVideoWriter(str(savepath), fps, shape[0], shape[1], dpi, size_inch)
    if box_times is not None:
        delta_t_boxes = np.min(np.diff(box_times)) / 2.0
    else:
        delta_t_boxes = None

    div, mod = divmod(n_frames_tot, n_frames_buffer)
    rr = div + int(mod != 0)
    delta_t_buffer = delta_t * n_frames_buffer
    t_now = t_start * 1e6
    t_vid = t_start * 1e6
    xlim = (-0.5, shape[1] - 0.5)
    ylim = (shape[0] - 0.5, -0.5)
    for ii in ProgressLogger(range(rr)):
        # last frames can be less than buffer size
        if ii == rr - 1 and mod != 0:
            n_frames_buffer = mod
            delta_t_buffer = n_frames_buffer * delta_t
        frames = events_to_frames(
            events,
            shape,
            n_frames=n_frames_buffer,
            t_start=t_now,
            t_end=t_now + delta_t_buffer * 1e3,
            binary_spikes=True,
        )
        frames = ev_frames_to_quad(frames)
        for jj, frame in enumerate(frames):
            writer.reset_frame()
            # it's faster to reset the frame than re-writing im.array!
            im = imshow_ev_quad(frame, writer.ax, im=None)
            if boxes is not None:
                t_start = t_vid - delta_t_boxes
                t_stop = t_vid + delta_t_boxes
                start, stop = np.searchsorted(box_times, [t_start, t_stop])
                # have to be [label, x_tl, y_tl, w, h]
                boxes_now = boxes[start:stop]
                add_boxes(
                    boxes_now,
                    10,
                    writer.ax,
                    text=add_boxes_text,
                    names=[str(ii) for ii in range(10)],
                    loc="topleft",
                    lw_scale=2.0,
                    box_kwargs=None,
                    cmap="Dark2",
                )
            if force_limits:
                if writer.ax.get_xlim() != xlim:
                    writer.ax.set_xlim(xlim)
                if writer.ax.get_ylim() != ylim:
                    writer.ax.set_ylim(ylim)
            writer.write()
            t_vid += delta_t
        t_now += delta_t_buffer * 1e3
    writer.close()


def save_event_video_moving_mnist(
    frames,
    frame_times,
    shape,
    savepath,
    boxes=None,
    box_times=None,
    class_names=None,
    t_start=0.0,
    t_stop=60.0,
    playback_speed=1.0,
    fps=30.0,
    dpi=100.0,
    size_inch=0.01,
    force_limits=True,
    add_boxes_text=True,
):
    n_classes = None if class_names is None else len(class_names)
    scaled_fps = fps / playback_speed
    delta_t = 1e3 / scaled_fps
    frame_start, frame_stop = np.searchsorted(frame_times, [t_start * 1e3, t_stop * 1e3])
    frame_times = frame_times[frame_start:frame_stop]
    frames = frames[frame_start:frame_stop]
    n_frames_tot = int((t_stop - t_start) * scaled_fps)
    writer = FigVideoWriter(str(savepath), fps, shape[0], shape[1], dpi, size_inch)
    if box_times is not None:
        delta_t_boxes_ms = np.min(np.diff(box_times)) / 2.0
    else:
        delta_t_boxes_ms = None

    t_vid = t_start * 1e3
    xlim = (-0.5, shape[1] - 0.5)
    ylim = (shape[0] - 0.5, -0.5)
    for jj in range(n_frames_tot):
        writer.reset_frame()
        idx_scaling_factor = (len(frames) - 1) / (n_frames_tot - 1)
        frame = frames[int(jj * idx_scaling_factor)]
        # it's faster to reset the frame than re-writing im.array!
        im = imshow_gs(frame, writer.ax, im=None, cmap="Greys")
        if boxes is not None:
            t_start = t_vid - delta_t_boxes_ms
            t_stop = t_vid + delta_t_boxes_ms
            start, stop = np.searchsorted(box_times, [t_start, t_stop])
            # have to be [label, x_tl, y_tl, w, h]
            boxes_now = boxes[start:stop]
            add_boxes(
                boxes_now,
                n_classes,
                writer.ax,
                text=add_boxes_text,
                names=class_names,
                loc="topleft",
                lw_scale=2.0,
                box_kwargs=None,
                cmap="Dark2",
            )
        if force_limits:
            if writer.ax.get_xlim() != xlim:
                writer.ax.set_xlim(xlim)
            if writer.ax.get_ylim() != ylim:
                writer.ax.set_ylim(ylim)
        writer.write()
        t_vid += delta_t
    writer.close()


def parse_args(args=None):
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--datadir", required=True)
    parser.add_argument("--idx", default=None)
    parser.add_argument("--train_val_test", default="train", choices=["train", "val", "test"])
    parser.add_argument("--savedir", type=str, required=True)
    parser.add_argument("--playback_speed", type=float, default=1.0)
    parser.add_argument("--size_inch", type=float, default=0.01, help="Scaling factor of pixels")
    parser.add_argument("--t_stop", type=float, default=None)
    parser.add_argument("--events", action="store_true")
    parser.add_argument("--text", action="store_true")
    args = parser.parse_args(args=args)
    return args


def main():

    logging.basicConfig(
        level=logging.WARN,
        format="%(asctime)s:%(levelname)s:%(name)s: %(message)s",
        datefmt="%y%m%d-%H:%M:%S",
    )
    args = parse_args()
    root = Path(args.datadir)
    idxs = args.idx
    if idxs is None:
        root = Path(root)
        meta_info = load_json(root / "meta_info.json")
        tvt_idx = {"train": 0, "val": 1, "test": 2}[args.train_val_test]
        idxs = range(meta_info["n_samples"][tvt_idx])
    else:
        idxs = [idxs]
    train_val_test = args.train_val_test

    for idx in ProgressLogger(idxs, name="tot"):
        (
            meta_info,
            frames,
            labels,
            bboxs,
            frame_times,
            bbox_times,
            bbox_idxs,
            speeds,
            stops,
        ) = load_rmmnist_by_index(root, idx, train_val_test)
        t_stop = args.t_stop if args.t_stop is not None else meta_info["time_s"]
        boxes = np.c_[labels, bboxs]
        makedirs(args.savedir)
        save_name = f"evvid_rmmnist_{train_val_test}_{idx:0>5}"
        if args.events:
            save_name += "_events"
        save_name += ".mp4"
        savepath = Path(args.savedir) / save_name
        if args.events:
            path_events = root / f"{train_val_test}/{idx:0>5}_events.npy"
            events = np.load(path_events)
            events = np.c_[events["t"] / 1e3, events["x"], events["y"], events["p"]]
            save_event_video_moving_mnist_events(
                events,
                meta_info["shape"],
                savepath,
                boxes=boxes,
                box_times=bbox_times,
                t_stop=t_stop,
                playback_speed=args.playback_speed,
                add_boxes_text=args.text,
            )
        else:
            save_event_video_moving_mnist(
                frames,
                frame_times,
                meta_info["shape"],
                savepath,
                boxes=boxes,
                box_times=bbox_times,
                class_names=meta_info["labels"],
                t_stop=t_stop,
                playback_speed=args.playback_speed,
                add_boxes_text=args.text,
            )


if __name__ == "__main__":
    matplotlib.use("agg")
    main()
