# Copyright (c) 2022 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

from pathlib import Path
import numpy as np
from imageio import imread
import matplotlib.pyplot as plt
import io
import imageio

from ecod.events.events import events_to_frames
from ecod.data.proph1mpx.toolbox.psee_loader import PSEELoader
from ecod.data.proph1mpx.load import load_boxes, proph_1mpx_boxes_to_array, trim_boxes, box_path_from_events_path
from ecod.plot.events import ev_frames_to_quad, imshow_ev_quad
from ecod.utils.data import get_dataset_attributes
from ecod.utils.general import ProgressLogger


class FigToArray:
    def __init__(self):
        self.buffer = io.BytesIO()

    def __call__(self, fig):
        self.buffer.seek(0)
        fig.savefig(self.buffer, format="jpeg")
        self.buffer.seek(0)
        array = np.array(imread(self.buffer), dtype=np.uint8)
        return array

    def close(self):
        self.buffer.close()

    def __del__(self):
        self.close()


class FigToCVImageConverter:
    def __init__(self):
        self.fig2array = FigToArray()

    def __call__(self, fig):
        array = self.fig2array(fig)
        # works only with format jpeg, png fails silently
        import cv2

        im = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
        return im

    def close(self):
        self.fig2array.close()

    def __del__(self):
        self.close()


class FigVideoWriter:
    def __init__(self, filepath, fps, height, width, dpi=100.0, size_inch=0.05, save_frames=False):
        self.filepath = filepath
        self.fps = fps
        self.height = height
        self.width = width
        self.save_frames = save_frames
        self.frame_shape = (int(size_inch * height * dpi), int(size_inch * width * dpi))
        self.video_writer = imageio.get_writer(str(filepath), format="FFMPEG", fps=fps)
        self.fig, self.ax = plt.subplots(figsize=(size_inch * width, size_inch * height), dpi=dpi)
        self.fig.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=1.0, wspace=0.0, hspace=0.0)
        self.fig2im = FigToArray()
        self.frames = []

    def write(self):
        im = self.fig2im(self.fig)
        self.video_writer.append_data(im)
        if self.save_frames:
            self.frames.append(im)

    def reset_frame(self):
        self.ax.cla()
        self.ax.axis("off")

    def close(self):
        self.video_writer.close()
        self.fig2im.close()
        if self.save_frames:
            pp = Path(self.filepath)
            savepath = pp.parent / f"frames_{pp.with_suffix('').name}.npz"
            np.savez_compressed(savepath, np.stack(self.frames))
        plt.close(self.fig)


def filter_small_boxes(boxes, min_size):
    boxes = boxes.copy()
    boxes = boxes[boxes["w"] > min_size]
    boxes = boxes[boxes["h"] > min_size]
    return boxes


class BoxSelector:
    def __init__(self, filepath, shape, min_size=0):
        box_handle = PSEELoader(box_path_from_events_path(filepath, bbox_suffix="filtered"))
        # tl_x, tl_y, width, height
        # [('t', '<u8'), ('x', '<f4'), ('y', '<f4'), ('w', '<f4'), ('h', '<f4'), ('class_id', 'u1'),
        #  ('class_confidence', '<f4'), ('track_id', '<u4')]
        self.boxes = filter_small_boxes(trim_boxes(load_boxes(box_handle), shape), min_size)
        self.box_times = np.unique(self.boxes["t"])
        last = len(self.box_times)
        self.box_locs = np.insert(np.searchsorted(self.boxes["t"], self.box_times), last, last)
        self.delta_t_boxes = np.diff(self.box_times).min()
        self.cur_box_time_idx = 0
        self.box_5c = np.zeros((0, 5))
        self.crossed = True

    def get_boxes(self, time):
        while (
            self.cur_box_time_idx < len(self.box_times)
            and self.box_times[self.cur_box_time_idx] + self.delta_t_boxes < time
        ):
            self.cur_box_time_idx += 1
            self.box_5c = np.zeros((0, 5))
            self.crossed = False
        # only update on new time
        if not self.crossed and (
            self.cur_box_time_idx < len(self.box_times)
            and time > self.box_times[self.cur_box_time_idx] - self.delta_t_boxes
        ):
            self.crossed = True
            start, stop = (
                self.box_locs[self.cur_box_time_idx],
                self.box_locs[self.cur_box_time_idx + 1],
            )
            boxes_this = self.boxes[start:stop]
            bboxs, labels = proph_1mpx_boxes_to_array(boxes_this)
            self.box_5c = np.c_[labels, bboxs]
        return self.box_5c


def save_event_video(
    filepath,
    savepath,
    t_start=0.0,
    t_stop=60.0,
    playback_speed=1.0,
    fps=30.0,
    dpi=100.0,
    size_inch=0.01,
    n_frames_buffer=1,
    with_boxes=True,
    force_limits=True,
    add_time=True,
    save_frames=False,
):
    events_handle = PSEELoader(str(filepath))
    events_handle.seek_time(t_start * 1e6)
    scaled_fps = fps / playback_speed
    delta_t = 1e6 / scaled_fps
    n_frames_tot = int((t_stop - t_start) * scaled_fps)
    attrs = get_dataset_attributes("proph_1mpx")
    shape = attrs["shape_max"]
    # -1 because classes account for background
    n_classes = attrs["n_classes"] - 1
    writer = FigVideoWriter(str(savepath), fps, shape[0], shape[1], dpi, size_inch, save_frames=save_frames)
    box_selector = BoxSelector(filepath, shape)

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
        events = events_handle.load_delta_t(delta_t_buffer)
        events = np.c_[events["t"], events["x"], events["y"], events["p"]].astype(np.float32)
        frames = events_to_frames(
            events,
            shape,
            n_frames=n_frames_buffer,
            t_start=t_now,
            t_end=t_now + delta_t_buffer,
            binary_spikes=True,
        )
        frames = ev_frames_to_quad(frames)
        for jj, frame in enumerate(frames):
            writer.reset_frame()
            # it's faster to reset the frame than re-writing im.array!
            im = imshow_ev_quad(frame, writer.ax, im=None)
            if with_boxes:
                box_5c = box_selector.get_boxes(t_vid)
                add_boxes(
                    box_5c,
                    n_classes,
                    writer.ax,
                    text=True,
                    names=attrs["class_names"][1:],
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
            if add_time:
                writer.ax.text(
                    0.96,
                    0.96,
                    f"{t_vid/1e3:.3f} ms",
                    horizontalalignment="right",
                    verticalalignment="top",
                    transform=writer.ax.transAxes,
                )
            if n_frames_buffer == 1:
                n_events = len(events)
                writer.ax.text(
                    0.96,
                    0.93,
                    f"{n_events:.0f} Ev",
                    horizontalalignment="right",
                    verticalalignment="top",
                    transform=writer.ax.transAxes,
                )
                writer.ax.text(
                    0.96,
                    0.90,
                    f"{n_events / delta_t *1e6:.0f} Ev/s",
                    horizontalalignment="right",
                    verticalalignment="top",
                    transform=writer.ax.transAxes,
                )
            writer.write()
            t_vid += delta_t
        t_now += delta_t_buffer
        if events_handle.done and ii < rr - 1:
            print("Finished early because no more events to read")
            break
    writer.close()
