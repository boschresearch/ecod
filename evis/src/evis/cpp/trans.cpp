// Copyright (c) 2022 Robert Bosch GmbH
// SPDX-License-Identifier: AGPL-3.0


#include <iostream>

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

#define ImageFloatType float
using Duration = std::uint64_t;
using real_t = double;
using Image = Eigen::Array<ImageFloatType, Eigen::Dynamic, Eigen::Dynamic>;
using Event = Eigen::Vector4d;
using Events = Eigen::Array<double, Eigen::Dynamic, 4>;
using Boxes = Eigen::Array<double, Eigen::Dynamic, 4>;
using EventsVec = std::vector<Event>;

bool event_negative(int polarity)
{
  return (polarity == 0 || polarity == -1);
}

bool img_neq_0(std::vector<Image> image_2c)
{
  return (image_2c.at(0).sum() != 0 || image_2c.at(1).sum() != 0);
}

int count_events_in_box(const Events &events, int x_start, int x_stop, int y_start, int y_stop,
                        real_t t_start_mus = -1, real_t t_stop_mus = -1)
{

  int count = 0;
  for (size_t idx = 0; idx < events.rows(); ++idx)
  {
    if (events(idx, 0) > t_stop_mus)
    {
      break;
    }
    if (events(idx, 0) > t_start_mus)
    {
      if (events(idx, 1) >= x_start && events(idx, 1) < x_stop)
      {
        if (events(idx, 2) >= y_start && events(idx, 2) < y_stop)
        {
          count++;
        }
      }
    }
  }
  return count;
}

py::array_t<int> count_events_in_boxes(const Events &events, const Boxes &boxes)
{

  auto counts = py::array_t<int>(boxes.rows());
  auto counts_it = counts.mutable_unchecked<1>();
  for (py::ssize_t i = 0; i < counts_it.shape(0); i++)
  {
    counts_it(i) = 0;
  }
  for (size_t idx = 0; idx < events.rows(); ++idx)
  {
    for (py::ssize_t i = 0; i < counts_it.shape(0); i++)
    {
      if (events(idx, 1) >= boxes(i, 0) && events(idx, 1) < boxes(i, 2))
      {
        if (events(idx, 2) >= boxes(i, 1) && events(idx, 2) < boxes(i, 3))
        {
          counts_it(i) += 1;
        }
      }
    }
  }
  return counts;
}

py::array_t<ImageFloatType> zeros_frames(int n_frames, int channels, int n_rows, int n_cols)
{
  auto result = py::array_t<ImageFloatType>({n_frames, channels, n_rows, n_cols});
  auto r = result.mutable_unchecked<4>();
  for (py::ssize_t i = 0; i < r.shape(0); i++)
    for (py::ssize_t j = 0; j < r.shape(1); j++)
      for (py::ssize_t k = 0; k < r.shape(2); k++)
        for (py::ssize_t l = 0; l < r.shape(3); l++)
          r(i, j, k, l) = 0.0;
  return result;
}

std::vector<size_t> time_idxs_from_events(Events events, double fps)
{
  auto times_mus = events.leftCols(1);
  real_t next_time_mus = 0;
  std::vector<size_t> idxs;
  for (size_t idx = 0; idx < events.rows(); ++idx)
  {
    if (times_mus(idx) >= next_time_mus)
    {
      idxs.push_back(idx);
      next_time_mus += 1e6 / fps; // times are in microsec
    }
  }
  return idxs;
}

int idx_from_time(Events events, real_t time)
{
  auto times_eigen = events.col(0);
  auto idx_pointer = std::lower_bound(times_eigen.data(), times_eigen.data() + times_eigen.rows(), time);
  int idx = idx_pointer - times_eigen.data();
  return idx;
}

py::array_t<ImageFloatType> events_to_n_frames_2c(const Events &events, int n_frames, int n_cols, int n_rows,
                                                  real_t t_start_mus = -1, real_t t_end_mus = -1, bool binary = false)
{
  // std::vector<std::vector<Image> >
  // be careful to not cause a memory error (80000 images at 32 GB)
  // std::cout << images.size() << std::endl;
  real_t t_start_events = events(0, 0);
  real_t t_end_events = events(events.rows() - 1, 0);
  int idx_start;
  if (t_start_mus >= 0)
  {
    idx_start = idx_from_time(events, t_start_mus);
  }
  else
  {
    idx_start = 0;
  }
  if (t_start_mus < 0)
  {
    t_start_mus = t_start_events;
  }
  if (t_end_mus < 0)
  {
    t_end_mus = t_end_events;
  }
  real_t t_step = (t_end_mus - t_start_mus) / static_cast<double>(n_frames);
  real_t next_time_mus = t_start_mus + t_step;
  int img_idx = 0;
  // std::vector<Image> image_2c(2, Image::Zero(n_cols, n_rows));
  // std::vector<std::vector<Image> > images(n_frames, image_2c);
  auto result = zeros_frames(n_frames, 2, n_rows, n_cols);
  auto r = result.mutable_unchecked<4>();
  // std::cout << *result.shape() << std::endl;

  // std::cout << images.size() << "; " << idx_start << "; " << events.rows() << std::endl;
  for (size_t idx = idx_start; idx < events.rows(); ++idx)
  {
    while (events(idx, 0) >= next_time_mus)
    {
      if (next_time_mus + t_step > t_end_mus)
      {
        break;
      }
      next_time_mus += t_step;
      ++img_idx;
    }
    // here
    if (events(idx, 0) > t_end_mus || img_idx >= n_frames)
    {
      break;
    }
    if (events(idx, 3) == 1)
    {
      // images.at(img_idx).at(0)(events(idx, 2), events(idx, 1)) += 1.;
      if (binary)
      {
        r(img_idx, 0, events(idx, 2), events(idx, 1)) = 1;
      }
      else
      {
        r(img_idx, 0, events(idx, 2), events(idx, 1)) += 1;
      }
    }
    else
    {
      // images.at(img_idx).at(1)(events(idx, 2), events(idx, 1)) += 1.;
      if (binary)
      {
        r(img_idx, 1, events(idx, 2), events(idx, 1)) = 1;
      }
      else
      {
        r(img_idx, 1, events(idx, 2), events(idx, 1)) += 1;
      }
    }
  }
  return result;
}

std::vector<std::vector<Image>> events_to_frames_2c_fps(const Events &events, double fps, int n_cols, int n_rows,
                                                        real_t t_start_mus = -1, real_t t_end_mus = -1)
{
  // return_last returns the last frame even if it is not complete
  // be careful to not cause a memory error (80000 images at 32 GB)
  // std::cout << images.size() << std::endl;
  real_t t_start_events = events(0, 0);
  real_t t_end_events = events(events.rows() - 1, 0);
  int idx_start;
  if (t_start_mus >= 0)
  {
    idx_start = idx_from_time(events, t_start_mus);
  }
  else
  {
    idx_start = 0;
  }
  if (t_start_mus < 0)
  {
    t_start_mus = t_start_events;
  }
  if (t_end_mus < 0)
  {
    t_end_mus = t_end_events;
  }
  real_t t_step = 1e6 / fps;
  real_t next_time_mus = t_start_mus + t_step;
  std::vector<Image> image_2c(2, Image::Zero(n_cols, n_rows));
  std::vector<std::vector<Image>> images;
  for (size_t idx = idx_start; idx < events.rows(); ++idx)
  {
    while (events(idx, 0) >= next_time_mus)
    {
      if (next_time_mus + t_step > t_end_mus)
      {
        break;
      }
      next_time_mus += t_step; // times are in microsec
      images.push_back(image_2c);
      image_2c.at(0) = Image::Zero(n_cols, n_rows);
      image_2c.at(1) = Image::Zero(n_cols, n_rows);
    }
    if (events(idx, 0) > t_end_mus)
    {
      break;
    }
    if (events(idx, 3) == 1)
    {
      image_2c.at(0)(events(idx, 2), events(idx, 1)) += 1.;
    }
    else if (event_negative(events(idx, 3)))
    {
      image_2c.at(1)(events(idx, 2), events(idx, 1)) += 1.;
    }
  }
  // push empty frames to end to fill up in case t_end > t_end_events
  while (next_time_mus <= t_end_mus)
  {
    next_time_mus += t_step; // times are in microsec
    images.push_back(image_2c);
    image_2c.at(0) = Image::Zero(n_cols, n_rows);
    image_2c.at(1) = Image::Zero(n_cols, n_rows);
  }
  return images;
}

std::vector<Image> events_to_frame_2c(const Events &events, int n_cols, int n_rows)
{
  auto times_mus = events.leftCols(1);
  std::vector<Image> frame;
  frame.push_back(Image::Zero(n_cols, n_rows));
  frame.push_back(Image::Zero(n_cols, n_rows));
  for (size_t idx = 0; idx < events.rows(); ++idx)
  {
    if (events(idx, 3) == 1)
    {
      frame.at(0)(events(idx, 2), events(idx, 1)) += 1.;
    }
    else if (event_negative(events(idx, 3)))
    {
      frame.at(1)(events(idx, 2), events(idx, 1)) += 1.;
    }
  }
  return frame;
}

std::vector<Image> events_to_frames_1c_fps(const Events &events, double fps, int n_cols, int n_rows, bool return_last = false)
{
  auto times_mus = events.leftCols(1);
  real_t next_time_mus = times_mus(0) + 1e6 / fps;
  Image img = Image::Zero(n_cols, n_rows);
  std::vector<Image> images;
  for (size_t idx = 0; idx < events.rows(); ++idx)
  {
    if (times_mus(idx) >= next_time_mus)
    {
      next_time_mus += 1e6 / fps; // times are in microsec
      images.push_back(img);
      img = Image::Zero(n_cols, n_rows);
    }
    if (events(idx, 3) == 1)
    {
      img(events(idx, 2), events(idx, 1)) += 1.;
    }
    else if (event_negative(events(idx, 3)))
    {
      img(events(idx, 2), events(idx, 1)) -= 1.;
    }
  }
  if (return_last and img.sum() != 0)
  {
    images.push_back(img);
  }
  return images;
}

Image events_to_frame_1c(const Events &events, int n_cols, int n_rows)
{
  Image img = Image::Zero(n_cols, n_rows);
  for (size_t idx = 0; idx < events.rows(); ++idx)
  {
    if (events(idx, 3) == 1)
    {
      img(events(idx, 2), events(idx, 1)) += 1.;
    }
    else if (event_negative(events(idx, 3)))
    {
      img(events(idx, 2), events(idx, 1)) -= 1.;
    }
  }
  return img;
}

int test_events_to_frames()
{
  double fps = 30;
  auto events = Events(5, 4);
  events << 0., 2., 3., 1.,
      10000., 0., 0., 1.,
      30000., 0., 5., -1.,
      60000., 1., 5., -1.,
      70000., 1., 9., -1.;
  events_to_frames_1c_fps(events, fps, 10, 10);
  return 0;
}

// WRAPPER
namespace py = pybind11;

PYBIND11_MODULE(trans, m)
{
  m.doc() = "Transform event camera data to frames, count events.";
  m.def("test_events_to_frames", &test_events_to_frames);
  m.def("events_to_n_frames_2c", &events_to_n_frames_2c);
  m.def("events_to_frames_2c_fps", &events_to_frames_2c_fps);
  m.def("events_to_frame_2c", &events_to_frame_2c);
  m.def("events_to_frames_1c_fps", &events_to_frames_1c_fps);
  m.def("events_to_frame_1c", &events_to_frame_1c);
  m.def("count_events_in_box", &count_events_in_box);
  m.def("count_events_in_boxes", &count_events_in_boxes);
}
