// based on https://github.com/uzh-rpg/rpg_esim
// MIT License

#include <iostream>
#include <random>
#include <tuple>
#include <functional>
#include <chrono>

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

// float should be precise enough for input images
// Range is [0, 1], deducted by looking at examples UnrealCV and Planar in ESIM repo
// I.e. log image is in [eps, 0] approx. [-7, 0]
#define ImageFloatType float
// using std:uint64_t allows for 2**64 = 18446744073709551615 nanoseconds, i.e. approx. 5e6 hours of
// recording. This should be fine for all scenarios
using Time = std::uint64_t;
using Duration = std::uint64_t;
using real_t = double;
using Image = Eigen::Array<ImageFloatType, Eigen::Dynamic, Eigen::Dynamic>;
// NEW VERSION Events
typedef Eigen::Matrix<Time, Eigen::Dynamic, 1> TimeVector;
typedef Eigen::Matrix<int, Eigen::Dynamic, 1> CoordVector;
typedef Eigen::Matrix<bool, Eigen::Dynamic, 1> PolVector;
struct Event
{
  Event(Time t, int x, int y, bool pol) : t_(t), x_(x), y_(y), pol_(pol){};
  Time t_;
  int x_;
  int y_;
  bool pol_;
};
// OLD VERSION Events
// Events are sent as double array, precision is reduced
// using Event = Eigen::Vector4d;
// using Events =  Eigen::Array<double, Eigen::Dynamic, 4>;
using EventsVec = std::vector<Event>;

struct Events
{
  Events() {}
  TimeVector getTime() { return times; }
  CoordVector getx() { return xs; }
  CoordVector gety() { return ys; }
  PolVector getPol() { return pols; }
  void setTime(TimeVector t) { times = t; }
  void setx(CoordVector x) { xs = x; }
  void sety(CoordVector y) { ys = y; }
  void setPol(PolVector p) { pols = p; }
  void addEvents(const EventsVec &vec)
  {
    times.conservativeResize(vec.size(), Eigen::NoChange);
    xs.conservativeResize(vec.size(), Eigen::NoChange);
    ys.conservativeResize(vec.size(), Eigen::NoChange);
    pols.conservativeResize(vec.size(), Eigen::NoChange);
    for (int i = 0; i < vec.size(); ++i)
    {
      // std::cout << i << " " << times[i] <<  " " << vec.at(i).t_ << std::endl;
      times[i] = vec.at(i).t_;
      xs[i] = vec.at(i).x_;
      ys[i] = vec.at(i).y_;
      pols[i] = vec.at(i).pol_;
    }
  }

  TimeVector times;
  CoordVector xs;
  CoordVector ys;
  PolVector pols;
};

template <typename T>
T sampleNormalDistribution(
    bool deterministic = false,
    T mean = T{0.0},
    T sigma = T{1.0})
{
  static std::mt19937 gen_nondeterministic(std::random_device{}());
  static std::mt19937 gen_deterministic(0);
  auto dist = std::normal_distribution<T>(mean, sigma);
  return deterministic ? dist(gen_deterministic) : dist(gen_nondeterministic);
}

class UniformSampler
{

public:
  UniformSampler(int seed, double b = 1, double a = 1)
      : seed_(seed),
        a_(a),
        b_(b)
  {
    gen_ = std::mt19937(seed_);
    dis_ = std::uniform_real_distribution<>(a_, b_);
  }

  std::vector<double> sample(int N)
  {
    std::vector<double> samples;
    for (int n = 0; n < N; ++n)
    {
      samples.push_back(dis_(gen_));
    }
    return samples;
  }

private:
  int seed_;
  std::mt19937 gen_;
  double a_ = 0;
  double b_ = 1;
  std::uniform_real_distribution<> dis_;
};

int64_t secToNanosec(real_t seconds)
{
  return static_cast<int64_t>(seconds * 1e9);
}

real_t nanosecToSecTrunc(int64_t nanoseconds)
{
  return static_cast<real_t>(nanoseconds) / 1e9;
}

struct Config
{
  Config()
      : Cp(0.6),
        Cm(0.6),
        sigma_Cp(0),
        sigma_Cm(0),
        refractory_period_ns(100000),
        use_log_image(true),
        log_eps(0.001)
  {
  }
  double Cp;
  double Cm;
  double sigma_Cp;
  double sigma_Cm;
  Duration refractory_period_ns;
  bool use_log_image;
  double log_eps;
};

class EventSimulator
{
public:
  using TimestampImage = Eigen::Matrix<Time, Eigen::Dynamic, Eigen::Dynamic>;

  EventSimulator(const Config &config)
      : config_(config),
        is_initialized_(false),
        current_time_(0)
  {
  }

  void init(const Image &img, Time time);
  Events imageCallback(const Image &img, Time time, Duration delta_t_ns_);

private:
  bool is_initialized_;
  Time current_time_;
  Image ref_values_;
  Image last_img_;
  TimestampImage last_event_timestamp_;
  Eigen::Vector2d size_;

  Config config_;
};

void EventSimulator::init(const Image &img, Time time = 0)
{
  // std::cout << "Initialized event camera simulator with sensor size: " << img.rows() << "x" << img.cols() << std::endl;
  // std::cout << "and contrast thresholds: C+ = " << config_.Cp << " , C- = " << config_.Cm << std::endl;
  is_initialized_ = true;
  // last_img_ = img.clone();
  // ref_values_ = img.clone();
  last_img_ = img;
  ref_values_ = img;
  size_ = Eigen::Vector2d(img.rows(), img.cols());
  last_event_timestamp_ = TimestampImage::Zero(size_(0), size_(1));
  current_time_ = time;
}

Events EventSimulator::imageCallback(const Image &img, Time time, Duration delta_t_ns_ = 0)
{
  // CHECK_GE(time, 0);
  Image preprocessed_img = img;
  if (config_.use_log_image)
  {
    preprocessed_img = (config_.log_eps + img).log();
  }

  if (!is_initialized_)
  {
    init(preprocessed_img, time);
    return {};
  }

  // For each pixel, check if new events need to be generated since the last image sample
  static constexpr ImageFloatType tolerance = 1e-6;
  EventsVec events_vec;
  Duration delta_t_ns;
  delta_t_ns = (delta_t_ns_ > 0) ? delta_t_ns_ : time - current_time_;

  // CHECK_GT(delta_t_ns, 0u);
  // CHECK_EQ(img.size(), size_);

  size_t n_drops = 0;
  for (int y = 0; y < size_(0); ++y)
  {
    for (int x = 0; x < size_(1); ++x)
    {

      ImageFloatType itdt = preprocessed_img(y, x);
      ImageFloatType it = last_img_(y, x);
      ImageFloatType prev_cross = ref_values_(y, x);

      if (std::fabs(it - itdt) > tolerance)
      {
        ImageFloatType pol = (itdt >= it) ? +1.0 : -1.0;
        ImageFloatType C = (pol > 0) ? config_.Cp : config_.Cm;
        ImageFloatType sigma_C = (pol > 0) ? config_.sigma_Cp : config_.sigma_Cm;
        if (sigma_C > 0)
        {
          C += sampleNormalDistribution<ImageFloatType>(false, 0, sigma_C);
          constexpr ImageFloatType minimum_contrast_threshold = 0.01;
          C = std::max(minimum_contrast_threshold, C);
        }
        ImageFloatType curr_cross = prev_cross;
        bool all_crossings = false;

        do
        {
          curr_cross += pol * C;

          if ((pol > 0 && curr_cross > it && curr_cross <= itdt) || (pol < 0 && curr_cross < it && curr_cross >= itdt))
          {
            // here is the reason why we need nanoseconds: Gives the most precise results
            Duration edt = (curr_cross - it) * delta_t_ns / (itdt - it);
            Time t = current_time_ + edt;

            // check that pixel (x,y) is not currently in a "refractory" state
            // i.e. |t-that last_timestamp(x,y)| >= refractory_period
            const Time last_stamp_at_xy = last_event_timestamp_(y, x);
            // CHECK_GE(t, last_stamp_at_xy);
            const Duration dt = t - last_stamp_at_xy;
            if (last_event_timestamp_(y, x) == 0 || dt >= config_.refractory_period_ns)
            {
              events_vec.push_back(Event(t, x, y, pol > 0));
              last_event_timestamp_(y, x) = t;
            }
            else
            {
              ++n_drops;
            }
            ref_values_(y, x) = curr_cross;
          }
          else
          {
            all_crossings = true;
          }
        } while (!all_crossings);
      } // end tolerance

    } // end for each pixel
  }

  // update simvars for next loop
  current_time_ = (delta_t_ns_ > 0) ? current_time_ + delta_t_ns_ : time;
  last_img_ = preprocessed_img;

  // Sort the events by increasing timestamps, since this is what
  // most event processing algorithms expect
  sort(events_vec.begin(), events_vec.end(),
       [](const Event &a, const Event &b) -> bool
       {
         // OLD VERSION Events
         //  return a(0) < b(0);
         return a.t_ < b.t_;
       });

  // OLD VERSION Events
  // Events events;
  // events.conservativeResize(events_vec.size(), Eigen::NoChange);
  // for(int i = 0; i < events_vec.size(); ++i){
  //  events.row(i) = events_vec.at(i);
  //}
  Events events;
  events.addEvents(events_vec);
  return events;
}

Events test_events_return_value()
{
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(0, 1);

  int number = 3;
  auto conf = Config();
  std::tuple<int, int> shape = std::make_tuple(180, 240);
  Image img = Image::NullaryExpr(std::get<0>(shape), std::get<1>(shape), [&]()
                                 { return dis(gen); });
  Time time = 0;        // nanosecs
  Duration dt = 100000; // nanosecs
  auto sim = EventSimulator(conf);
  double total = 0.;
  Events events_last;
  for (size_t ii = 0; ii < number; ++ii)
  {
    auto t1 = std::chrono::high_resolution_clock::now();
    Events events = sim.imageCallback(img, time);
    events_last = events;
    auto t2 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1e6;
    total += duration;
    Image imgnew = Image::NullaryExpr(std::get<0>(shape), std::get<1>(shape), [&]()
                                      { return dis(gen); });
    img = imgnew;
    time = time + dt * ii;
  }
  return events_last;
}

int main()
{
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(0, 1);

  int number = 50;
  auto conf = Config();
  std::tuple<int, int> shape = std::make_tuple(180, 240);
  Image img = Image::NullaryExpr(std::get<0>(shape), std::get<1>(shape), [&]()
                                 { return dis(gen); });
  Time time = 0;        // nanosecs
  Duration dt = 100000; // nanosecs
  auto sim = EventSimulator(conf);
  double total = 0.;
  for (size_t ii = 0; ii < number; ++ii)
  {
    auto t1 = std::chrono::high_resolution_clock::now();
    Events events = sim.imageCallback(img, time, dt);
    auto t2 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1e6;
    std::cout << ii << " " << duration << " r: " << events.times.rows() << std::endl;
    total += duration;
    Image imgnew = Image::NullaryExpr(std::get<0>(shape), std::get<1>(shape), [&]()
                                      { return dis(gen); });
    img = imgnew;
    time = time + dt * ii;
  }
  std::cout << "Executing " << number << " iterations took " << total << " s" << std::endl;
  return 0;
}

// WRAPPER
namespace py = pybind11;

PYBIND11_MODULE(simulator, m)
{
  m.doc() = "Simulate events from frames";

  // m.def("sim_events", &sim_events);
  m.def("main", &main);
  m.def("test_events_return_value", &test_events_return_value);
  py::class_<Config>(m, "Config")
      .def(py::init<>())
      .def_readwrite("Cp", &Config::Cp)
      .def_readwrite("Cm", &Config::Cm)
      .def_readwrite("sigma_Cp", &Config::sigma_Cp)
      .def_readwrite("sigma_Cm", &Config::sigma_Cm)
      .def_readwrite("refractory_period_ns", &Config::refractory_period_ns)
      .def_readwrite("use_log_image", &Config::use_log_image)
      .def_readwrite("log_eps", &Config::log_eps);
  py::class_<EventSimulator>(m, "EventSimulator")
      .def(py::init<const Config &>())
      .def("simulate", &EventSimulator::imageCallback);
  // no specific reason to define getters and setters, could also have used def_readwrite()
  py::class_<Events>(m, "Events")
      .def(py::init<>())
      .def("getTime", &Events::getTime)
      .def("getx", &Events::getx)
      .def("gety", &Events::gety)
      .def("getPol", &Events::getPol)
      .def("setTime", &Events::setTime)
      .def("setx", &Events::setx)
      .def("sety", &Events::sety)
      .def("setPol", &Events::setPol)
      .def_property("times", &Events::getTime, &Events::setTime)
      .def_property("xs", &Events::getx, &Events::setx)
      .def_property("ys", &Events::gety, &Events::sety)
      .def_property("pols", &Events::getPol, &Events::setPol);
}
