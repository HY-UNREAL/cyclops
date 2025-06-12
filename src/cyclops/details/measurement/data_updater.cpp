#include "cyclops/details/measurement/data_updater.hpp"
#include "cyclops/details/measurement/data_queue.hpp"
#include "cyclops/details/estimation/propagation.hpp"
#include "cyclops/details/estimation/state/accessor_read.hpp"
#include "cyclops/details/config.hpp"

#include <range/v3/all.hpp>

namespace cyclops::measurement {
  using estimation::ImuPropagationUpdateHandler;
  using estimation::StateVariableReadAccessor;

  class MeasurementDataUpdaterImpl: public MeasurementDataUpdater {
  private:
    std::shared_ptr<CyclopsConfig const> _config;
    std::shared_ptr<MeasurementDataQueue> _measurement_queue;
    std::shared_ptr<ImuPropagationUpdateHandler> _propagator;
    std::shared_ptr<StateVariableReadAccessor const> _state_reader;

    double _landmark_dt_sum = 0;
    std::deque<Timestamp> _landmark_update_timestamps;

    bool throttleLandmarkUpdateFps(Timestamp timestamp);

  public:
    MeasurementDataUpdaterImpl(
      std::shared_ptr<CyclopsConfig const> config,
      std::shared_ptr<MeasurementDataQueue> measurement_queue,
      std::shared_ptr<ImuPropagationUpdateHandler> propagator,
      std::shared_ptr<StateVariableReadAccessor const> state_reader);
    ~MeasurementDataUpdaterImpl();
    void reset() override;

    void updateImu(ImuData const& data) override;

    std::optional<FrameID> updateLandmark(ImageData const& data) override;
    void repropagate(FrameID last_frame, Timestamp timestamp) override;

    std::map<FrameID, Timestamp> frames() const override;
  };

  MeasurementDataUpdaterImpl::MeasurementDataUpdaterImpl(
    std::shared_ptr<CyclopsConfig const> config,
    std::shared_ptr<MeasurementDataQueue> measurement_queue,
    std::shared_ptr<ImuPropagationUpdateHandler> propagator,
    std::shared_ptr<StateVariableReadAccessor const> state_reader)
      : _config(config),
        _measurement_queue(measurement_queue),
        _propagator(propagator),
        _state_reader(state_reader) {
  }

  MeasurementDataUpdaterImpl::~MeasurementDataUpdaterImpl() = default;

  bool MeasurementDataUpdaterImpl::throttleLandmarkUpdateFps(
    Timestamp timestamp) {
    auto const& throttle_config = _config->update_throttling;

    if (_landmark_update_timestamps.empty()) {
      _landmark_update_timestamps.emplace_back(timestamp);
      return true;
    }

    auto dt = timestamp - _landmark_update_timestamps.back();
    auto dt_sum = _landmark_dt_sum + dt;
    auto dt_avg = dt_sum / _landmark_update_timestamps.size();
    auto dt_target = 1 / throttle_config.update_rate_target;

    if (dt_avg <= dt_target)
      return false;

    _landmark_dt_sum = dt_sum;
    _landmark_update_timestamps.emplace_back(timestamp);

    while (!_landmark_update_timestamps.empty()) {
      auto t_i = _landmark_update_timestamps.front();
      auto t_f = _landmark_update_timestamps.back();

      auto windowsize = t_f - t_i;
      auto max_windowsize = throttle_config.update_rate_smoothing_window_size;
      if (windowsize <= max_windowsize)
        break;

      auto t1 = _landmark_update_timestamps.at(0);
      auto t2 = _landmark_update_timestamps.size() == 1
        ? _landmark_update_timestamps.at(0)
        : _landmark_update_timestamps.at(1);
      auto dt_pop = t2 - t1;

      _landmark_dt_sum -= dt_pop;
      _landmark_update_timestamps.pop_front();
    }
    return true;
  }

  void MeasurementDataUpdaterImpl::updateImu(ImuData const& data) {
    _measurement_queue->updateImu(data);
    _propagator->updateImuData(data);
  }

  std::optional<FrameID> MeasurementDataUpdaterImpl::updateLandmark(
    ImageData const& data) {
    if (!throttleLandmarkUpdateFps(data.timestamp))
      return std::nullopt;
    return _measurement_queue->updateLandmark(data);
  }

  void MeasurementDataUpdaterImpl::repropagate(
    FrameID last_frame_id, Timestamp timestamp) {
    auto maybe_state_block = _state_reader->motionFrame(last_frame_id);
    if (!maybe_state_block)
      return;
    _propagator->updateOptimization(timestamp, *maybe_state_block);
  }

  std::map<FrameID, Timestamp> MeasurementDataUpdaterImpl::frames() const {
    namespace views = ranges::views;

    auto const& keyframes = _measurement_queue->keyframes();
    auto const& pending_frames = _measurement_queue->pendingFrames();
    return views::concat(keyframes, views::all(pending_frames)) |
      ranges::to<std::map<FrameID, Timestamp>>;
  }

  void MeasurementDataUpdaterImpl::reset() {
    _measurement_queue->reset();
    _propagator->reset();
  }

  std::unique_ptr<MeasurementDataUpdater> MeasurementDataUpdater::Create(
    std::shared_ptr<CyclopsConfig const> config,
    std::shared_ptr<MeasurementDataQueue> measurement_queue,
    std::shared_ptr<ImuPropagationUpdateHandler> propagator,
    std::shared_ptr<StateVariableReadAccessor const> state_reader) {
    return std::make_unique<MeasurementDataUpdaterImpl>(
      config, measurement_queue, propagator, state_reader);
  }
}  // namespace cyclops::measurement
