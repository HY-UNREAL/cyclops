#include "cyclops/details/measurement/data_queue.hpp"
#include "cyclops/details/measurement/data_provider.hpp"
#include "cyclops/details/measurement/keyframe.hpp"
#include "cyclops/details/measurement/preintegration.hpp"

#include "cyclops/details/estimation/state/accessor_read.hpp"
#include "cyclops/details/utils/vision.hpp"

#include "cyclops/details/config.hpp"
#include "cyclops/details/logging.hpp"

#include <spdlog/spdlog.h>
#include <range/v3/all.hpp>

namespace cyclops::measurement {
  using std::set;

  namespace views = ranges::views;

  class MeasurementDataQueueImpl: public MeasurementDataQueue {
  private:
    std::shared_ptr<CyclopsConfig const> _config;
    std::shared_ptr<MeasurementDataProvider> _data_provider;
    std::shared_ptr<KeyframeManager> _keyframe_manager;
    std::shared_ptr<estimation::StateVariableReadAccessor const> _state;

    std::map<Timestamp, ImuData> _imu_history;

    std::unique_ptr<ImuPreintegration> popImuPreintegration(
      Timestamp start_time, Timestamp end_time, Eigen::Vector3d const& bias_acc,
      Eigen::Vector3d const& bias_gyr);
    std::optional<FrameID> queryFrame(Timestamp timestamp, double max_dt) const;

    std::map<FrameID, Timestamp> allFrames() const;

  public:
    MeasurementDataQueueImpl(
      std::shared_ptr<CyclopsConfig const> config,
      std::shared_ptr<MeasurementDataProvider> measurements,
      std::shared_ptr<KeyframeManager> keyframe_manager,
      std::shared_ptr<estimation::StateVariableReadAccessor const> state);
    ~MeasurementDataQueueImpl();
    void reset() override;

    void updateImu(ImuData const&) override;
    std::optional<FrameID> updateLandmark(ImageData const&) override;

    bool detectKeyframe(FrameID candidate_frame) const override;
    void acceptCurrentPendingKeyframe() override;

    void marginalize(FrameID drop_frame) override;
    void marginalizeKeyframe(
      FrameID drop_frame, set<LandmarkID> const& drop_landmarks,
      FrameID inserted_keyframe) override;
    void marginalizePendingFrame(
      FrameID drop_frame, set<LandmarkID> const& drop_landmarks) override;

    std::map<FrameID, Timestamp> const& keyframes() const override;
    std::map<FrameID, Timestamp> const& pendingFrames() const override;
  };

  MeasurementDataQueueImpl::~MeasurementDataQueueImpl() = default;

  MeasurementDataQueueImpl::MeasurementDataQueueImpl(
    std::shared_ptr<CyclopsConfig const> config,
    std::shared_ptr<MeasurementDataProvider> measurements,
    std::shared_ptr<KeyframeManager> keyframe_manager,
    std::shared_ptr<estimation::StateVariableReadAccessor const> state)
      : _config(config),
        _data_provider(measurements),
        _keyframe_manager(keyframe_manager),
        _state(state) {
  }

  void MeasurementDataQueueImpl::reset() {
    _imu_history.clear();
    _data_provider->reset();
    _keyframe_manager->reset();
  }

  std::map<FrameID, Timestamp> MeasurementDataQueueImpl::allFrames() const {
    return views::concat(keyframes(), pendingFrames()) |
      ranges::to<std::map<FrameID, Timestamp>>;
  }

  ImuPreintegration::UniquePtr MeasurementDataQueueImpl::popImuPreintegration(
    Timestamp t_s, Timestamp t_e, Eigen::Vector3d const& b_a,
    Eigen::Vector3d const& b_w) {
    auto result = std::make_unique<ImuPreintegration>(
      b_a, b_w,
      ImuNoise {
        .acc_white_noise = _config->noise.acc_white_noise,
        .gyr_white_noise = _config->noise.gyr_white_noise,
      });

    if (_imu_history.empty()) {
      __logger__->warn(
        "Empty IMU history queue while evaluating IMU preintegration");
      return nullptr;
    }

    auto s = _imu_history.upper_bound(t_s);
    auto e = _imu_history.upper_bound(t_e);
    __logger__->info(
      "Updating keyframe; #IMU data points = {}", std::distance(s, e));

    if (e == _imu_history.end()) {
      __logger__->info("Incomplete IMU data while updating keyframe");
      return nullptr;
    }

    for (auto j = s; j != e; j++) {
      auto i = j == _imu_history.begin() ? j : std::prev(j);
      auto const& [t1, prev] = *i;
      auto const& [t2, curr] = *j;
      auto dt = t2 - t1;
      auto a_hat = ((prev.accel + curr.accel) / 2).eval();
      auto w_hat = ((prev.rotat + curr.rotat) / 2).eval();
      result->propagate(dt, a_hat, w_hat);
    }

    if (result->time_delta == 0)
      return nullptr;

    if (e != _imu_history.begin())
      _imu_history.erase(_imu_history.begin(), std::prev(e));
    return result;
  }

  void MeasurementDataQueueImpl::updateImu(ImuData const& imu) {
    _imu_history.emplace(imu.timestamp, imu);
  }

  std::optional<FrameID> MeasurementDataQueueImpl::updateLandmark(
    ImageData const& landmark) {
    auto frames = allFrames();
    if (frames.empty()) {
      auto frame_id = _keyframe_manager->createNewFrame(landmark.timestamp);
      _data_provider->updateFrame(frame_id, landmark);
      _keyframe_manager->setKeyframe(frame_id);
      return frame_id;
    }
    auto [prev_frame_id, prev_t_] = *frames.rbegin();

    auto t_delay = _config->extrinsics.imu_camera_time_delay;
    auto prev_t = prev_t_ - t_delay;
    auto curr_t = landmark.timestamp - t_delay;

    auto zero = Eigen::Vector3d::Zero().eval();
    auto b_a = _state->motionFrames().empty()
      ? zero
      : estimation::getAccBias(_state->lastMotionFrameBlock());
    auto b_w = _state->motionFrames().empty()
      ? zero
      : estimation::getGyrBias(_state->lastMotionFrameBlock());

    auto preintegration = popImuPreintegration(prev_t, curr_t, b_a, b_w);
    if (preintegration == nullptr) {
      __logger__->warn(
        "Empty IMU data after received landmark keyframe. Skipping update...");
      return std::nullopt;
    }
    auto curr_frame_id = _keyframe_manager->createNewFrame(landmark.timestamp);
    _data_provider->updateFrame(
      prev_frame_id, curr_frame_id, landmark, std::move(preintegration));

    // we simply treat all motion frames as a keyframe during initialization.
    if (_state->motionFrames().empty())
      _keyframe_manager->setKeyframe(curr_frame_id);

    // if keyframes are empty, force set this frame to a keyframe.
    if (keyframes().empty())
      _keyframe_manager->setKeyframe(curr_frame_id);

    return curr_frame_id;
  }

  std::optional<FrameID> MeasurementDataQueueImpl::queryFrame(
    Timestamp timestamp, double max_dt) const {
    auto frames = allFrames();
    if (frames.empty())
      return std::nullopt;

    auto time_sorted_frames =  //
      frames | views::transform([](auto const& _) {
        auto const& [frame_id, frame_timestamp] = _;
        return std::make_pair(frame_timestamp, frame_id);
      }) |
      ranges::to<std::map<Timestamp, FrameID>>;

    auto e = time_sorted_frames.upper_bound(timestamp);
    auto s = e;
    if (s != time_sorted_frames.begin())
      s = std::prev(s);

    auto [t_s, f_s] = *s;
    if (e == time_sorted_frames.end()) {
      if (std::abs(t_s - timestamp) < max_dt)
        return f_s;
      return std::nullopt;
    }
    auto [t_e, f_e] = *e;

    auto dt_s = std::abs(t_s - timestamp);
    auto dt_e = std::abs(t_e - timestamp);
    if (dt_s < dt_e && dt_s < max_dt)
      return f_s;
    if (dt_e < dt_s && dt_e < max_dt)
      return f_e;
    return std::nullopt;
  }

  bool MeasurementDataQueueImpl::detectKeyframe(FrameID candidate_frame) const {
    auto const& tracks = _data_provider->tracks();
    if (keyframes().empty())
      return false;

    auto [last_keyframe, _] = *keyframes().rbegin();

    auto motion_statistics =
      evaluateKeyframeMotionStatistics(tracks, last_keyframe, candidate_frame);

    auto const& threshold = _config->keyframe_detection;
    auto min_novel_landmarks = threshold.min_novel_landmarks;
    auto min_average_parallax = threshold.min_average_parallax;

    return motion_statistics.new_features > min_novel_landmarks ||
      motion_statistics.average_parallax > min_average_parallax;
  }

  void MeasurementDataQueueImpl::acceptCurrentPendingKeyframe() {
    if (pendingFrames().empty())
      return;

    auto [curr_pending_frame, _] = *pendingFrames().begin();
    _keyframe_manager->setKeyframe(curr_pending_frame);
  }

  void MeasurementDataQueueImpl::marginalize(FrameID drop_frame) {
    auto drop_landmarks =  //
      _data_provider->tracks() | views::filter([&](auto const& id_track) {
        auto const& [_, track] = id_track;
        if (track.empty())
          return true;
        auto last_observation_frame = track.rbegin()->first;
        return last_observation_frame <= drop_frame;
      }) |
      views::keys | ranges::to<set>;

    _data_provider->marginalize(drop_frame, drop_landmarks);
    _keyframe_manager->removeFrame(drop_frame);
  }

  void MeasurementDataQueueImpl::marginalizeKeyframe(
    FrameID drop_frame, set<LandmarkID> const& drop_landmarks,
    FrameID new_keyframe) {
    _data_provider->marginalize(drop_frame, drop_landmarks);
    _keyframe_manager->removeFrame(drop_frame);
    _keyframe_manager->setKeyframe(new_keyframe);
  }

  void MeasurementDataQueueImpl::marginalizePendingFrame(
    FrameID drop_frame, set<LandmarkID> const& drop_landmarks) {
    _data_provider->marginalize(drop_frame, drop_landmarks);
    _keyframe_manager->removeFrame(drop_frame);
  }

  std::map<FrameID, Timestamp> const& MeasurementDataQueueImpl::keyframes()
    const {
    return _keyframe_manager->keyframes();
  }

  std::map<FrameID, Timestamp> const& MeasurementDataQueueImpl::pendingFrames()
    const {
    return _keyframe_manager->pendingFrames();
  }

  std::unique_ptr<MeasurementDataQueue> MeasurementDataQueue::Create(
    std::shared_ptr<CyclopsConfig const> config,
    std::shared_ptr<MeasurementDataProvider> measurements,
    std::shared_ptr<KeyframeManager> keyframe_manager,
    std::shared_ptr<estimation::StateVariableReadAccessor const> state) {
    return std::make_unique<MeasurementDataQueueImpl>(
      config, measurements, keyframe_manager, state);
  }
}  // namespace cyclops::measurement
