#include "cyclops_tests/mockups/data_provider.hpp"
#include "cyclops_tests/data/imu.hpp"
#include "cyclops_tests/data/landmark.hpp"

#include "cyclops/details/measurement/preintegration.hpp"

namespace cyclops::measurement {
  using Mockup = MeasurementDataProviderMockup;

  using Eigen::Matrix2d;
  using Eigen::Vector3d;

  void Mockup::reset() {
    _imu.clear();
    _tracks.clear();
  }

  void Mockup::updateFrame(FrameID id, ImageData const&) {
    _updated_frames.insert(id);
  }

  void Mockup::updateFrame(
    FrameID prev_frame, FrameID curr_frame, ImageData const&,
    ImuPreintegration::UniquePtr) {
    _updated_frames.insert(curr_frame);
    _updated_imu_frames.emplace(std::make_tuple(prev_frame, curr_frame));
  }

  void Mockup::updateImuBias() {
    _bias_update_count++;
  }

  void Mockup::updateImuBias(
    Vector3d const& bias_acc, Vector3d const& bias_gyr) {
    _bias_update_count++;
  }

  void Mockup::marginalize(
    FrameID drop_frame, std::set<LandmarkID> const& drop_landmarks) {
    _dropped_frames.insert(drop_frame);
    _dropped_landmarks.insert(drop_landmarks.begin(), drop_landmarks.end());
    _marginalization_count++;
  }

  ImuMotions const& Mockup::imu() const {
    return _imu;
  }

  FeatureTracks const& Mockup::tracks() const {
    return _tracks;
  }

  std::unique_ptr<MeasurementDataProviderMockup> makeMeasurementProviderMockup(
    PoseSignal pose_signal, SE3Transform const& extrinsic,
    LandmarkPositions const& landmarks,
    std::map<FrameID, Timestamp> const& frames) {
    auto imu_motions =
      makeImuMotions(Vector3d::Zero(), Vector3d::Zero(), pose_signal, frames);
    auto tracks = makeLandmarkTracks(pose_signal, extrinsic, landmarks, frames);

    auto result = std::make_unique<Mockup>();
    result->_imu = std::move(imu_motions);
    result->_tracks = tracks;
    return result;
  }
}  // namespace cyclops::measurement
