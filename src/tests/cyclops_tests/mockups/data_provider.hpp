#pragma once

#include "cyclops/details/measurement/data_provider.hpp"
#include "cyclops/details/measurement/type.hpp"

#include "cyclops_tests/signal.hpp"

#include <map>
#include <memory>
#include <random>
#include <set>
#include <tuple>

namespace cyclops {
  struct SensorStatistics;
  struct SE3Transform;
}  // namespace cyclops

namespace cyclops::measurement {
  struct ImuPreintegration;

  struct MeasurementDataProviderMockup: public MeasurementDataProvider {
    ImuMotions _imu;
    FeatureTracks _tracks;

    std::set<FrameID> _updated_frames;
    std::set<std::tuple<FrameID, FrameID>> _updated_imu_frames;
    size_t _bias_update_count = 0;
    size_t _marginalization_count = 0;

    std::set<FrameID> _dropped_frames;
    std::set<LandmarkID> _dropped_landmarks;

    void reset() override;

    void updateFrame(FrameID id, ImageData const&) override;
    void updateFrame(
      FrameID prev_frame, FrameID curr_frame,
      ImageData const& landmark_measurement,
      std::unique_ptr<ImuPreintegration> imu_motion) override;

    void updateImuBias() override;
    void updateImuBias(
      Eigen::Vector3d const& bias_acc,
      Eigen::Vector3d const& bias_gyr) override;

    void marginalize(
      FrameID drop_frame, std::set<LandmarkID> const& drop_landmarks) override;

    ImuMotions const& imu() const override;
    FeatureTracks const& tracks() const override;
  };

  std::unique_ptr<MeasurementDataProviderMockup> makeMeasurementProviderMockup(
    PoseSignal pose_signal, SE3Transform const& extrinsic,
    LandmarkPositions const&, std::map<FrameID, Timestamp> const&);
}  // namespace cyclops::measurement
