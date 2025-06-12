#pragma once

#include "cyclops/details/measurement/type.hpp"

#include <memory>
#include <set>

namespace cyclops {
  struct CyclopsConfig;
}

namespace cyclops::estimation {
  struct StateVariableReadAccessor;
}  // namespace cyclops::estimation

namespace cyclops::measurement {
  class MeasurementDataProvider {
  public:
    virtual ~MeasurementDataProvider() = default;
    virtual void reset() = 0;

    virtual void updateFrame(
      FrameID frame_id, ImageData const& image_data) = 0;
    virtual void updateFrame(
      FrameID prev_frame_id, FrameID curr_frame_id,
      ImageData const& image, std::unique_ptr<ImuPreintegration> imu) = 0;

    virtual void updateImuBias() = 0;
    virtual void updateImuBias(
      Eigen::Vector3d const& bias_acc, Eigen::Vector3d const& bias_gyr) = 0;

    virtual void marginalize(
      FrameID drop_frame, std::set<LandmarkID> const& drop_landmarks) = 0;

    virtual ImuMotions const& imu() const = 0;
    virtual FeatureTracks const& tracks() const = 0;

    static std::unique_ptr<MeasurementDataProvider> Create(
      std::shared_ptr<CyclopsConfig const> config,
      std::shared_ptr<estimation::StateVariableReadAccessor const> state);
  };
}  // namespace cyclops::measurement
