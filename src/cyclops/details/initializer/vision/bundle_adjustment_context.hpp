#pragma once

#include "cyclops/details/type.hpp"

#include <map>
#include <memory>

namespace ceres::internal {
  struct ResidualBlock;
}

namespace ceres {
  struct Problem;
}

namespace cyclops {
  struct CyclopsConfig;
}

namespace cyclops::initializer {
  struct BundleAdjustmentOptimizationState;
  struct MultiViewGeometry;
  struct TwoViewImuRotationConstraint;

  class BundleAdjustmentOptimizationContext {
  private:
    struct Impl;
    std::unique_ptr<Impl> _pimpl;

  public:
    BundleAdjustmentOptimizationContext(
      CyclopsConfig const& config, MultiViewGeometry const& geometry_guess);
    ~BundleAdjustmentOptimizationContext();

    ceres::Problem& problem();
    BundleAdjustmentOptimizationState& state();

    std::map<LandmarkID, double*>& landmarks();
    std::map<FrameID, double*>& frames();
    std::vector<ceres::internal::ResidualBlock*>& residuals();

    int nLandmarkMeasurements() const;
    int nGyroMotionConstraints() const;

    bool construct(
      std::map<FrameID, std::map<LandmarkID, FeaturePoint>> const& features,
      std::map<FrameID, TwoViewImuRotationConstraint> const& gyro_motions);
  };
}  // namespace cyclops::initializer
