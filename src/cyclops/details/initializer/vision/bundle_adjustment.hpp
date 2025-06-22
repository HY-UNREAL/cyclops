#pragma once

#include "cyclops/details/type.hpp"

namespace cyclops {
  struct CyclopsConfig;
}  // namespace cyclops

namespace cyclops::initializer {
  struct MultiViewGeometry;
  struct MSfMSolution;
  struct TwoViewImuRotationConstraint;

  std::optional<MSfMSolution> solveBundleAdjustment(
    CyclopsConfig const& config, MultiViewGeometry const& guess,
    std::map<FrameID, std::map<LandmarkID, FeaturePoint>> const& data,
    std::map<FrameID, TwoViewImuRotationConstraint> const& imu_prior);
}  // namespace cyclops::initializer
