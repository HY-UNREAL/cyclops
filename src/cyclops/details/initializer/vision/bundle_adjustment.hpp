#pragma once

#include "cyclops/details/type.hpp"
#include <memory>

namespace cyclops {
  struct CyclopsConfig;
}  // namespace cyclops

namespace cyclops::initializer {
  struct MultiViewGeometry;
  struct BundleAdjustmentSolution;
  struct GyroMotionConstraint;

  class BundleAdjustmentSolver {
  public:
    virtual ~BundleAdjustmentSolver() = default;
    virtual std::optional<BundleAdjustmentSolution> solve(
      MultiViewGeometry const& guess,
      std::map<FrameID, std::map<LandmarkID, FeaturePoint>> const& features,
      std::map<FrameID, GyroMotionConstraint> const& gyro_motion) = 0;

    static std::unique_ptr<BundleAdjustmentSolver> Create(
      std::shared_ptr<CyclopsConfig const> config);
  };
}  // namespace cyclops::initializer
