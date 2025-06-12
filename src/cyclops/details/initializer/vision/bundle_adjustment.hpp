#pragma once

#include "cyclops/details/type.hpp"

namespace cyclops::config::initializer {
  struct VisionSolverConfig;
}  // namespace cyclops::config::initializer

namespace cyclops::initializer {
  struct MultiViewGeometry;
  struct MSfMSolution;

  std::optional<MSfMSolution> solveBundleAdjustment(
    config::initializer::VisionSolverConfig const& config,
    MultiViewGeometry const& guess,
    std::map<FrameID, std::map<LandmarkID, FeaturePoint>> const& data);
}  // namespace cyclops::initializer
