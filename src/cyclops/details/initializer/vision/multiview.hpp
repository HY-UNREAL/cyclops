#pragma once

#include "cyclops/details/type.hpp"

#include <map>
#include <memory>
#include <random>
#include <vector>

namespace cyclops {
  struct CyclopsConfig;
}

namespace cyclops::telemetry {
  struct InitializerTelemetry;
}

namespace cyclops::initializer {
  struct GyroMotionConstraint;

  struct MultiViewGeometry {
    std::map<FrameID, SE3Transform> camera_motions;
    LandmarkPositions landmarks;
  };

  class MultiviewVisionGeometrySolver {
  public:
    virtual ~MultiviewVisionGeometrySolver() = default;
    virtual void reset() = 0;

    // Returns a sequence of possible multiview geometries.
    virtual std::vector<MultiViewGeometry> solve(
      std::map<FrameID, std::map<LandmarkID, FeaturePoint>> const& features,
      std::map<FrameID, GyroMotionConstraint> const& gyro_motions) = 0;

    static std::unique_ptr<MultiviewVisionGeometrySolver> Create(
      std::shared_ptr<CyclopsConfig const> config,
      std::shared_ptr<std::mt19937> rgen,
      std::shared_ptr<telemetry::InitializerTelemetry> telemetry);
  };
}  // namespace cyclops::initializer
