#pragma once

#include "cyclops/details/type.hpp"

#include <random>
#include <map>
#include <memory>
#include <vector>

namespace cyclops {
  struct CyclopsConfig;
}  // namespace cyclops

namespace cyclops::telemetry {
  struct InitializerTelemetry;
}

namespace cyclops::initializer {
  struct MSfMSolution;
  struct GyroMotionConstraint;

  class VisionInitializer {
  public:
    virtual ~VisionInitializer() = default;
    virtual void reset() = 0;

    virtual std::vector<MSfMSolution> solve(
      std::map<FrameID, std::map<LandmarkID, FeaturePoint>> const& features,
      std::map<FrameID, GyroMotionConstraint> const& gyro_motions) = 0;

    static std::unique_ptr<VisionInitializer> Create(
      std::shared_ptr<CyclopsConfig const> config,
      std::shared_ptr<std::mt19937> rgen,
      std::shared_ptr<telemetry::InitializerTelemetry> telemetry);
  };
}  // namespace cyclops::initializer
