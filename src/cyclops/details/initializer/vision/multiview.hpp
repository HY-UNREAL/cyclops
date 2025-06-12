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
  struct MultiViewGeometry;
  struct TwoViewImuRotationConstraint;

  class MultiviewVisionGeometrySolver {
  public:
    using MultiViewImageData =
      std::map<FrameID, std::map<LandmarkID, FeaturePoint>>;
    using CameraRotationPriorLookup =
      std::map<FrameID, TwoViewImuRotationConstraint>;

  public:
    virtual ~MultiviewVisionGeometrySolver() = default;
    virtual void reset() = 0;

    // returns a sequence of possible multiview geometries.
    virtual std::vector<MultiViewGeometry> solve(
      MultiViewImageData const& multiview_data,
      CameraRotationPriorLookup const& camera_rotations) = 0;

    static std::unique_ptr<MultiviewVisionGeometrySolver> Create(
      std::shared_ptr<CyclopsConfig const> config,
      std::shared_ptr<std::mt19937> rgen,
      std::shared_ptr<telemetry::InitializerTelemetry> telemetry);
  };
}  // namespace cyclops::initializer
