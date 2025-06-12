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
  struct TwoViewImuRotationConstraint;

  class VisionBootstrapSolver {
  public:
    using MultiViewImageData =
      std::map<FrameID, std::map<LandmarkID, FeaturePoint>>;
    using CameraRotations = std::map<FrameID, TwoViewImuRotationConstraint>;

  public:
    virtual ~VisionBootstrapSolver() = default;
    virtual void reset() = 0;

    virtual std::vector<MSfMSolution> solve(
      MultiViewImageData const& image_data,
      CameraRotations const& camera_rotation_prior) = 0;

    static std::unique_ptr<VisionBootstrapSolver> Create(
      std::shared_ptr<CyclopsConfig const> config,
      std::shared_ptr<std::mt19937> rgen,
      std::shared_ptr<telemetry::InitializerTelemetry> telemetry);
  };
}  // namespace cyclops::initializer
