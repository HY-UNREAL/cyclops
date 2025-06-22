#pragma once

#include "cyclops/details/initializer/vision_imu/translation.hpp"

#include <map>
#include <memory>
#include <optional>

namespace cyclops {
  struct CyclopsConfig;
}

namespace cyclops::telemetry {
  struct InitializerTelemetry;
}

namespace cyclops::initializer {
  struct MSfMSolution;

  struct ImuMatchSolution {
    ImuRotationMatch rotation_match;
    std::vector<ImuTranslationMatch> translation_match;
  };

  class ImuMatchSolver {
  public:
    virtual ~ImuMatchSolver() = default;
    virtual void reset() = 0;

    virtual std::optional<ImuMatchSolution> solve(
      MSfMSolution const& msfm,
      measurement::ImuMotionRefs const& imu_motions) = 0;

    static std::unique_ptr<ImuMatchSolver> Create(
      std::shared_ptr<CyclopsConfig const> config,
      std::shared_ptr<telemetry::InitializerTelemetry> telemetry);
  };
}  // namespace cyclops::initializer
