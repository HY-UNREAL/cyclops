#pragma once

#include "cyclops/details/measurement/type.hpp"

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
  struct ImuMatchMotionPrior;
  struct ImuMatchResult;

  class ImuMatchSolver {
  public:
    virtual ~ImuMatchSolver() = default;
    virtual void reset() = 0;

    virtual std::optional<std::vector<ImuMatchResult>> solve(
      measurement::ImuMotionRefs const& motions,
      ImuMatchMotionPrior const& camera_prior) = 0;

    static std::unique_ptr<ImuMatchSolver> Create(
      std::shared_ptr<CyclopsConfig const> config,
      std::shared_ptr<telemetry::InitializerTelemetry> telemetry);
  };
}  // namespace cyclops::initializer
