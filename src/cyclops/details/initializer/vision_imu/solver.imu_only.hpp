#pragma once

#include "cyclops/details/initializer/vision_imu/solver.hpp"

#include <map>
#include <memory>
#include <optional>

namespace cyclops::initializer {
  class ImuOnlyMatchSolver: public ImuMatchSolver {
  public:
    virtual ~ImuOnlyMatchSolver() = default;

    static std::unique_ptr<ImuOnlyMatchSolver> Create(
      std::shared_ptr<CyclopsConfig const> config,
      std::shared_ptr<telemetry::InitializerTelemetry> telemetry);
  };
}  // namespace cyclops::initializer
