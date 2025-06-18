#pragma once

#include "cyclops/details/initializer/vision_imu/translation.hpp"

#include <map>
#include <memory>
#include <optional>

namespace cyclops::initializer {
  class ImuOnlyTranslationMatchSolver: public ImuTranslationMatchSolver {
  public:
    virtual ~ImuOnlyTranslationMatchSolver() = default;

    static std::unique_ptr<ImuOnlyTranslationMatchSolver> Create(
      std::shared_ptr<CyclopsConfig const> config,
      std::shared_ptr<telemetry::InitializerTelemetry> telemetry);
  };
}  // namespace cyclops::initializer
