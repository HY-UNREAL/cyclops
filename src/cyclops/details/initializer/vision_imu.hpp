#pragma once

#include "cyclops/details/initializer/vision_imu/type.hpp"
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
  struct MSfMSolution;
  struct ImuMatchResult;

  class VisionImuInitializer {
  public:
    virtual ~VisionImuInitializer() = default;
    virtual void reset() = 0;

    virtual std::optional<std::vector<ImuMatchResult>> solve(
      MSfMSolution const& msfm,
      measurement::ImuMotionRefs const& imu_motions) = 0;

    static std::unique_ptr<VisionImuInitializer> Create(
      std::shared_ptr<CyclopsConfig const> config,
      std::shared_ptr<telemetry::InitializerTelemetry> telemetry);
  };
}  // namespace cyclops::initializer
