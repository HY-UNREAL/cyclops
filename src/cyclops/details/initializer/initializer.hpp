#pragma once

#include "cyclops/details/type.hpp"

#include <map>
#include <memory>

namespace cyclops::measurement {
  struct KeyframeManager;
}

namespace cyclops::telemetry {
  struct InitializerTelemetry;
}

namespace cyclops::initializer {
  class InitializerCandidateSolver;

  struct InitializationSolution {
    Eigen::Vector3d acc_bias;
    Eigen::Vector3d gyr_bias;
    LandmarkPositions landmarks;
    std::map<FrameID, ImuMotionState> motions;
  };

  class InitializerMain {
  public:
    virtual ~InitializerMain() = default;
    virtual void reset() = 0;

    virtual std::optional<InitializationSolution> solve() = 0;

    static std::unique_ptr<InitializerMain> Create(
      std::unique_ptr<InitializerCandidateSolver> candidate_solver,
      std::shared_ptr<measurement::KeyframeManager const> keyframe_manager,
      std::shared_ptr<telemetry::InitializerTelemetry> telemetry);
  };
}  // namespace cyclops::initializer
