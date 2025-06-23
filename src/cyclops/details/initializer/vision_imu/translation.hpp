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
  struct ImuMatchCameraMotionPrior;

  struct ImuMatchSolution {
    double scale;
    double cost;
    Eigen::Vector3d gravity;
    Eigen::Vector3d acc_bias;
    Eigen::Vector3d gyr_bias;
    std::map<FrameID, Eigen::Vector3d> body_velocities;
    std::map<FrameID, Eigen::Quaterniond> body_orientations;
    std::map<FrameID, Eigen::Vector3d> sfm_positions;
  };

  struct ImuMatchResult {
    bool accept;
    ImuMatchSolution solution;
  };

  class ImuMatchSolver {
  public:
    virtual ~ImuMatchSolver() = default;
    virtual void reset() = 0;

    virtual std::optional<std::vector<ImuMatchResult>> solve(
      measurement::ImuMotionRefs const& motions,
      ImuMatchCameraMotionPrior const& camera_prior) = 0;

    static std::unique_ptr<ImuMatchSolver> Create(
      std::shared_ptr<CyclopsConfig const> config,
      std::shared_ptr<telemetry::InitializerTelemetry> telemetry);
  };
}  // namespace cyclops::initializer
