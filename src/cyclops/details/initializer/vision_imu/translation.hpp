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
  struct ImuMatchCameraTranslationPrior;
  struct ImuRotationMatch;

  struct ImuTranslationMatchSolution {
    double scale;
    double cost;
    Eigen::Vector3d gravity;
    Eigen::Vector3d acc_bias;
    std::map<FrameID, Eigen::Vector3d> imu_body_velocities;
    std::map<FrameID, Eigen::Vector3d> sfm_positions;
  };

  struct ImuRotationMatch {
    Eigen::Vector3d gyro_bias;
    std::map<FrameID, Eigen::Quaterniond> body_orientations;
  };

  struct ImuTranslationMatch {
    bool accept;
    ImuTranslationMatchSolution solution;
  };

  class ImuTranslationMatchSolver {
  public:
    virtual ~ImuTranslationMatchSolver() = default;
    virtual void reset() = 0;

    virtual std::optional<std::vector<ImuTranslationMatch>> solve(
      measurement::ImuMotionRefs const& motions,
      ImuRotationMatch const& rotations,
      ImuMatchCameraTranslationPrior const& camera_prior) = 0;

    static std::unique_ptr<ImuTranslationMatchSolver> Create(
      std::shared_ptr<CyclopsConfig const> config,
      std::shared_ptr<telemetry::InitializerTelemetry> telemetry);
  };
}  // namespace cyclops::initializer
