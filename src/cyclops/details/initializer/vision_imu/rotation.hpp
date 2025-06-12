#pragma once

#include "cyclops/details/measurement/type.hpp"

#include <map>
#include <memory>
#include <optional>

namespace cyclops {
  struct CyclopsConfig;
}  // namespace cyclops

namespace cyclops::initializer {
  struct ImuMatchCameraRotationPrior;

  struct ImuRotationMatch {
    Eigen::Vector3d gyro_bias;
    std::map<FrameID, Eigen::Quaterniond> body_orientations;
  };

  class ImuRotationMatchSolver {
  public:
    virtual ~ImuRotationMatchSolver() = default;
    virtual void reset() = 0;

    virtual std::optional<ImuRotationMatch> solve(
      measurement::ImuMotionRefs const& motions,
      ImuMatchCameraRotationPrior const& prior) = 0;

    static std::unique_ptr<ImuRotationMatchSolver> Create(
      std::shared_ptr<CyclopsConfig const> config);
  };
}  // namespace cyclops::initializer
