#pragma once

#include "cyclops/details/type.hpp"

#include <map>
#include <tuple>

namespace cyclops::initializer {
  struct MSfMSolution;

  struct ImuMatchCameraRotationPrior {
    std::map<FrameID, Eigen::Quaterniond> rotations;
    Eigen::MatrixXd weight;
  };

  struct ImuMatchCameraTranslationPrior {
    std::map<FrameID, Eigen::Vector3d> translations;
    Eigen::MatrixXd weight;
  };

  std::tuple<ImuMatchCameraRotationPrior, ImuMatchCameraTranslationPrior>
  makeImuMatchCameraMotionPrior(MSfMSolution const& msfm);
}  // namespace cyclops::initializer
