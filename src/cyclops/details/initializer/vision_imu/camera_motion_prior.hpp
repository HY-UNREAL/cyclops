#pragma once

#include "cyclops/details/type.hpp"

#include <map>
#include <tuple>

namespace cyclops::initializer {
  struct MSfMSolution;

  struct ImuMatchCameraTranslationPrior {
    std::map<FrameID, Eigen::Vector3d> translations;
    Eigen::MatrixXd weight;
  };

  ImuMatchCameraTranslationPrior makeImuMatchCameraMotionPrior(
    MSfMSolution const& msfm);
}  // namespace cyclops::initializer
