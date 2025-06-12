#pragma once

#include "cyclops/details/type.hpp"
#include <memory>
#include <vector>

namespace cyclops {
  struct CyclopsConfig;
  struct LandmarkGenerationArgument;

  Eigen::Quaterniond makeDefaultCameraRotation();
  SE3Transform makeDefaultImuCameraExtrinsic();

  std::shared_ptr<CyclopsConfig> makeDefaultConfig();
  std::vector<LandmarkGenerationArgument> makeDefaultLandmarkSet();
}  // namespace cyclops
