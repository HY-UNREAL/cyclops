#pragma once

#include "cyclops/details/type.hpp"

#include <map>
#include <tuple>

namespace cyclops::initializer {
  struct MSfMSolution;

  struct ImuMatchCameraMotionPrior {
    std::map<FrameID, Eigen::Quaterniond> imu_orientations;
    std::map<FrameID, Eigen::Vector3d> camera_positions;
    Eigen::Vector3d gyro_bias;
    Eigen::MatrixXd weight;
  };

  ImuMatchCameraMotionPrior makeImuMatchCameraMotionPrior(
    MSfMSolution const& msfm, SE3Transform const& camera_extrinsic);
}  // namespace cyclops::initializer
