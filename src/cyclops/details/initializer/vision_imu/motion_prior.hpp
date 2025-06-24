#pragma once

#include "cyclops/details/type.hpp"

#include <map>
#include <tuple>

namespace cyclops::initializer {
  struct BundleAdjustmentSolution;

  struct ImuMatchMotionPrior {
    std::map<FrameID, Eigen::Quaterniond> imu_orientations;
    std::map<FrameID, Eigen::Vector3d> camera_positions;
    Eigen::Vector3d gyro_bias;
    Eigen::MatrixXd weight;
  };

  ImuMatchMotionPrior makeImuMatchMotionPrior(
    BundleAdjustmentSolution const& msfm, SE3Transform const& extrinsic);
}  // namespace cyclops::initializer
