#pragma once

#include "cyclops/details/type.hpp"

namespace cyclops::initializer {
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
}  // namespace cyclops::initializer
