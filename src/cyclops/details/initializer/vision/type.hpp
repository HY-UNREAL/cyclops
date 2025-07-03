#pragma once

#include "cyclops/details/type.hpp"

#include <map>
#include <tuple>

namespace cyclops::initializer {
  struct GyroMotionConstraint {
    FrameID init_frame_id;
    FrameID term_frame_id;

    Eigen::Quaterniond value;
    Eigen::Matrix3d covariance;

    Eigen::Vector3d bias_nominal;
    Eigen::Matrix3d bias_jacobian;
  };

  struct BundleAdjustmentSolution {
    bool acceptable;

    double solution_significant_probability;
    double measurement_inlier_ratio;

    int n_inliers;
    int n_outliers;

    std::map<FrameID, SE3Transform> camera_motions;
    Eigen::MatrixXd motion_information_weight;
    Eigen::Vector3d gyro_bias;

    LandmarkPositions landmarks;
  };
}  // namespace cyclops::initializer
