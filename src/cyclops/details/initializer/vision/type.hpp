#pragma once

#include "cyclops/details/type.hpp"

#include <map>
#include <tuple>

namespace cyclops::initializer {
  using TwoViewFeaturePair = std::tuple<Eigen::Vector2d, Eigen::Vector2d>;

  struct TwoViewImuRotationData {
    Eigen::Quaterniond value;
    Eigen::Matrix3d covariance;
  };

  struct TwoViewImuRotationConstraint {
    FrameID init_frame_id;
    FrameID term_frame_id;

    TwoViewImuRotationData rotation;
  };

  struct TwoViewCorrespondenceData {
    TwoViewImuRotationData rotation_prior;
    std::map<LandmarkID, TwoViewFeaturePair> features;
  };

  struct MultiViewCorrespondences {
    FrameID reference_frame;
    std::map<FrameID, TwoViewCorrespondenceData> view_frames;
  };

  struct TwoViewGeometry {
    SE3Transform camera_motion;
    LandmarkPositions landmarks;
  };

  struct MultiViewGeometry {
    std::map<FrameID, SE3Transform> camera_motions;
    LandmarkPositions landmarks;
  };

  struct MSfMSolution {
    bool acceptable;

    double solution_significant_probability;
    double measurement_inlier_ratio;

    MultiViewGeometry geometry;
    Eigen::MatrixXd motion_information_weight;
  };
}  // namespace cyclops::initializer
