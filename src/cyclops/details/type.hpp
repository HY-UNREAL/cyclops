#pragma once

#include <Eigen/Dense>

#include <cstdlib>
#include <map>

namespace cyclops {
  using Timestamp = double;

  using FrameID = std::size_t;
  using LandmarkID = std::size_t;

  using LandmarkPositions = std::map<LandmarkID, Eigen::Vector3d>;

  struct ImuData {
    Timestamp timestamp;
    Eigen::Vector3d accel;
    Eigen::Vector3d rotat;
  };

  struct FeaturePoint {
    Eigen::Vector2d point;
    Eigen::Matrix2d weight;
  };

  struct ImageData {
    Timestamp timestamp;
    std::map<LandmarkID, FeaturePoint> features;
  };

  struct SE3Transform {
    Eigen::Vector3d translation;
    Eigen::Quaterniond rotation;

    static SE3Transform Identity();
  };

  struct ImuMotionState {
    Eigen::Quaterniond orientation;
    Eigen::Vector3d position;
    Eigen::Vector3d velocity;
  };
}  // namespace cyclops
