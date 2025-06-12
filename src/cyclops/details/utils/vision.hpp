#pragma once

#include "cyclops/details/measurement/type.hpp"

namespace cyclops {
  struct KeyframeMotionStatistics {
    int new_features;
    int common_features;

    double average_parallax;
  };

  struct RotationPositionPair {
    Eigen::Matrix3d rotation;
    Eigen::Vector3d translation;
  };

  KeyframeMotionStatistics evaluateKeyframeMotionStatistics(
    measurement::FeatureTracks const& tracks, FrameID frame1, FrameID frame2);
  KeyframeMotionStatistics evaluateKeyframeMotionStatistics(
    std::map<LandmarkID, FeaturePoint> const& frame1,
    std::map<LandmarkID, FeaturePoint> const& frame2);

  std::optional<Eigen::Vector3d> triangulatePoint(
    measurement::FeatureTrack const& features,
    std::map<FrameID, RotationPositionPair> const& camera_pose_lookup);

  std::map<LandmarkID, std::tuple<Eigen::Vector2d, Eigen::Vector2d>>
  compileTwoViewFeaturePairs(
    std::map<LandmarkID, FeaturePoint> const& frame1,
    std::map<LandmarkID, FeaturePoint> const& frame2);
}  // namespace cyclops
