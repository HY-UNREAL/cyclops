#pragma once

#include "cyclops/details/type.hpp"

#include <set>
#include <map>
#include <vector>

namespace cyclops {
  struct RotationPositionPair;
}  // namespace cyclops

namespace cyclops::initializer {
  struct TwoViewFeaturePair;

  struct EpipolarAnalysis {
    double expected_inliers;

    Eigen::Matrix3d essential_matrix;
    std::set<LandmarkID> inliers;
  };

  EpipolarAnalysis analyzeTwoViewEpipolar(
    double sigma, std::vector<std::set<LandmarkID>> const& ransac_batch,
    std::map<LandmarkID, TwoViewFeaturePair> const& features);

  std::vector<RotationPositionPair> solveEpipolarMotionHypothesis(
    Eigen::Matrix3d const& essential_matrix);
}  // namespace cyclops::initializer
