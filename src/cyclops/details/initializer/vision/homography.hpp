#pragma once

#include "cyclops/details/type.hpp"

#include <map>
#include <set>
#include <vector>

namespace cyclops {
  struct RotationPositionPair;
}  // namespace cyclops

namespace cyclops::initializer {
  struct TwoViewFeaturePair;

  struct HomographyAnalysis {
    double expected_inliers;

    Eigen::Matrix3d homography;
    std::set<LandmarkID> inliers;
  };

  HomographyAnalysis analyzeTwoViewHomography(
    double sigma, std::vector<std::set<LandmarkID>> const& ransac_batch,
    std::map<LandmarkID, TwoViewFeaturePair> const& features);

  std::vector<RotationPositionPair> solveHomographyMotionHypothesis(
    Eigen::Matrix3d const& H);
}  // namespace cyclops::initializer
