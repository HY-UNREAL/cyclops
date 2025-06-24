#pragma once

#include "cyclops/details/initializer/vision/type.hpp"
#include "cyclops/details/type.hpp"

#include <map>
#include <set>

namespace cyclops {
  struct RotationPositionPair;
}

namespace cyclops::config::initializer {
  struct VisionSolverConfig;
}

namespace cyclops::initializer {
  struct TwoViewFeaturePair;

  struct TwoViewTriangulation {
    /* failure statistics. */
    int n_triangulation_failure;
    int n_error_probability_test_failure;

    double expected_inliers;

    /**
     * actually accepted triangulations. i.e. triangulations that passed both
     * reprojection test and direction test, with enough parallax.
     */
    LandmarkPositions landmarks;
  };

  TwoViewTriangulation triangulateTwoViewFeaturePairs(
    config::initializer::VisionSolverConfig const& config,
    std::map<LandmarkID, TwoViewFeaturePair> const& feature_pairs,
    std::set<LandmarkID> const& feature_ids,
    RotationPositionPair const& camera_motion);
}  // namespace cyclops::initializer
