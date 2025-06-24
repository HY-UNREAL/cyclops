#pragma once

#include "cyclops/details/type.hpp"

#include <map>
#include <set>

namespace cyclops::initializer {
  struct TwoViewFeaturePair;

  Eigen::Matrix3d refineHomographyGeometry(
    double noise,  //
    Eigen::Matrix3d const& H_initial, std::set<LandmarkID> const& ids,
    std::map<LandmarkID, TwoViewFeaturePair> const& features);
}  // namespace cyclops::initializer
