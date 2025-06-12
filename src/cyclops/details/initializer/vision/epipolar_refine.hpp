#pragma once

#include "cyclops/details/initializer/vision/type.hpp"

#include <map>
#include <set>

namespace cyclops::initializer {
  Eigen::Matrix3d refineEpipolarGeometry(
    Eigen::Matrix3d const& E_initial, std::set<LandmarkID> const& ids,
    std::map<LandmarkID, TwoViewFeaturePair> const& features);
}  // namespace cyclops::initializer
