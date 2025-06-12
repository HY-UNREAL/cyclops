#pragma once

#include "cyclops/details/type.hpp"

#include <map>
#include <optional>

namespace cyclops {
  struct RotationPositionPair;
}  // namespace cyclops

namespace cyclops::initializer {
  struct PnpImagePoint {
    Eigen::Vector3d position;
    Eigen::Vector2d observation;
  };

  std::optional<RotationPositionPair> solvePnpCameraPose(
    std::map<LandmarkID, PnpImagePoint> const& image_point_set,
    int gauss_newton_refinement_iterations = 5);
}  // namespace cyclops::initializer
