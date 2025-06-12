#pragma once

#include "cyclops/details/type.hpp"

#include <string>
#include <map>
#include <vector>

namespace cyclops {
  struct SE3Transform;

  std::string serialize(std::vector<Eigen::Vector3d> const&);
  std::string serialize(std::vector<SE3Transform> const&);
  std::string serialize(
    std::map<LandmarkID, std::map<FrameID, FeaturePoint>> const&);
}  // namespace cyclops
