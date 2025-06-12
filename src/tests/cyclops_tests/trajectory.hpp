#pragma once

#include "cyclops_tests/signal.hpp"
#include <vector>

namespace cyclops {
  Vector3Signal bezier(double T, std::vector<Eigen::Vector3d> const& points);
}  // namespace cyclops
