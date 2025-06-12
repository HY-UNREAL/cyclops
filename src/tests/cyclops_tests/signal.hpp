#pragma once

#include "cyclops/details/type.hpp"

#include <Eigen/Dense>
#include <functional>

namespace cyclops {
  using ScalarSignal = std::function<double(Timestamp)>;
  using Vector3Signal = std::function<Eigen::Vector3d(Timestamp)>;
  using QuaternionSignal = std::function<Eigen::Quaterniond(Timestamp)>;

  struct PoseSignal {
    Vector3Signal position;
    QuaternionSignal orientation;

    SE3Transform evaluate(Timestamp t) const;
  };
}  // namespace cyclops
