#include "cyclops/details/type.hpp"

namespace cyclops {
  SE3Transform SE3Transform::Identity() {
    return SE3Transform {
      .translation = Eigen::Vector3d::Zero(),
      .rotation = Eigen::Quaterniond::Identity(),
    };
  }
}  // namespace cyclops
