#pragma once

#include "cyclops_tests/signal.hpp"

#include <map>

namespace cyclops::initializer {
  struct TwoViewImuRotationConstraint;
}

namespace cyclops {
  struct SE3Transform;

  std::map<FrameID, initializer::TwoViewImuRotationConstraint>
  makeMultiViewRotationPrior(
    PoseSignal const& pose_signal, SE3Transform const& camera_extrinsic,
    std::map<FrameID, Timestamp> const& frame_timestamps);
}  // namespace cyclops
