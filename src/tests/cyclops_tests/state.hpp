#pragma once

#include "cyclops/details/estimation/state/state_block.hpp"

namespace cyclops {
  static estimation::MotionFrameParameterBlock makeMotionFrameParameter(
    Eigen::Quaterniond const& q, Eigen::Vector3d const& p,
    Eigen::Vector3d const& v) {
    estimation::MotionFrameParameterBlock x;
    (Eigen::Map<Eigen::Quaterniond>(x.data())) = q;
    (Eigen::Map<Eigen::Vector3d>(x.data() + 4)) = p;
    (Eigen::Map<Eigen::Vector3d>(x.data() + 7)) = v;
    (Eigen::Map<Eigen::Vector3d>(x.data() + 10)).setZero();
    (Eigen::Map<Eigen::Vector3d>(x.data() + 13)).setZero();
    return x;
  }

  static estimation::MotionFrameParameterBlock makeMotionFrameParameter(
    ImuMotionState const& x) {
    return makeMotionFrameParameter(x.orientation, x.position, x.velocity);
  }

  static estimation::LandmarkParameterBlock makeLandmarkParameter(
    Eigen::Vector3d const& f) {
    estimation::LandmarkParameterBlock f_;
    Eigen::Map<Eigen::Vector3d>(f_.data()) = f;
    return f_;
  }
}  // namespace cyclops
