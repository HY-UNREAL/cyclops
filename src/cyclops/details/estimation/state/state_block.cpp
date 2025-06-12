#include "cyclops/details/estimation/state/state_block.hpp"

namespace cyclops::estimation {
  using Eigen::Quaterniond;
  using Eigen::Vector3d;

  static Quaterniond getOrientation(double const* frame_ptr) {
    return decltype(getOrientation(frame_ptr))(frame_ptr);
  }

  static Vector3d getPosition(double const* frame_ptr) {
    return decltype(getPosition(frame_ptr))(frame_ptr + 4);
  }

  static Vector3d getVelocity(double const* frame_ptr) {
    return decltype(getVelocity(frame_ptr))(frame_ptr + 7);
  }

  static Vector3d getAccBias(double const* frame_ptr) {
    return decltype(getAccBias(frame_ptr))(frame_ptr + 10);
  }

  static Vector3d getGyrBias(double const* frame_ptr) {
    return decltype(getGyrBias(frame_ptr))(frame_ptr + 13);
  }
}  // namespace cyclops::estimation

namespace cyclops::estimation::buffer::motion_frame {
  SE3Transform getSE3Transform(double const* frame_ptr) {
    return {
      .translation = getPosition(frame_ptr),
      .rotation = getOrientation(frame_ptr),
    };
  }
}  // namespace cyclops::estimation::buffer::motion_frame

namespace cyclops::estimation {
  SE3Transform getSE3Transform(MotionFrameParameterBlock const& block) {
    return buffer::motion_frame::getSE3Transform(block.data());
  }

  Vector3d getPosition(LandmarkParameterBlock const& block) {
    return Vector3d(block.data());
  }

  ImuMotionState getMotionState(MotionFrameParameterBlock const& frame) {
    auto frame_ptr = frame.data();
    return ImuMotionState {
      .orientation = getOrientation(frame_ptr),
      .position = getPosition(frame_ptr),
      .velocity = getVelocity(frame_ptr),
    };
  }

  Vector3d getAccBias(MotionFrameParameterBlock const& block) {
    return getAccBias(block.data());
  }

  Vector3d getGyrBias(MotionFrameParameterBlock const& block) {
    return getGyrBias(block.data());
  }
}  // namespace cyclops::estimation
