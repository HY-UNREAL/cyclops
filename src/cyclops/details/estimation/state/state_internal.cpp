#include "cyclops/details/estimation/state/state_internal.hpp"

namespace cyclops::estimation {
  MotionFrameParameterBlocks const& StateVariableInternal::motionFrames()
    const {
    return _motion_frames;
  }

  MotionFrameParameterBlocks& StateVariableInternal::motionFrames() {
    return _motion_frames;
  }

  LandmarkParameterBlocks const& StateVariableInternal::landmarks() const {
    return _landmarks;
  }

  LandmarkParameterBlocks& StateVariableInternal::landmarks() {
    return _landmarks;
  }

  LandmarkPositions const& StateVariableInternal::mappedLandmarks() const {
    return _mapped_landmarks;
  }

  LandmarkPositions& StateVariableInternal::mappedLandmarks() {
    return _mapped_landmarks;
  }
}  // namespace cyclops::estimation
