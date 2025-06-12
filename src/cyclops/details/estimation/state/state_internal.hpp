#pragma once

#include "cyclops/details/estimation/state/state_block.hpp"

namespace cyclops::estimation {
  class StateVariableInternal {
  private:
    MotionFrameParameterBlocks _motion_frames;
    LandmarkParameterBlocks _landmarks;
    LandmarkPositions _mapped_landmarks;

  public:
    MotionFrameParameterBlocks const& motionFrames() const;
    MotionFrameParameterBlocks& motionFrames();

    LandmarkParameterBlocks const& landmarks() const;
    LandmarkParameterBlocks& landmarks();

    LandmarkPositions const& mappedLandmarks() const;
    LandmarkPositions& mappedLandmarks();
  };
}  // namespace cyclops::estimation
