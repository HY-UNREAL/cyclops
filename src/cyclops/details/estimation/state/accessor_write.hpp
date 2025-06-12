#pragma once

#include "cyclops/details/estimation/state/state_block.hpp"

#include <memory>
#include <set>

namespace cyclops::estimation {
  class StateVariableInternal;

  class StateVariableWriteAccessor {
  private:
    template <typename value_t>
    using MaybeRef = std::optional<std::reference_wrapper<value_t>>;

  private:
    std::shared_ptr<StateVariableInternal> _state;

  public:
    explicit StateVariableWriteAccessor(
      std::shared_ptr<StateVariableInternal> state);
    ~StateVariableWriteAccessor();

    std::tuple<std::set<FrameID>, std::set<LandmarkID>> prune(
      std::set<FrameID> const& current_frames,
      std::set<LandmarkID> const& current_landmarks);
    void reset();

    void updateMotionFrameGuess(MotionFrameParameterBlocks const& frames);
    void updateLandmarkGuess(LandmarkParameterBlocks const& landmarks);
    void updateMappedLandmarks(LandmarkPositions const& positions);

    MaybeRef<MotionFrameParameterBlock> motionFrame(FrameID id);
    MaybeRef<LandmarkParameterBlock> landmark(LandmarkID id);
  };
}  // namespace cyclops::estimation
