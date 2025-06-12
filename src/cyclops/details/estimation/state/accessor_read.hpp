#pragma once

#include "cyclops/details/estimation/state/state_block.hpp"
#include <memory>

namespace cyclops::estimation {
  class ImuPropagationUpdateHandler;
  class StateVariableInternal;

  class StateVariableReadAccessor {
  private:
    std::shared_ptr<StateVariableInternal const> _state;
    std::shared_ptr<ImuPropagationUpdateHandler const> _propagator;

  private:
    template <typename value_t>
    using MaybeCRef = std::optional<std::reference_wrapper<value_t const>>;

  public:
    StateVariableReadAccessor(
      std::shared_ptr<StateVariableInternal const> state_internal,
      std::shared_ptr<ImuPropagationUpdateHandler const> propagation_accessor);
    ~StateVariableReadAccessor();

    MaybeCRef<MotionFrameParameterBlock> motionFrame(FrameID id) const;
    MaybeCRef<LandmarkParameterBlock> landmark(LandmarkID id) const;

    FrameID lastMotionFrameId() const;
    MotionFrameParameterBlock const& lastMotionFrameBlock() const;

    MotionFrameParameterBlocks const& motionFrames() const;
    LandmarkParameterBlocks const& landmarks() const;
    LandmarkPositions const& mappedLandmarks() const;

    std::optional<std::tuple<Timestamp, ImuMotionState>> propagatedState()
      const;
  };
}  // namespace cyclops::estimation
