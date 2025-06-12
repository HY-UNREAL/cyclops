#pragma once

#include "cyclops/details/estimation/state/state_block.hpp"

#include <memory>
#include <tuple>

namespace cyclops {
  struct CyclopsConfig;
}

namespace cyclops::estimation {
  struct ImuPropagationUpdateHandler;
  struct StateVariableReadAccessor;
  struct StateVariableWriteAccessor;

  class StateVariableAccessor {
  private:
    std::shared_ptr<StateVariableReadAccessor> _reader;
    std::shared_ptr<StateVariableWriteAccessor> _writer;

  private:
    template <typename value_t>
    using MaybeRef = std::optional<std::reference_wrapper<value_t>>;

  public:
    StateVariableAccessor(
      std::shared_ptr<StateVariableReadAccessor> reader,
      std::shared_ptr<StateVariableWriteAccessor> writer);
    ~StateVariableAccessor();

    void reset();
    std::shared_ptr<StateVariableReadAccessor> deriveReader();
    std::shared_ptr<StateVariableWriteAccessor> deriveWriter();

    MaybeRef<MotionFrameParameterBlock> motionFrame(FrameID id);
    MaybeRef<LandmarkParameterBlock> landmark(LandmarkID id);

    MotionFrameParameterBlocks const& motionFrames() const;
    LandmarkParameterBlocks const& landmarks() const;
    LandmarkPositions const& mappedLandmarks() const;

    std::optional<std::tuple<Timestamp, ImuMotionState>> propagatedState()
      const;

    static std::unique_ptr<StateVariableAccessor> Create(
      std::shared_ptr<CyclopsConfig const> config,
      std::shared_ptr<ImuPropagationUpdateHandler> propagator);
  };
}  // namespace cyclops::estimation
