#include "cyclops/details/estimation/state/accessor_read.hpp"
#include "cyclops/details/estimation/state/state_internal.hpp"
#include "cyclops/details/estimation/propagation.hpp"

namespace cyclops::estimation {
  template <typename value_t>
  using MaybeCRef = StateVariableReadAccessor::MaybeCRef<value_t>;

  template <typename key_t, typename container_t>
  static auto maybeFind(container_t&& container, key_t key)
    -> MaybeCRef<std::remove_reference_t<decltype(container.at(key))>> {
    auto i = container.find(key);
    if (i == container.end())
      return std::nullopt;

    auto& [_, result] = *i;
    return result;
  }

  StateVariableReadAccessor::StateVariableReadAccessor(
    std::shared_ptr<StateVariableInternal const> state,
    std::shared_ptr<ImuPropagationUpdateHandler const> propagator)
      : _state(state), _propagator(propagator) {
  }

  StateVariableReadAccessor::~StateVariableReadAccessor() = default;

  MaybeCRef<MotionFrameParameterBlock> StateVariableReadAccessor::motionFrame(
    FrameID id) const {
    return maybeFind(_state->motionFrames(), id);
  }

  MaybeCRef<LandmarkParameterBlock> StateVariableReadAccessor::landmark(
    LandmarkID id) const {
    return maybeFind(_state->landmarks(), id);
  }

  FrameID StateVariableReadAccessor::lastMotionFrameId() const {
    return _state->motionFrames().rbegin()->first;
  }

  MotionFrameParameterBlock const&
  StateVariableReadAccessor::lastMotionFrameBlock() const {
    return _state->motionFrames().rbegin()->second;
  }

  MotionFrameParameterBlocks const& StateVariableReadAccessor::motionFrames()
    const {
    return _state->motionFrames();
  }

  LandmarkParameterBlocks const& StateVariableReadAccessor::landmarks() const {
    return _state->landmarks();
  }

  LandmarkPositions const& StateVariableReadAccessor::mappedLandmarks() const {
    return _state->mappedLandmarks();
  }

  std::optional<std::tuple<Timestamp, ImuMotionState>>
  StateVariableReadAccessor::propagatedState() const {
    return _propagator->get();
  }
}  // namespace cyclops::estimation
