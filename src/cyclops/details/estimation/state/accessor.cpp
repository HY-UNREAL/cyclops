#include "cyclops/details/estimation/state/accessor.hpp"
#include "cyclops/details/estimation/state/accessor_read.hpp"
#include "cyclops/details/estimation/state/accessor_write.hpp"
#include "cyclops/details/estimation/state/state_internal.hpp"

namespace cyclops::estimation {
  using std::shared_ptr;

  StateVariableAccessor::StateVariableAccessor(
    std::shared_ptr<StateVariableReadAccessor> reader,
    std::shared_ptr<StateVariableWriteAccessor> writer)
      : _reader(reader), _writer(writer) {
  }

  StateVariableAccessor::~StateVariableAccessor() = default;

  std::shared_ptr<StateVariableReadAccessor>
  StateVariableAccessor::deriveReader() {
    return _reader;
  }

  std::shared_ptr<StateVariableWriteAccessor>
  StateVariableAccessor::deriveWriter() {
    return _writer;
  }

  void StateVariableAccessor::reset() {
    _writer->reset();
  }

  StateVariableAccessor::MaybeRef<MotionFrameParameterBlock>
  StateVariableAccessor::motionFrame(FrameID id) {
    return _writer->motionFrame(id);
  }

  StateVariableAccessor::MaybeRef<LandmarkParameterBlock>
  StateVariableAccessor::landmark(LandmarkID id) {
    return _writer->landmark(id);
  }

  MotionFrameParameterBlocks const& StateVariableAccessor::motionFrames()
    const {
    return _reader->motionFrames();
  }

  LandmarkParameterBlocks const& StateVariableAccessor::landmarks() const {
    return _reader->landmarks();
  }

  LandmarkPositions const& StateVariableAccessor::mappedLandmarks() const {
    return _reader->mappedLandmarks();
  }

  std::optional<std::tuple<Timestamp, ImuMotionState>>
  StateVariableAccessor::propagatedState() const {
    return _reader->propagatedState();
  }

  std::unique_ptr<StateVariableAccessor> StateVariableAccessor::Create(
    std::shared_ptr<CyclopsConfig const> config,
    std::shared_ptr<ImuPropagationUpdateHandler> propagator) {
    shared_ptr state_internal = std::make_shared<StateVariableInternal>();

    return std::make_unique<StateVariableAccessor>(
      std::make_shared<StateVariableReadAccessor>(state_internal, propagator),
      std::make_shared<StateVariableWriteAccessor>(state_internal));
  }
}  // namespace cyclops::estimation
