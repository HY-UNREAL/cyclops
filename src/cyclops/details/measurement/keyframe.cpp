#include "cyclops/details/measurement/keyframe.hpp"
#include "cyclops/details/telemetry/keyframe.hpp"

namespace cyclops::measurement {
  class KeyframeManagerImpl: public KeyframeManager {
  private:
    std::shared_ptr<telemetry::KeyframeTelemetry> _telemetry;

    FrameSequence _keyframes;
    FrameSequence _pending_frames;
    FrameID _frame_id_ctr = 0;

  public:
    explicit KeyframeManagerImpl(
      std::shared_ptr<telemetry::KeyframeTelemetry> telemetry)
        : _telemetry(telemetry) {
    }

    void reset() override;

    FrameID createNewFrame(Timestamp timestamp) override;
    void setKeyframe(FrameID id) override;
    void removeFrame(FrameID frame) override;

    FrameSequence const& keyframes() const override;
    FrameSequence const& pendingFrames() const override;
  };

  void KeyframeManagerImpl::reset() {
    _keyframes.clear();
    _pending_frames.clear();
    _telemetry->reset();
  }

  FrameID KeyframeManagerImpl::createNewFrame(Timestamp timestamp) {
    _pending_frames.emplace(_frame_id_ctr, timestamp);
    _telemetry->onNewMotionFrame({
      .frame_id = _frame_id_ctr,
      .timestamp = timestamp,
    });

    return _frame_id_ctr++;
  }

  void KeyframeManagerImpl::setKeyframe(FrameID id) {
    auto i = _pending_frames.find(id);
    if (i == _pending_frames.end())
      return;
    auto timestamp = i->second;

    _pending_frames.erase(i);
    _keyframes.emplace(id, timestamp);
  }

  void KeyframeManagerImpl::removeFrame(FrameID frame) {
    _pending_frames.erase(frame);
    _keyframes.erase(frame);
  }

  KeyframeManagerImpl::FrameSequence const& KeyframeManagerImpl::keyframes()
    const {
    return _keyframes;
  }

  KeyframeManagerImpl::FrameSequence const& KeyframeManagerImpl::pendingFrames()
    const {
    return _pending_frames;
  }

  std::unique_ptr<KeyframeManager> KeyframeManager::Create(
    std::shared_ptr<telemetry::KeyframeTelemetry> telemetry) {
    return std::make_unique<KeyframeManagerImpl>(telemetry);
  }
}  // namespace cyclops::measurement
