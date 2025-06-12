#pragma once

#include "cyclops/details/type.hpp"

#include <map>
#include <memory>

namespace cyclops::telemetry {
  class KeyframeTelemetry;
}

namespace cyclops::measurement {
  class KeyframeManager {
  public:
    virtual ~KeyframeManager() = default;
    virtual void reset() = 0;

    virtual FrameID createNewFrame(Timestamp timestamp) = 0;
    virtual void setKeyframe(FrameID id) = 0;
    virtual void removeFrame(FrameID frame) = 0;

    using FrameSequence = std::map<FrameID, Timestamp>;
    virtual FrameSequence const& keyframes() const = 0;
    virtual FrameSequence const& pendingFrames() const = 0;

    static std::unique_ptr<KeyframeManager> Create(
      std::shared_ptr<telemetry::KeyframeTelemetry> telemetry);
  };
}  // namespace cyclops::measurement
