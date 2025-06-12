#pragma once

#include "cyclops/details/type.hpp"
#include <memory>

namespace cyclops::telemetry {
  class KeyframeTelemetry {
  public:
    virtual ~KeyframeTelemetry() = default;
    virtual void reset();

    struct OnNewMotionFrame {
      FrameID frame_id;
      Timestamp timestamp;
    };
    virtual void onNewMotionFrame(OnNewMotionFrame const& argument);

    static std::unique_ptr<KeyframeTelemetry> CreateDefault();
  };
}  // namespace cyclops::telemetry
