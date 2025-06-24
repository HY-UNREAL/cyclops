#include "cyclops/details/telemetry/keyframe.hpp"

namespace cyclops::telemetry {
  void KeyframeTelemetry::reset() {
    // Nothing.
  }

  void KeyframeTelemetry::onNewMotionFrame(OnNewMotionFrame const& argument) {
    // Nothing.
  }

  std::unique_ptr<KeyframeTelemetry> KeyframeTelemetry::CreateDefault() {
    return std::make_unique<KeyframeTelemetry>();
  }
}  // namespace cyclops::telemetry
