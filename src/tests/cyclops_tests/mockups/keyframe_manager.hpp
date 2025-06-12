#pragma once

#include "cyclops/details/measurement/keyframe.hpp"

namespace cyclops::measurement {
  struct KeyframeManagerMock: public KeyframeManager {
    FrameID _last_frame_id = 0;
    FrameSequence _keyframes;
    FrameSequence _pending_frames;

    void reset() override;

    FrameID createNewFrame(Timestamp timestamp) override;
    void setKeyframe(FrameID id) override;
    void removeFrame(FrameID frame) override;

    FrameSequence const& keyframes() const override;
    FrameSequence const& pendingFrames() const override;
  };
}  // namespace cyclops::measurement
