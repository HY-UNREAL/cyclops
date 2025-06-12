#pragma once

#include "cyclops/details/type.hpp"

#include <map>
#include <memory>
#include <optional>
#include <set>

namespace cyclops {
  struct CyclopsConfig;
}

namespace cyclops::estimation {
  struct StateVariableReadAccessor;
}

namespace cyclops::measurement {
  struct MeasurementDataProvider;
  struct KeyframeManager;

  class MeasurementDataQueue {
  public:
    virtual ~MeasurementDataQueue() = default;
    virtual void reset() = 0;

    virtual void updateImu(ImuData const&) = 0;
    virtual std::optional<FrameID> updateLandmark(ImageData const&) = 0;

    virtual bool detectKeyframe(FrameID candidate_frame) const = 0;
    virtual void acceptCurrentPendingKeyframe() = 0;

    virtual void marginalize(FrameID drop_frame) = 0;
    virtual void marginalizeKeyframe(
      FrameID drop_frame, std::set<LandmarkID> const& drop_landmarks,
      FrameID inserted_frame) = 0;
    virtual void marginalizePendingFrame(
      FrameID drop_frame, std::set<LandmarkID> const& drop_landmarks) = 0;

    virtual std::map<FrameID, Timestamp> const& keyframes() const = 0;
    virtual std::map<FrameID, Timestamp> const& pendingFrames() const = 0;

    static std::unique_ptr<MeasurementDataQueue> Create(
      std::shared_ptr<CyclopsConfig const> config,
      std::shared_ptr<MeasurementDataProvider> measurements,
      std::shared_ptr<KeyframeManager> keyframe_manager,
      std::shared_ptr<estimation::StateVariableReadAccessor const> state);
  };
}  // namespace cyclops::measurement
