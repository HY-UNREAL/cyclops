#pragma once

#include "cyclops/details/type.hpp"

#include <map>
#include <memory>
#include <optional>

namespace cyclops {
  struct CyclopsConfig;
  struct ImuData;
  struct ImageData;
}  // namespace cyclops

namespace cyclops::estimation {
  class ImuPropagationUpdateHandler;
  class StateVariableReadAccessor;
}  // namespace cyclops::estimation

namespace cyclops::measurement {
  class MeasurementDataQueue;

  class MeasurementDataUpdater {
  public:
    virtual ~MeasurementDataUpdater() = default;
    virtual void reset() = 0;

    virtual void updateImu(ImuData const& data) = 0;
    virtual std::optional<FrameID> updateLandmark(ImageData const& data) = 0;
    virtual void repropagate(FrameID last_frame, Timestamp timestamp) = 0;

    virtual std::map<FrameID, Timestamp> frames() const = 0;

    static std::unique_ptr<MeasurementDataUpdater> Create(
      std::shared_ptr<CyclopsConfig const> config,
      std::shared_ptr<MeasurementDataQueue> measurement_queue,
      std::shared_ptr<estimation::ImuPropagationUpdateHandler> propagator,
      std::shared_ptr<estimation::StateVariableReadAccessor const>
        state_reader);
  };
}  // namespace cyclops::measurement
