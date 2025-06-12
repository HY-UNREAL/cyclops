#pragma once

#include "cyclops/details/estimation/state/state_block.hpp"

#include <memory>
#include <optional>
#include <tuple>

namespace cyclops {
  struct CyclopsConfig;
}  // namespace cyclops

namespace cyclops::estimation {
  class ImuPropagationUpdateHandler {
  public:
    virtual ~ImuPropagationUpdateHandler() = default;
    virtual void reset() = 0;

    virtual void updateOptimization(
      Timestamp last_timestamp,
      MotionFrameParameterBlock const& last_state) = 0;
    virtual void updateImuData(ImuData const& data) = 0;

    virtual std::optional<std::tuple<Timestamp, ImuMotionState>> get()
      const = 0;

    static std::unique_ptr<ImuPropagationUpdateHandler> Create(
      std::shared_ptr<CyclopsConfig const> config);
  };
}  // namespace cyclops::estimation
