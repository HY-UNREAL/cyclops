#pragma once

#include <memory>

namespace cyclops {
  struct CyclopsConfig;
}

namespace cyclops::measurement {
  class MeasurementDataQueue;
}

namespace cyclops::estimation {
  class FactorGraphInstance;
  class StateVariableReadAccessor;

  struct GaussianPrior;

  class MarginalizationManager {
  public:
    virtual ~MarginalizationManager() = default;
    virtual void reset() = 0;

    virtual void marginalize(FactorGraphInstance& graph_instance) = 0;
    virtual void marginalize() = 0;
    virtual GaussianPrior const& prior() const = 0;

    static std::unique_ptr<MarginalizationManager> Create(
      std::shared_ptr<CyclopsConfig const> config,
      std::shared_ptr<StateVariableReadAccessor const> state,
      std::shared_ptr<measurement::MeasurementDataQueue> data_queue);
  };
}  // namespace cyclops::estimation
