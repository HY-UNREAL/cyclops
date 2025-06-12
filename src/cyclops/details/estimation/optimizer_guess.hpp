#pragma once

#include "cyclops/details/estimation/state/state_block.hpp"
#include <memory>

namespace cyclops {
  struct CyclopsConfig;
}

namespace cyclops::measurement {
  class MeasurementDataProvider;
}

namespace cyclops::initializer {
  class InitializerMain;
}

namespace cyclops::estimation {
  class StateVariableReadAccessor;

  class OptimizerSolutionGuessPredictor {
  public:
    virtual ~OptimizerSolutionGuessPredictor() = default;
    virtual void reset() = 0;

    struct Solution {
      MotionFrameParameterBlocks motions;
      LandmarkParameterBlocks landmarks;
    };
    virtual std::optional<Solution> solve() = 0;

    static std::unique_ptr<OptimizerSolutionGuessPredictor> Create(
      std::unique_ptr<initializer::InitializerMain> initializer,
      std::shared_ptr<CyclopsConfig const> config,
      std::shared_ptr<StateVariableReadAccessor const> state,
      std::shared_ptr<measurement::MeasurementDataProvider> measurement);
  };
}  // namespace cyclops::estimation
