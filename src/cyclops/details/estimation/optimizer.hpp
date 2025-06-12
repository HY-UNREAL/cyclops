#pragma once

#include "cyclops/details/estimation/sanity.hpp"
#include "cyclops/details/type.hpp"

#include <memory>
#include <set>
#include <optional>

namespace cyclops {
  struct CyclopsConfig;
}

namespace cyclops::measurement {
  class MeasurementDataProvider;
}

namespace cyclops::estimation {
  struct GaussianPrior;

  class FactorGraphInstance;
  class StateVariableWriteAccessor;
  class OptimizerSolutionGuessPredictor;

  class LikelihoodOptimizer {
  public:
    virtual ~LikelihoodOptimizer() = default;
    virtual void reset() = 0;

    struct OptimizationResult {
      std::unique_ptr<FactorGraphInstance> graph;

      LandmarkSanityStatistics landmark_sanity_statistics;
      OptimizerSanityStatistics optimizer_sanity_statistics;

      double solve_time;
      double optimizer_time;
      std::string optimizer_report;

      std::set<FrameID> motion_frames;
      std::set<LandmarkID> active_landmarks;
      std::set<LandmarkID> mapped_landmarks;
    };
    virtual std::optional<OptimizationResult> optimize(
      GaussianPrior const& prior) = 0;

    static std::unique_ptr<LikelihoodOptimizer> Create(
      std::unique_ptr<OptimizerSolutionGuessPredictor> predictor,
      std::shared_ptr<CyclopsConfig const> config,
      std::shared_ptr<StateVariableWriteAccessor> state_accessor,
      std::shared_ptr<measurement::MeasurementDataProvider> measurement);
  };
}  // namespace cyclops::estimation
