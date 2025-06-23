#pragma once

#include <optional>
#include <memory>

namespace cyclops {
  struct CyclopsConfig;
}  // namespace cyclops

namespace cyclops::initializer {
  struct ImuMatchScaleEvaluationContext;

  struct ImuMatchScaleRefinement {
    double scale;
    double cost;
  };

  class ImuMatchLocalOptimizer {
  private:
    std::shared_ptr<CyclopsConfig const> _config;

  public:
    explicit ImuMatchLocalOptimizer(
      std::shared_ptr<CyclopsConfig const> config);
    ~ImuMatchLocalOptimizer();
    void reset();

    std::optional<ImuMatchScaleRefinement> optimize(
      ImuMatchScaleEvaluationContext const& evaluator, double s0);

    static std::unique_ptr<ImuMatchLocalOptimizer> Create(
      std::shared_ptr<CyclopsConfig const> config);
  };
}  // namespace cyclops::initializer
