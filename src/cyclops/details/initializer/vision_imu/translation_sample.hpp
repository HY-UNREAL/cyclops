#pragma once

#include "cyclops/details/type.hpp"

#include <memory>
#include <optional>
#include <set>
#include <vector>

namespace cyclops {
  struct CyclopsConfig;
}

namespace cyclops::telemetry {
  struct InitializerTelemetry;
}

namespace cyclops::initializer {
  struct ImuTranslationMatchAnalysis;
  struct ImuTranslationMatchAnalysisCache;

  struct ImuMatchScaleSampleSolution {
    double scale;
    double cost;

    Eigen::VectorXd inertial_state;
    Eigen::VectorXd visual_state;

    Eigen::MatrixXd hessian;
  };

  class ImuMatchScaleSampleSolver {
  public:
    virtual ~ImuMatchScaleSampleSolver() = default;
    virtual void reset() = 0;

    virtual std::optional<std::vector<ImuMatchScaleSampleSolution>> solve(
      std::set<FrameID> const& motion_frames,
      ImuTranslationMatchAnalysis const& analysis,
      ImuTranslationMatchAnalysisCache const& cache) const = 0;

    static std::unique_ptr<ImuMatchScaleSampleSolver> Create(
      std::shared_ptr<CyclopsConfig const> config,
      std::shared_ptr<telemetry::InitializerTelemetry> telemetry);
  };
}  // namespace cyclops::initializer
