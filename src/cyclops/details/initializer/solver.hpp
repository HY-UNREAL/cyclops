#pragma once

#include "cyclops/details/measurement/type.hpp"

#include <Eigen/Dense>

#include <memory>
#include <random>
#include <tuple>
#include <vector>

namespace cyclops {
  struct CyclopsConfig;
}

namespace cyclops::measurement {
  struct MeasurementDataProvider;
}

namespace cyclops::telemetry {
  struct InitializerTelemetry;
}

namespace cyclops::initializer {
  struct MSfMSolution;

  struct InitializationSolverResult {
    std::vector<MSfMSolution> msfm_solutions;

    struct SolutionCandidate {
      int msfm_solution_index;

      bool acceptance;
      double cost;
      double scale;
      Eigen::Vector3d gravity;
      Eigen::Vector3d gyr_bias;
      Eigen::Vector3d acc_bias;

      LandmarkPositions landmarks;
      std::map<FrameID, ImuMotionState> motions;
    };
    std::vector<SolutionCandidate> solution_candidates;
  };

  class InitializationSolverInternal {
  public:
    virtual ~InitializationSolverInternal() = default;
    virtual void reset() = 0;

    virtual InitializationSolverResult solve() = 0;

    static std::unique_ptr<InitializationSolverInternal> Create(
      std::shared_ptr<std::mt19937> rgen,
      std::shared_ptr<CyclopsConfig const> config,
      std::shared_ptr<measurement::MeasurementDataProvider const> data_provider,
      std::shared_ptr<telemetry::InitializerTelemetry> telemetry);
  };
}  // namespace cyclops::initializer
