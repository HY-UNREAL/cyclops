#pragma once

#include "cyclops/details/estimation/optimizer_guess.hpp"
#include "cyclops_tests/signal.hpp"

#include <random>
#include <memory>
#include <map>
#include <optional>

namespace cyclops::estimation {
  class OptimizerSolutionGuessPredictorMock:
      public OptimizerSolutionGuessPredictor {
  private:
    std::shared_ptr<std::mt19937> _rgen;

    LandmarkPositions _landmarks;
    std::map<FrameID, ImuMotionState> _motions;

  public:
    OptimizerSolutionGuessPredictorMock(
      std::shared_ptr<std::mt19937> rgen, LandmarkPositions const& landmarks,
      PoseSignal const& pose_signal,
      std::map<FrameID, Timestamp> frame_timestamps);

    void reset() override;

    std::optional<Solution> solve() override;
  };
}  // namespace cyclops::estimation
