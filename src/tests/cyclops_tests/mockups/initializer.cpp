#include "cyclops_tests/mockups/initializer.hpp"
#include "cyclops_tests/random.hpp"
#include "cyclops_tests/signal.ipp"

#include <range/v3/all.hpp>

namespace cyclops::estimation {
  namespace views = ranges::views;

  using estimation::LandmarkParameterBlocks;
  using estimation::MotionFrameParameterBlocks;

  static auto makeLocalLandmarks(
    LandmarkPositions const& landmarks, PoseSignal const& pose_signal,
    std::map<FrameID, Timestamp> frame_timestamps) {
    if (frame_timestamps.empty())
      return LandmarkPositions {};

    auto [_, t0] = *frame_timestamps.begin();
    auto x0 = pose_signal.evaluate(t0);

    return  //
      landmarks | views::transform([&](auto const& pair) {
        auto const& [landmark_id, f] = pair;
        auto const& [p0, q0] = x0;

        auto f_bar = (q0.conjugate() * (f - p0)).eval();
        return std::make_pair(landmark_id, f_bar);
      }) |
      ranges::to<LandmarkPositions>;
  }

  static auto makeMotionStateLookup(
    PoseSignal const& pose_signal,
    std::map<FrameID, Timestamp> frame_timestamps) {
    if (frame_timestamps.empty())
      return std::map<FrameID, ImuMotionState> {};

    auto [_, t0] = *frame_timestamps.begin();
    auto x0 = pose_signal.evaluate(t0);
    auto const& p0 = x0.translation;
    auto const& q0 = x0.rotation;

    auto const& p = pose_signal.position;
    auto const& q = pose_signal.orientation;
    auto v = numericDerivative(p);

    return  //
      frame_timestamps | views::transform([&](auto pair) {
        auto const& [frame_id, t] = pair;
        auto x = ImuMotionState {
          .orientation = q0.conjugate() * q(t),
          .position = q0.conjugate() * (p(t) - p0),
          .velocity = q0.conjugate() * v(t),
        };
        return std::make_pair(frame_id, x);
      }) |
      ranges::to<std::map<FrameID, ImuMotionState>>;
  }

  OptimizerSolutionGuessPredictorMock::OptimizerSolutionGuessPredictorMock(
    std::shared_ptr<std::mt19937> rgen, LandmarkPositions const& landmarks,
    PoseSignal const& pose_signal,
    std::map<FrameID, Timestamp> frame_timestamps)
      : _rgen(rgen),
        _landmarks(
          makeLocalLandmarks(landmarks, pose_signal, frame_timestamps)),
        _motions(makeMotionStateLookup(pose_signal, frame_timestamps)) {
  }

  void OptimizerSolutionGuessPredictorMock::reset() {
    // does nothing.
  }

  std::optional<OptimizerSolutionGuessPredictor::Solution>
  OptimizerSolutionGuessPredictorMock::solve() {
    if (_motions.empty())
      return std::nullopt;

    auto initial_frame_id = _motions.begin()->first;
    auto motions =
      _motions | views::transform([&](auto const& id_motion) {
        auto const& [id, motion] = id_motion;
        double const perturbation = id == initial_frame_id ? 0. : 0.05;
        return std::make_pair(
          id, makePerturbatedFrameState(motion, perturbation, *_rgen));
      }) |
      ranges::to<MotionFrameParameterBlocks>;

    auto landmarks =
      _landmarks | views::transform([&](auto const& id_landmark) {
        auto const& [id, landmark] = id_landmark;
        return std::make_pair(
          id, makePerturbatedLandmarkState(landmark, 0.05, *_rgen));
      }) |
      ranges::to<LandmarkParameterBlocks>;

    return Solution {motions, landmarks};
  }
}  // namespace cyclops::estimation
