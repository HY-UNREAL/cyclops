#include "cyclops/details/initializer/initializer.hpp"
#include "cyclops/details/initializer/candidate.hpp"
#include "cyclops/details/initializer/vision.hpp"
#include "cyclops/details/initializer/vision_imu.hpp"

#include "cyclops/details/measurement/keyframe.hpp"
#include "cyclops/details/telemetry/initializer.hpp"

#include "cyclops/details/logging.hpp"

#include <range/v3/all.hpp>
#include <spdlog/spdlog.h>

namespace cyclops::initializer {
  using Eigen::Matrix3d;
  using Eigen::Quaterniond;
  using Eigen::Vector3d;

  namespace views = ranges::views;

  using VisionTelemetryDigest =
    telemetry::InitializerTelemetry::VisionSolutionCandidateDigest;
  using ImuTelemetryDigest =
    telemetry::InitializerTelemetry::ImuSolutionCandidateDigest;

  using MatchCandidate = InitializerCandidatePairs::ImuMatchCandidate;

  class InitializerMainImpl: public InitializerMain {
  private:
    std::unique_ptr<InitializerCandidateSolver> _candidate_solver;

    std::shared_ptr<measurement::KeyframeManager const> _keyframe_manager;
    std::shared_ptr<telemetry::InitializerTelemetry> _telemetry;

    void reportFailureTelemetry(InitializerCandidatePairs const& solution);
    std::optional<MatchCandidate> solveAndReportTelemetry();

  public:
    InitializerMainImpl(
      std::unique_ptr<InitializerCandidateSolver> candidate_solver,
      std::shared_ptr<measurement::KeyframeManager const> keyframe_manager,
      std::shared_ptr<telemetry::InitializerTelemetry> telemetry);
    ~InitializerMainImpl();
    void reset() override;

    std::optional<InitializationSolution> solve() override;
  };

  InitializerMainImpl::InitializerMainImpl(
    std::unique_ptr<InitializerCandidateSolver> candidate_solver,
    std::shared_ptr<measurement::KeyframeManager const> keyframe_manager,
    std::shared_ptr<telemetry::InitializerTelemetry> telemetry)
      : _candidate_solver(std::move(candidate_solver)),
        _keyframe_manager(keyframe_manager),
        _telemetry(telemetry) {
  }

  InitializerMainImpl::~InitializerMainImpl() = default;

  void InitializerMainImpl::reset() {
    _candidate_solver->reset();
    _telemetry->reset();
  }

  void InitializerMainImpl::reportFailureTelemetry(
    InitializerCandidatePairs const& solution) {
    auto vision_solutions =  //
      solution.msfm_solutions | views::transform([](auto const& sol) {
        return VisionTelemetryDigest {
          .acceptable = sol.acceptable,
          .keyframes = sol.camera_motions | views::keys | ranges::to<std::set>,
        };
      }) |
      ranges::to_vector;

    auto imu_solutions =
      solution.imu_match_solutions | views::transform([](auto const& solution) {
        return ImuTelemetryDigest {
          .vision_solution_index = solution.msfm_solution_index,
          .acceptable = solution.acceptance,
          .scale = solution.scale,
          .keyframes = solution.motions | views::keys | ranges::to<std::set>,
        };
      }) |
      ranges::to_vector;

    __logger__->debug("Reporting failure...");
    _telemetry->onFailure({
      .vision_solutions = vision_solutions,
      .imu_solutions = imu_solutions,
    });
    __logger__->debug("Reported failure.");
  }

  std::optional<MatchCandidate> InitializerMainImpl::solveAndReportTelemetry() {
    auto solution = _candidate_solver->solve();
    __logger__->debug("Initialization solution obtained.");

    if (solution.imu_match_solutions.empty()) {
      reportFailureTelemetry(solution);
      return std::nullopt;
    }
    if (solution.imu_match_solutions.size() > 1) {
      reportFailureTelemetry(solution);
      return std::nullopt;
    }

    auto const& candidate = solution.imu_match_solutions.front();
    auto const& vision_solution =
      solution.msfm_solutions.at(candidate.msfm_solution_index);

    if (!candidate.acceptance || !vision_solution.acceptable) {
      reportFailureTelemetry(solution);
      return std::nullopt;
    }

    if (candidate.motions.empty()) {
      __logger__->error("IMU bootstrap returned empty motion.");

      reportFailureTelemetry(solution);
      return std::nullopt;
    }

    auto initial_motion_frame_id = candidate.motions.begin()->first;
    auto initial_motion_frame_timestamp =
      _keyframe_manager->keyframes().at(initial_motion_frame_id);

    _telemetry->onSuccess({
      .vision_solution_index = candidate.msfm_solution_index,
      .initial_motion_frame_id = initial_motion_frame_id,
      .initial_motion_frame_timestamp = initial_motion_frame_timestamp,
      .sfm_camera_pose = vision_solution.camera_motions,
      .cost = candidate.cost,
      .scale = candidate.scale,
      .gravity = candidate.gravity,
      .motions = candidate.motions,
    });
    return candidate;
  }

  /**
   * Returns a rotation matrix such that the third column (the z direction)
   * matches the given argument `g`, and the first column (the x direction) lies
   * within the plane spanned by the given arguments `g` and `x`.
   */
  static Matrix3d solveVisionOriginRotation(
    Vector3d const& g, Vector3d const& x) {
    Vector3d r3 = g.normalized();
    Vector3d r2 = r3.cross(x).normalized();
    Vector3d r1 = r2.cross(r3);

    if (x.dot(r1) < 0)
      r1 = -r1;

    Matrix3d R;
    R << r1, r2, r3;
    if (R.determinant() < 0)
      R.col(1) = -R.col(1);

    return R;
  }

  struct InitializationGravityRotation {
    LandmarkPositions landmarks;
    std::map<FrameID, ImuMotionState> motions;
  };

  static InitializationGravityRotation rotateGravity(
    MatchCandidate const& imu_matching) {
    auto const& g = imu_matching.gravity;

    auto R_vb0 = [&]() -> Matrix3d {
      if (imu_matching.motions.empty())
        return Matrix3d::Identity();

      auto const& [_, x_vb0] = *imu_matching.motions.begin();
      return x_vb0.orientation.matrix();
    }();

    auto R_vw = solveVisionOriginRotation(g, R_vb0.col(0));
    auto q_vw = Quaterniond(R_vw);
    auto q_wv = q_vw.conjugate();

    std::map<FrameID, ImuMotionState> motions = std::move(imu_matching.motions);
    LandmarkPositions landmarks = std::move(imu_matching.landmarks);

    for (auto& [_, motion] : motions) {
      motion.orientation = q_wv * motion.orientation;
      motion.position = q_wv * motion.position;
      motion.velocity = q_wv * motion.velocity;
    }
    for (auto& [_, landmark] : landmarks)
      landmark = q_wv * landmark;
    return {landmarks, motions};
  }

  std::optional<InitializationSolution> InitializerMainImpl::solve() {
    auto maybe_imu_match = solveAndReportTelemetry();
    if (!maybe_imu_match)
      return std::nullopt;

    auto const& imu_match = *maybe_imu_match;
    auto [landmarks, motions] = rotateGravity(imu_match);

    return InitializationSolution {
      .acc_bias = imu_match.acc_bias,
      .gyr_bias = imu_match.gyr_bias,
      .landmarks = std::move(landmarks),
      .motions = std::move(motions),
    };
  }

  std::unique_ptr<InitializerMain> InitializerMain::Create(
    std::unique_ptr<InitializerCandidateSolver> candidate_solver,
    std::shared_ptr<measurement::KeyframeManager const> keyframe_manager,
    std::shared_ptr<telemetry::InitializerTelemetry> telemetry) {
    return std::make_unique<InitializerMainImpl>(
      std::move(candidate_solver), keyframe_manager, telemetry);
  }
}  // namespace cyclops::initializer
