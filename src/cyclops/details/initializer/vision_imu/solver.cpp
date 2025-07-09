#include "cyclops/details/initializer/vision_imu/solver.hpp"

#include "cyclops/details/initializer/vision_imu/acceptance.hpp"
#include "cyclops/details/initializer/vision_imu/motion_prior.hpp"
#include "cyclops/details/initializer/vision_imu/analysis.hpp"
#include "cyclops/details/initializer/vision_imu/analysis_cache.hpp"
#include "cyclops/details/initializer/vision_imu/scale_sample.hpp"
#include "cyclops/details/initializer/vision_imu/type.hpp"
#include "cyclops/details/initializer/vision_imu/uncertainty.hpp"

#include "cyclops/details/telemetry/initializer.hpp"
#include "cyclops/details/config.hpp"

#include <range/v3/all.hpp>

#include <algorithm>
#include <set>
#include <vector>

namespace cyclops::initializer {
  using Eigen::Vector3d;

  namespace views = ranges::views;

  class ImuMatchSolutionParseContext {
  private:
    CyclopsConfig const& config;
    ImuMatchAcceptDiscriminator const& acceptor;

    ImuMatchMotionPrior const& camera_prior;
    ImuMatchAnalysis const& analysis;

    ImuMatchSolution parseSolution(ImuMatchScaleSampleSolution const& solution);

  public:
    ImuMatchSolutionParseContext(
      CyclopsConfig const& config, ImuMatchAcceptDiscriminator const& acceptor,
      ImuMatchMotionPrior const& camera_prior, ImuMatchAnalysis const& analysis)
        : config(config),
          acceptor(acceptor),
          camera_prior(camera_prior),
          analysis(analysis) {
    }

    std::vector<ImuMatchResult> parse(
      std::vector<ImuMatchScaleSampleSolution> const& candidates);
  };

  ImuMatchSolution ImuMatchSolutionParseContext::parseSolution(
    ImuMatchScaleSampleSolution const& solution) {
    auto const& extrinsic = config.extrinsics.imu_camera_transform;

    auto const& x_I = solution.inertial_state;
    auto const& x_V = solution.visual_state;

    auto body_velocities =  //
      views::enumerate(camera_prior.camera_positions | views::keys) |
      views::transform([&](auto const& element) {
        auto [n, frame_id] = element;
        auto v_n = x_I.segment(6 + 3 * n, 3).eval();

        return std::make_pair(frame_id, v_n);
      }) |
      ranges::to<std::map<FrameID, Vector3d>>;

    auto camera_position_delta = [&](auto n) -> Vector3d {
      if (n == 0)
        return Vector3d::Zero();
      return x_V.segment(3 * (n - 1), 3);
    };

    auto camera_positions =  //
      views::enumerate(camera_prior.camera_positions) |
      views::transform([&](auto const& enumeration) {
        auto const& [n, element] = enumeration;
        auto const& [frame_id, p_c] = element;
        auto const& q_b = camera_prior.imu_orientations.at(frame_id);
        auto q_c = q_b * extrinsic.rotation;

        auto delta_p = camera_position_delta(n);
        return std::make_pair(frame_id, (p_c + q_c * delta_p).eval());
      }) |
      ranges::to<std::map<FrameID, Vector3d>>;

    return ImuMatchSolution {
      .scale = solution.scale,
      .cost = solution.cost,
      .gravity = x_I.head(3),
      .acc_bias = x_I.segment(3, 3),
      .gyr_bias = camera_prior.gyro_bias,
      .body_velocities = body_velocities,
      .body_orientations = camera_prior.imu_orientations,
      .sfm_positions = camera_positions,
    };
  }

  std::vector<ImuMatchResult> ImuMatchSolutionParseContext::parse(
    std::vector<ImuMatchScaleSampleSolution> const& solutions) {
    auto candidates =  //
      solutions | views::transform([&](auto const& candidate) {
        return std::make_tuple(
          parseSolution(candidate),
          analyzeImuMatchUncertainty(analysis, candidate));
      }) |
      ranges::to_vector;

    return acceptor.determineAcceptance(candidates);
  }

  class ImuMatchSolverImpl: public ImuMatchSolver {
  private:
    std::unique_ptr<ImuMatchAnalyzer> _analyzer;
    std::unique_ptr<ImuMatchAcceptDiscriminator> _acceptor;
    std::unique_ptr<ImuMatchScaleSampleSolver> _sample_solver;

    std::shared_ptr<CyclopsConfig const> _config;

  public:
    ImuMatchSolverImpl(
      std::unique_ptr<ImuMatchAnalyzer> analyzer,
      std::unique_ptr<ImuMatchAcceptDiscriminator> acceptor,
      std::unique_ptr<ImuMatchScaleSampleSolver> sample_solver,
      std::shared_ptr<CyclopsConfig const> config);
    void reset() override;

    std::optional<std::vector<ImuMatchResult>> solve(
      measurement::ImuMotionRefs const& motions,
      ImuMatchMotionPrior const& camera_prior) override;
  };

  ImuMatchSolverImpl::ImuMatchSolverImpl(
    std::unique_ptr<ImuMatchAnalyzer> analyzer,
    std::unique_ptr<ImuMatchAcceptDiscriminator> acceptor,
    std::unique_ptr<ImuMatchScaleSampleSolver> sample_solver,
    std::shared_ptr<CyclopsConfig const> config)
      : _analyzer(std::move(analyzer)),
        _acceptor(std::move(acceptor)),
        _sample_solver(std::move(sample_solver)),
        _config(config) {
  }

  void ImuMatchSolverImpl::reset() {
    _analyzer->reset();
    _acceptor->reset();
    _sample_solver->reset();
  }

  std::optional<std::vector<ImuMatchResult>> ImuMatchSolverImpl::solve(
    measurement::ImuMotionRefs const& motions,
    ImuMatchMotionPrior const& camera_prior) {
    auto const& extrinsic = _config->extrinsics.imu_camera_transform;

    auto analysis = _analyzer->analyze(motions, camera_prior);

    auto cache = ImuMatchAnalysisCache(analysis);
    auto motion_keyframes =
      camera_prior.camera_positions | views::keys | ranges::to<std::set>;

    auto solutions = _sample_solver->solve(motion_keyframes, analysis, cache);
    if (!solutions.has_value())
      return std::nullopt;

    auto candidate_parse_context = ImuMatchSolutionParseContext(
      *_config, *_acceptor, camera_prior, analysis);
    return candidate_parse_context.parse(*solutions);
  }

  std::unique_ptr<ImuMatchSolver> ImuMatchSolver::Create(
    std::shared_ptr<CyclopsConfig const> config,
    std::shared_ptr<telemetry::InitializerTelemetry> telemetry) {
    return std::make_unique<ImuMatchSolverImpl>(
      ImuMatchAnalyzer::Create(config),
      ImuMatchAcceptDiscriminator::Create(config, telemetry),
      ImuMatchScaleSampleSolver::Create(config, telemetry), config);
  }
}  // namespace cyclops::initializer
