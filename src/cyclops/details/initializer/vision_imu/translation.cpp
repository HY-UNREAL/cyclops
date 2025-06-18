#include "cyclops/details/initializer/vision_imu/translation.hpp"
#include "cyclops/details/initializer/vision_imu/acceptance.hpp"
#include "cyclops/details/initializer/vision_imu/camera_motion_prior.hpp"
#include "cyclops/details/initializer/vision_imu/rotation.hpp"
#include "cyclops/details/initializer/vision_imu/translation_analysis.hpp"
#include "cyclops/details/initializer/vision_imu/translation_cache.hpp"
#include "cyclops/details/initializer/vision_imu/translation_sample.hpp"
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

  class ImuTranslationMatchSolutionParseContext {
  private:
    CyclopsConfig const& config;
    ImuTranslationMatchAcceptDiscriminator const& acceptor;

    ImuMatchCameraTranslationPrior const& camera_prior;
    ImuTranslationMatchAnalysis const& analysis;

    ImuTranslationMatchSolution parseSolution(
      ImuRotationMatch const& rotations,
      ImuMatchScaleSampleSolution const& solution);

  public:
    ImuTranslationMatchSolutionParseContext(
      CyclopsConfig const& config,
      ImuTranslationMatchAcceptDiscriminator const& acceptor,
      ImuMatchCameraTranslationPrior const& camera_prior,
      ImuTranslationMatchAnalysis const& analysis)
        : config(config),
          acceptor(acceptor),
          camera_prior(camera_prior),
          analysis(analysis) {
    }

    std::optional<ImuTranslationMatch> parse(
      ImuRotationMatch const& rotations,
      std::vector<ImuMatchScaleSampleSolution> const& candidates);
  };

  ImuTranslationMatchSolution
  ImuTranslationMatchSolutionParseContext::parseSolution(
    ImuRotationMatch const& rotations,
    ImuMatchScaleSampleSolution const& solution) {
    auto const& extrinsic = config.extrinsics.imu_camera_transform;
    auto const& imu_orientations = rotations.body_orientations;
    auto const& camera_position_nominals = camera_prior.translations;

    auto keyframes =
      camera_position_nominals | views::keys | ranges::to<std::set>;
    auto keyvalue_reverse_transform = views::transform(
      [](auto const& kv) { return std::make_pair(kv.second, kv.first); });
    auto keyframe_indexmap = views::enumerate(keyframes) |
      keyvalue_reverse_transform | ranges::to<std::map<FrameID, int>>;

    auto keyframe_index_transform = views::transform(
      [&](auto frame_id) { return keyframe_indexmap.at(frame_id); });

    auto const& x_I = solution.inertial_state;
    auto const& x_V = solution.visual_state;

    auto body_velocities =
      keyframes | keyframe_index_transform | views::transform([&](auto i) {
        return static_cast<Vector3d>(x_I.segment(6 + 3 * i, 3));
      });
    auto camera_orientations =
      keyframes | views::transform([&](auto frame_id) -> Eigen::Quaterniond {
        return imu_orientations.at(frame_id) * extrinsic.rotation;
      });

    auto camera_position_perturbation_transform =
      keyframe_index_transform | views::transform([&](auto i) -> Vector3d {
        if (i == 0)
          return Vector3d::Zero();
        return x_V.segment(3 * (i - 1), 3);
      });
    auto camera_positions =
      views::zip(
        camera_orientations, camera_position_nominals | views::values,
        keyframes | camera_position_perturbation_transform) |
      views::transform([](auto const& pair) -> Vector3d {
        auto const& [q_c, p_c, delta_p] = pair;
        return p_c + q_c * delta_p;
      });

    return ImuTranslationMatchSolution {
      .scale = solution.scale,
      .cost = solution.cost,
      .gravity = x_I.head(3),
      .acc_bias = x_I.segment(3, 3),
      .imu_body_velocities = views::zip(keyframes, body_velocities) |
        ranges::to<std::map<FrameID, Vector3d>>,
      .sfm_positions = views::zip(keyframes, camera_positions) |
        ranges::to<std::map<FrameID, Vector3d>>,
    };
  }

  std::optional<ImuTranslationMatch>
  ImuTranslationMatchSolutionParseContext::parse(
    ImuRotationMatch const& rotations,
    std::vector<ImuMatchScaleSampleSolution> const& solutions) {
    auto candidates =  //
      solutions | views::transform([&](auto const& candidate) {
        return std::make_tuple(
          parseSolution(rotations, candidate),
          analyzeImuTranslationMatchUncertainty(analysis, candidate));
      }) |
      ranges::to_vector;

    return acceptor.determineAcceptance(rotations, candidates);
  }

  class ImuTranslationMatchSolverImpl: public ImuTranslationMatchSolver {
  private:
    std::unique_ptr<ImuTranslationMatchAnalyzer> _analyzer;
    std::unique_ptr<ImuTranslationMatchAcceptDiscriminator> _acceptor;
    std::unique_ptr<ImuMatchScaleSampleSolver> _sample_solver;

    std::shared_ptr<CyclopsConfig const> _config;

  public:
    ImuTranslationMatchSolverImpl(
      std::unique_ptr<ImuTranslationMatchAnalyzer> analyzer,
      std::unique_ptr<ImuTranslationMatchAcceptDiscriminator> acceptor,
      std::unique_ptr<ImuMatchScaleSampleSolver> sample_solver,
      std::shared_ptr<CyclopsConfig const> config);
    void reset() override;

    std::optional<ImuTranslationMatch> solve(
      measurement::ImuMotionRefs const& motions,
      ImuRotationMatch const& rotations,
      ImuMatchCameraTranslationPrior const& camera_prior) override;
  };

  ImuTranslationMatchSolverImpl::ImuTranslationMatchSolverImpl(
    std::unique_ptr<ImuTranslationMatchAnalyzer> analyzer,
    std::unique_ptr<ImuTranslationMatchAcceptDiscriminator> acceptor,
    std::unique_ptr<ImuMatchScaleSampleSolver> sample_solver,
    std::shared_ptr<CyclopsConfig const> config)
      : _analyzer(std::move(analyzer)),
        _acceptor(std::move(acceptor)),
        _sample_solver(std::move(sample_solver)),
        _config(config) {
  }

  void ImuTranslationMatchSolverImpl::reset() {
    _analyzer->reset();
    _acceptor->reset();
    _sample_solver->reset();
  }

  std::optional<ImuTranslationMatch> ImuTranslationMatchSolverImpl::solve(
    measurement::ImuMotionRefs const& motions,
    ImuRotationMatch const& rotations,
    ImuMatchCameraTranslationPrior const& camera_prior) {
    auto const& extrinsic = _config->extrinsics.imu_camera_transform;

    auto analysis = _analyzer->analyze(motions, rotations, camera_prior);

    auto cache = ImuTranslationMatchAnalysisCache(analysis);
    auto motion_keyframes =
      camera_prior.translations | views::keys | ranges::to<std::set>;

    auto solutions = _sample_solver->solve(motion_keyframes, analysis, cache);
    if (!solutions.has_value())
      return std::nullopt;

    auto candidate_parse_context = ImuTranslationMatchSolutionParseContext(
      *_config, *_acceptor, camera_prior, analysis);
    return candidate_parse_context.parse(rotations, *solutions);
  }

  std::unique_ptr<ImuTranslationMatchSolver> ImuTranslationMatchSolver::Create(
    std::shared_ptr<CyclopsConfig const> config,
    std::shared_ptr<telemetry::InitializerTelemetry> telemetry) {
    return std::make_unique<ImuTranslationMatchSolverImpl>(
      ImuTranslationMatchAnalyzer::Create(config),
      ImuTranslationMatchAcceptDiscriminator::Create(config, telemetry),
      ImuMatchScaleSampleSolver::Create(config, telemetry), config);
  }
}  // namespace cyclops::initializer
