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
  using std::vector;

  using Eigen::Vector3d;
  namespace views = ranges::views;

  using AcceptDecision = ImuTranslationMatchAcceptDiscriminator::AcceptDecision;
  using RejectReason =
    telemetry::InitializerTelemetry::ImuMatchCandidateRejectReason;

  class ImuTranslationMatchSolutionParseContext {
  private:
    CyclopsConfig const& config;
    ImuTranslationMatchAcceptDiscriminator const& acceptor;
    telemetry::InitializerTelemetry& telemetry;

    ImuRotationMatch const& rotation_match;
    ImuMatchCameraTranslationPrior const& camera_prior;
    ImuTranslationMatchAnalysis const& analysis;

    vector<std::optional<ImuTranslationMatchUncertainty>>
    evaluateSolutionCandidateUncertainty(
      vector<ImuMatchScaleSampleSolution> const& candidates);

    ImuTranslationMatchSolution parseMatchSolution(
      ImuMatchScaleSampleSolution const& solution);

    std::optional<telemetry::InitializerTelemetry::ImuMatchReject>
    makeRejectTelemetry(
      ImuTranslationMatchSolution const& solution,
      ImuTranslationMatchUncertainty const& uncertainty,
      AcceptDecision decision);

    std::optional<ImuTranslationMatch> filterAndReportAccepts(
      vector<ImuTranslationMatchSolution> const& solutions,
      vector<std::optional<ImuTranslationMatchUncertainty>> const&
        uncertainties);

    telemetry::InitializerTelemetry::ImuMatchSolutionPoint
    makeTelemetrySolutionPoint(ImuTranslationMatchSolution const& solution);

    telemetry::InitializerTelemetry::ImuMatchUncertainty
    makeTelemetrySolutionUncertainty(
      ImuTranslationMatchSolution const& solution,
      ImuTranslationMatchUncertainty const& uncertainty);

  public:
    ImuTranslationMatchSolutionParseContext(
      CyclopsConfig const& config,
      ImuTranslationMatchAcceptDiscriminator const& acceptor,
      telemetry::InitializerTelemetry& telemetry,
      ImuRotationMatch const& rotation_match,
      ImuMatchCameraTranslationPrior const& camera_prior,
      ImuTranslationMatchAnalysis const& analysis)
        : config(config),
          acceptor(acceptor),
          telemetry(telemetry),
          rotation_match(rotation_match),
          camera_prior(camera_prior),
          analysis(analysis) {
    }

    std::optional<ImuTranslationMatch> parse(
      vector<ImuMatchScaleSampleSolution> const& candidates);
  };

  vector<std::optional<ImuTranslationMatchUncertainty>>
  ImuTranslationMatchSolutionParseContext::evaluateSolutionCandidateUncertainty(
    vector<ImuMatchScaleSampleSolution> const& candidates) {
    return  //
      candidates | views::transform([&](auto const& candidate) {
        return analyzeImuTranslationMatchUncertainty(analysis, candidate);
      }) |
      ranges::to_vector;
  }

  ImuTranslationMatchSolution
  ImuTranslationMatchSolutionParseContext::parseMatchSolution(
    ImuMatchScaleSampleSolution const& solution) {
    auto const& extrinsic = config.extrinsics.imu_camera_transform;
    auto const& imu_orientations = rotation_match.body_orientations;
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

  telemetry::InitializerTelemetry::ImuMatchSolutionPoint
  ImuTranslationMatchSolutionParseContext::makeTelemetrySolutionPoint(
    ImuTranslationMatchSolution const& solution) {
    return {
      .scale = solution.scale,
      .cost = solution.cost,

      .gravity = solution.gravity,
      .acc_bias = solution.acc_bias,
      .gyr_bias = rotation_match.gyro_bias,
      .imu_orientations = rotation_match.body_orientations,
      .imu_body_velocities = solution.imu_body_velocities,
      .sfm_positions = solution.sfm_positions,
    };
  }

  static auto maxVelocity(ImuTranslationMatchSolution const& solution) {
    double result = 1e-6;
    for (auto const& [_, v] : solution.imu_body_velocities)
      result = std::max<double>(v.norm(), result);
    return result;
  }

  telemetry::InitializerTelemetry::ImuMatchUncertainty
  ImuTranslationMatchSolutionParseContext::makeTelemetrySolutionUncertainty(
    ImuTranslationMatchSolution const& solution,
    ImuTranslationMatchUncertainty const& uncertainty) {
    auto gravity_norm = config.gravity_norm;
    auto sigma_g = uncertainty.gravity_tangent_deviation(0) / gravity_norm;
    auto sigma_v =
      uncertainty.body_velocity_deviation(1) / maxVelocity(solution);

    return {
      .final_cost_significant_probability =
        uncertainty.final_cost_significant_probability,
      .scale_log_deviation = uncertainty.scale_log_deviation,
      .gravity_max_deviation = sigma_g,
      .bias_max_deviation = uncertainty.bias_deviation(0),
      .body_velocity_max_deviation = sigma_v,
      .scale_symmetric_translation_error_max_deviation =
        uncertainty.translation_scale_symmetric_deviation(0),
    };
  }

  std::optional<telemetry::InitializerTelemetry::ImuMatchReject>
  ImuTranslationMatchSolutionParseContext::makeRejectTelemetry(
    ImuTranslationMatchSolution const& solution,
    ImuTranslationMatchUncertainty const& uncertainty,
    AcceptDecision decision) {
    switch (decision) {
    case AcceptDecision::ACCEPT:
      return std::nullopt;

    case AcceptDecision::REJECT_COST_PROBABILITY_INSIGNIFICANT:
      return telemetry::InitializerTelemetry::ImuMatchReject {
        .reason = RejectReason::COST_PROBABILITY_INSIGNIFICANT,
        .solution = makeTelemetrySolutionPoint(solution),
        .uncertainty = makeTelemetrySolutionUncertainty(solution, uncertainty),
      };

    case AcceptDecision::REJECT_UNDERINFORMATIVE_PARAMETER:
      return telemetry::InitializerTelemetry::ImuMatchReject {
        .reason = RejectReason::UNDERINFORMATIVE_PARAMETER,
        .solution = makeTelemetrySolutionPoint(solution),
        .uncertainty = makeTelemetrySolutionUncertainty(solution, uncertainty),
      };

    case AcceptDecision::REJECT_SCALE_LESS_THAN_ZERO:
      return telemetry::InitializerTelemetry::ImuMatchReject {
        .reason = RejectReason::SCALE_LESS_THAN_ZERO,
        .solution = makeTelemetrySolutionPoint(solution),
        .uncertainty = makeTelemetrySolutionUncertainty(solution, uncertainty),
      };
    }
    return std::nullopt;
  }

  std::optional<ImuTranslationMatch>
  ImuTranslationMatchSolutionParseContext::filterAndReportAccepts(
    vector<ImuTranslationMatchSolution> const& solutions,
    vector<std::optional<ImuTranslationMatchUncertainty>> const&
      uncertainties) {
    auto accepts =  //
      views::zip(solutions, uncertainties) |
      views::filter([&](auto const& pair) {
        auto const& [candidate, maybe_uncertainty] = pair;
        if (!maybe_uncertainty.has_value()) {
          telemetry.onImuMatchCandidateReject({
            .reason = RejectReason::UNCERTAINTY_EVALUATION_FAILED,
            .solution = makeTelemetrySolutionPoint(candidate),
            .uncertainty = std::nullopt,
          });
          return false;
        }
        auto const& uncertainty = maybe_uncertainty.value();

        auto decision = acceptor.determineCandidate(candidate, uncertainty);
        auto reject_telemetry =
          makeRejectTelemetry(candidate, uncertainty, decision);
        if (reject_telemetry.has_value()) {
          telemetry.onImuMatchCandidateReject(*reject_telemetry);
          return false;
        }
        return true;
      }) |
      ranges::to_vector;

    if (accepts.size() != 1) {
      auto solutions =  //
        accepts | views::transform([&](auto const& _) {
          auto const& [solution, uncertainty] = _;
          return makeTelemetrySolutionPoint(solution);
        }) |
        ranges::to_vector;
      auto uncertainties =  //
        accepts | views::transform([&](auto const& _) {
          auto const& [solution, uncertainty] = _;
          return makeTelemetrySolutionUncertainty(solution, *uncertainty);
        }) |
        ranges::to_vector;

      telemetry.onImuMatchAmbiguity({solutions, uncertainties});
      return std::nullopt;
    }

    auto const& [solution, uncertainty] = accepts.front();
    auto decision = acceptor.determineAccept(solution, *uncertainty);

    auto reject_telemetry =
      makeRejectTelemetry(solution, *uncertainty, decision);
    if (reject_telemetry.has_value()) {
      telemetry.onImuMatchReject(*reject_telemetry);
      return ImuTranslationMatch {false, solution};
    }

    telemetry.onImuMatchAccept({
      makeTelemetrySolutionPoint(solution),
      makeTelemetrySolutionUncertainty(solution, *uncertainty),
    });
    return ImuTranslationMatch {true, solution};
  }

  std::optional<ImuTranslationMatch>
  ImuTranslationMatchSolutionParseContext::parse(
    vector<ImuMatchScaleSampleSolution> const& candidates) {
    auto uncertainties = evaluateSolutionCandidateUncertainty(candidates);
    auto solutions =  //
      candidates | views::transform([&](auto const& solution) {
        return parseMatchSolution(solution);
      }) |
      ranges::to_vector;
    return filterAndReportAccepts(solutions, uncertainties);
  }

  class ImuTranslationMatchSolverImpl: public ImuTranslationMatchSolver {
  private:
    std::unique_ptr<ImuTranslationMatchAnalyzer> _analyzer;
    std::unique_ptr<ImuTranslationMatchAcceptDiscriminator> _acceptor;
    std::unique_ptr<ImuMatchScaleSampleSolver> _sample_solver;

    std::shared_ptr<CyclopsConfig const> _config;
    std::shared_ptr<telemetry::InitializerTelemetry> _telemetry;

  public:
    ImuTranslationMatchSolverImpl(
      std::unique_ptr<ImuTranslationMatchAnalyzer> analyzer,
      std::unique_ptr<ImuTranslationMatchAcceptDiscriminator> acceptor,
      std::unique_ptr<ImuMatchScaleSampleSolver> sample_solver,
      std::shared_ptr<CyclopsConfig const> config,
      std::shared_ptr<telemetry::InitializerTelemetry> telemetry);
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
    std::shared_ptr<CyclopsConfig const> config,
    std::shared_ptr<telemetry::InitializerTelemetry> telemetry)
      : _analyzer(std::move(analyzer)),
        _acceptor(std::move(acceptor)),
        _sample_solver(std::move(sample_solver)),
        _config(config),
        _telemetry(telemetry) {
  }

  void ImuTranslationMatchSolverImpl::reset() {
    _analyzer->reset();
    _acceptor->reset();
    _sample_solver->reset();
    _telemetry->reset();
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

    auto maybe_candidates =
      _sample_solver->solve(motion_keyframes, analysis, cache);
    if (!maybe_candidates.has_value())
      return std::nullopt;

    auto candidate_parse_context = ImuTranslationMatchSolutionParseContext(
      *_config, *_acceptor, *_telemetry, rotations, camera_prior, analysis);
    return candidate_parse_context.parse(maybe_candidates.value());
  }

  std::unique_ptr<ImuTranslationMatchSolver> ImuTranslationMatchSolver::Create(
    std::shared_ptr<CyclopsConfig const> config,
    std::shared_ptr<telemetry::InitializerTelemetry> telemetry) {
    return std::make_unique<ImuTranslationMatchSolverImpl>(
      ImuTranslationMatchAnalyzer::Create(config),
      ImuTranslationMatchAcceptDiscriminator::Create(config),
      ImuMatchScaleSampleSolver::Create(config, telemetry), config, telemetry);
  }
}  // namespace cyclops::initializer
