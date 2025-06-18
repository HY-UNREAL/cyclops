#include "cyclops/details/initializer/vision_imu/translation.imu_only.hpp"
#include "cyclops/details/initializer/vision_imu/acceptance.imu_only.hpp"

#include "cyclops/details/initializer/vision_imu/camera_motion_prior.hpp"
#include "cyclops/details/initializer/vision_imu/rotation.hpp"
#include "cyclops/details/initializer/vision_imu/translation_analysis.hpp"
#include "cyclops/details/initializer/vision_imu/translation_hessian.hpp"
#include "cyclops/details/initializer/vision_imu/uncertainty.hpp"

#include "cyclops/details/utils/qcqp1.hpp"

#include "cyclops/details/config.hpp"

#include <range/v3/all.hpp>

#include <algorithm>
#include <set>
#include <vector>

namespace cyclops::initializer {
  using Eigen::Matrix3d;
  using Eigen::MatrixXd;
  using Eigen::Vector3d;
  using Eigen::VectorXd;

  using measurement::ImuMotionRefs;

  namespace views = ranges::views;

  using MsfmPositions = std::map<FrameID, Vector3d>;

  using Scale = double;
  using Cost = double;
  using ImuOnlyMatchSolution = std::tuple<VectorXd, Scale, Cost>;

  class ImuOnlyTranslationMatchSolverImpl:
      public ImuOnlyTranslationMatchSolver {
  private:
    std::unique_ptr<ImuTranslationMatchAnalyzer> _analyzer;
    std::unique_ptr<ImuOnlyTranslationMatchAcceptDiscriminator> _acceptor;

    std::shared_ptr<CyclopsConfig const> _config;

    std::optional<ImuOnlyMatchSolution> solve(
      ImuTranslationMatchAnalysis const& analysis);

    ImuOnlyMatchSolution parseSolution(
      ImuTranslationMatchAnalysis const& analysis, VectorXd const& x) const;
    ImuTranslationMatchSolution parseSolution(
      MsfmPositions const& msfm_positions,
      ImuOnlyMatchSolution const& match_solution) const;

    std::optional<ImuTranslationMatchUncertainty> analyzeUncertainty(
      ImuTranslationMatchAnalysis const& analysis,
      ImuOnlyMatchSolution const& match_solution) const;

  public:
    ImuOnlyTranslationMatchSolverImpl(
      std::unique_ptr<ImuTranslationMatchAnalyzer> analyzer,
      std::unique_ptr<ImuOnlyTranslationMatchAcceptDiscriminator> acceptor,
      std::shared_ptr<CyclopsConfig const> config);

    std::optional<ImuTranslationMatch> solve(
      ImuMotionRefs const& motions, ImuRotationMatch const& rotation_match,
      ImuMatchCameraTranslationPrior const& camera_prior) override;
    void reset() override;
  };

  std::optional<ImuOnlyMatchSolution> ImuOnlyTranslationMatchSolverImpl::solve(
    ImuTranslationMatchAnalysis const& analysis) {
    auto const& [_1, _2, _3, A_I, _4, _5, alpha, beta] = analysis;
    auto gravity_norm = _config->gravity_norm;

    MatrixXd H_I = A_I.transpose() * A_I;
    VectorXd A_I_T__beta = A_I.transpose() * beta;

    auto n = A_I.cols();
    if (n <= 2)
      return std::nullopt;

    auto P = MatrixXd(n + 1, n + 1);
    // clang-format off
    P <<
      H_I, A_I_T__beta,
      A_I_T__beta.transpose(), beta.dot(beta);
    // clang-format on

    auto p = VectorXd(n + 1);
    p << 2 * A_I.transpose() * alpha, 2 * beta.dot(alpha);

    Matrix3d P_brev = P.topLeftCorner(3, 3);
    MatrixXd F = P.topRightCorner(3, n - 2);
    MatrixXd Q = P.bottomRightCorner(n - 2, n - 2);

    Vector3d p_brev = p.head(3);
    Vector3d q = p.tail(n - 2);

    Eigen::JacobiSVD<MatrixXd> Q_inv(
      Q, Eigen::ComputeThinU | Eigen::ComputeThinV);

    Matrix3d P_dagger = P_brev - F * Q_inv.solve(F.transpose());
    Vector3d p_dagger = p_brev - F * Q_inv.solve(q);

    auto rho_min = -P_dagger.selfadjointView<Eigen::Upper>().eigenvalues()[0];
    auto solution = solveNormConstrainedQcqp1(
      P_dagger, p_dagger, gravity_norm * gravity_norm, rho_min);

    if (!solution.success)
      return std::nullopt;

    VectorXd result = VectorXd(n + 1);
    result << solution.x, -Q_inv.solve(F.transpose() * solution.x + q);

    return parseSolution(analysis, result);
  }

  ImuOnlyMatchSolution ImuOnlyTranslationMatchSolverImpl::parseSolution(
    ImuTranslationMatchAnalysis const& analysis, VectorXd const& x) const {
    auto const& [n_frames, _2, _3, A_I, _4, _5, alpha, beta] = analysis;

    auto x_I = x.head(6 + 3 * n_frames).eval();
    auto s = x(6 + 3 * n_frames);
    auto r = (A_I * x_I + beta * s + alpha).eval();

    return std::make_tuple(x_I, s, r.dot(r));
  }

  ImuTranslationMatchSolution ImuOnlyTranslationMatchSolverImpl::parseSolution(
    MsfmPositions const& msfm_positions,
    ImuOnlyMatchSolution const& match_solution) const {
    auto const& [x_I, s, cost] = match_solution;

    auto velocities =  //
      views::enumerate(msfm_positions | views::keys) |
      views::transform([x_I = x_I](auto const& element) {
        auto [n, frame_id] = element;
        auto v_n = x_I.segment(6 + 3 * n, 3).eval();

        return std::make_pair(frame_id, v_n);
      }) |
      ranges::to<std::map<FrameID, Vector3d>>;

    return ImuTranslationMatchSolution {
      .scale = s,
      .cost = cost,
      .gravity = x_I.head(3),
      .acc_bias = x_I.segment(3, 3),
      .imu_body_velocities = velocities,
      .sfm_positions = msfm_positions,
    };
  }

  std::optional<ImuTranslationMatchUncertainty>
  ImuOnlyTranslationMatchSolverImpl::analyzeUncertainty(
    ImuTranslationMatchAnalysis const& analysis,
    ImuOnlyMatchSolution const& match_solution) const {
    auto const& [n_frames, _1, _2, _3, _4, A_V, alpha, beta] = analysis;
    auto const& [x_I, s, cost] = match_solution;

    auto x_V = VectorXd::Zero(A_V.cols()).eval();
    auto n_V = x_V.size();

    // Since we are applying virtual constraint x_V = 0 during the IMU-only
    // match, assign high information intensity to the x_V part of the Hessian
    // matrix that will be used for the parameter observability test.
    auto H = evaluateImuTranslationMatchHessian(analysis, s, x_I, x_V);
    H.bottomRightCorner(n_V, n_V) = 1e12 * MatrixXd::Identity(n_V, n_V);

    int residual_dimension = 3 + 6 * (n_frames - 1);
    int parameter_dimension = 6 + 3 * n_frames;

    auto cost_p_value = analyzeImuTranslationMatchCostProbability(
      residual_dimension, parameter_dimension, cost);
    if (!cost_p_value.has_value())
      return std::nullopt;

    return analyzeImuTranslationMatchUncertainty(
      n_frames, H, cost_p_value.value());
  }

  std::optional<ImuTranslationMatch> ImuOnlyTranslationMatchSolverImpl::solve(
    ImuMotionRefs const& motions, ImuRotationMatch const& rotation_match,
    ImuMatchCameraTranslationPrior const& camera_prior) {
    auto analysis = _analyzer->analyze(motions, rotation_match, camera_prior);

    auto maybe_solution = solve(analysis);
    if (!maybe_solution.has_value())
      return std::nullopt;

    auto maybe_uncertainty =
      analyzeUncertainty(analysis, maybe_solution.value());
    auto translation_match =
      parseSolution(camera_prior.translations, maybe_solution.value());

    auto accept = _acceptor->determineAccept(
      rotation_match, std::make_tuple(translation_match, maybe_uncertainty));

    return ImuTranslationMatch {
      .accept = accept,
      .solution = translation_match,
    };
  }

  void ImuOnlyTranslationMatchSolverImpl::reset() {
    _analyzer->reset();
    _acceptor->reset();
  }

  ImuOnlyTranslationMatchSolverImpl::ImuOnlyTranslationMatchSolverImpl(
    std::unique_ptr<ImuTranslationMatchAnalyzer> analyzer,
    std::unique_ptr<ImuOnlyTranslationMatchAcceptDiscriminator> acceptor,
    std::shared_ptr<CyclopsConfig const> config)
      : _analyzer(std::move(analyzer)),
        _acceptor(std::move(acceptor)),
        _config(config) {
  }

  std::unique_ptr<ImuOnlyTranslationMatchSolver>
  ImuOnlyTranslationMatchSolver::Create(
    std::shared_ptr<CyclopsConfig const> config,
    std::shared_ptr<telemetry::InitializerTelemetry> telemetry) {
    return std::make_unique<ImuOnlyTranslationMatchSolverImpl>(
      ImuTranslationMatchAnalyzer::Create(config),
      ImuOnlyTranslationMatchAcceptDiscriminator::Create(config, telemetry),
      config);
  }
}  // namespace cyclops::initializer
