#include "cyclops/details/initializer/vision_imu/solver.imu_only.hpp"
#include "cyclops/details/initializer/vision_imu/acceptance.imu_only.hpp"
#include "cyclops/details/initializer/vision_imu/analysis.hpp"
#include "cyclops/details/initializer/vision_imu/hessian.hpp"
#include "cyclops/details/initializer/vision_imu/motion_prior.hpp"
#include "cyclops/details/initializer/vision_imu/type.hpp"
#include "cyclops/details/initializer/vision_imu/uncertainty.hpp"

#include "cyclops/details/utils/qcqp1.hpp"

#include "cyclops/details/config.hpp"
#include "cyclops/details/logging.hpp"

#include <range/v3/all.hpp>
#include <spdlog/spdlog.h>

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

  class ImuOnlyMatchSolverImpl: public ImuOnlyMatchSolver {
  private:
    std::unique_ptr<ImuMatchAnalyzer> _analyzer;
    std::unique_ptr<ImuOnlyMatchAcceptDiscriminator> _acceptor;

    std::shared_ptr<CyclopsConfig const> _config;

    std::optional<ImuOnlyMatchSolution> solve(ImuMatchAnalysis const& analysis);

    ImuOnlyMatchSolution parseSolution(
      ImuMatchAnalysis const& analysis, VectorXd const& x) const;
    ImuMatchSolution parseSolution(
      ImuMatchMotionPrior const& camera_prior,
      ImuOnlyMatchSolution const& match_solution) const;

    std::optional<ImuMatchUncertainty> analyzeUncertainty(
      ImuMatchAnalysis const& analysis,
      ImuOnlyMatchSolution const& match_solution) const;

  public:
    ImuOnlyMatchSolverImpl(
      std::unique_ptr<ImuMatchAnalyzer> analyzer,
      std::unique_ptr<ImuOnlyMatchAcceptDiscriminator> acceptor,
      std::shared_ptr<CyclopsConfig const> config);

    std::optional<std::vector<ImuMatchResult>> solve(
      ImuMotionRefs const& motions,
      ImuMatchMotionPrior const& camera_prior) override;
    void reset() override;
  };

  std::optional<ImuOnlyMatchSolution> ImuOnlyMatchSolverImpl::solve(
    ImuMatchAnalysis const& analysis) {
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
    VectorXd q = p.tail(n - 2);

    Eigen::JacobiSVD<MatrixXd> Q_inv(
      Q, Eigen::ComputeThinU | Eigen::ComputeThinV);

    Matrix3d P_dagger = P_brev - F * Q_inv.solve(F.transpose());
    Vector3d p_dagger = p_brev - F * Q_inv.solve(q);

    auto rho_min = -P_dagger.selfadjointView<Eigen::Upper>().eigenvalues()[0];
    auto solution = solveNormConstrainedQcqp1(
      P_dagger, p_dagger, gravity_norm * gravity_norm, rho_min, 100, 1e-5);

    if (!solution.success) {
      __logger__->info(
        "IMU-only scale initialization QCQP-1C evaluation failed.");
      return std::nullopt;
    }

    VectorXd result = VectorXd(n + 1);
    result << solution.x, -Q_inv.solve(F.transpose() * solution.x + q);

    return parseSolution(analysis, result);
  }

  ImuOnlyMatchSolution ImuOnlyMatchSolverImpl::parseSolution(
    ImuMatchAnalysis const& analysis, VectorXd const& x) const {
    auto const& [n_frames, _2, _3, A_I, _4, _5, alpha, beta] = analysis;

    auto x_I = x.head(6 + 3 * n_frames).eval();
    auto s = x(6 + 3 * n_frames);
    auto r = (A_I * x_I + beta * s + alpha).eval();

    return std::make_tuple(x_I, s, r.dot(r));
  }

  ImuMatchSolution ImuOnlyMatchSolverImpl::parseSolution(
    ImuMatchMotionPrior const& camera_prior,
    ImuOnlyMatchSolution const& match_solution) const {
    auto const& [x_I, s, cost] = match_solution;

    auto velocities =  //
      views::enumerate(camera_prior.camera_positions | views::keys) |
      views::transform([x_I = x_I](auto const& element) {
        auto [n, frame_id] = element;
        auto v_n = x_I.segment(6 + 3 * n, 3).eval();

        return std::make_pair(frame_id, v_n);
      }) |
      ranges::to<std::map<FrameID, Vector3d>>;

    return ImuMatchSolution {
      .scale = s,
      .cost = cost,
      .gravity = x_I.head(3),
      .acc_bias = x_I.segment(3, 3),
      .gyr_bias = camera_prior.gyro_bias,
      .body_velocities = velocities,
      .body_orientations = camera_prior.imu_orientations,
      .sfm_positions = camera_prior.camera_positions,
    };
  }

  std::optional<ImuMatchUncertainty> ImuOnlyMatchSolverImpl::analyzeUncertainty(
    ImuMatchAnalysis const& analysis,
    ImuOnlyMatchSolution const& match_solution) const {
    auto const& [n_frames, _1, _2, _3, _4, A_V, alpha, beta] = analysis;
    auto const& [x_I, s, cost] = match_solution;
    auto n_I = x_I.size();

    auto x_V = VectorXd::Zero(A_V.cols()).eval();

    // Since we are applying virtual constraint x_V = 0 during the IMU-only
    // match, assign high information intensity to the x_V part of the Hessian
    // matrix that will be used for the parameter observability test.
    auto H = evaluateImuMatchHessian(analysis, s, x_I, x_V);
    auto H_I = H.topLeftCorner(n_I, n_I).eval();

    int residual_dimension = 3 + 6 * (n_frames - 1);
    int parameter_dimension = 6 + 3 * n_frames;

    auto cost_p_value = analyzeImuMatchCostProbability(
      residual_dimension, parameter_dimension, cost);
    if (!cost_p_value.has_value())
      return std::nullopt;

    return analyzeImuOnlyMatchUncertainty(n_frames, H_I, cost_p_value.value());
  }

  std::optional<std::vector<ImuMatchResult>> ImuOnlyMatchSolverImpl::solve(
    ImuMotionRefs const& motions, ImuMatchMotionPrior const& camera_prior) {
    auto analysis = _analyzer->analyze(motions, camera_prior);

    auto maybe_solution = solve(analysis);
    if (!maybe_solution.has_value())
      return std::nullopt;

    auto uncertainty = analyzeUncertainty(analysis, maybe_solution.value());
    auto solution = parseSolution(camera_prior, maybe_solution.value());
    auto candidate = std::make_tuple(solution, uncertainty);

    auto result = ImuMatchResult {
      .accept = _acceptor->determineAccept(candidate),
      .solution = solution,
    };
    return std::vector<ImuMatchResult> {result};
  }

  void ImuOnlyMatchSolverImpl::reset() {
    _analyzer->reset();
    _acceptor->reset();
  }

  ImuOnlyMatchSolverImpl::ImuOnlyMatchSolverImpl(
    std::unique_ptr<ImuMatchAnalyzer> analyzer,
    std::unique_ptr<ImuOnlyMatchAcceptDiscriminator> acceptor,
    std::shared_ptr<CyclopsConfig const> config)
      : _analyzer(std::move(analyzer)),
        _acceptor(std::move(acceptor)),
        _config(config) {
  }

  std::unique_ptr<ImuOnlyMatchSolver> ImuOnlyMatchSolver::Create(
    std::shared_ptr<CyclopsConfig const> config,
    std::shared_ptr<telemetry::InitializerTelemetry> telemetry) {
    return std::make_unique<ImuOnlyMatchSolverImpl>(
      ImuMatchAnalyzer::Create(config),
      ImuOnlyMatchAcceptDiscriminator::Create(config, telemetry), config);
  }
}  // namespace cyclops::initializer
