#include "cyclops/details/initializer/vision_imu/translation_evaluation.hpp"
#include "cyclops/details/initializer/vision_imu/translation_analysis.hpp"
#include "cyclops/details/initializer/vision_imu/translation_cache.hpp"
#include "cyclops/details/utils/qcqp1.hpp"

namespace cyclops::initializer {
  using Eigen::Matrix3d;
  using Eigen::MatrixXd;
  using Eigen::Vector3d;
  using Eigen::VectorXd;

  ImuMatchScaleEvaluationContext::ImuMatchScaleEvaluationContext(
    double gravity_norm, ImuMatchAnalysis const& analysis,
    ImuMatchAnalysisCache const& cache)
      : gravity_norm(gravity_norm), analysis(analysis), cache(cache) {
  }

  /*
   * represents the following Schur-decomposed system of linear equations,
   * 1. (H_g_bar + mu * I3) * g + b_g_bar = 0,
   * 2. x_q = -(F_q * g + z_q).
   *
   * that is originated from the following system of linear equations,
   *              (H_I_bar + mu * C_I) * x_I + b_I_bar = 0
   *                               <=>
   *       [H_g + mu * I3,   F_g] * [  g  ] + [ b_g ]  = [ 0 ]
   *       [F_g.T,           H_q]   [ x_q ]   [ b_q ]    [ 0 ].
   */
  struct GravityTermReduction {
    Matrix3d H_g_bar;
    Vector3d b_g_bar;
    MatrixXd F_q;
    VectorXd z_q;
  };

  static GravityTermReduction reduceToGravityTerm(
    MatrixXd const& H_I_bar, VectorXd const& b_I_bar) {
    auto dim = H_I_bar.cols();
    auto H_g = H_I_bar.block(0, 0, 3, 3).eval();
    auto H_r = H_I_bar.block(0, 3, 3, dim - 3).eval();
    auto H_q = H_I_bar.block(3, 3, dim - 3, dim - 3).eval();

    Eigen::LDLT<MatrixXd> H_q_inv(H_q);

    auto b_g = b_I_bar.segment(0, 3).eval();
    auto b_q = b_I_bar.segment(3, dim - 3).eval();

    return {
      .H_g_bar = H_g - H_r * H_q_inv.solve(H_r.transpose()),
      .b_g_bar = b_g - H_r * H_q_inv.solve(b_q),
      .F_q = H_q_inv.solve(H_r.transpose()),
      .z_q = H_q_inv.solve(b_q),
    };
  }

  static VectorXd concatenate(VectorXd const& a, VectorXd const& b) {
    VectorXd c(a.size() + b.size());
    c << a, b;
    return c;
  }

  std::optional<ImuMatchScaleEvaluation>
  ImuMatchScaleEvaluationContext::evaluate(double s) const {
    auto [H_I_bar, b_I_bar, F_V, z_V] = cache.inflatePrimal(s);
    auto [H_g_bar, b_g_bar, F_q, z_q] = reduceToGravityTerm(H_I_bar, b_I_bar);
    auto rho_min = -H_g_bar.selfadjointView<Eigen::Upper>().eigenvalues()[0];

    auto [success, rho_solved, g_solved] = solveNormConstrainedQcqp1(
      H_g_bar, 2 * b_g_bar, gravity_norm * gravity_norm, rho_min);

    if (!success)
      return std::nullopt;

    VectorXd x_q_solved = -(F_q * g_solved + z_q);
    VectorXd x_I_solved = concatenate(g_solved, x_q_solved);
    VectorXd x_V_solved = -(F_V * x_I_solved + z_V) * s;

    auto const& [_1, _2, _3, A_I, B_I, A_V, alpha, beta] = analysis;
    auto residual = concatenate(
      A_I * x_I_solved + B_I * x_V_solved * s + alpha + beta * s,
      A_V * x_V_solved);

    auto cost = residual.dot(residual);

    return ImuMatchScaleEvaluation {
      .multiplier = rho_solved,
      .cost = cost,
      .inertial_solution = x_I_solved,
      .visual_solution = x_V_solved,
    };
  }

  double ImuMatchScaleEvaluationContext::evaluateDerivative(
    ImuMatchScaleEvaluation const& evaluation, double s) const {
    auto const& [mu, p, x_I, x_V] = evaluation;
    auto [r_s__dot, b_I_s__dot, b_V_s__dot, F_I, D_I] =
      cache.inflateDerivative(s);

    auto b_s_dot_T__x = b_I_s__dot.dot(x_I) + b_V_s__dot.dot(x_V);
    auto x_T__H_s_dot__x = 2 * (x_I.dot(F_I * x_V) + x_V.dot(D_I * x_V) * s);
    return 2 * b_s_dot_T__x + x_T__H_s_dot__x + r_s__dot;
  }
}  // namespace cyclops::initializer
