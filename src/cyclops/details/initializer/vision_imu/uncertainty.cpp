#include "cyclops/details/initializer/vision_imu/uncertainty.hpp"
#include "cyclops/details/initializer/vision_imu/analysis.hpp"
#include "cyclops/details/initializer/vision_imu/scale_sample.hpp"
#include "cyclops/details/initializer/vision_imu/type.hpp"
#include "cyclops/details/utils/math.hpp"

#include "cyclops/details/logging.hpp"
#include <spdlog/spdlog.h>
#include <spdlog/fmt/ostr.h>

namespace cyclops::initializer {
  using Eigen::MatrixXd;
  using Eigen::VectorXd;

  static std::optional<std::tuple<MatrixXd, MatrixXd>>
  computeMarginalInformationPair(MatrixXd const& H, int p) {
    __logger__->trace("Computing marginal information pair");
    __logger__->trace("Matrix size: <{}, {}>", H.rows(), H.cols());
    __logger__->trace("Partition dimension: {}", p);

    int n = H.rows();
    int m = H.cols();

    if (n != m) {
      __logger__->error(
        "Input matrix is not square during marginal information pair "
        "computation");
      return std::nullopt;
    }

    if (n < p) {
      __logger__->error(
        "Partition dimension exceeds the matrix dimension during marginal "
        "information pair computation");
      return std::nullopt;
    }

    auto q = n - p;

#define H_a (H.topLeftCorner(p, p))
#define H_b (H.bottomRightCorner(q, q))
#define H_r (H.topRightCorner(p, q))

    Eigen::LDLT<MatrixXd> H_b_llt(H_b);
    if (H_b_llt.info() != Eigen::Success) {
      __logger__->debug("Cholesky decomposition failed.");
      return std::nullopt;
    }
    MatrixXd H_a_bar = H_a - H_r * H_b_llt.solve(H_r.transpose());

    Eigen::LDLT<MatrixXd> H_a_llt(H_a);
    if (H_a_llt.info() != Eigen::Success) {
      __logger__->debug("Cholesky decomposition failed.");
      return std::nullopt;
    }
    MatrixXd H_b_bar = H_b - H_r.transpose() * H_a_llt.solve(H_r);

#undef H_r
#undef H_b
#undef H_a

    return std::make_tuple(H_a_bar, H_b_bar);
  }

  static auto marginalizeOrDie(MatrixXd const& H, int dim, std::string tag) {
    auto result = computeMarginalInformationPair(H, dim);
    if (!result)
      __logger__->debug("{} marginalization failed", tag);
    return result;
  }

  static std::optional<std::vector<MatrixXd>> evaluateMarginalizationChain(
    Eigen::MatrixXd const& matrix,
    std::vector<std::tuple<int, std::string>> const& chain) {
    Eigen::MatrixXd H = matrix;

    std::vector<MatrixXd> result;
    for (auto [dimension, tag] : chain) {
      auto marginalization = marginalizeOrDie(H, dimension, tag);
      if (!marginalization.has_value())
        return std::nullopt;

      auto const& [current, rest] = marginalization.value();

      result.push_back(current);
      H = rest;
    }
    result.push_back(H);

    return result;
  }

  static std::optional<VectorXd> computePositiveEigenvalues(
    MatrixXd const& matrix) {
    Eigen::SelfAdjointEigenSolver<MatrixXd> eigen(matrix);
    if (eigen.info() != Eigen::Success) {
      __logger__->debug(
        "Eigendecomposition failed during IMU match uncertainty analysis");
      return std::nullopt;
    }

    VectorXd lambda = eigen.eigenvalues();
    if (lambda.size() != 0 && lambda(0) <= 0) {
      __logger__->debug(
        "Semi-or-indefinite information matrix during IMU match uncertainty "
        "analysis");
      return std::nullopt;
    }

    return lambda;
  }

  static auto eigenvalueOrDie(MatrixXd const& matrix, std::string tag) {
    auto maybe_lambda = computePositiveEigenvalues(matrix);
    if (!maybe_lambda)
      __logger__->debug("{} eigenvalue computation failed.", tag);
    return maybe_lambda;
  };

  static std::optional<std::vector<VectorXd>>
  evaluateChainMarginalizationEigenvalues(
    std::vector<std::tuple<MatrixXd, std::string>> const& chain) {
    std::vector<VectorXd> result;
    for (auto const& [H, tag] : chain) {
      auto maybe_lambda = eigenvalueOrDie(H, tag);
      if (!maybe_lambda.has_value())
        return std::nullopt;

      result.push_back(maybe_lambda.value());
    }
    return result;
  }

  static bool checkHessianDimension(MatrixXd const& H, int dimension) {
    if (H.rows() != H.cols()) {
      __logger__->error("IMU match hessian is not a square matrix");
      return false;
    }

    auto n = H.cols();
    if (n != dimension) {
      __logger__->error("IMU match hessian dimension mismatch");
      __logger__->debug("{} vs {}", n, dimension);
      return false;
    }
    return true;
  }

  std::optional<double> analyzeImuMatchCostProbability(
    int residual_dimension, int parameter_dimension, double cost) {
    int degrees_of_freedom = residual_dimension - parameter_dimension;
    if (degrees_of_freedom <= 0)
      return std::nullopt;

    __logger__->debug("Degrees of freedom: {}", degrees_of_freedom);
    return 1 - chiSquaredCdf(degrees_of_freedom, cost);
  }

  std::optional<ImuMatchUncertainty> analyzeImuOnlyMatchUncertainty(
    int frames_count, Eigen::MatrixXd const& H, double cost_p_value) {
    __logger__->debug("Analyzing the uncertainty of IMU-only match");

    if (!checkHessianDimension(H, 3 * frames_count + 6))
      return std::nullopt;

    auto chain = evaluateMarginalizationChain(
      H, {{1, "Scale"}, {2, "Gravity"}, {3, "Bias"}});
    if (!chain.has_value())
      return std::nullopt;

    auto const& H_s = chain->at(0);
    auto const& H_g = chain->at(1);
    auto const& H_b = chain->at(2);
    auto const& H_v = chain->at(3);

    auto lambda_s = H_s(0, 0);
    if (lambda_s <= 0) {
      __logger__->debug("Scale information is negative definite.");
      return std::nullopt;
    }

    auto eigenvalues = evaluateChainMarginalizationEigenvalues(
      {{H_g, "Gravity"}, {H_b, "Bias"}, {H_v, "Velocity"}});
    if (!eigenvalues.has_value())
      return std::nullopt;

    auto const& lambda_g = eigenvalues->at(0);
    auto const& lambda_b = eigenvalues->at(1);
    auto const& lambda_v = eigenvalues->at(2);

    return ImuMatchUncertainty {
      .final_cost_significant_probability = cost_p_value,
      .scale_log_deviation = 1 / std::sqrt(lambda_s),
      .gravity_tangent_deviation = lambda_g.cwiseSqrt().cwiseInverse(),
      .bias_deviation = lambda_b.cwiseSqrt().cwiseInverse(),
      .body_velocity_deviation = lambda_v.cwiseSqrt().cwiseInverse(),
      .translation_scale_symmetric_deviation =
        VectorXd::Zero(3 * frames_count - 3),
    };
  }

  std::optional<ImuMatchUncertainty> analyzeImuMatchUncertainty(
    int frames_count, Eigen::MatrixXd const& H, double cost_p_value) {
    __logger__->debug("Analyzing the uncertainty of IMU match");

    if (!checkHessianDimension(H, 6 * frames_count + 3))
      return std::nullopt;

    auto v_dim = 3 * frames_count;
    auto chain = evaluateMarginalizationChain(
      H, {{1, "Scale"}, {2, "Gravity"}, {3, "Bias"}, {v_dim, "Velocity"}});

    if (!chain.has_value())
      return std::nullopt;

    auto const& H_s = chain->at(0);
    auto const& H_g = chain->at(1);
    auto const& H_b = chain->at(2);
    auto const& H_v = chain->at(3);
    auto const& H_p = chain->at(4);

    auto lambda_s = H_s(0, 0);
    if (lambda_s <= 0) {
      __logger__->debug("Scale information is negative definite.");
      return std::nullopt;
    }

    auto eigenvalues = evaluateChainMarginalizationEigenvalues(
      {{H_g, "Gravity"}, {H_b, "Bias"}, {H_v, "Velocity"}, {H_p, "Position"}});
    if (!eigenvalues.has_value())
      return std::nullopt;

    auto const& lambda_g = eigenvalues->at(0);
    auto const& lambda_b = eigenvalues->at(1);
    auto const& lambda_v = eigenvalues->at(2);
    auto const& lambda_p = eigenvalues->at(3);

    return ImuMatchUncertainty {
      .final_cost_significant_probability = cost_p_value,
      .scale_log_deviation = 1 / std::sqrt(lambda_s),
      .gravity_tangent_deviation = lambda_g.cwiseSqrt().cwiseInverse(),
      .bias_deviation = lambda_b.cwiseSqrt().cwiseInverse(),
      .body_velocity_deviation = lambda_v.cwiseSqrt().cwiseInverse(),
      .translation_scale_symmetric_deviation =
        lambda_p.cwiseSqrt().cwiseInverse(),
    };
  }

  std::optional<ImuMatchUncertainty> analyzeImuMatchUncertainty(
    ImuMatchAnalysis const& analysis,
    ImuMatchScaleSampleSolution const& solution) {
    auto s = solution.scale;
    auto const& x_I = solution.inertial_state;
    auto const& x_V = solution.visual_state;
    auto const& [_1, _2, _3, A_I, B_I, A_V, alpha, beta] = analysis;

    auto r_I = (A_I * x_I + B_I * x_V * s + alpha + beta * s).eval();
    auto r_V = (A_V * x_V).eval();
    auto cost = r_I.dot(r_I) + r_V.dot(r_V);

    auto cost_p_value = analyzeImuMatchCostProbability(
      analysis.residual_dimension, analysis.parameter_dimension, cost);
    if (!cost_p_value.has_value())
      return std::nullopt;

    return analyzeImuMatchUncertainty(
      analysis.frames_count, solution.hessian, cost_p_value.value());
  }
}  // namespace cyclops::initializer
