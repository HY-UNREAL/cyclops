#include "cyclops/details/initializer/vision_imu/solver.cpp"
#include "cyclops/details/initializer/vision_imu/scale_evaluation.hpp"
#include "cyclops/details/measurement/keyframe.hpp"

#include "cyclops/details/config.hpp"
#include "cyclops/details/logging.hpp"
#include "cyclops/details/telemetry/initializer.hpp"

#include "cyclops_tests/mockups/keyframe_manager.hpp"

#include <Eigen/Dense>
#include <spdlog/fmt/ostr.h>
#include <spdlog/spdlog.h>

#include <random>

#include <doctest/doctest.h>

namespace cyclops::initializer {
  using Eigen::MatrixXd;
  using Eigen::VectorXd;

  using doctest::Approx;

  using measurement::ImuMotionRefs;

  static double randnum(std::mt19937& rgen) {
    return std::normal_distribution<double>(0, 1)(rgen);
  }

  static MatrixXd randmat(std::mt19937& rgen, int n, int m) {
    MatrixXd result(n, m);
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < m; j++)
        result(i, j) = randnum(rgen);
    }
    return result;
  }

  static VectorXd randvec(std::mt19937& rgen, int n) {
    VectorXd result(n);
    for (int i = 0; i < n; i++)
      result(i) = randnum(rgen);
    return result;
  }

  static MatrixXd randmatOrthogonal(std::mt19937& rgen, int n) {
    return randmat(rgen, n, n).fullPivHouseholderQr().matrixQ();
  }

  static MatrixXd randmatWellConditioned(
    std::mt19937& rgen, int n, int m, double average_radius, int order = 8) {
    MatrixXd Sigma = MatrixXd::Zero(n, m);
    for (auto i = 0; i < std::min(n, m); i++) {
      auto r = 0.;
      for (auto _ = 0; _ < order; _++) {
        auto s = randnum(rgen);
        r += s * s;
      }
      r /= order;
      r *= average_radius;
      Sigma(i, i) = r;
    }
    auto U = randmatOrthogonal(rgen, n);
    auto V = randmatOrthogonal(rgen, m);
    return U * Sigma * V;
  }

  template <int rows, int cols>
  static Eigen::Matrix<double, rows, cols> makechol(
    Eigen::Matrix<double, rows, cols> const& A) {
    auto H = (A.transpose() * A).eval();

    Eigen::LLT<Eigen::Matrix<double, rows, cols>> _(H);
    return _.matrixU();
  }

  struct ImuMatchAnalyzerMock: public ImuMatchAnalyzer {
    ImuMatchAnalysis analysis;

    explicit ImuMatchAnalyzerMock(ImuMatchAnalysis analysis)
        : analysis(analysis) {
    }

    void reset() override {
    }

    ImuMatchAnalysis analyze(
      ImuMotionRefs const& _1, ImuMatchMotionPrior const& _2) override {
      return analysis;
    }
  };

  class ImuMatchAcceptDiscriminatorMock: public ImuMatchAcceptDiscriminator {
  public:
    std::vector<ImuMatchResult> determineAcceptance(
      std::vector<ImuMatchCandidate> const& matches) const override {
      auto const& [solution, _] = matches.front();

      return {ImuMatchResult {
        .accept = true,
        .solution = solution,
      }};
    }

    void reset() override {
    }
  };

  TEST_CASE("Test IMU match") {
    std::mt19937 rgen(20220803);
    auto A_I = randmatWellConditioned(rgen, 54, 36, 1e4);
    auto B_I = randmatWellConditioned(rgen, 54, 27, 1e4);

    auto A_V = makechol(randmatWellConditioned(rgen, 27, 27, 1e4));

    auto s_seed = std::uniform_real_distribution<double>(-1, +1)(rgen);
    auto s = 0.1 + std::pow(s_seed, 2) * 9.9;

    CAPTURE(s);
    __logger__->debug("s truth: {}", s);

    auto x_I = randvec(rgen, 36);
    x_I.head(3) = x_I.head(3).normalized().eval();
    x_I.head(3) *= 9.81;

    auto x_V = VectorXd::Zero(27).eval();

    auto alpha = randvec(rgen, 54);
    auto beta = (-(A_I * x_I + B_I * x_V * s + alpha) / s).eval();

    auto analysis_mock = std::make_unique<ImuMatchAnalyzerMock>(
      ImuMatchAnalysis {10, 81, 63, A_I, B_I, A_V, alpha, beta});
    {
      auto cache = ImuMatchAnalysisCache(analysis_mock->analysis);
      auto evaluator =
        ImuMatchScaleEvaluationContext(9.81, analysis_mock->analysis, cache);
      auto maybe_p_primal = evaluator.evaluate(s);
      REQUIRE(static_cast<bool>(maybe_p_primal));
      REQUIRE(maybe_p_primal->cost == Approx(0.));
    }

    auto config = std::make_shared<CyclopsConfig>();
    config->gravity_norm = 9.81;
    config->initialization.imu = config::initializer::ImuSolverConfig {
      .sampling =
        {
          .sampling_domain_lowerbound = 0.001,
          .sampling_domain_upperbound = 100,
          .samples_count = 200,
          .min_evaluation_success_rate = 0.8,
        },
      .refinement =
        {
          .max_iteration = 100,
          .stepsize_tolerance = 1e-6,
          .gradient_tolerance = 1e-8,
        },
      .candidate_test =
        {
          .cost_significance = 0,
        },
      .acceptance_test =
        {
          .max_scale_log_deviation = 1e6,
          .max_normalized_gravity_deviation = 1e6,
          .max_normalized_velocity_deviation = 1e6,
          .max_sfm_perturbation = 1e6,
          .translation_match_min_p_value = 0,
        },
    };

    std::shared_ptr telemetry =
      telemetry::InitializerTelemetry::CreateDefault();
    auto solver = ImuMatchSolverImpl(
      std::move(analysis_mock),
      std::make_unique<ImuMatchAcceptDiscriminatorMock>(),
      ImuMatchScaleSampleSolver::Create(config, telemetry), config);
    auto maybe_solution = solver.solve({}, {});
    REQUIRE(maybe_solution.has_value());
    REQUIRE(maybe_solution->size() == 1);

    auto const& solution = maybe_solution->front();
    REQUIRE(solution.accept);

    auto const& s_optimized = solution.solution.scale;
    {
      CAPTURE(s_optimized);
      CHECK(std::abs(s_optimized - s) < 1e-6);
    }
    {
      auto g = x_I.head(3).eval();
      auto const& g_optimized = solution.solution.gravity;

      CAPTURE(g.transpose());
      CAPTURE(g_optimized.transpose());
      CHECK(g_optimized.isApprox(g, 1e-6));
    }
    {
      auto b_a = x_I.segment(3, 3).eval();
      auto const& b_a_optimized = solution.solution.acc_bias;

      CAPTURE(b_a.transpose());
      CAPTURE(b_a_optimized.transpose());
      CHECK(b_a_optimized.isApprox(b_a, 1e-6));
    }
  }
}  // namespace cyclops::initializer
