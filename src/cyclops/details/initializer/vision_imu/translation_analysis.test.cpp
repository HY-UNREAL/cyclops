#include "cyclops/details/initializer/vision_imu/translation_analysis.hpp"
#include "cyclops/details/initializer/vision_imu/camera_motion_prior.hpp"
#include "cyclops/details/initializer/vision_imu/rotation.hpp"

#include "cyclops/details/measurement/preintegration.hpp"
#include "cyclops/details/config.hpp"

#include <range/v3/all.hpp>
#include <doctest/doctest.h>

namespace cyclops::initializer {
  namespace views = ranges::views;

  using Eigen::MatrixXd;
  using Eigen::Quaterniond;
  using Eigen::Vector3d;
  using Eigen::VectorXd;

  using measurement::ImuMotionRef;
  using measurement::ImuMotions;
  using measurement::ImuMotion;

  template <
    typename orientation_signal_t, typename velocity_signal_t,
    typename position_signal_t>
  static auto makeImuMotions(
    orientation_signal_t q, velocity_signal_t v, position_signal_t p,
    Vector3d const& gravity, std::vector<double> times) {
    auto n = static_cast<int>(times.size());
    auto time_pairs =
      views::zip(times | views::slice(0, n - 1), times | views::slice(1, n));
    auto index_pairs = views::zip(views::ints(0, n - 1), views::ints(1, n));

    ImuMotions result;
    for (auto const& [time_pair, index_pair] :
         views::zip(time_pairs, index_pairs)) {
      auto const& [t_s, t_e] = time_pair;
      auto const& [n_s, n_e] = index_pair;
      auto dt = (t_e - t_s);

      result.emplace_back(ImuMotion {
        static_cast<FrameID>(n_s), static_cast<FrameID>(n_e),
        std::make_unique<measurement::ImuPreintegration>(
          Vector3d::Zero(), Vector3d::Zero(),
          measurement::ImuNoise {0.0, 0.0})});
      auto const& imu = result.back().data;
      imu->time_delta = dt;
      imu->rotation_delta = q(t_s).conjugate() * q(t_e);
      imu->velocity_delta =
        q(t_s).conjugate() * (v(t_e) - v(t_s) + gravity * dt);
      imu->position_delta = q(t_s).conjugate() *
        (p(t_e) - p(t_s) - v(t_s) * dt + gravity * dt * dt / 2);
      imu->covariance = Eigen::Matrix<double, 9, 9>::Identity();
      imu->bias_jacobian = Eigen::Matrix<double, 9, 6>::Zero();
    }
    return result;
  }

  template <int dim>
  using Vector = Eigen::Matrix<double, dim, 1>;

  template <int dim>
  static VectorXd concatenate(Vector<dim> const& v) {
    return v;
  }

  template <int vector1_dim, int vector2_dim, int... rest_dims>
  static auto concatenate(
    Vector<vector1_dim> const& v1, Vector<vector2_dim> const& v2,
    Vector<rest_dims> const&... rest) {
    VectorXd v3(v1.size() + v2.size());
    v3 << v1, v2;
    return concatenate(v3, rest...);
  }

  template <int dim>
  static VectorXd concatenate(std::vector<Vector<dim>> const& vs) {
    VectorXd result(ranges::accumulate(
      vs | views::transform([](auto const& v) { return v.size(); }), 0));

    int n = 0;
    for (auto const& v : vs) {
      result.segment(n, v.size()) = v;
      n += v.size();
    }
    return result;
  }

  TEST_CASE("test visual-inertial translation matching analysis") {
    std::mt19937 rgen(20220511);
    auto rand = [&rgen]() {
      return std::uniform_real_distribution<double>(-1, 1)(rgen);
    };

    auto const rotation_axis =
      Vector3d(rand(), rand(), rand()).normalized().eval();
    auto const translation_axis =
      Vector3d(rand(), rand(), rand()).normalized().eval();
    auto const extrinsic = SE3Transform {
      Vector3d(rand(), rand(), rand()),
      Quaterniond(rand(), rand(), rand(), rand()).normalized()};
    auto const scale = 0.4;

    auto gravity =
      (9.81 * Vector3d(rand(), rand(), rand()).normalized()).eval();

    auto orientation_signal = [&](double t) -> Quaterniond {
      return Quaterniond(Eigen::AngleAxisd(0.1 * t * t, rotation_axis));
    };
    auto position_signal = [&](double t) -> Vector3d {
      auto x = (1 + 0.1 * sin(t / M_PI)) * t;
      return x * translation_axis;
    };
    auto velocity_signal = [&](auto t) -> Vector3d {
      auto constexpr h = 1e-6;
      return (position_signal(t + h) - position_signal(t - h)) / 2 / h;
    };

    auto camera_position_signal = [&](double t) -> Vector3d {
      auto const& [p_bc, q_bc] = extrinsic;
      auto p_b = position_signal(t);
      auto q_b = orientation_signal(t);
      return p_b + q_b * p_bc;
    };
    auto camera_orientation_signal = [&](double t) -> Quaterniond {
      auto const& [p_bc, q_bc] = extrinsic;
      auto q_b = orientation_signal(t);
      return q_b * q_bc;
    };
    auto body_velocity_signal = [&](double t) -> Vector3d {
      auto q_b = orientation_signal(t);
      return q_b.conjugate() * velocity_signal(t);
    };

    auto position_perturbations = std::vector {
      (0.1 * Vector3d(rand(), rand(), rand())).eval(),
      (0.1 * Vector3d(rand(), rand(), rand())).eval(),
    };
    auto make_camera_position_prior = [&](auto i) -> Vector3d {
      auto p_c = camera_position_signal(0.1 * i);
      auto q_c = camera_orientation_signal(0.1 * i);
      if (i == 0)
        return p_c / scale;

      auto const& dp = position_perturbations.at(i - 1);
      return (p_c / scale - q_c * dp);
    };
    auto position_prior = ImuMatchCameraTranslationPrior {
      .translations =
        {
          {0, make_camera_position_prior(0)},
          {1, make_camera_position_prior(1)},
          {2, make_camera_position_prior(2)},
        },
      .weight = 1e4 * MatrixXd::Identity(6, 6),
    };
    auto orientations = ImuRotationMatch {
      .gyro_bias = Vector3d::Zero(),
      .body_orientations =
        {
          {0, orientation_signal(0.0)},
          {1, orientation_signal(0.1)},
          {2, orientation_signal(0.2)},
        },
    };
    auto times = std::vector {0.0, 0.1, 0.2};

    auto imu_motions = makeImuMotions(
      orientation_signal, velocity_signal, position_signal, gravity,
      std::vector {0.0, 0.1, 0.2});

    auto config = std::make_shared<CyclopsConfig>();
    config->gravity_norm = 9.81;
    config->extrinsics.imu_camera_transform = extrinsic;
    config->noise.acc_bias_prior_stddev = 1.;

    auto analyzer = ImuTranslationMatchAnalyzer::Create(config);
    auto [_1, _2, _3, A_I, B_I, A_V, alpha, beta] = analyzer->analyze(
      imu_motions |
        views::transform([](auto const& _) -> ImuMotionRef { return _; }) |
        ranges::to_vector,
      orientations, position_prior);

    auto x_I = concatenate(
      gravity, Vector3d::Zero().eval(),
      concatenate(
        times | views::transform(body_velocity_signal) | ranges::to_vector));
    auto x_p = concatenate(position_perturbations);
    auto x = concatenate(x_I, x_p);

    auto A_s = MatrixXd(A_I.rows(), A_I.cols() + B_I.cols());
    A_s << A_I, B_I * scale;
    auto b_s = (alpha + beta * scale).eval();

    auto r = (A_s * x + b_s).eval();
    CAPTURE(x.transpose());
    CAPTURE(r.transpose());
    CHECK(r.norm() < 1e-6);
  }
}  // namespace cyclops::initializer
