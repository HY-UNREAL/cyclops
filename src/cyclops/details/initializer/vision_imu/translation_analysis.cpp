#include "cyclops/details/initializer/vision_imu/translation_analysis.hpp"
#include "cyclops/details/initializer/vision_imu/camera_motion_prior.hpp"
#include "cyclops/details/initializer/vision_imu/translation.hpp"

#include "cyclops/details/measurement/preintegration.hpp"
#include "cyclops/details/utils/math.hpp"
#include "cyclops/details/config.hpp"

#include <range/v3/all.hpp>

namespace cyclops::initializer {
  using Eigen::Matrix3d;
  using Eigen::MatrixXd;
  using Eigen::Vector3d;
  using Eigen::VectorXd;

  using Matrix6d = Eigen::Matrix<double, 6, 6>;
  using Vector6d = Eigen::Matrix<double, 6, 1>;

  using measurement::ImuMotionRef;

  class ImuMatchAnalyzerImpl: public ImuMatchAnalyzer {
  private:
    std::shared_ptr<CyclopsConfig const> _config;

  public:
    explicit ImuMatchAnalyzerImpl(std::shared_ptr<CyclopsConfig const> config)
        : _config(config) {
    }
    void reset() override;

    ImuMatchAnalysis analyze(
      std::vector<ImuMotionRef> const& imu_motions,
      ImuMatchCameraMotionPrior const& camera_prior) override;
  };

  void ImuMatchAnalyzerImpl::reset() {
    // does nothing.
  }

  struct ImuMotionAnalysis {
    MatrixXd A_I;
    MatrixXd B_I;
    Vector6d alpha;
    Vector6d beta;
  };

  static ImuMotionAnalysis analyzeImuMotion(
    ImuMatchCameraMotionPrior const& camera_prior, int n, int i,
    SE3Transform const& extrinsic, ImuMotionRef const& imu_motion) {
    auto A_I_bar = MatrixXd::Zero(6, 6 + 3 * n).eval();
    auto B_I_bar = MatrixXd::Zero(3, 3 * n).eval();

    auto const& [from, to, data] = imu_motion.get();
    auto const& [p_bc, q_bc] = extrinsic;

    auto const& q_i_bar = camera_prior.imu_orientations.at(from);
    auto const& q_j_bar = camera_prior.imu_orientations.at(to);
    auto q_ij_bar = q_i_bar.conjugate() * q_j_bar;

    auto const& p_i_hat = camera_prior.camera_positions.at(from);
    auto const& p_j_hat = camera_prior.camera_positions.at(to);

    auto y_R_T = data->rotation_delta.conjugate().matrix().eval();
    auto const& y_v = data->velocity_delta;
    auto const& y_p = data->position_delta;

    auto delta_theta = so3Logmap(data->rotation_delta.conjugate() * q_ij_bar);
    auto N_inv = so3LeftJacobianInverse(delta_theta);
    auto N_inv__y_R_T = (N_inv * y_R_T).eval();

    auto R_bc = q_bc.matrix().eval();
    auto R_i_bar_T = q_i_bar.conjugate().matrix().eval();
    auto R_ij_bar = q_ij_bar.matrix().eval();

    auto N_inv__y_R_T__R_i_bar_T = (N_inv__y_R_T * R_i_bar_T).eval();

    auto dt = data->time_delta;

    auto G_a = data->bias_jacobian.block<6, 3>(3, 0).eval();
    auto G_w = data->bias_jacobian.block<6, 3>(3, 3).eval();

    auto delta_b_w = (camera_prior.gyro_bias - data->gyrBias()).eval();

    A_I_bar.block(0, 0, 3, 3) = N_inv__y_R_T__R_i_bar_T * dt * dt / 2;
    A_I_bar.block(0, 3, 6, 3) = -G_a;
    A_I_bar.block(0, 6 + 3 * i, 3, 3) = -N_inv__y_R_T * dt;

    A_I_bar.block(3, 0, 3, 3) = N_inv__y_R_T__R_i_bar_T * dt;
    A_I_bar.block(3, 6 + 3 * i, 3, 3) = -N_inv__y_R_T;
    A_I_bar.block(3, 9 + 3 * i, 3, 3) = N_inv__y_R_T * R_ij_bar;

    B_I_bar.block(0, 0 + 3 * i, 3, 3) = -N_inv__y_R_T * R_bc;
    B_I_bar.block(0, 3 + 3 * i, 3, 3) = N_inv__y_R_T * R_ij_bar * R_bc;

    Vector6d alpha_bar;
    // clang-format off
    alpha_bar <<
      -N_inv__y_R_T * (y_p - p_bc + R_ij_bar * p_bc),
      -N_inv__y_R_T * y_v;
    // clang-format on
    alpha_bar -= G_w * delta_b_w;

    Vector3d beta_bar = N_inv__y_R_T * R_i_bar_T * (p_j_hat - p_i_hat);

    Matrix6d P = data->covariance.bottomRightCorner<6, 6>();
    Matrix6d L_inv = P.llt().matrixL().solve(Matrix6d::Identity());

    return {
      .A_I = L_inv * A_I_bar,
      .B_I = L_inv.leftCols<3>() * B_I_bar.rightCols(3 * (n - 1)),
      .alpha = L_inv * alpha_bar,
      .beta = L_inv.leftCols<3>() * beta_bar,
    };
  }

  ImuMatchAnalysis ImuMatchAnalyzerImpl::analyze(
    std::vector<ImuMotionRef> const& imu_motions,
    ImuMatchCameraMotionPrior const& camera_prior) {
    auto const& extrinsic = _config->extrinsics.imu_camera_transform;
    auto acc_bias_prior_weight = 1 / _config->noise.acc_bias_prior_stddev;

    auto n = imu_motions.size() + 1;

    auto A_I = MatrixXd(6 * (n - 1) + 3, 3 * (n + 2));
    auto B_I = MatrixXd(6 * (n - 1) + 3, 3 * (n - 1));
    auto alpha = VectorXd(6 * (n - 1) + 3);
    auto beta = VectorXd(6 * (n - 1) + 3);

    A_I.topRows(3).setZero();
    B_I.topRows(3).setZero();
    alpha.head(3).setZero();
    beta.head(3).setZero();
    A_I.block(0, 3, 3, 3) = acc_bias_prior_weight * Matrix3d::Identity();

    for (auto const& [i, motion] : ranges::views::enumerate(imu_motions)) {
      auto analysis = analyzeImuMotion(camera_prior, n, i, extrinsic, motion);
      A_I.middleRows(6 * i + 3, 6) = analysis.A_I;
      B_I.middleRows(6 * i + 3, 6) = analysis.B_I;
      alpha.segment(6 * i + 3, 6) = analysis.alpha;
      beta.segment(6 * i + 3, 6) = analysis.beta;
    }
    auto const& H_V = camera_prior.weight;

    return {
      .frames_count = n,
      .residual_dimension = 9 * (n - 1) + 3,
      .parameter_dimension = 6 * n + 3,
      .inertial_weight = A_I,
      .translational_weight = B_I,
      .visual_weight = Eigen::LLT<MatrixXd>(H_V).matrixU(),
      .inertial_perturbation = alpha,
      .translation_perturbation = beta,
    };
  }

  std::unique_ptr<ImuMatchAnalyzer> ImuMatchAnalyzer::Create(
    std::shared_ptr<CyclopsConfig const> config) {
    return std::make_unique<ImuMatchAnalyzerImpl>(config);
  }
}  // namespace cyclops::initializer
