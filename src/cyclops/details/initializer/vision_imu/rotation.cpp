#include "cyclops/details/initializer/vision_imu/rotation.hpp"
#include "cyclops/details/initializer/vision_imu/camera_motion_prior.hpp"

#include "cyclops/details/measurement/preintegration.hpp"

#include "cyclops/details/utils/math.hpp"
#include "cyclops/details/utils/type.hpp"

#include "cyclops/details/config.hpp"
#include "cyclops/details/logging.hpp"

#include <ceres/ceres.h>

#include <spdlog/fmt/ostr.h>
#include <spdlog/spdlog.h>

#include <range/v3/all.hpp>

#include <map>

namespace cyclops::initializer {
  namespace views = ranges::views;

  using ceres::AutoDiffCostFunction;
  using ceres::DynamicAutoDiffCostFunction;
  using ceres::EigenQuaternionParameterization;

  using Eigen::Matrix3d;
  using Eigen::MatrixXd;
  using Eigen::Quaterniond;
  using Eigen::Vector3d;

  using GyroBiasBlock = std::array<double, 3>;
  using OrientationBlock = std::array<double, 4>;
  using OrientationBlocks = std::map<FrameID, OrientationBlock>;

  using measurement::ImuMotionRefs;

  template <typename value_t>
  using MaybeRef = std::optional<std::reference_wrapper<value_t>>;

  static double degrees(double radian) {
    return radian * 180 / M_PI;
  }

  static Matrix3d computeInverseCholeskyMatrixU(Matrix3d const& mat) {
    return Eigen::LLT<Matrix3d>(mat.inverse()).matrixU();
  }

  struct GyroBiasZeroPriorCost {
    double const _weight;

    explicit GyroBiasZeroPriorCost(double weight): _weight(weight) {
    }

    template <typename scalar_t>
    bool operator()(scalar_t const* const b_w, scalar_t* const r) const {
      using Vector3 = Eigen::Matrix<scalar_t, 3, 1>;
      (Eigen::Map<Vector3>(r)) = static_cast<scalar_t>(_weight) * Vector3(b_w);
      return true;
    }
  };

  struct ImuRotationMatchGyroCost {
    measurement::ImuPreintegration const& _data;

    explicit ImuRotationMatchGyroCost(
      measurement::ImuPreintegration const& data)
        : _data(data) {
    }

    template <typename scalar_t>
    bool operator()(
      scalar_t const* const q0_, scalar_t const* const q1_,
      scalar_t const* const b_w, scalar_t* const r) const {
      using Vector3 = Eigen::Matrix<scalar_t, 3, 1>;
      using Quaternion = Eigen::Quaternion<scalar_t>;

      auto q0 = Quaternion(q0_);
      auto q1 = Quaternion(q1_);
      auto db_w = (Vector3(b_w) - _data.gyrBias().cast<scalar_t>()).eval();
      auto G_R = _data.bias_jacobian.block<3, 3>(0, 3).cast<scalar_t>().eval();

      Quaternion y_q = _data.rotation_delta.cast<scalar_t>();
      Vector3 u = so3Logmap(y_q.conjugate() * q0.conjugate() * q1) - G_R * db_w;

      auto weight =
        computeInverseCholeskyMatrixU(_data.covariance.topLeftCorner<3, 3>());
      (Eigen::Map<Vector3>(r)) = weight.cast<scalar_t>() * u;
      return true;
    }
  };

  struct CameraRotationPriorCost {
    SE3Transform const& _extrinsic;
    ImuMatchCameraRotationPrior const& _prior;
    MatrixXd const _weight;

    CameraRotationPriorCost(
      SE3Transform const& extrinsic, ImuMatchCameraRotationPrior const& prior)
        : _extrinsic(extrinsic),
          _prior(prior),
          _weight(prior.weight.llt().matrixU()) {
    }

    template <typename scalar_t>
    bool operator()(
      scalar_t const* const* body_rotations, scalar_t* const residual) const {
      using VectorX = Eigen::Matrix<scalar_t, Eigen::Dynamic, 1>;
      using Quaternion = Eigen::Quaternion<scalar_t>;

      int n = _prior.rotations.size();
      if (n < 1)
        return false;

      auto dimension = 3 * std::max(0, n - 1);
      auto q_bc = _extrinsic.rotation.cast<scalar_t>();
      auto delta = VectorX(dimension);

      for (auto const& [i, q_c_hat] : views::enumerate(
             _prior.rotations | views::values | views::slice(1, n))) {
        auto q_b = Quaternion(body_rotations[i]);
        auto q_c = q_b * q_bc;
        delta.segment(3 * i, 3) =
          so3Logmap(q_c_hat.template cast<scalar_t>().conjugate() * q_c);
      }
      (Eigen::Map<VectorX>(residual, dimension)) =
        _weight.cast<scalar_t>() * delta;
      return true;
    }
  };

  static GyroBiasBlock makeGyroBiasBlock() {
    GyroBiasBlock result;
    std::fill(result.begin(), result.end(), 0);
    return result;
  }

  static OrientationBlock makeOrientationBlock(Quaterniond const& q) {
    OrientationBlock result;
    (Eigen::Map<Quaterniond>(result.data())) = q;
    return result;
  }

  static OrientationBlocks makeOrientationBlocks(
    ImuMatchCameraRotationPrior const& prior, SE3Transform const& extrinsic) {
    return  //
      prior.rotations | views::transform([&extrinsic](auto const& id_rotation) {
        auto const& [id, rotation] = id_rotation;
        return std::make_pair(
          id, makeOrientationBlock(rotation * extrinsic.rotation.conjugate()));
      }) |
      ranges::to<OrientationBlocks>;
  }

  static MaybeRef<OrientationBlock> tryFindFrame(
    OrientationBlocks& blocks, FrameID frame_id) {
    auto i = blocks.find(frame_id);
    if (i == blocks.end()) {
      __logger__->warn(
        "Unknown camera pose for frame {} in IMU rotation match", frame_id);
      return std::nullopt;
    }
    auto& [_, block] = *i;
    return std::ref(block);
  }

  static ceres::ResidualBlockId constructGyroBiasZeroPriorCost(
    ceres::Problem& problem, GyroBiasBlock& block, double weight) {
    auto cost = new AutoDiffCostFunction<GyroBiasZeroPriorCost, 3, 3>(
      new GyroBiasZeroPriorCost(weight));
    return problem.AddResidualBlock(cost, nullptr, block.data());
  }

  static std::optional<std::vector<ceres::ResidualBlockId>>
  constructImuMotionCost(
    ceres::Problem& problem, GyroBiasBlock& gyro_bias,
    OrientationBlocks& orientations, ImuMotionRefs const& imu_motions) {
    std::vector<ceres::ResidualBlockId> result;
    for (auto const& imu_motion_ref : imu_motions) {
      auto const& [from, to, data] = imu_motion_ref.get();

      auto maybe_q0 = tryFindFrame(orientations, from);
      auto maybe_q1 = tryFindFrame(orientations, to);
      if (!maybe_q0 || !maybe_q1)
        return std::nullopt;

      auto cost =
        new AutoDiffCostFunction<ImuRotationMatchGyroCost, 3, 4, 4, 3>(
          new ImuRotationMatchGyroCost(*data));
      auto residual_id = problem.AddResidualBlock(
        cost, nullptr, maybe_q0->get().data(), maybe_q1->get().data(),
        gyro_bias.data());
      result.push_back(residual_id);
    }
    return result;
  }

  static ceres::ResidualBlockId constructCameraRotationPriorCost(
    ceres::Problem& problem, OrientationBlocks& orientation_blocks,
    SE3Transform const& extrinsic,
    ImuMatchCameraRotationPrior const& camera_rotation_prior) {
    auto cost = new DynamicAutoDiffCostFunction<CameraRotationPriorCost>(
      new CameraRotationPriorCost(extrinsic, camera_rotation_prior));

    int n = orientation_blocks.size();
    cost->SetNumResiduals(3 * std::max(0, n - 1));

    std::vector<double*> parameters;
    for (auto& block :
         orientation_blocks | views::values | views::slice(1, n)) {
      cost->AddParameterBlock(4);
      parameters.emplace_back(block.data());
    }
    return problem.AddResidualBlock(cost, nullptr, parameters);
  }

  static void solveProblem(ceres::Problem& problem) {
    ceres::Solver::Options options;
    ceres::Solver::Summary summary;
    options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
    ceres::Solve(options, &problem, &summary);

    __logger__->trace(
      "Finished gyro bias iteration: {}", summary.BriefReport());
  }

  static MatrixXd evaluateRotationMatchCostAndHessian(
    ceres::Problem& problem,
    std::vector<ceres::ResidualBlockId> const& residual_ids,
    GyroBiasBlock& gyro_bias, OrientationBlocks& orientations) {
    ceres::Problem::EvaluateOptions opt;

    std::vector<double*> parameters = {gyro_bias.data()};
    for (auto& [_, orientation] : orientations)
      parameters.push_back(orientation.data());

    opt.parameter_blocks = std::move(parameters);
    opt.residual_blocks = residual_ids;

    ceres::CRSMatrix jacobian;
    problem.Evaluate(opt, nullptr, nullptr, nullptr, &jacobian);

    auto eigen_jacobian = Eigen::Map<EigenCRSMatrix>(
      jacobian.num_rows, jacobian.num_cols, jacobian.values.size(),
      jacobian.rows.data(), jacobian.cols.data(), jacobian.values.data());
    return eigen_jacobian.transpose() * eigen_jacobian;
  }

  static std::optional<Eigen::VectorXd> evaluateRotationMatchDeviation(
    ceres::Problem& problem,
    std::vector<ceres::ResidualBlockId> const& residuals,
    GyroBiasBlock& gyro_bias, OrientationBlocks& orientations) {
    __logger__->trace("Evaluating vision-IMU rotation match uncertainty");

    auto H = evaluateRotationMatchCostAndHessian(
      problem, residuals, gyro_bias, orientations);
    if (H.rows() != H.cols()) {
      __logger__->error("Vision-IMU rotation match: Non-square Hessian matrix");
      return std::nullopt;
    }

    auto n = H.rows();
    if (n <= 6 || n % 3 != 0) {
      __logger__->error(
        "Vision-IMU rotation match: wrong Hessian dimension. n = {}", n);
      return std::nullopt;
    }
    auto q_dim = n - 3;

    // The first rotation is assumed constant to handle the gauge symmetry.
    auto effective_q_dim = q_dim - 3;

    Matrix3d H_b = H.topLeftCorner(3, 3);
    MatrixXd H_r = H.topRightCorner(3, effective_q_dim);
    MatrixXd H_q = H.bottomRightCorner(effective_q_dim, effective_q_dim);

    MatrixXd H_bar = H_q - H_r.transpose() * H_b.ldlt().solve(H_r);
    Eigen::SelfAdjointEigenSolver<MatrixXd> H_bar_eigen(H_bar);

    if (H_bar_eigen.eigenvalues()(0) <= 0) {
      __logger__->trace("Vision-IMU rotation match: indefinite Hessian");
      return std::nullopt;
    }
    return H_bar_eigen.eigenvalues().cwiseSqrt().cwiseInverse();
  }

  static bool checkRotationMatchSolutionImuConsistency(
    double threshold,  //
    OrientationBlocks const& solution, ImuMotionRefs const& motions) {
    for (auto const& motion_ref : motions) {
      auto const& motion = motion_ref.get();
      auto const& imu_rotation = motion.data->rotation_delta;

      auto orientation1 = Quaterniond(solution.at(motion.from).data());
      auto orientation2 = Quaterniond(solution.at(motion.to).data());
      auto rotation_solution = orientation1.conjugate() * orientation2;

      auto error = imu_rotation.conjugate() * rotation_solution;
      if (Eigen::AngleAxisd(error).angle() > threshold)
        return false;
    }
    return true;
  }

  static bool checkRotationMatchSolutionVisionConsistency(
    double threshold,  //
    OrientationBlocks const& solution, Quaterniond const& extrinsic,
    std::map<FrameID, Quaterniond> const& vision_prior) {
    for (auto const& [frame_id, camera_orientation] : vision_prior) {
      auto imu_orientation = camera_orientation * extrinsic.conjugate();
      auto solution_orientation = Quaterniond(solution.at(frame_id).data());

      auto error = imu_orientation.conjugate() * solution_orientation;
      if (Eigen::AngleAxisd(error).angle() > threshold)
        return false;
    }
    return true;
  }

  static ImuRotationMatch makeRotationMatch(
    GyroBiasBlock const& gyro_bias, OrientationBlocks const& orientations) {
    return {
      .gyro_bias = Vector3d(gyro_bias.data()),
      .body_orientations =
        orientations | views::transform([](auto const& id_block) {
          auto const& [id, block] = id_block;
          return std::make_pair(id, Quaterniond(block.data()));
        }) |
        ranges::to<std::map<FrameID, Quaterniond>>,
    };
  }

  class ImuRotationMatchSolverImpl: public ImuRotationMatchSolver {
  private:
    std::shared_ptr<CyclopsConfig const> _config;

    bool checkInputDataSolutionConsistency(
      OrientationBlocks const& solution, ImuMotionRefs const& imu_data,
      std::map<FrameID, Quaterniond> const& vision_prior) const;

  public:
    explicit ImuRotationMatchSolverImpl(
      std::shared_ptr<CyclopsConfig const> config);
    void reset() override;

    std::optional<ImuRotationMatch> solve(
      ImuMotionRefs const& motions,
      ImuMatchCameraRotationPrior const& prior) override;
  };

  ImuRotationMatchSolverImpl::ImuRotationMatchSolverImpl(
    std::shared_ptr<CyclopsConfig const> config)
      : _config(config) {
  }

  void ImuRotationMatchSolverImpl::reset() {
    // does nothing.
  }

  bool ImuRotationMatchSolverImpl::checkInputDataSolutionConsistency(
    OrientationBlocks const& solution, ImuMotionRefs const& imu_data,
    std::map<FrameID, Quaterniond> const& vision_prior) const {
    auto const& rotation_match_config =
      _config->initialization.imu.rotation_match;
    auto const& threshold =
      rotation_match_config.vision_imu_rotation_consistency_angle_threshold;

    auto const& extrinsic = _config->extrinsics.imu_camera_transform;

    if (!checkRotationMatchSolutionImuConsistency(
          threshold, solution, imu_data)) {
      __logger__->debug(
        "Vision-imu rotation match failure: imu rotation inconsistency");
      return false;
    }

    if (!checkRotationMatchSolutionVisionConsistency(
          threshold, solution, extrinsic.rotation, vision_prior)) {
      __logger__->debug(
        "Vision-imu rotation match failure: vision rotation inconsistency");
      __logger__->trace("<frame_id>: [<solution>], [<sfm prior>]");

      for (auto const& [frame_id, camera_orientation] : vision_prior) {
        auto solution_orientation =
          Eigen::Map<Quaterniond const>(solution.at(frame_id).data());
        auto prior_orientation =
          camera_orientation * extrinsic.rotation.conjugate();

        __logger__->trace(
          "{}: [{}], [{}]", frame_id, solution_orientation.coeffs().transpose(),
          prior_orientation.coeffs().transpose());
      }
      return false;
    }
    return true;
  }

  std::optional<ImuRotationMatch> ImuRotationMatchSolverImpl::solve(
    ImuMotionRefs const& motions, ImuMatchCameraRotationPrior const& prior) {
    auto const& extrinsic = _config->extrinsics.imu_camera_transform;
    auto gyro_bias_prior = 1 / _config->noise.gyr_bias_prior_stddev;

    auto const& accept_config = _config->initialization.imu.acceptance_test;
    auto accept_max_deviation = accept_config.max_rotation_deviation;
    auto accept_min_p_value = accept_config.rotation_match_min_p_value;

    auto gyro_bias = makeGyroBiasBlock();
    auto orientations = makeOrientationBlocks(prior, extrinsic);

    ceres::Problem problem;
    problem.AddParameterBlock(gyro_bias.data(), 3);
    for (auto& orientation : orientations | views::values) {
      problem.AddParameterBlock(
        orientation.data(), 4, new EigenQuaternionParameterization());
    }

    // fix reference frame
    auto& [_, reference] = *orientations.begin();
    problem.SetParameterBlockConstant(reference.data());

    auto maybe_residuals =
      constructImuMotionCost(problem, gyro_bias, orientations, motions);
    if (!maybe_residuals)
      return std::nullopt;
    auto& residuals = *maybe_residuals;

    residuals.push_back(
      constructGyroBiasZeroPriorCost(problem, gyro_bias, gyro_bias_prior));
    residuals.push_back(constructCameraRotationPriorCost(
      problem, orientations, extrinsic, prior));

    solveProblem(problem);

    Eigen::Map<Vector3d> b_w(gyro_bias.data());
    if (std::isnan(b_w.norm()))
      return std::nullopt;

    if (!checkInputDataSolutionConsistency(
          orientations, motions, prior.rotations)) {
      return std::nullopt;
    }

    auto maybe_deviation = evaluateRotationMatchDeviation(
      problem, residuals, gyro_bias, orientations);
    if (!maybe_deviation) {
      __logger__->debug("Solving gyro bias");
      return std::nullopt;
    }
    auto const& deviation = *maybe_deviation;

    if (deviation(0) > accept_max_deviation) {
      __logger__->debug("Vision-imu rotation match failure: large deviation");
      __logger__->debug(
        "{}° > {}°", degrees(deviation(0)), degrees(accept_max_deviation));
      return std::nullopt;
    }

    __logger__->trace(
      "Successed to solve gyro bias; [{}, {}, {}]", b_w.x(), b_w.y(), b_w.z());
    return makeRotationMatch(gyro_bias, orientations);
  }

  std::unique_ptr<ImuRotationMatchSolver> ImuRotationMatchSolver::Create(
    std::shared_ptr<CyclopsConfig const> config) {
    return std::make_unique<ImuRotationMatchSolverImpl>(config);
  }
}  // namespace cyclops::initializer
