#include "cyclops/details/initializer/vision_imu/motion_prior.hpp"
#include "cyclops/details/initializer/vision/type.hpp"

#include <range/v3/all.hpp>

namespace cyclops::initializer {
  using Eigen::MatrixXd;
  using Vector6i = Eigen::Matrix<int, 6, 1>;

  namespace views = ranges::views;

  static MatrixXd evaluateCameraPositionInformationWeight(
    BundleAdjustmentSolution const& msfm) {
    // discount by one to handle symmetry
    auto n = static_cast<int>(msfm.camera_motions.size() - 1);

    Eigen::VectorXi orientation_indices(3 * n);
    Eigen::VectorXi translation_indices(3 * n);
    for (int i = 0; i < n; i++) {
      auto m = 6 * i + 6;
      orientation_indices.segment(3 * i, 3) << m + 0, m + 1, m + 2;
      translation_indices.segment(3 * i, 3) << m + 3, m + 4, m + 5;
    }
    Vector6i drop_indices;
    drop_indices << 0, 1, 2, 3, 4, 5;

    Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> perm(6 * (n + 1));
    perm.indices() << translation_indices, orientation_indices, drop_indices;

    auto const& weight = msfm.motion_information_weight;
    MatrixXd H = (perm.transpose() * weight * perm).topLeftCorner(6 * n, 6 * n);

    MatrixXd const H_pp = H.block(0, 0, 3 * n, 3 * n);
    MatrixXd const H_pr = H.block(0, 3 * n, 3 * n, 3 * n);
    MatrixXd const H_rp = H.block(3 * n, 0, 3 * n, 3 * n);
    MatrixXd const H_rr = H.block(3 * n, 3 * n, 3 * n, 3 * n);

    Eigen::LDLT<MatrixXd> H_rr__inv(H_rr);
    return H_pp - H_pr * H_rr__inv.solve(H_rp);
  }

  ImuMatchMotionPrior makeImuMatchMotionPrior(
    BundleAdjustmentSolution const& msfm, SE3Transform const& extrinsic) {
    if (msfm.camera_motions.size() == 0)
      return {};

    auto imu_orientations =  //
      msfm.camera_motions | views::transform([&](auto const& element) {
        auto const& [frame_id, motion] = element;
        auto const& q_c = motion.rotation;
        auto const& q_bc = extrinsic.rotation;
        return std::make_pair(frame_id, q_c * q_bc.conjugate());
      }) |
      ranges::to<std::map<FrameID, Eigen::Quaterniond>>;
    auto camera_positions =  //
      msfm.camera_motions | views::transform([](auto const& id_frame) {
        auto const& [id, frame] = id_frame;
        return std::make_pair(id, frame.translation);
      }) |
      ranges::to<std::map<FrameID, Eigen::Vector3d>>;

    return ImuMatchMotionPrior {
      .imu_orientations = imu_orientations,
      .camera_positions = camera_positions,
      .gyro_bias = msfm.gyro_bias,
      .weight = evaluateCameraPositionInformationWeight(msfm),
    };
  }
}  // namespace cyclops::initializer
