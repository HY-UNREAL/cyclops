#include "cyclops/details/initializer/vision_imu/camera_motion_prior.hpp"
#include "cyclops/details/initializer/vision/type.hpp"

#include <range/v3/all.hpp>
#include <set>

#include <doctest/doctest.h>

namespace cyclops::initializer {
  namespace views = ranges::views;

  using std::set;

  using Eigen::Matrix3d;
  using Eigen::MatrixXd;
  using Eigen::Quaterniond;
  using Eigen::Vector3d;

  static Matrix3d makeRandomPositiveDefiniteMatrix(std::mt19937& rgen) {
    Matrix3d S;
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++)
        S(i, j) = std::uniform_real_distribution(-1.0, 1.0)(rgen);
    }
    return S.transpose() * S;
  }

  TEST_CASE("test camera motion prior generation") {
    MSfMSolution msfm;
    msfm.geometry.camera_motions = {
      {0, SE3Transform {Vector3d::UnitX(), Quaterniond(1, 0, 0, 0)}},
      {1, SE3Transform {Vector3d::UnitY(), Quaterniond(0, 1, 0, 0)}},
    };
    msfm.motion_information_weight = MatrixXd::Zero(12, 12);

    std::mt19937 rgen(20220510);
    auto W_R2 = makeRandomPositiveDefiniteMatrix(rgen);
    auto W_p2 = makeRandomPositiveDefiniteMatrix(rgen);

    msfm.motion_information_weight.block(6, 6, 3, 3) = W_R2;
    msfm.motion_information_weight.block(9, 9, 3, 3) = W_p2;

    auto [rotation_prior, translation_prior] =
      makeImuMatchCameraMotionPrior(msfm);

    REQUIRE(
      (rotation_prior.rotations | views::keys | ranges::to<set>) ==
      set<FrameID> {0, 1});
    REQUIRE(
      (translation_prior.translations | views::keys | ranges::to<set>) ==
      set<FrameID> {0, 1});

    CHECK(rotation_prior.rotations.at(0).isApprox(Quaterniond(1, 0, 0, 0), 0));
    CHECK(rotation_prior.rotations.at(1).isApprox(Quaterniond(0, 1, 0, 0), 0));
    CHECK(translation_prior.translations.at(0) == Vector3d(1, 0, 0));
    CHECK(translation_prior.translations.at(1) == Vector3d(0, 1, 0));

    CHECK(translation_prior.weight == W_p2);
    CHECK(rotation_prior.weight == W_R2);
  }
}  // namespace cyclops::initializer
