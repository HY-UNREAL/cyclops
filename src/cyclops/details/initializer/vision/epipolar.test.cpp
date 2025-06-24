#include "cyclops/details/initializer/vision/epipolar.cpp"
#include "cyclops_tests/random.hpp"

#include <range/v3/all.hpp>
#include <doctest/doctest.h>

namespace cyclops::initializer {
  using Eigen::Quaterniond;

  static auto project(Vector3d const& x) {
    return (x.head<2>() / x.z()).eval();
  }

  TEST_CASE("Essential matrix computation") {
    std::mt19937 rgen(20240513001);

    GIVEN("Random landmarks and camera movement") {
      std::map<LandmarkID, Vector3d> landmarks;
      for (auto feature_id : ranges::views::ints(0, 32)) {
        std::uniform_real_distribution<double> dist(-1, 1);
        landmarks.emplace(
          feature_id, Vector3d(dist(rgen), dist(rgen), dist(rgen)));
      }

      auto p = perturbate(Vector3d::Zero().eval(), 1, rgen);
      auto R = perturbate(Quaterniond::Identity(), 1, rgen).matrix().eval();

      GIVEN("Perfectly correct feature observations") {
        std::map<LandmarkID, TwoViewFeaturePair> feature_frame;
        for (auto const& [feature_id, landmark] : landmarks) {
          auto f_1 = project(landmark);
          auto f_2 = project(R.transpose() * (landmark - p));
          feature_frame.emplace(feature_id, TwoViewFeaturePair {f_1, f_2});
        }

        std::vector<std::set<LandmarkID>> batch = {
          {0, 7, 25, 28, 3, 8, 19, 26},
          {5, 12, 24, 31, 4, 9, 17, 27},
          {2, 15, 18, 23, 11, 13, 21, 29},
          {6, 14, 16, 1, 10, 22, 20, 30},
        };

        WHEN("Solve epipolar two-view geometry by random consensus") {
          auto result = analyzeTwoViewEpipolar(1e-6, batch, feature_frame);
          THEN("The resulting essential matrix is correct") {
            Matrix3d E = (R.inverse() * skew3d(p)).normalized();

            CAPTURE(E);
            CAPTURE(result.essential_matrix);
            CHECK(result.expected_inliers > 0.);
            CHECK(result.essential_matrix.isApprox(E));
            CHECK(result.inliers.size() == 32);
          }
        }
      }
    }
  }
}  // namespace cyclops::initializer
