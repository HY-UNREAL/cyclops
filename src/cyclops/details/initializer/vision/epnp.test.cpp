#include "cyclops/details/initializer/vision/epnp.cpp"
#include "cyclops_tests/data/landmark.hpp"

#include <doctest/doctest.h>

namespace cyclops::initializer {
  namespace views = ranges::views;
  using Eigen::AngleAxisd;

  static auto makePnpImagePointSet(
    LandmarkPositions const& landmarks,
    std::map<LandmarkID, FeaturePoint> const& features) {
    return  //
      features | views::transform([&](auto const& _) {
        auto const& [feature_id, feature] = _;
        auto pnp_point =
          PnpImagePoint {landmarks.at(feature_id), feature.point};
        return std::make_pair(feature_id, pnp_point);
      }) |
      ranges::to<std::map<LandmarkID, PnpImagePoint>>;
  }

  TEST_CASE("Perspective-n-point camera pose reconstruction") {
    std::mt19937 rgen(20240513005);

    GIVEN("Random distributed landmarks") {
      auto uniform_distribution = std::uniform_real_distribution<>(-1, 1);
      auto rnd = [&]() { return 0.5 * uniform_distribution(rgen); };
      auto landmarks = generateLandmarks(
        views::ints(0, 200) | ranges::to<std::set<LandmarkID>>,
        [&](auto _) -> Vector3d {
          return Vector3d(0.5, 0, 1) + Vector3d(rnd(), rnd(), rnd());
        });

      GIVEN("Feature observations in random camera pose") {
        auto p = Vector3d(1.0, 0, 0);
        auto R = AngleAxisd(-M_PI / 4, Vector3d::UnitY()).matrix().eval();
        auto features = generateLandmarkObservations(R, p, landmarks);
        REQUIRE(features.size() >= 5);

        WHEN(
          "Solved the camera pose from the landmark "
          "position-observation pair set") {
          auto maybe_camera_pose =
            solvePnpCameraPose(makePnpImagePointSet(landmarks, features), 10);
          REQUIRE(static_cast<bool>(maybe_camera_pose));

          THEN("Camera rotation is correct") {
            CAPTURE(R);
            CAPTURE(maybe_camera_pose->rotation);
            CHECK(maybe_camera_pose->rotation.isApprox(R));
          }

          THEN("Camera position is correct") {
            CAPTURE(p.transpose());
            CAPTURE(maybe_camera_pose->translation.transpose());
            CHECK(maybe_camera_pose->translation.isApprox(p));
          }
        }
      }
    }
  }
}  // namespace cyclops::initializer
