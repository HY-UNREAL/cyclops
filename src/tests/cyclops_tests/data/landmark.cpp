#include "cyclops_tests/data/landmark.hpp"
#include "cyclops_tests/data/typefwd.hpp"
#include "cyclops_tests/random.hpp"

#include "cyclops/details/utils/math.hpp"

#include <range/v3/all.hpp>

namespace cyclops {
  using std::function;
  using std::map;
  using std::set;
  using std::vector;

  using Eigen::Matrix2d;
  using Eigen::Matrix3d;
  using Eigen::Vector2d;
  using Eigen::Vector3d;

  namespace views = ranges::views;

  using measurement::FeatureTracks;

  static Vector3d makeUniformRandom3(std::mt19937& rgen) {
    std::uniform_real_distribution<> rand(-1, 1);
    return Vector3d(rand(rgen), rand(rgen), rand(rgen));
  }

  LandmarkPositions generateLandmarks(
    std::mt19937& rgen, LandmarkGenerationArgument const& arg) {
    return generateLandmarks(rgen, LandmarkGenerationArguments {arg});
  }

  LandmarkPositions generateLandmarks(
    std::mt19937& rgen, LandmarkGenerationArguments const& args) {
    std::map<LandmarkID, Vector3d> result;

    int _id = 0;
    auto id_generator = views::transform([&](auto _) {
      _id += std::uniform_int_distribution<>(1, 2)(rgen);
      return _id;
    });
    for (auto const& arg : args) {
      for (auto const id : views::iota(0, arg.count) | id_generator) {
        auto const& A = arg.concentration;
        auto const& C = arg.center;
        result.emplace(id, C + A * makeUniformRandom3(rgen));
      }
    }
    return result;
  }

  LandmarkPositions generateLandmarks(
    set<LandmarkID> ids, function<Vector3d(LandmarkID)> gen) {
    return ids |
      views::transform([gen](auto id) { return std::make_pair(id, gen(id)); }) |
      ranges::to<std::map<LandmarkID, Vector3d>>();
  }

  static auto makeLandmarkObservationRange(
    Matrix3d const& R, Vector3d const& p, LandmarkPositions const& landmarks) {
    return views::for_each(landmarks, [&](auto const& id_landmark) {
      auto const& [id, landmark] = id_landmark;
      Vector3d const f = R.transpose() * (landmark - p);
      Vector2d const u = f.head<2>() / f.z();

      auto&& contained = [](auto x, auto a, auto b) { return x > a && x < b; };
      return ranges::yield_if(
        f.z() > 1e-3 && contained(u.x(), -1, 1) && contained(u.y(), -1, 1),
        std::make_pair(id, u));
    });
  }

  static Matrix2d makeDefaultLandmarkWeight() {
    return Vector2d(2.5e5, 2.5e5).asDiagonal();
  }

  std::map<LandmarkID, FeaturePoint> generateLandmarkObservations(
    Matrix3d const& R, Vector3d const& p, LandmarkPositions const& landmarks) {
    // clang-format off
    return makeLandmarkObservationRange(R, p, landmarks)
      | views::transform([](auto const& id_point) {
          auto const& [id, u] = id_point;
          return std::make_pair(
            id, FeaturePoint {u, makeDefaultLandmarkWeight()});
        })
      | ranges::to<std::map<LandmarkID, FeaturePoint>>;
    // clang-format on
  }

  std::map<LandmarkID, FeaturePoint> generateLandmarkObservations(
    std::mt19937& rgen, Matrix2d const& cov, Matrix3d const& R,
    Vector3d const& p, LandmarkPositions const& landmarks) {
    Matrix2d weight = cov.inverse();
    Matrix2d spread = cov.llt().matrixL();

    // clang-format off
    return makeLandmarkObservationRange(R, p, landmarks)
      | views::transform([&](auto const& id_point) {
          auto const& [id, u] = id_point;
          return std::make_pair(
            id, FeaturePoint {perturbate(u, spread, rgen), weight});
        })
      | ranges::to<std::map<LandmarkID, FeaturePoint>>;
    // clang-format on
  }

  static auto makeLandmarkFrame(
    PoseSignal pose_signal, SE3Transform const& extrinsic,
    LandmarkPositions const& landmarks, Timestamp t) {
    auto const [p, q] = pose_signal;
    auto const [p_c, q_c] = compose({p(t), q(t)}, extrinsic);
    return generateLandmarkObservations(q_c.matrix(), p_c, landmarks);
  }

  static auto makeLandmarkFrame(
    PoseSignal pose_signal, SE3Transform const& extrinsic,
    LandmarkPositions const& landmarks, Timestamp t, std::mt19937& rgen,
    Matrix2d const& cov) {
    auto const [p, q] = pose_signal;
    auto const [p_c, q_c] = compose({p(t), q(t)}, extrinsic);
    return generateLandmarkObservations(
      rgen, cov, q_c.matrix(), p_c, landmarks);
  }

  vector<ImageData> makeLandmarkFrames(
    PoseSignal pose_signal, SE3Transform const& extrinsic,
    LandmarkPositions const& landmarks, vector<Timestamp> const& times) {
    auto transform = views::transform([&](auto timestamp) {
      return ImageData {
        timestamp,
        makeLandmarkFrame(pose_signal, extrinsic, landmarks, timestamp)};
    });
    return times | transform | ranges::to_vector;
  }

  vector<ImageData> makeLandmarkFrames(
    PoseSignal pose_signal, SE3Transform const& extrinsic,
    LandmarkPositions const& landmarks, vector<Timestamp> const& times,
    std::mt19937& rgen, Matrix2d const& cov) {
    auto transform = views::transform([&](auto timestamp) {
      return ImageData {
        timestamp,
        makeLandmarkFrame(
          pose_signal, extrinsic, landmarks, timestamp, rgen, cov)};
    });
    return times | transform | ranges::to_vector;
  }

  static FeatureTracks makeLandmarkTracks(
    map<FrameID, Timestamp> const& frames,
    vector<ImageData> const& landmark_frames) {
    FeatureTracks tracks;
    for (auto const& [frame_id, landmark_frame] :
         views::zip(frames | views::keys, landmark_frames)) {
      for (auto const& [feature_id, feature] : landmark_frame.features)
        tracks[feature_id].emplace(frame_id, feature);
    }
    return tracks;
  }

  FeatureTracks makeLandmarkTracks(
    PoseSignal pose_signal, SE3Transform const& extrinsic,
    LandmarkPositions const& landmarks, map<FrameID, Timestamp> const& frames) {
    auto landmark_frames = makeLandmarkFrames(
      pose_signal, extrinsic, landmarks,
      frames | views::values | ranges::to_vector);
    return makeLandmarkTracks(frames, landmark_frames);
  }

  FeatureTracks makeLandmarkTracks(
    PoseSignal pose_signal, SE3Transform const& extrinsic,
    LandmarkPositions const& landmarks, map<FrameID, Timestamp> const& frames,
    std::mt19937& rgen, Matrix2d const& cov) {
    auto landmark_frames = makeLandmarkFrames(
      pose_signal, extrinsic, landmarks,
      frames | views::values | ranges::to_vector, rgen, cov);
    return makeLandmarkTracks(frames, landmark_frames);
  }

  map<FrameID, map<LandmarkID, FeaturePoint>> makeLandmarkMultiviewObservation(
    PoseSignal pose_signal, SE3Transform const& extrinsic,
    LandmarkPositions const& landmarks,
    std::map<FrameID, Timestamp> const& frame_times) {
    auto feature_observation_transform = views::transform([&](auto const& _) {
      auto const& [frame_id, timestamp] = _;
      auto x = SE3Transform {
        .translation = pose_signal.position(timestamp),
        .rotation = pose_signal.orientation(timestamp),
      };
      auto [p_c, q_c] = compose(x, extrinsic);
      auto R_c = q_c.matrix().eval();

      auto features = generateLandmarkObservations(R_c, p_c, landmarks);
      return std::make_pair(frame_id, features);
    });

    return frame_times | feature_observation_transform |
      ranges::to<map<FrameID, map<LandmarkID, FeaturePoint>>>;
  }
}  // namespace cyclops
