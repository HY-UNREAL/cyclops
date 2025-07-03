#include "cyclops/details/initializer/vision/multiview.hpp"
#include "cyclops/details/initializer/vision/twoview.hpp"
#include "cyclops/details/initializer/vision/twoview_selection.hpp"
#include "cyclops/details/initializer/vision/triangulation.hpp"
#include "cyclops/details/initializer/vision/epnp.hpp"
#include "cyclops/details/initializer/vision/type.hpp"

#include "cyclops/details/telemetry/initializer.hpp"
#include "cyclops/details/utils/vision.hpp"

#include "cyclops/details/config.hpp"
#include "cyclops/details/logging.hpp"

#include <spdlog/spdlog.h>
#include <range/v3/all.hpp>

#include <functional>

namespace cyclops::initializer {
  namespace views = ranges::views;

  using telemetry::InitializerTelemetry;

  using Eigen::Matrix3d;
  using Eigen::Quaterniond;

  using MultiViewImageData =
    std::map<FrameID, std::map<LandmarkID, FeaturePoint>>;
  using MultiViewGyroMotionData = std::map<FrameID, GyroMotionConstraint>;

  using TwoViewSolution =
    std::tuple<std::optional<FrameID>, std::vector<TwoViewGeometry>>;

  class MultiviewVisionGeometrySolverImpl:
      public MultiviewVisionGeometrySolver {
  private:
    std::unique_ptr<TwoViewVisionGeometrySolver> _two_view_solver;
    std::shared_ptr<CyclopsConfig const> _config;
    std::shared_ptr<InitializerTelemetry> _telemetry;

    MultiViewCorrespondences makeMultiViewCorrespondences(
      MultiViewImageData const& features,
      MultiViewGyroMotionData const& gyro_motions);

    TwoViewSolution solveTwoView(
      std::set<FrameID> frame_ids,
      MultiViewCorrespondences const& correspondences);

    std::optional<MultiViewGeometry> solveMultiview(
      FrameID twoview_frame_id, TwoViewGeometry const& twoview_solution,
      MultiViewCorrespondences const& multiview_correspondences);

    void reportBestTwoViewSelection(
      std::set<FrameID> frame_ids, FrameID best_frame_id);
    void reportTwoViewHypothesis(
      std::set<FrameID> frame_ids, FrameID best_frame_id,
      std::vector<TwoViewGeometry> const& hypothesis);
    void reportTwoViewSolverSuccess(
      std::set<FrameID> frame_ids, TwoViewCorrespondenceData const& view,
      TwoViewGeometrySolverResult const& solution);
    void reportVisionFailure(
      std::set<FrameID> frame_ids,
      InitializerTelemetry::VisionBootstrapFailureReason reason);

  public:
    MultiviewVisionGeometrySolverImpl(
      std::unique_ptr<TwoViewVisionGeometrySolver> two_view_solver,
      std::shared_ptr<CyclopsConfig const> config,
      std::shared_ptr<InitializerTelemetry> telemetry);
    ~MultiviewVisionGeometrySolverImpl();
    void reset() override;

    std::vector<MultiViewGeometry> solve(
      MultiViewImageData const& features,
      MultiViewGyroMotionData const& gyro_motions) override;
  };

  static std::optional<TwoViewImuRotationData> makeTwoViewRotationPrior(
    MultiViewGyroMotionData const& gyro_motions, FrameID reference_view_frame,
    FrameID best_view_frame) {
    FrameID frame_id = best_view_frame;

    Quaterniond q_best_to_reference = Quaterniond::Identity();
    Matrix3d P_best_to_reference = Matrix3d::Identity();

    while (true) {
      if (frame_id == reference_view_frame)
        break;

      auto i = gyro_motions.find(frame_id);
      if (i == gyro_motions.end())
        return std::nullopt;

      auto const& [_1, rotation] = *i;
      auto const& q_delta = rotation.value;
      auto const& P_delta = rotation.covariance;

      auto R_delta = q_delta.matrix().eval();

      q_best_to_reference = q_best_to_reference * q_delta;
      P_best_to_reference =
        R_delta.transpose() * P_best_to_reference * R_delta + P_delta;
      frame_id = rotation.term_frame_id;
    }

    Matrix3d R_br = q_best_to_reference.matrix();
    return TwoViewImuRotationData {
      .value = q_best_to_reference.conjugate(),
      .covariance = R_br * P_best_to_reference * R_br.transpose(),
    };
  }

  MultiViewCorrespondences
  MultiviewVisionGeometrySolverImpl::makeMultiViewCorrespondences(
    MultiViewImageData const& features,
    MultiViewGyroMotionData const& gyro_motions) {
    auto const& [frame1_id, frame1] = *features.rbegin();

    MultiViewCorrespondences result;
    result.reference_frame = frame1_id;

    auto image_frames = features | views::drop_last(1);
    for (auto const& [frame2_id, frame2] : image_frames) {
      auto maybe_rotation_prior =
        makeTwoViewRotationPrior(gyro_motions, frame1_id, frame2_id);
      if (!maybe_rotation_prior) {
        __logger__->error("Two view rotation prior generation failed");
        return {};
      }

      auto& view_frame = result.view_frames[frame2_id];
      view_frame.rotation_prior = *maybe_rotation_prior;

      for (auto const& [landmark_id, u2] : frame2) {
        auto i = frame1.find(landmark_id);
        if (i == frame1.end())
          continue;
        auto const& u1 = i->second;

        view_frame.features.emplace(
          landmark_id, TwoViewFeaturePair {u1.point, u2.point});
      }
    }
    return result;
  }

  std::optional<MultiViewGeometry>
  MultiviewVisionGeometrySolverImpl::solveMultiview(
    FrameID twoview_frame_id, TwoViewGeometry const& twoview_solution,
    MultiViewCorrespondences const& correspondences) {
    auto motions = std::map<FrameID, SE3Transform> {
      {correspondences.reference_frame, SE3Transform::Identity()},
      {twoview_frame_id, twoview_solution.camera_motion},
    };
    auto landmarks = twoview_solution.landmarks;

    auto solve_pnp = [&](auto const& view) {
      std::map<LandmarkID, PnpImagePoint> pnp_image_set;
      for (auto const& [feature_id, feature] : view.features) {
        auto i = landmarks.find(feature_id);
        if (i == landmarks.end())
          continue;
        auto const& [_, f] = *i;
        auto const& [u, v] = feature;

        pnp_image_set.emplace(feature_id, PnpImagePoint {f, v});
      }
      return solvePnpCameraPose(pnp_image_set);
    };

    for (auto const& [frame_id, view_frame] : correspondences.view_frames) {
      auto maybe_camera_pose = solve_pnp(view_frame);
      if (!maybe_camera_pose) {
        __logger__->info("EPnP camera pose reconstruction failed.");
        __logger__->debug("Frame id: {}", frame_id);
        return std::nullopt;
      }

      auto const& camera_pose = *maybe_camera_pose;
      auto const& [R, p] = camera_pose;
      motions.emplace(frame_id, SE3Transform {p, Quaterniond(R)});

      auto const& features = view_frame.features;
      auto const& vision_config = _config->initialization.vision;

      auto unknown_landmarks =
        views::set_difference(features | views::keys, landmarks | views::keys) |
        ranges::to<std::set>;
      auto triangulation = triangulateTwoViewFeaturePairs(
        vision_config, features, unknown_landmarks, camera_pose);
      auto new_landmarks = triangulation.landmarks;

      landmarks.insert(new_landmarks.begin(), new_landmarks.end());
    }
    return MultiViewGeometry {
      .camera_motions = motions,
      .landmarks = landmarks,
    };
  }

  MultiviewVisionGeometrySolverImpl::MultiviewVisionGeometrySolverImpl(
    std::unique_ptr<TwoViewVisionGeometrySolver> two_view_solver,
    std::shared_ptr<CyclopsConfig const> config,
    std::shared_ptr<InitializerTelemetry> telemetry)
      : _two_view_solver(std::move(two_view_solver)),
        _config(config),
        _telemetry(telemetry) {
  }

  MultiviewVisionGeometrySolverImpl::~MultiviewVisionGeometrySolverImpl() =
    default;

  void MultiviewVisionGeometrySolverImpl::reset() {
    _two_view_solver->reset();
  }

  static auto geometryModelAsTelemetry(
    TwoViewGeometrySolverResult::GeometryModel model) {
    return model == TwoViewGeometrySolverResult::EPIPOLAR
      ? InitializerTelemetry::EPIPOLAR
      : InitializerTelemetry::HOMOGRAPHY;
  }

  static auto twoViewGeometryAsTelemetry(TwoViewGeometry const& geometry) {
    return InitializerTelemetry::TwoViewGeometry {
      .acceptable = geometry.acceptable,
      .rotation_prior_test_passed = geometry.gyro_prior_test_passed,
      .triangulation_test_passed = geometry.triangulation_test_passed,
      .rotation_prior_p_value = geometry.gyro_prior_p_value,
      .triangulation_success_count =
        static_cast<int>(geometry.landmarks.size()),
      .motion = geometry.camera_motion,
    };
  }

  void MultiviewVisionGeometrySolverImpl::reportBestTwoViewSelection(
    std::set<FrameID> frame_ids, FrameID best_frame_id) {
    _telemetry->onBestTwoViewSelection({
      .frames = frame_ids,
      .frame_id_1 = *frame_ids.rbegin(),
      .frame_id_2 = best_frame_id,
    });
  }

  void MultiviewVisionGeometrySolverImpl::reportTwoViewHypothesis(
    std::set<FrameID> frame_ids, FrameID best_frame_id,
    std::vector<TwoViewGeometry> const& hypothesis) {
    _telemetry->onTwoViewMotionHypothesis({
      .frames = frame_ids,
      .frame_id_1 = *frame_ids.rbegin(),
      .frame_id_2 = best_frame_id,
      .candidates = hypothesis | views::transform(twoViewGeometryAsTelemetry) |
        ranges::to_vector,
    });
  }

  void MultiviewVisionGeometrySolverImpl::reportTwoViewSolverSuccess(
    std::set<FrameID> frame_ids, TwoViewCorrespondenceData const& view,
    TwoViewGeometrySolverResult const& solution) {
    _telemetry->onTwoViewSolverSuccess({
      .frames = frame_ids,
      .initial_selected_model =
        geometryModelAsTelemetry(solution.initial_selected_model),
      .final_selected_model =
        geometryModelAsTelemetry(solution.final_selected_model),
      .landmarks_count = static_cast<int>(view.features.size()),
      .homography_expected_inliers = solution.homography_expected_inliers,
      .epipolar_expected_inliers = solution.epipolar_expected_inliers,
      .candidates = solution.candidates |
        views::transform(twoViewGeometryAsTelemetry) | ranges::to_vector,
    });
  }

  void MultiviewVisionGeometrySolverImpl::reportVisionFailure(
    std::set<FrameID> frame_ids,
    InitializerTelemetry::VisionBootstrapFailureReason reason) {
    _telemetry->onVisionFailure({frame_ids, reason});
  }

  TwoViewSolution MultiviewVisionGeometrySolverImpl::solveTwoView(
    std::set<FrameID> frame_ids,
    MultiViewCorrespondences const& correspondences) {
    auto maybe_best_view = selectBestTwoViewPair(correspondences);
    if (!maybe_best_view) {
      return std::make_tuple(std::nullopt, std::vector<TwoViewGeometry> {});
    }

    auto const& [best_frame_id, best_view] = maybe_best_view->get();
    reportBestTwoViewSelection(frame_ids, best_frame_id);

    auto solution = _two_view_solver->solve(best_view);
    if (!solution.has_value())
      return std::make_tuple(best_frame_id, std::vector<TwoViewGeometry> {});

    auto const& hypotheses = solution->candidates;
    reportTwoViewHypothesis(frame_ids, best_frame_id, solution->candidates);

    auto solutions = solution->candidates |
      views::filter([](auto const& _) { return _.acceptable; }) |
      ranges::to_vector;

    if (!solutions.empty())
      reportTwoViewSolverSuccess(frame_ids, best_view, *solution);
    return std::make_tuple(best_frame_id, solutions);
  }

  std::vector<MultiViewGeometry> MultiviewVisionGeometrySolverImpl::solve(
    MultiViewImageData const& features,
    MultiViewGyroMotionData const& gyro_motions) {
    if (features.empty())
      return {};
    auto frame_ids = features | views::keys | ranges::to<std::set>;
    auto correspondences = makeMultiViewCorrespondences(features, gyro_motions);

    auto [maybe_best_frame_id, twoview_solutions] =
      solveTwoView(frame_ids, correspondences);
    if (!maybe_best_frame_id.has_value()) {
      reportVisionFailure(
        frame_ids, InitializerTelemetry::BEST_TWO_VIEW_SELECTION_FAILED);
      return {};
    }

    if (twoview_solutions.empty()) {
      reportVisionFailure(
        frame_ids, InitializerTelemetry::TWO_VIEW_GEOMETRY_FAILED);
      return {};
    }

    auto best_frame_id = maybe_best_frame_id.value();
    auto multiview_solutions = twoview_solutions |
      views::transform(std::bind(
        &MultiviewVisionGeometrySolverImpl::solveMultiview, this, best_frame_id,
        std::placeholders::_1, correspondences)) |
      ranges::to_vector;

    auto multiview_solutions_successful = multiview_solutions |
      views::filter([](auto const& maybe) { return maybe.has_value(); }) |
      views::transform([](auto const& maybe) { return maybe.value(); }) |
      ranges::to_vector;
    if (multiview_solutions_successful.empty()) {
      reportVisionFailure(
        frame_ids, InitializerTelemetry::MULTI_VIEW_GEOMETRY_FAILED);
      return {};
    }

    return multiview_solutions_successful;
  }

  std::unique_ptr<MultiviewVisionGeometrySolver>
  MultiviewVisionGeometrySolver::Create(
    std::shared_ptr<CyclopsConfig const> config,
    std::shared_ptr<std::mt19937> rgen,
    std::shared_ptr<telemetry::InitializerTelemetry> telemetry) {
    return std::make_unique<MultiviewVisionGeometrySolverImpl>(
      TwoViewVisionGeometrySolver::Create(config, rgen), config, telemetry);
  }
}  // namespace cyclops::initializer
