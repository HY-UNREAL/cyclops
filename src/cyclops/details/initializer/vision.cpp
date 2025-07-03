#include "cyclops/details/initializer/vision.hpp"
#include "cyclops/details/initializer/vision/bundle_adjustment.hpp"
#include "cyclops/details/initializer/vision/multiview.hpp"
#include "cyclops/details/initializer/vision/type.hpp"

#include "cyclops/details/telemetry/initializer.hpp"

#include "cyclops/details/utils/algorithm.hpp"
#include "cyclops/details/utils/debug.hpp"
#include "cyclops/details/utils/vision.hpp"

#include "cyclops/details/config.hpp"
#include "cyclops/details/logging.hpp"

#include <spdlog/spdlog.h>
#include <spdlog/fmt/ostr.h>
#include <range/v3/all.hpp>

namespace cyclops::initializer {
  namespace views = ranges::views;

  using telemetry::InitializerTelemetry;
  using KeyframeMotionStatisticsLookup =
    std::map<FrameID, KeyframeMotionStatistics>;

  using MultiViewImageData =
    std::map<FrameID, std::map<LandmarkID, FeaturePoint>>;
  using MultiViewGyroMotionData = std::map<FrameID, GyroMotionConstraint>;

  template <typename key_t, typename value_t>
  static std::set<key_t> projectKeys(std::map<key_t, value_t> const& map) {
    return map | views::keys | ranges::to<std::set<key_t>>;
  }

  class VisionInitializerImpl: public VisionInitializer {
  private:
    std::unique_ptr<MultiviewVisionGeometrySolver> _multiview_solver;
    std::unique_ptr<BundleAdjustmentSolver> _bundle_adjustment_solver;
    std::shared_ptr<CyclopsConfig const> _config;
    std::shared_ptr<InitializerTelemetry> _telemetry;

    KeyframeMotionStatisticsLookup compileMotionStatistics(
      MultiViewImageData const& features) const;

    KeyframeMotionStatisticsLookup filterConnectedImageTrackSequence(
      config::initializer::ObservabilityPretestThreshold const& threshold,
      KeyframeMotionStatisticsLookup const& lookup) const;
    bool detectCameraMotion(
      config::initializer::ObservabilityPretestThreshold const& threshold,
      KeyframeMotionStatisticsLookup const& lookup) const;

    std::vector<BundleAdjustmentSolution> filterDuplicateSolutions(
      std::vector<BundleAdjustmentSolution> const& solutions);

    void reportBundleAdjustmentTelemetry(
      std::set<FrameID> const& input_frames,
      std::vector<BundleAdjustmentSolution> const& solutions);

  public:
    VisionInitializerImpl(
      std::unique_ptr<MultiviewVisionGeometrySolver> multiview_solver,
      std::unique_ptr<BundleAdjustmentSolver> bundle_adjustment_solver,
      std::shared_ptr<CyclopsConfig const> config,
      std::shared_ptr<InitializerTelemetry> telemetry);
    ~VisionInitializerImpl();
    void reset() override;

    std::vector<BundleAdjustmentSolution> solve(
      MultiViewImageData const& features,
      MultiViewGyroMotionData const& gyro_motions) override;
  };

  VisionInitializerImpl::~VisionInitializerImpl() = default;

  VisionInitializerImpl::VisionInitializerImpl(
    std::unique_ptr<MultiviewVisionGeometrySolver> multiview_solver,
    std::unique_ptr<BundleAdjustmentSolver> bundle_adjustment_solver,
    std::shared_ptr<CyclopsConfig const> config,
    std::shared_ptr<InitializerTelemetry> telemetry)
      : _multiview_solver(std::move(multiview_solver)),
        _bundle_adjustment_solver(std::move(bundle_adjustment_solver)),
        _config(config),
        _telemetry(telemetry) {
  }

  void VisionInitializerImpl::reset() {
    _multiview_solver->reset();
  }

  KeyframeMotionStatisticsLookup VisionInitializerImpl::compileMotionStatistics(
    MultiViewImageData const& features) const {
    return  //
      features | views::drop_last(1) |
      views::transform([&](auto const& id_frame) {
        auto const& [frame_id, frame] = id_frame;
        auto const& [_, last_frame] = *features.rbegin();

        return std::make_pair(
          frame_id, evaluateKeyframeMotionStatistics(frame, last_frame));
      }) |
      ranges::to<KeyframeMotionStatisticsLookup>;
  }

  static auto makeImageObservabilityStatisticsTelemetry(
    KeyframeMotionStatistics const& motion_statistics) {
    return InitializerTelemetry::ImageObservabilityStatistics {
      .common_features = motion_statistics.common_features,
      .motion_parallax = motion_statistics.average_parallax,
    };
  }

  KeyframeMotionStatisticsLookup
  VisionInitializerImpl::filterConnectedImageTrackSequence(
    config::initializer::ObservabilityPretestThreshold const& threshold,
    KeyframeMotionStatisticsLookup const& lookup) const {
    if (lookup.empty())
      return {};

    KeyframeMotionStatisticsLookup result;
    for (auto i = lookup.rbegin(); i != lookup.rend(); i++) {
      auto const& [frame_id, statistics] = *i;
      if (statistics.common_features < threshold.min_landmark_overlap)
        break;

      result.emplace(frame_id, statistics);
    }

    auto telemetry_frames =  //
      lookup | views::transform([](auto const& pair) {
        auto const& [frame_id, statistics] = pair;
        return std::make_pair(
          frame_id, makeImageObservabilityStatisticsTelemetry(statistics));
      }) |
      ranges::to<
        std::map<FrameID, InitializerTelemetry::ImageObservabilityStatistics>>;

    _telemetry->onImageObservabilityPretest({
      .frames = telemetry_frames,
      .connected_frames = projectKeys(result),
    });
    return result;
  }

  bool VisionInitializerImpl::detectCameraMotion(
    config::initializer::ObservabilityPretestThreshold const& threshold,
    KeyframeMotionStatisticsLookup const& lookup) const {
    auto average_parallax = ranges::max(  //
      lookup | views::values |
      views::transform([](auto const& _) { return _.average_parallax; }));
    auto min_average_parallax = threshold.min_average_parallax;

    if (average_parallax < min_average_parallax) {
      __logger__->debug(
        "Not enough image motion parallax ({} < {})",  //
        average_parallax, min_average_parallax);
      return false;
    }
    return true;
  }

  std::vector<BundleAdjustmentSolution>
  VisionInitializerImpl::filterDuplicateSolutions(
    std::vector<BundleAdjustmentSolution> const& solutions) {
    auto n = static_cast<int>(solutions.size());
    if (n <= 1)
      return solutions;

    __logger__->info("Bundle adjustment solutions");
    for (auto const& solution : solutions) {
      for (auto const& [frame_id, motion] : solution.camera_motions) {
        auto const& p = motion.translation;
        __logger__->info("  {}", p.transpose());
      }
    }

    auto const& multiview_config = _config->initialization.vision.multiview;
    auto duplicate_max_position_error =
      multiview_config.duplicate_solution_max_position_error;

    auto partition = DisjointSetPartitionContext(
      views::cartesian_product(views::ints(0, n), views::ints(0, n)) |
      views::filter([&](auto pair) {
        auto [i, j] = pair;
        if (i > j)
          return false;
        if (i == j)
          return true;

        if (
          solutions.at(i).camera_motions.size() !=
          solutions.at(j).camera_motions.size()) {
          throw std::length_error(
            "Bundle adjustment solution candidates motion duration mismatch");
        }

        auto error = ranges::max(
          views::zip(
            solutions.at(i).camera_motions | views::values,
            solutions.at(j).camera_motions | views::values) |
          views::transform([](auto const& pair) {
            auto const& [motion_1, motion_2] = pair;
            auto const& p1 = motion_1.translation;
            auto const& p2 = motion_2.translation;
            return (p2 - p1).norm();
          }));
        return error < duplicate_max_position_error;
      }) |
      views::transform([](auto pair) {
        auto [i, j] = pair;
        return std::set<int> {i, j};
      }) |
      ranges::to_vector);
    auto duplicate_solution_index_sets = partition();

    return  //
      duplicate_solution_index_sets |
      views::filter([](auto const& set) { return !set.empty(); }) |
      views::transform([&](auto const& set) {
        auto best = ranges::max(set, std::less<int> {}, [&](auto const& i) {
          return solutions.at(i).n_inliers;
        });
        return solutions.at(best);
      }) |
      ranges::to_vector;
  }

  void VisionInitializerImpl::reportBundleAdjustmentTelemetry(
    std::set<FrameID> const& input_frames,
    std::vector<BundleAdjustmentSolution> const& solutions) {
    if (solutions.empty()) {
      _telemetry->onVisionFailure({
        .frames = input_frames,
        .reason = InitializerTelemetry::BUNDLE_ADJUSTMENT_FAILED,
      });
      return;
    }

    for (auto const& solution : solutions) {
      _telemetry->onBundleAdjustmentSuccess({
        .camera_motions = solution.camera_motions,
        .landmarks = solution.landmarks,
      });
    }

    auto sanity_telemetry =  //
      solutions | views::transform([](auto const& solution) {
        return InitializerTelemetry::BundleAdjustmentSanity {
          .acceptable = solution.acceptable,
          .inlier_ratio = solution.measurement_inlier_ratio,
          .final_cost_significant_probability =
            solution.solution_significant_probability,
        };
      }) |
      ranges::to_vector;

    _telemetry->onBundleAdjustmentSanity({
      .frames = input_frames,
      .candidates_sanity = sanity_telemetry,
    });
  }

  std::vector<BundleAdjustmentSolution> VisionInitializerImpl::solve(
    MultiViewImageData const& features,
    MultiViewGyroMotionData const& gyro_motions) {
    auto tic = ::cyclops::tic();

    auto const& threshold = _config->initialization.observability_pretest;
    auto connected_tracks = filterConnectedImageTrackSequence(
      threshold, compileMotionStatistics(features));
    if ((connected_tracks.size() + 1) < threshold.min_keyframes) {
      _telemetry->onVisionFailure({
        .frames = projectKeys(features),
        .reason = InitializerTelemetry::NOT_ENOUGH_CONNECTED_IMAGE_FRAMES,
      });
      return {};
    }
    if (!detectCameraMotion(threshold, connected_tracks)) {
      _telemetry->onVisionFailure({
        .frames = projectKeys(connected_tracks),
        .reason = InitializerTelemetry::NOT_ENOUGH_MOTION_PARALLAX,
      });
      return {};
    }

    auto const& [last_motion_frame, _] = *features.rbegin();
    auto image_data_filtered =
      views::concat(
        connected_tracks | views::keys, views::single(last_motion_frame)) |
      views::transform([&](auto frame_id) {
        return std::make_pair(frame_id, features.at(frame_id));
      }) |
      ranges::to<MultiViewImageData>;

    auto multiview_solutions =
      _multiview_solver->solve(image_data_filtered, gyro_motions);
    if (multiview_solutions.empty())
      return {};

    __logger__->debug(
      "Obtained {} multi-view solutions", multiview_solutions.size());

    auto maybe_bundle_adjustments =
      multiview_solutions | views::transform([&](auto const& solution) {
        return _bundle_adjustment_solver->solve(
          solution, image_data_filtered, gyro_motions);
      }) |
      ranges::to_vector;

    auto successful_bundle_adjustments = filterDuplicateSolutions(
      maybe_bundle_adjustments |
      views::filter([](auto const& maybe_x) { return maybe_x.has_value(); }) |
      views::transform([](auto const& maybe_x) { return maybe_x.value(); }) |
      ranges::to_vector);

    reportBundleAdjustmentTelemetry(
      projectKeys(image_data_filtered), successful_bundle_adjustments);

    __logger__->debug("Vision bootstrap time: {}[s]", toc(tic));
    return successful_bundle_adjustments;
  }

  std::unique_ptr<VisionInitializer> VisionInitializer::Create(
    std::shared_ptr<CyclopsConfig const> config,
    std::shared_ptr<std::mt19937> rgen,
    std::shared_ptr<InitializerTelemetry> telemetry) {
    return std::make_unique<VisionInitializerImpl>(
      MultiviewVisionGeometrySolver::Create(config, rgen, telemetry),  //
      BundleAdjustmentSolver::Create(config), config, telemetry);
  }
}  // namespace cyclops::initializer
