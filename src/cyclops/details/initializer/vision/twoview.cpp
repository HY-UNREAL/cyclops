#include "cyclops/details/initializer/vision/twoview.hpp"

#include "cyclops/details/initializer/vision/epipolar.hpp"
#include "cyclops/details/initializer/vision/homography.hpp"
#include "cyclops/details/initializer/vision/hypothesis.hpp"
#include "cyclops/details/utils/vision.hpp"
#include "cyclops/details/utils/math.hpp"

#include "cyclops/details/config.hpp"
#include "cyclops/details/logging.hpp"

#include <spdlog/fmt/ostr.h>
#include <spdlog/spdlog.h>

#include <range/v3/all.hpp>

namespace cyclops::initializer {
  namespace views = ranges::views;

  using std::set;
  using std::vector;

  using Eigen::Matrix2d;
  using Eigen::Matrix3d;

  using TwoViewData = std::map<LandmarkID, TwoViewFeaturePair>;
  using GeometryModel = TwoViewGeometrySolverResult::GeometryModel;

  static vector<set<LandmarkID>> makeRansacBatch(
    int size, set<LandmarkID> const& features, std::mt19937& rgen) {
    if (features.size() < 8)
      return {};

    return  //
      views::ints(0, size) | views::transform([&](auto _) {
        return features | views::sample(8, rgen) | ranges::to<set>;
      }) |
      ranges::to_vector;
  }

  class TwoViewVisionGeometrySolverImpl: public TwoViewVisionGeometrySolver {
  private:
    std::unique_ptr<TwoViewMotionHypothesisSelector> _motion_selector;

    std::shared_ptr<CyclopsConfig const> _config;
    std::shared_ptr<std::mt19937> _rgen;

    std::optional<GeometryModel> selectGeometryModel(
      EpipolarAnalysis const& epipolar,
      HomographyAnalysis const& homography) const;

    vector<std::tuple<GeometryModel, TwoViewGeometry>> solveHomography(
      HomographyAnalysis const& homography, TwoViewData const& features,
      TwoViewImuRotationData const& rotation_prior);
    vector<std::tuple<GeometryModel, TwoViewGeometry>> solveEpipolar(
      EpipolarAnalysis const& epipolar, TwoViewData const& features,
      TwoViewImuRotationData const& rotation_prior);

  public:
    TwoViewVisionGeometrySolverImpl(
      std::unique_ptr<TwoViewMotionHypothesisSelector> motion_selector,
      std::shared_ptr<CyclopsConfig const> config,
      std::shared_ptr<std::mt19937> rgen);
    ~TwoViewVisionGeometrySolverImpl();
    void reset() override;

    std::optional<TwoViewGeometrySolverResult> solve(
      TwoViewCorrespondenceData const& correspondence) override;
  };

  TwoViewVisionGeometrySolverImpl::TwoViewVisionGeometrySolverImpl(
    std::unique_ptr<TwoViewMotionHypothesisSelector> motion_selector,
    std::shared_ptr<CyclopsConfig const> config,
    std::shared_ptr<std::mt19937> rgen)
      : _motion_selector(std::move(motion_selector)),
        _config(config),
        _rgen(rgen) {
  }

  TwoViewVisionGeometrySolverImpl::~TwoViewVisionGeometrySolverImpl() = default;

  void TwoViewVisionGeometrySolverImpl::reset() {
    _motion_selector->reset();
  }

  std::optional<GeometryModel>
  TwoViewVisionGeometrySolverImpl::selectGeometryModel(
    EpipolarAnalysis const& epipolar,
    HomographyAnalysis const& homography) const {
    auto S_H = homography.expected_inliers;
    auto S_E = epipolar.expected_inliers;

    __logger__->trace("Selecting two-view vision initialization model.");
    __logger__->trace("Homography expected number of inliers: {}", S_H);
    __logger__->trace("Epipolar expected number of inliers: {}", S_E);

    if (S_H <= 0 && S_E <= 0) {
      // both of them failed. abort
      __logger__->debug(
        "Initialization model selection failed; S_H: {}, S_E: {}", S_H, S_E);
      return std::nullopt;
    }

    auto R_H = S_H / (S_H + S_E);
    auto const& selection_config =
      _config->initialization.vision.two_view.model_selection;
    __logger__->debug("Model selection score: {}", R_H);

    if (R_H > selection_config.homography_selection_score_threshold) {
      __logger__->debug("Selecting homography model");
      return GeometryModel::HOMOGRAPHY;
    } else if (R_H < selection_config.epipolar_selection_score_threshold) {
      __logger__->debug("Selecting epipolar model");
      return GeometryModel::EPIPOLAR;
    }

    __logger__->debug("Selecting both homography and epipolar model");
    return GeometryModel::BOTH;
  }

  vector<std::tuple<GeometryModel, TwoViewGeometry>>
  TwoViewVisionGeometrySolverImpl::solveHomography(
    HomographyAnalysis const& homography, TwoViewData const& data,
    TwoViewImuRotationData const& rotation_prior) {
    auto const& [expected_inliers, H, inliers] = homography;
    if (expected_inliers <= 0)
      return {};

    auto hypotheses = solveHomographyMotionHypothesis(H);
    if (hypotheses.size() != 8)
      return {};

    auto motions = _motion_selector->selectPossibleMotions(
      hypotheses, data, inliers, rotation_prior);
    return  //
      motions | views::transform([](auto const& motion) {
        return std::make_tuple(GeometryModel::HOMOGRAPHY, motion);
      }) |
      ranges::to_vector;
  }

  vector<std::tuple<GeometryModel, TwoViewGeometry>>
  TwoViewVisionGeometrySolverImpl::solveEpipolar(
    EpipolarAnalysis const& epipolar, TwoViewData const& data,
    TwoViewImuRotationData const& rotation_prior) {
    auto const& [expected_inliers, E, inliers] = epipolar;
    if (expected_inliers <= 0)
      return {};

    auto hypotheses = solveEpipolarMotionHypothesis(E);
    if (hypotheses.size() != 4)
      return {};

    auto motions = _motion_selector->selectPossibleMotions(
      hypotheses, data, inliers, rotation_prior);
    return  //
      motions | views::transform([](auto const& motion) {
        return std::make_tuple(GeometryModel::EPIPOLAR, motion);
      }) |
      ranges::to_vector;
  }

  static bool isTwoViewGeometryAcceptable(
    vector<std::tuple<GeometryModel, TwoViewGeometry>> const& candidates) {
    if (candidates.empty())
      return false;

    return ranges::any_of(candidates, [](auto const& pair) {
      auto const& [_, geometry] = pair;
      return geometry.acceptable;
    });
  }

  std::optional<TwoViewGeometrySolverResult>
  TwoViewVisionGeometrySolverImpl::solve(
    TwoViewCorrespondenceData const& correspondence) {
    auto tic = std::chrono::steady_clock::now();
    auto const& vision_config = _config->initialization.vision;
    auto const& features = correspondence.features;
    auto const& rotation_prior = correspondence.rotation_prior;

    __logger__->trace("Solving two-view structure for vision initialization.");
    __logger__->trace("Number of common features: {}", features.size());

    auto rotation_vector = so3Logmap(rotation_prior.value);
    __logger__->debug(
      "Two-view rotation prior: {}", rotation_vector.transpose());

    auto landmark_ids = features | views::keys | ranges::to<set>;
    auto ransac_batch = makeRansacBatch(
      vision_config.two_view.model_selection.ransac_batch_size, landmark_ids,
      *_rgen);
    auto homography = analyzeTwoViewHomography(
      vision_config.feature_point_isotropic_noise, ransac_batch, features);
    auto epipolar = analyzeTwoViewEpipolar(
      vision_config.feature_point_isotropic_noise, ransac_batch, features);

    auto model = selectGeometryModel(epipolar, homography);
    auto toc = std::chrono::steady_clock::now();
    auto dt = std::chrono::duration<double>(toc - tic).count();

    __logger__->debug(
      "Two-view geometry model selection complete. duration: {}", dt);

    if (!model.has_value())  // Selection failure.
      return std::nullopt;

    auto result =
      [&](auto initial_model, auto final_model, auto const& solutions) {
        return TwoViewGeometrySolverResult {
          .initial_selected_model = initial_model,
          .final_selected_model = final_model,
          .homography_expected_inliers = homography.expected_inliers,
          .epipolar_expected_inliers = epipolar.expected_inliers,
          .candidates = solutions,
        };
      };

    switch (*model) {
    case GeometryModel::EPIPOLAR: {
      auto solutions = solveEpipolar(epipolar, features, rotation_prior);
      if (isTwoViewGeometryAcceptable(solutions)) {
        return result(
          GeometryModel::EPIPOLAR, GeometryModel::EPIPOLAR, solutions);
      }

      __logger__->debug(
        "Epipolar reconstruction failed. Fallback to homography model");
      return result(
        GeometryModel::EPIPOLAR, GeometryModel::HOMOGRAPHY,
        solveHomography(homography, features, rotation_prior));
    }

    case GeometryModel::HOMOGRAPHY: {
      auto solutions = solveHomography(homography, features, rotation_prior);
      if (isTwoViewGeometryAcceptable(solutions)) {
        return result(
          GeometryModel::HOMOGRAPHY, GeometryModel::HOMOGRAPHY, solutions);
      }

      __logger__->debug(
        "Homography reconstruction failed. Fallback to epipolar model");
      return result(
        GeometryModel::HOMOGRAPHY, GeometryModel::EPIPOLAR,
        solveEpipolar(epipolar, features, rotation_prior));
    }

    case GeometryModel::BOTH: {
      auto homography_solutions =
        solveHomography(homography, features, rotation_prior);
      auto epipolar_solutions =
        solveEpipolar(epipolar, features, rotation_prior);
      auto solutions = views::concat(homography_solutions, epipolar_solutions) |
        ranges::to_vector;

      return result(GeometryModel::BOTH, GeometryModel::BOTH, solutions);
    }
    }
  }

  std::unique_ptr<TwoViewVisionGeometrySolver>
  TwoViewVisionGeometrySolver::Create(
    std::shared_ptr<CyclopsConfig const> config,
    std::shared_ptr<std::mt19937> rgen) {
    return std::make_unique<TwoViewVisionGeometrySolverImpl>(
      TwoViewMotionHypothesisSelector::Create(config), config, rgen);
  }
}  // namespace cyclops::initializer
