#include "cyclops/details/initializer/candidate.hpp"

#include "cyclops/details/initializer/vision.hpp"
#include "cyclops/details/initializer/vision_imu.hpp"

#include "cyclops/details/measurement/data_provider.hpp"
#include "cyclops/details/measurement/preintegration.hpp"
#include "cyclops/details/telemetry/initializer.hpp"

#include "cyclops/details/config.hpp"

#include <range/v3/all.hpp>

namespace cyclops::initializer {
  namespace views = ranges::views;

  using measurement::MeasurementDataProvider;

  using FrameIDs = std::set<FrameID>;
  using ImuMotionRefs = std::vector<measurement::ImuMotionRef>;

  using MultiViewFeatures =
    std::map<FrameID, std::map<LandmarkID, FeaturePoint>>;

  class InitializerCandidateSolverImpl: public InitializerCandidateSolver {
  private:
    std::unique_ptr<VisionInitializer> _vision_solver;
    std::unique_ptr<VisionImuInitializer> _imu_solver;

    std::shared_ptr<CyclopsConfig const> _config;
    std::shared_ptr<MeasurementDataProvider const> _data_provider;

    MultiViewFeatures reorderMultiviewImageObservations() const;
    ImuMotionRefs filterImageObservedIMU(FrameIDs image_frames) const;

    std::map<FrameID, GyroMotionConstraint> makeCameraRotationPriorLookup(
      ImuMotionRefs const& imu_motions) const;

    InitializerCandidatePairs::ImuMatchCandidate parseSolutionCandidate(
      int msfm_index, MSfMSolution const& msfm_solution,
      ImuMatchResult const& imu_match) const;

  public:
    InitializerCandidateSolverImpl(
      std::unique_ptr<initializer::VisionInitializer> vision_solver,
      std::unique_ptr<VisionImuInitializer> imu_solver,
      std::shared_ptr<CyclopsConfig const> config,
      std::shared_ptr<MeasurementDataProvider const> data_provider);
    ~InitializerCandidateSolverImpl();
    void reset() override;

    InitializerCandidatePairs solve() override;
  };

  InitializerCandidateSolverImpl::InitializerCandidateSolverImpl(
    std::unique_ptr<initializer::VisionInitializer> vision_solver,
    std::unique_ptr<VisionImuInitializer> imu_solver,
    std::shared_ptr<CyclopsConfig const> config,
    std::shared_ptr<MeasurementDataProvider const> data_provider)
      : _vision_solver(std::move(vision_solver)),
        _imu_solver(std::move(imu_solver)),
        _config(config),
        _data_provider(data_provider) {
  }

  InitializerCandidateSolverImpl::~InitializerCandidateSolverImpl() = default;

  void InitializerCandidateSolverImpl::reset() {
    _vision_solver->reset();
    _imu_solver->reset();
  }

  MultiViewFeatures
  InitializerCandidateSolverImpl::reorderMultiviewImageObservations() const {
    MultiViewFeatures result;

    for (auto const& [landmark_id, track] : _data_provider->tracks()) {
      for (auto const& [frame_id, feature] : track)
        result[frame_id].emplace(landmark_id, feature);
    }
    return result;
  }

  ImuMotionRefs InitializerCandidateSolverImpl::filterImageObservedIMU(
    FrameIDs image_frames) const {
    return  //
      _data_provider->imu() | views::filter([&](auto const& motion) {
        auto init_frame_exists =
          image_frames.find(motion.from) != image_frames.end();
        auto term_frame_exists =
          image_frames.find(motion.to) != image_frames.end();

        return init_frame_exists && term_frame_exists;
      }) |
      views::transform(
        [](auto const& _) -> measurement::ImuMotionRef { return _; }) |
      ranges::to_vector;
  }

  std::map<FrameID, GyroMotionConstraint>
  InitializerCandidateSolverImpl::makeCameraRotationPriorLookup(
    ImuMotionRefs const& imu_motions) const {
    return  //
      imu_motions | views::transform([&](auto const& ref) {
        auto const& data = ref.get();

        auto const& q = data.data->rotation_delta;
        auto const& G = data.data->bias_jacobian;
        auto const& q_ext = _config->extrinsics.imu_camera_transform.rotation;

        auto P = data.data->covariance.template topLeftCorner<3, 3>().eval();
        auto R_ext = q_ext.matrix().eval();

        auto G_w = G.template block<3, 3>(0, 3).eval();

        return GyroMotionConstraint {
          .init_frame_id = data.from,
          .term_frame_id = data.to,
          .value = q_ext.conjugate() * q * q_ext,
          .covariance = R_ext.transpose() * P * R_ext,
          .bias_nominal = data.data->gyrBias(),
          .bias_jacobian = R_ext.transpose() * G_w,
        };
      }) |
      views::transform([](auto const& rotation) {
        return std::make_pair(rotation.init_frame_id, rotation);
      }) |
      ranges::to<std::map<FrameID, GyroMotionConstraint>>;
  }

  InitializerCandidatePairs::ImuMatchCandidate
  InitializerCandidateSolverImpl::parseSolutionCandidate(
    int msfm_index, MSfMSolution const& msfm_solution,
    ImuMatchResult const& imu_match) const {
    auto const& solution = imu_match.solution;
    auto s = solution.scale;

    auto landmarks =  //
      msfm_solution.geometry.landmarks |
      views::transform([&](auto const& id_landmark) {
        auto [landmark_id, f] = id_landmark;
        return std::make_pair(landmark_id, (s * f).eval());
      }) |
      ranges::to<LandmarkPositions>;

    auto motions =
      views::zip(
        solution.body_velocities | views::keys,
        solution.body_orientations | views::values,
        solution.body_velocities | views::values,
        solution.sfm_positions | views::values) |
      views::transform([&](auto const& pair) {
        auto const& [frame_id, q_b, v_body, p_c] = pair;
        auto v = (q_b * v_body).eval();

        auto const& [p_bc, _] = _config->extrinsics.imu_camera_transform;
        Eigen::Vector3d p = p_c * s - q_b * p_bc;
        return std::make_pair(frame_id, ImuMotionState {q_b, p, v});
      }) |
      ranges::to<std::map<FrameID, ImuMotionState>>;

    return {
      .msfm_solution_index = msfm_index,

      .acceptance = imu_match.accept,
      .cost = solution.cost,
      .scale = solution.scale,
      .gravity = solution.gravity,

      .gyr_bias = solution.gyr_bias,
      .acc_bias = solution.acc_bias,

      .landmarks = landmarks,
      .motions = motions,
    };
  }

  InitializerCandidatePairs InitializerCandidateSolverImpl::solve() {
    auto image_data = reorderMultiviewImageObservations();
    auto image_motion_frames = image_data | views::keys | ranges::to<std::set>;
    auto imu_motions = filterImageObservedIMU(image_motion_frames);

    auto rotation_prior = makeCameraRotationPriorLookup(imu_motions);

    InitializerCandidatePairs result;
    result.msfm_solutions = _vision_solver->solve(image_data, rotation_prior);
    result.imu_match_solutions.reserve(result.msfm_solutions.size());

    auto n_msfm_solutions = result.msfm_solutions.size();
    for (int msfm_index = 0; msfm_index < n_msfm_solutions; msfm_index++) {
      auto const& msfm_solution = result.msfm_solutions.at(msfm_index);
      auto matches = _imu_solver->solve(msfm_solution, imu_motions);
      if (!matches.has_value())
        continue;

      for (auto const& match : *matches) {
        result.imu_match_solutions.push_back(
          parseSolutionCandidate(msfm_index, msfm_solution, match));
      }
    }
    return result;
  }

  std::unique_ptr<InitializerCandidateSolver>
  InitializerCandidateSolver::Create(
    std::shared_ptr<std::mt19937> rgen,
    std::shared_ptr<CyclopsConfig const> config,
    std::shared_ptr<MeasurementDataProvider const> data_provider,
    std::shared_ptr<telemetry::InitializerTelemetry> telemetry) {
    return std::make_unique<InitializerCandidateSolverImpl>(
      initializer::VisionInitializer::Create(config, rgen, telemetry),
      VisionImuInitializer::Create(config, telemetry), config, data_provider);
  }
}  // namespace cyclops::initializer
