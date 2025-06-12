#include "cyclops/details/estimation/optimizer_guess.hpp"
#include "cyclops/details/estimation/state/accessor_read.hpp"

#include "cyclops/details/initializer/initializer.hpp"
#include "cyclops/details/measurement/data_provider.hpp"
#include "cyclops/details/measurement/preintegration.hpp"
#include "cyclops/details/measurement/type.hpp"
#include "cyclops/details/utils/debug.hpp"
#include "cyclops/details/utils/math.hpp"
#include "cyclops/details/utils/vision.hpp"

#include "cyclops/details/config.hpp"
#include "cyclops/details/logging.hpp"

#include <spdlog/spdlog.h>

#include <range/v3/all.hpp>

namespace cyclops::estimation {
  namespace views = ranges::views;
  namespace actions = ranges::actions;

  using std::map;

  using Eigen::Quaterniond;
  using Eigen::Vector3d;

  struct ImuStatePropagation {
    ImuMotionState motion_state;
    Vector3d b_a;
    Vector3d b_w;
  };

  using MotionStatePropagationLookup = map<FrameID, ImuStatePropagation>;
  using CameraPoseLookup = map<FrameID, RotationPositionPair>;

  class OptimizerSolutionGuessPredictorImpl:
      public OptimizerSolutionGuessPredictor {
  private:
    std::unique_ptr<initializer::InitializerMain> _initializer;
    std::shared_ptr<CyclopsConfig const> _config;
    std::shared_ptr<StateVariableReadAccessor const> _state;
    std::shared_ptr<measurement::MeasurementDataProvider> _data_provider;

    std::optional<Solution> bootstrap();

    std::optional<ImuStatePropagation> propagateMotionState(
      measurement::ImuMotion const& imu_motion) const;
    MotionStatePropagationLookup propagateUnknownMotions() const;
    CameraPoseLookup makeCameraPoseLookup(
      MotionStatePropagationLookup const& motion_states) const;

    measurement::FeatureTracks collectUnknownSolvableLandmarks(
      CameraPoseLookup const& pose_lookup) const;

  public:
    OptimizerSolutionGuessPredictorImpl(
      std::unique_ptr<initializer::InitializerMain> initializer,
      std::shared_ptr<CyclopsConfig const> config,
      std::shared_ptr<StateVariableReadAccessor const> state,
      std::shared_ptr<measurement::MeasurementDataProvider> data_provider);
    ~OptimizerSolutionGuessPredictorImpl();
    void reset() override;

    std::optional<Solution> solve() override;
  };

  OptimizerSolutionGuessPredictorImpl::OptimizerSolutionGuessPredictorImpl(
    std::unique_ptr<initializer::InitializerMain> initializer,
    std::shared_ptr<CyclopsConfig const> config,
    std::shared_ptr<StateVariableReadAccessor const> state,
    std::shared_ptr<measurement::MeasurementDataProvider> data_provider)
      : _initializer(std::move(initializer)),
        _config(std::move(config)),
        _state(std::move(state)),
        _data_provider(std::move(data_provider)) {
  }

  OptimizerSolutionGuessPredictorImpl::~OptimizerSolutionGuessPredictorImpl() =
    default;

  void OptimizerSolutionGuessPredictorImpl::reset() {
    _initializer->reset();
    _data_provider->reset();
  }

  static auto makeFrameStateBlock(
    FrameID frame_id, Quaterniond const& q, Vector3d const& p,
    Vector3d const& v, Vector3d const& b_a, Vector3d const& b_w) {
    MotionFrameParameterBlock result;
    Eigen::Map<Quaterniond>(result.data()) = q;
    Eigen::Map<Vector3d>(result.data() + 4) = p;
    Eigen::Map<Vector3d>(result.data() + 7) = v;
    Eigen::Map<Vector3d>(result.data() + 10) = b_a;
    Eigen::Map<Vector3d>(result.data() + 13) = b_w;
    return std::make_pair(frame_id, result);
  }

  static auto makeFrameStateBlock(
    Vector3d const& b_a, Vector3d const& b_w, FrameID frame_id,
    ImuMotionState const& motion) {
    return makeFrameStateBlock(
      frame_id, motion.orientation, motion.position, motion.velocity, b_a, b_w);
  }

  static auto makeLandmarkStateBlock(
    LandmarkPositions::value_type const& id_position) {
    auto [id, position] = id_position;
    LandmarkParameterBlock result;
    Eigen::Map<Vector3d>(result.data()) = position;

    return std::make_pair(id, result);
  }

  std::optional<OptimizerSolutionGuessPredictor::Solution>
  OptimizerSolutionGuessPredictorImpl::bootstrap() {
    auto tic = ::cyclops::tic();
    auto maybe_bootstrap = _initializer->solve();
    __logger__->info("VIO bootstrap time: {}[s]", toc(tic));

    if (!maybe_bootstrap.has_value()) {
      __logger__->info("Failed VIO bootstrap; skipping state guess");
      return std::nullopt;
    }

    auto const& b_a = maybe_bootstrap->acc_bias;
    auto const& b_w = maybe_bootstrap->gyr_bias;

    auto motion_blocks =  //
      maybe_bootstrap->motions | views::transform([&](auto const& _) {
        auto [frame_id, motion] = _;
        return makeFrameStateBlock(b_a, b_w, frame_id, motion);
      }) |
      ranges::to<MotionFrameParameterBlocks>;

    auto landmark_blocks = maybe_bootstrap->landmarks |
      views::transform(makeLandmarkStateBlock) |
      ranges::to<LandmarkParameterBlocks>;

    return Solution {motion_blocks, landmark_blocks};
  }

  std::optional<ImuStatePropagation>
  OptimizerSolutionGuessPredictorImpl::propagateMotionState(
    measurement::ImuMotion const& imu_motion) const {
    auto const maybe_frame = _state->motionFrame(imu_motion.from);
    if (!maybe_frame)
      return std::nullopt;

    auto const& frame = maybe_frame->get();
    auto [q, p, v] = estimation::getMotionState(frame);

    auto const& dt = imu_motion.data->time_delta;
    auto const& alpha = imu_motion.data->position_delta;
    auto const& beta = imu_motion.data->velocity_delta;
    auto const& gamma = imu_motion.data->rotation_delta;

    auto g = Vector3d(0, 0, _config->gravity_norm);

    auto motion_state = ImuMotionState {
      .orientation = q * gamma,
      .position = p + v * dt - 0.5 * g * dt * dt + q * alpha,
      .velocity = v - g * dt + q * beta,
    };
    return ImuStatePropagation {
      .motion_state = motion_state,
      .b_a = estimation::getAccBias(frame),
      .b_w = estimation::getGyrBias(frame),
    };
  }

  MotionStatePropagationLookup
  OptimizerSolutionGuessPredictorImpl::propagateUnknownMotions() const {
    map<FrameID, ImuStatePropagation> result;
    for (auto const& motion : _data_provider->imu()) {
      if (_state->motionFrame(motion.to))  // if already initialized, skip.
        continue;

      auto maybe_propagation = propagateMotionState(motion);
      if (!maybe_propagation)
        continue;
      result.emplace(motion.to, *maybe_propagation);
    }
    return result;
  }

  CameraPoseLookup OptimizerSolutionGuessPredictorImpl::makeCameraPoseLookup(
    MotionStatePropagationLookup const& propagations) const {
    auto const& extrinsic = _config->extrinsics.imu_camera_transform;

    std::set<FrameID> frames;
    for (auto const& track : _data_provider->tracks() | views::values)
      actions::insert(frames, track | views::keys);

    map<FrameID, RotationPositionPair> result;
    for (auto frame_id : frames) {
      auto i = propagations.find(frame_id);
      if (i != propagations.end()) {
        auto const& [_, state] = *i;
        auto const& x = state.motion_state;
        auto [p, q] = compose({x.position, x.orientation}, extrinsic);
        result.emplace(frame_id, RotationPositionPair {q.matrix(), p});
        continue;
      }

      auto maybe_frame = _state->motionFrame(frame_id);
      if (!maybe_frame)
        continue;

      auto const& frame = maybe_frame->get();
      auto [p, q] = compose(estimation::getSE3Transform(frame), extrinsic);
      result.emplace(frame_id, RotationPositionPair {q.matrix(), p});
    }
    return result;
  }

  measurement::FeatureTracks
  OptimizerSolutionGuessPredictorImpl::collectUnknownSolvableLandmarks(
    CameraPoseLookup const& pose_lookup) const {
    return  //
      _data_provider->tracks() | views::transform([&](auto const& id_track) {
        auto const& [landmark_id, track] = id_track;
        auto track_filtered =  //
          track | views::filter([&](auto const& id_observation) {
            auto const& [frame_id, _] = id_observation;
            return pose_lookup.find(frame_id) != pose_lookup.end();
          }) |
          ranges::to<measurement::FeatureTrack>;

        return std::make_pair(landmark_id, track_filtered);
      }) |
      views::filter([&](auto const& id_track) {
        auto const& [landmark_id, track] = id_track;
        auto maybe_landmark = _state->landmark(landmark_id);
        return !maybe_landmark && track.size() >= 2;
      }) |
      ranges::to<measurement::FeatureTracks>;
  }

  static bool determineLandmarkOutlier(
    measurement::FeatureTrack const& track,
    CameraPoseLookup const& camera_pose_lookup, Vector3d const& f,
    double feature_min_depth) {
    for (auto const& frame_id : track | views::keys) {
      auto const& [R, p] = camera_pose_lookup.at(frame_id);
      Vector3d z = R.transpose() * (f - p);

      if (z.z() <= feature_min_depth)
        return true;
    }
    return false;
  }

  static LandmarkPositions triangulateLandmarks(
    CameraPoseLookup const& camera_pose_lookup,
    measurement::FeatureTracks const& tracks, double feature_min_depth) {
    LandmarkPositions result;
    for (auto const& [landmark_id, track] : tracks) {
      auto maybe_point = triangulatePoint(track, camera_pose_lookup);
      if (!maybe_point)
        continue;
      if (determineLandmarkOutlier(
            track, camera_pose_lookup, *maybe_point, feature_min_depth))
        continue;
      result.emplace(landmark_id, *maybe_point);
    }
    return result;
  }

  std::optional<OptimizerSolutionGuessPredictor::Solution>
  OptimizerSolutionGuessPredictorImpl::solve() {
    if (_state->motionFrames().empty())
      return bootstrap();
    auto propagated_motions = propagateUnknownMotions();
    auto camera_pose_lookup = makeCameraPoseLookup(propagated_motions);

    auto unknown_features = collectUnknownSolvableLandmarks(camera_pose_lookup);
    auto landmark_positions = triangulateLandmarks(
      camera_pose_lookup, unknown_features,
      _config->estimation.landmark_acceptance.inlier_min_depth);

    auto motion_blocks =  //
      propagated_motions | views::transform([](auto const& id_propagation) {
        auto const& [frame_id, propagated_state] = id_propagation;
        auto const& [motion_state, b_a, b_w] = propagated_state;
        auto const& [q, p, v] = motion_state;
        return makeFrameStateBlock(frame_id, q, p, v, b_a, b_w);
      }) |
      ranges::to<MotionFrameParameterBlocks>;
    auto landmark_blocks = landmark_positions |
      views::transform(makeLandmarkStateBlock) |
      ranges::to<LandmarkParameterBlocks>;

    return Solution {motion_blocks, landmark_blocks};
  }

  std::unique_ptr<OptimizerSolutionGuessPredictor>
  OptimizerSolutionGuessPredictor::Create(
    std::unique_ptr<initializer::InitializerMain> initializer,
    std::shared_ptr<CyclopsConfig const> config,
    std::shared_ptr<StateVariableReadAccessor const> state,
    std::shared_ptr<measurement::MeasurementDataProvider> data_provider) {
    return std::make_unique<OptimizerSolutionGuessPredictorImpl>(
      std::move(initializer), config, state, data_provider);
  }
}  // namespace cyclops::estimation
