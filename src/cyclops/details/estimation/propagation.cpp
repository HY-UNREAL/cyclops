#include "cyclops/details/estimation/propagation.hpp"

#include "cyclops/details/measurement/preintegration.hpp"
#include "cyclops/details/config.hpp"

#include <cmath>
#include <functional>
#include <iterator>

namespace cyclops::estimation {
  using Eigen::Vector3d;

  using measurement::ImuNoise;
  using measurement::ImuPreintegration;

  using TimestampedMotionState = std::tuple<Timestamp, ImuMotionState>;

  class ImuPropagationUpdateHandlerImpl: public ImuPropagationUpdateHandler {
  private:
    std::shared_ptr<CyclopsConfig const> _config;

    struct PropagationState {
      Timestamp timestamp;

      ImuMotionState motion_state;
      Vector3d bias_acc;
      Vector3d bias_gyr;

      ImuPreintegration integrator;

      PropagationState(
        Timestamp timestamp, ImuMotionState const& motion_state,
        Vector3d const& b_a, Vector3d const& b_w, ImuNoise const& noise);
    };

    std::unique_ptr<PropagationState> _propagation_state;
    std::map<Timestamp, ImuData> _imu_queue;

    void propagateIMU(ImuData const& imu_prev, ImuData const& imu_next);

    std::map<Timestamp, ImuData>::const_iterator findIMUQueuePointRightBefore(
      Timestamp time) const;

  public:
    explicit ImuPropagationUpdateHandlerImpl(
      std::shared_ptr<CyclopsConfig const> config);
    ~ImuPropagationUpdateHandlerImpl();
    void reset() override;

    void updateOptimization(
      Timestamp last_timestamp,
      MotionFrameParameterBlock const& last_state) override;
    void updateImuData(ImuData const& data) override;

    std::optional<TimestampedMotionState> get() const override;
  };

  ImuPropagationUpdateHandlerImpl::PropagationState::PropagationState(
    Timestamp timestamp, ImuMotionState const& motion_state,
    Vector3d const& b_a, Vector3d const& b_w, ImuNoise const& noise)
      : timestamp(timestamp),
        motion_state(motion_state),
        bias_acc(b_a),
        bias_gyr(b_w),
        integrator(b_a, b_w, noise) {
  }

  std::map<Timestamp, ImuData>::const_iterator
  ImuPropagationUpdateHandlerImpl::findIMUQueuePointRightBefore(
    Timestamp time) const {
    auto i = _imu_queue.upper_bound(time);
    if (i == _imu_queue.begin())
      return _imu_queue.end();
    return std::prev(i);
  }

  static Vector3d interpolate(
    Timestamp t_eval, Timestamp t_init, Vector3d const& v_init,
    Timestamp t_term, Vector3d const& v_term) {
    if (t_term - t_init < 1e-6) {
      if (std::abs(t_term - t_eval) < std::abs(t_init - t_eval))
        return v_term;
      return v_init;
    }

    Vector3d delta = v_term - v_init;
    auto n = t_eval - t_init;
    auto d = t_term - t_init;
    return v_init + std::min(1., std::max(0., n / d)) * delta;
  }

  void ImuPropagationUpdateHandlerImpl::propagateIMU(
    ImuData const& imu_prev, ImuData const& imu_next) {
    auto const& t_curr = _propagation_state->timestamp;
    auto const& [t_prev, a_prev, w_prev] = imu_prev;
    auto const& [t_next, a_next, w_next] = imu_next;

    auto dt = t_next - t_curr;
    if (dt <= 0)
      return;

    auto a_curr = interpolate(t_curr, t_prev, a_prev, t_next, a_next);
    auto w_curr = interpolate(t_curr, t_prev, w_prev, t_next, w_next);

    Vector3d a_hat = (a_curr + a_next) / 2;
    Vector3d w_hat = (w_curr + w_next) / 2;

    _propagation_state->integrator.propagate(dt, a_hat, w_hat);
    _propagation_state->timestamp = t_next;
  }

  ImuPropagationUpdateHandlerImpl::ImuPropagationUpdateHandlerImpl(
    std::shared_ptr<CyclopsConfig const> config)
      : _config(config) {
  }

  ImuPropagationUpdateHandlerImpl::~ImuPropagationUpdateHandlerImpl() = default;

  void ImuPropagationUpdateHandlerImpl::updateOptimization(
    Timestamp timestamp, MotionFrameParameterBlock const& state_block) {
    auto motion_state = estimation::getMotionState(state_block);
    auto bias_acc = estimation::getAccBias(state_block);
    auto bias_gyr = estimation::getGyrBias(state_block);

    _propagation_state = std::make_unique<PropagationState>(
      timestamp, motion_state, bias_acc, bias_gyr,
      ImuNoise {
        .acc_white_noise = _config->noise.acc_white_noise,
        .gyr_white_noise = _config->noise.gyr_white_noise,
      });

    auto i = _imu_queue.upper_bound(timestamp);
    if (i != _imu_queue.begin())
      _imu_queue.erase(_imu_queue.begin(), std::prev(i));
    if (i == _imu_queue.end())
      return;

    for (; std::next(i) != _imu_queue.end(); i++) {
      auto const& [_1, imu_prev] = *i;
      auto const& [_2, imu_next] = *std::next(i);
      propagateIMU(imu_prev, imu_next);
    }
  }

  void ImuPropagationUpdateHandlerImpl::updateImuData(ImuData const& imu_next) {
    _imu_queue.emplace_hint(_imu_queue.end(), imu_next.timestamp, imu_next);
    if (_propagation_state == nullptr)
      return;
    auto const& t_curr = _propagation_state->timestamp;

    auto i_prev = findIMUQueuePointRightBefore(t_curr);
    if (i_prev == _imu_queue.end())
      return;
    auto const& imu_prev = i_prev->second;

    propagateIMU(imu_prev, imu_next);
  }

  std::optional<TimestampedMotionState> ImuPropagationUpdateHandlerImpl::get()
    const {
    if (!_propagation_state)
      return std::nullopt;

    auto g = Vector3d(0, 0, _config->gravity_norm);
    auto const& y_q = _propagation_state->integrator.rotation_delta;
    auto const& y_p = _propagation_state->integrator.position_delta;
    auto const& y_v = _propagation_state->integrator.velocity_delta;
    auto const& dt = _propagation_state->integrator.time_delta;

    auto const& [q, p, v] = _propagation_state->motion_state;
    auto propagated_state = ImuMotionState {
      .orientation = q * y_q,
      .position = p + v * dt - 0.5 * g * dt * dt + q * y_p,
      .velocity = v - g * dt + q * y_v,
    };
    return std::make_tuple(_propagation_state->timestamp, propagated_state);
  }

  void ImuPropagationUpdateHandlerImpl::reset() {
    _propagation_state = nullptr;
    _imu_queue.clear();
  }

  std::unique_ptr<ImuPropagationUpdateHandler>
  ImuPropagationUpdateHandler::Create(
    std::shared_ptr<CyclopsConfig const> config) {
    return std::make_unique<ImuPropagationUpdateHandlerImpl>(config);
  }
}  // namespace cyclops::estimation
