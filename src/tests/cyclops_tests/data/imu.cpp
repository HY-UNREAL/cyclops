#include "cyclops_tests/data/imu.hpp"
#include "cyclops_tests/random.hpp"
#include "cyclops_tests/range.ipp"
#include "cyclops_tests/signal.ipp"

#include "cyclops/details/measurement/preintegration.hpp"
#include "cyclops/details/config.hpp"

namespace cyclops {
  using std::map;
  using std::vector;

  using Eigen::Vector3d;

  using measurement::ImuMotion;
  using measurement::ImuMotions;
  using measurement::ImuNoise;
  using measurement::ImuPreintegration;

  namespace views = ranges::views;

  static auto makeImuSignal(PoseSignal const& pose_signal) {
    auto a = numericSecondDerivative(pose_signal.position);
    auto q = pose_signal.orientation;

    auto a_b = [a, q](Timestamp t) {
      auto g = Vector3d(0, 0, 9.81);
      return (q(t).inverse() * (a(t) + g)).eval();
    };
    auto w_b = numericDerivative(q);
    return std::make_tuple(a_b, w_b);
  }

  template <typename imu_data_gen_t>
  static ImuMockupSequence makeImuMockup(
    vector<Timestamp> const& timestamps, imu_data_gen_t&& gen) {
    if (timestamps.empty())
      return {};

    ImuMockupSequence result;

    auto const t_s = timestamps.front();
    auto const t_e = timestamps.back();
    auto const dt = (t_e - t_s) / timestamps.size();
    for (auto const t : timestamps)
      result.emplace(t, gen(t, dt));
    return result;
  }

  ImuMockupSequence generateImuData(
    PoseSignal pose_signal, std::vector<Timestamp> const& timestamps,
    Vector3d const& bias_acc, Vector3d const& bias_gyr, std::mt19937& rgen,
    SensorStatistics const& noise) {
    auto signal = makeImuSignal(pose_signal);
    Vector3d b_a = bias_acc;
    Vector3d b_w = bias_gyr;

    return makeImuMockup(timestamps, [&](Timestamp t, Timestamp dt) {
      auto const& [a, w] = signal;
      auto const a_m =
        perturbate((a(t) + b_a).eval(), noise.acc_white_noise, rgen);
      auto const w_m =
        perturbate((w(t) + b_w).eval(), noise.gyr_white_noise, rgen);

      auto const data_frame = ImuMockup {
        .bias_acc = b_a,
        .bias_gyr = b_w,
        .measurement = {t, a_m, w_m},
      };
      b_a = perturbate(b_a, dt * noise.acc_random_walk, rgen);
      b_w = perturbate(b_w, dt * noise.gyr_random_walk, rgen);
      return data_frame;
    });
  }

  ImuMockupSequence generateImuData(
    PoseSignal pose_signal, vector<Timestamp> const& timestamps,
    std::mt19937& rgen, SensorStatistics const& noise) {
    return generateImuData(
      pose_signal, timestamps, Vector3d::Zero(), Vector3d::Zero(), rgen, noise);
  }

  ImuMockupSequence generateImuData(
    PoseSignal pose_signal, std::vector<Timestamp> const& timestamps,
    Vector3d const& bias_acc, Vector3d const& bias_gyr) {
    auto signal = makeImuSignal(pose_signal);

    return makeImuMockup(timestamps, [&](Timestamp t, Timestamp dt) {
      auto const& [a, w] = signal;
      auto const a_m = (a(t) + bias_acc).eval();
      auto const w_m = (w(t) + bias_gyr).eval();
      auto const data_frame = ImuMockup {
        .bias_acc = bias_acc,
        .bias_gyr = bias_gyr,
        .measurement = {t, a_m, w_m},
      };
      return data_frame;
    });
  }

  ImuMockupSequence generateImuData(
    PoseSignal pose_signal, vector<Timestamp> const& timestamps) {
    auto signal = makeImuSignal(pose_signal);
    return makeImuMockup(timestamps, [&](Timestamp t, Timestamp dt) {
      auto const& [a, w] = signal;
      return ImuMockup {
        .bias_acc = Vector3d::Zero(),
        .bias_gyr = Vector3d::Zero(),
        .measurement = {t, a(t), w(t)},
      };
    });
  }

  static std::unique_ptr<ImuPreintegration> makeImuPreintegration(
    ImuNoise const& noise, ImuMockupSequence const& imu_sequence) {
    auto samples = imu_sequence.size();
    if (samples <= 1) {
      return std::make_unique<ImuPreintegration>(
        Vector3d::Zero(), Vector3d::Zero(), noise);
    }

    auto const& [_, i0] = *imu_sequence.begin();
    auto result = std::make_unique<ImuPreintegration>(
      Vector3d::Zero(), Vector3d::Zero(), noise);
    for (auto const& [prev, curr] : views::zip(
           views::slice(imu_sequence, 0, samples - 1),
           views::slice(imu_sequence, 1, samples))) {
      auto const& [t_prev, data_prev] = prev;
      auto const& [t_curr, data_curr] = curr;
      auto const& m_prev = data_prev.measurement;
      auto const& m_curr = data_curr.measurement;

      auto dt = t_curr - t_prev;
      auto a = ((m_prev.accel + m_curr.accel) / 2).eval();
      auto w = ((m_prev.rotat + m_curr.rotat) / 2).eval();
      result->propagate(dt, a, w);
    }
    return result;
  }

  static std::unique_ptr<ImuPreintegration> makeImuPreintegration(
    ImuNoise const& noise, ImuMockupSequence const& imu_sequence,
    PoseSignal pose_signal, Timestamp t_s, Timestamp t_e) {
    auto [p, q] = pose_signal;
    auto v = numericDerivative(p);
    auto p_s = p(t_s);
    auto v_s = v(t_s);
    auto q_s = q(t_s);

    auto p_e = p(t_e);
    auto v_e = v(t_e);
    auto q_e = q(t_e);

    auto dt = t_e - t_s;
    auto g = Vector3d(0, 0, 9.81);

    auto y_q = q_s.conjugate() * q_e;
    auto y_p =
      (q_s.conjugate() * (p_e - p_s - v_s * dt + 0.5 * g * dt * dt)).eval();
    auto y_v = (q_s.conjugate() * (v_e - v_s + g * dt)).eval();

    auto data = makeImuPreintegration(noise, imu_sequence);
    data->rotation_delta = y_q;
    data->position_delta = y_p;
    data->velocity_delta = y_v;
    data->time_delta = dt;
    return data;
  }

  static auto makeImuTimestamps(Timestamp t_s, Timestamp t_e) {
    auto timedelta = t_e - t_s;
    auto samples = std::max(20, static_cast<int>(timedelta * 200));
    return linspace(t_s, t_e, samples) | ranges::to_vector;
  }

  std::unique_ptr<ImuPreintegration> makeImuPreintegration(
    std::mt19937& rgen, SensorStatistics const& noise, Vector3d const& bias_acc,
    Vector3d const& bias_gyr, PoseSignal pose_signal, Timestamp t_s,
    Timestamp t_e) {
    auto imu_sequence = generateImuData(
      pose_signal, makeImuTimestamps(t_s, t_e), bias_acc, bias_gyr, rgen,
      noise);
    return makeImuPreintegration(
      ImuNoise {
        .acc_white_noise = noise.acc_white_noise,
        .gyr_white_noise = noise.gyr_white_noise,
      },
      imu_sequence);
  }

  std::unique_ptr<ImuPreintegration> makeImuPreintegration(
    std::mt19937& rgen, SensorStatistics const& noise, PoseSignal pose_signal,
    Timestamp t_s, Timestamp t_e) {
    auto imu_sequence =
      generateImuData(pose_signal, makeImuTimestamps(t_s, t_e), rgen, noise);
    return makeImuPreintegration(
      ImuNoise {
        .acc_white_noise = noise.acc_white_noise,
        .gyr_white_noise = noise.gyr_white_noise,
      },
      imu_sequence);
  }

  std::unique_ptr<ImuPreintegration> makeImuPreintegration(
    Vector3d const& bias_acc, Vector3d const& bias_gyr, PoseSignal pose_signal,
    Timestamp t_s, Timestamp t_e) {
    auto imu_sequence = generateImuData(
      pose_signal, makeImuTimestamps(t_s, t_e), bias_acc, bias_gyr);
    return makeImuPreintegration(ImuNoise {1e-3, 1e-3}, imu_sequence);
  }

  std::unique_ptr<ImuPreintegration> makeImuPreintegration(
    SensorStatistics const& noise, PoseSignal pose_signal, Timestamp t_s,
    Timestamp t_e) {
    auto imu_noise = ImuNoise {
      .acc_white_noise = noise.acc_white_noise,
      .gyr_white_noise = noise.gyr_white_noise,
    };
    return makeImuPreintegration(
      imu_noise, generateImuData(pose_signal, makeImuTimestamps(t_s, t_e)),
      pose_signal, t_s, t_e);
  }

  std::unique_ptr<ImuPreintegration> makeImuPreintegration(
    PoseSignal pose_signal, Timestamp t_s, Timestamp t_e) {
    auto imu_sequence =
      generateImuData(pose_signal, makeImuTimestamps(t_s, t_e));
    auto noise = ImuNoise {1e-3, 1e-3};
    return makeImuPreintegration(noise, imu_sequence, pose_signal, t_s, t_e);
  }

  template <typename preintegrator_t>
  static auto makeImuMotions(
    map<FrameID, Timestamp> const& frames,
    preintegrator_t const& preintegrator) {
    auto n = frames.size();
    auto prevs = views::slice(frames, 0, n - 1);
    auto currs = views::slice(frames, 1, n);

    auto transform = views::transform([&](auto const& frame_pair) {
      auto const& [s, e] = frame_pair;
      auto const& [id_s, t_s] = s;
      auto const& [id_e, t_e] = e;
      return ImuMotion {
        .from = id_s,
        .to = id_e,
        .data = preintegrator(t_s, t_e),
      };
    });
    return views::zip(prevs, currs) | transform | ranges::to<ImuMotions>;
  }

  ImuMotions makeImuMotions(
    PoseSignal pose_signal, map<FrameID, Timestamp> const& frames) {
    return makeImuMotions(frames, [&](auto t_s, auto t_e) {
      return makeImuPreintegration(pose_signal, t_s, t_e);
    });
  }

  ImuMotions makeImuMotions(
    SensorStatistics const& noise, PoseSignal pose_signal,
    map<FrameID, Timestamp> const& frames) {
    return makeImuMotions(frames, [&](auto t_s, auto t_e) {
      return makeImuPreintegration(noise, pose_signal, t_s, t_e);
    });
  }

  ImuMotions makeImuMotions(
    Vector3d const& b_a, Vector3d const& b_w, PoseSignal pose_signal,
    map<FrameID, Timestamp> const& frames) {
    return makeImuMotions(frames, [&](auto t_s, auto t_e) {
      return makeImuPreintegration(b_a, b_w, pose_signal, t_s, t_e);
    });
  }

  ImuMotions makeImuMotions(
    std::mt19937& rgen, SensorStatistics const& noise, PoseSignal pose_signal,
    map<FrameID, Timestamp> const& frames) {
    return makeImuMotions(frames, [&](auto t_s, auto t_e) {
      return makeImuPreintegration(rgen, noise, pose_signal, t_s, t_e);
    });
  }

  ImuMotions makeImuMotions(
    std::mt19937& rgen, SensorStatistics const& noise, Vector3d const& bias_acc,
    Vector3d const& bias_gyr, PoseSignal pose_signal,
    std::map<FrameID, Timestamp> const& frames) {
    return makeImuMotions(frames, [&](auto t_s, auto t_e) {
      return makeImuPreintegration(
        rgen, noise, bias_acc, bias_gyr, pose_signal, t_s, t_e);
    });
  }
}  // namespace cyclops
