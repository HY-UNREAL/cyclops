#pragma once

#include "cyclops/details/config.hpp"
#include "cyclops/details/logging.hpp"
#include "cyclops/details/type.hpp"

#include "cyclops/details/telemetry/initializer.hpp"
#include "cyclops/details/telemetry/keyframe.hpp"
#include "cyclops/details/telemetry/optimizer.hpp"

#include <map>
#include <memory>
#include <optional>
#include <vector>

namespace cyclops {
  using telemetry::InitializerTelemetry;
  using telemetry::KeyframeTelemetry;
  using telemetry::OptimizerTelemetry;

  struct ImageUpdateHandle {
    FrameID frame_id;
    Timestamp timestamp;
  };

  struct KeyframeState {
    Timestamp timestamp;

    Eigen::Vector3d acc_bias;
    Eigen::Vector3d gyr_bias;
    ImuMotionState motion_state;
  };

  struct PropagationState {
    Timestamp timestamp;
    ImuMotionState motion_state;
  };

  struct MainArgument {
    std::shared_ptr<CyclopsConfig const> config;
    std::optional<uint32_t> seed = std::nullopt;

    std::shared_ptr<OptimizerTelemetry> optimizer_telemetry = nullptr;
    std::shared_ptr<KeyframeTelemetry> keyframe_telemetry = nullptr;
    std::shared_ptr<InitializerTelemetry> initializer_telemetry = nullptr;
  };

  struct EstimationUpdateResult {
    bool reset;
    std::vector<ImageUpdateHandle> update_handles;
  };

  class CyclopsMain {
  public:
    virtual ~CyclopsMain() = default;

    /*
     * ========================= Data thread methods ========================
     *
     * The following four methods are intended to be invoked in a "data thread",
     * a thread that spins in parallel to the optimizer thread.
     *
     * In addition, each IMU and landmark data is assumed to be updated in
     * time-aligned order. That is, if t1 and t2 are timestamps of two
     * consecutive IMU data updates, then t1 <= t2 must hold, and if s1, s2 are
     * timestamps of two consecutive landmark updates, then s1 <= s2.
     */
    virtual void enqueueLandmarkData(ImageData const& data) = 0;
    virtual void enqueueImuData(ImuData const& data) = 0;

    /*
     * Enqueue external reset request. This reset request is handled in the next
     * `updateEstimation()` call.
     */
    virtual void enqueueResetRequest() = 0;

    /*
     * Get the current IMU-rate propagated motion states.
     */
    virtual std::optional<PropagationState> propagation() const = 0;
    /* ========================= Data thread methods ======================== */

    /*
     * ====================== Optimizer thread methods ======================
     *
     * The following three methods are intended to be invoked in the optimizer
     * thread.
     */
    virtual EstimationUpdateResult updateEstimation() = 0;
    virtual LandmarkPositions mappedLandmarks() const = 0;
    virtual std::map<FrameID, KeyframeState> motions() const = 0;
    /* ====================== Optimizer thread methods ====================== */

    static std::unique_ptr<CyclopsMain> Create(MainArgument args);
  };
}  // namespace cyclops
